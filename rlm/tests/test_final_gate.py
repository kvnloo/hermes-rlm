"""Actual registered Hermes tool middleware and finalizer contract regressions.

reuse-checked: turn_context._collect_pre_llm_call_context, model_tools._execute_tool,
turn_finalizer.apply_llm_output_transform and the existing host cancellation fixture.
"""
import json
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('RLM_HERMES_SOURCE'),
                                reason='requires actual Hermes hook/finalizer APIs')


@pytest.fixture
def host(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.syspath_prepend(os.environ['RLM_HERMES_SOURCE'])
    (home / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': ['rlm']},
                                                 'telemetry': {'enabled': False}}))
    shutil.copytree(Path(__file__).parents[1], home / 'plugins' / 'rlm',
                    ignore=shutil.ignore_patterns('tests', '__pycache__'))
    from hermes_cli.plugins import get_plugin_manager
    from agent.turn_context import _collect_pre_llm_call_context
    from agent.turn_finalizer import apply_llm_output_transform
    from model_tools import _execute_tool, _CallIds

    manager = get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins['rlm']
    assert loaded.enabled and not loaded.error, loaded.error
    source = tmp_path / 'source.txt'
    source.write_text('answer=42\n', encoding='utf-8')

    def begin(session='s', task='task', turn='turn-1'):
        agent = SimpleNamespace(session_id=session, model='test', platform='cli')
        _collect_pre_llm_call_context(agent, effective_task_id=task, turn_id=turn,
                                     original_user_message='question', messages=[],
                                     conversation_history=[])

    def tool(name, args, session='s', task='task', turn='turn-1', bypass=False):
        result = _execute_tool(name, args, args, _CallIds(task_id=task, session_id=session,
                               turn_id=turn, tool_call_id='call'), user_task='question',
                               enabled_tools=['rlm_load', 'rlm_exec'],
                               skip_tool_execution_middleware=bypass)
        return json.loads(result)

    def final(text='uncited root prose', session='s', turn='turn-1'):
        agent = SimpleNamespace(session_id=session, model='test', platform='cli')
        return apply_llm_output_transform(agent, text, turn_id=turn,
                                          logger=logging.getLogger('test.final_gate'))[0]

    yield SimpleNamespace(begin=begin, tool=tool, final=final, source=source,
                          plugin=loaded.module, manager=manager)
    manager.unload()


def load(host, **ids):
    host.begin(**ids)
    result = host.tool('rlm_load', {'paths': [str(host.source)]}, **ids)
    assert result.get('loaded'), result


def submit(host, **ids):
    out = host.tool('rlm_exec', {'code': "FINAL('42', citations=[cite('source.txt', 7, 9)])"}, **ids)
    assert out.get('final') == '42', out


def test_uncited_root_prose_is_withheld_by_real_finalizer(host):
    load(host)
    result = host.final('The answer is 999 without FINAL.')
    assert 'withheld' in result.lower() and 'FINAL' in result
    assert '999' not in result


def test_validated_answer_overrides_root_and_formats_verified_quotes(host):
    load(host)
    submit(host)
    result = host.final('Contradictory root says 999.')
    assert result.startswith('42\n')
    assert 'source.txt[7:9]' in result and '"42"' in result
    assert '999' not in result


def test_other_session_and_non_rlm_session_unchanged(host):
    load(host)
    submit(host)
    host.begin(session='other')
    assert host.final('ordinary answer', session='other') == 'ordinary answer'
    assert host.final('never registered', session='unknown') == 'never registered'


def test_next_turn_cannot_reuse_a_validated_final(host):
    load(host)
    submit(host)
    host.begin(turn='turn-2')
    assert 'withheld' in host.final('stale 42', turn='turn-2').lower()
    assert 'withheld' in host.final('late old turn', turn='turn-1').lower()


@pytest.mark.parametrize('code', [
    "FINAL('forged', citations=[cite('source.txt',0,999)])",
    "FINAL_VAR('missing')", "1/0",
])
def test_failed_cell_invalidates_previous_final(host, code):
    load(host)
    submit(host)
    assert host.tool('rlm_exec', {'code': code})['error']
    assert 'withheld' in host.final('forged root').lower()


def test_reload_and_session_start_clear_final(host):
    load(host)
    submit(host)
    host.tool('rlm_load', {'paths': [str(host.source)]})
    assert 'withheld' in host.final().lower()
    submit(host)
    host.manager.invoke_hook('on_session_start', session_id='s', model='test', platform='cli')
    assert host.final('fresh session') == 'fresh session'


def test_timeout_invalidates_previous_final_and_logs_citations(host):
    load(host)
    submit(host)
    session = next(iter(host.plugin._sessions.values()))
    cells = [json.loads(line) for line in session.log_path.read_text().splitlines()
             if json.loads(line)['event'] == 'cell']
    assert cells[-1]['citations'][0]['quote'] == '42'
    session.lim['timeout_s'] = 0.1
    assert 'killed' in host.tool('rlm_exec', {'code': 'while True: pass'})['error']
    assert session.last_validated_final is None
    assert 'withheld' in host.final().lower()


def test_dispatch_that_skips_middleware_cannot_bypass_final_gate(host):
    host.begin()
    out = host.tool('rlm_load', {'paths': [str(host.source)]}, bypass=True)
    assert out.get('loaded'), out
    assert 'withheld' in host.final('uncited bypass prose').lower()
    host.tool('rlm_exec', {'code': "FINAL('42', citations=[cite('source.txt',7,9)])"}, bypass=True)
    assert 'withheld' in host.final('no middleware turn authorization').lower()
    submit(host)
    host.tool('rlm_load', {'paths': [str(host.source)]}, bypass=True)
    assert 'withheld' in host.final('stale before bypass reload').lower()
    submit(host)
    host.tool('rlm_exec', {'code': "FINAL('42', citations=[cite('source.txt',7,9)])"}, bypass=True)
    assert 'withheld' in host.final('same text is not the same authorization').lower()


def test_empty_final_does_not_fall_through_to_root_prose(host):
    host.source.write_text('')
    load(host)
    out = host.tool('rlm_exec', {'code': "FINAL('')"})
    assert out.get('final') == ''
    result = host.final('unvalidated root fallback')
    assert 'withheld' in result.lower() and 'empty' in result.lower()
    assert 'unvalidated root fallback' not in result


def test_new_turn_can_authorize_only_its_own_final(host):
    load(host)
    submit(host)
    host.begin(turn='turn-2')
    submit(host, turn='turn-2')
    assert host.final(turn='turn-2').startswith('42\n')
    assert 'withheld' in host.final(turn='turn-1').lower()


def test_tool_identity_mismatch_cannot_authorize_current_turn(host):
    load(host)
    host.tool('rlm_exec', {'code': "FINAL('wrong', citations=[cite('source.txt',7,9)])"},
              task='different')
    assert 'withheld' in host.final().lower()
    host.tool('rlm_exec', {'code': "FINAL('wrong', citations=[cite('source.txt',7,9)])"},
              turn='old')
    assert 'withheld' in host.final().lower()
