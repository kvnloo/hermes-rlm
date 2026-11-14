"""Paper protocol and cancellation regressions, runnable without Hermes or Linux.

reuse-checked: existing test_rlm.py fixtures and Hermes plugin_llm async facade.
"""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from test_rlm import rlm


@pytest.fixture
def make_session(tmp_path, request):
    sessions = []

    def make(**kwargs):
        s = rlm.Session(str(len(sessions)), log_dir=tmp_path / 'runs',
                        allow_unsandboxed=not request.config.getoption('--rlm-sandbox'), **kwargs)
        sessions.append(s)
        return s

    yield make
    for s in sessions:
        s.kill()


def test_settings_precede_deprecated_env(monkeypatch):
    monkeypatch.setenv('RLM_MAX_SUBCALLS', '99')
    ctx = SimpleNamespace(get_config=lambda key, default=None: {'max_subcalls': 0,
        'max_depth': 1, 'max_tokens': 12, 'timeout_s': 0.5,
        'max_cost': 0.0, 'subcall_task': 'custom'}.get(key, default))
    settings = rlm.limits(ctx)
    assert settings['max_subcalls'] == 0
    assert settings['max_tokens'] == 12
    assert settings['timeout_s'] == 0.5
    assert settings['max_cost'] == 0.0
    assert settings['subcall_task'] == 'custom'


def test_final_var_and_validated_unicode_spans(make_session, tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('αβ answer=42\n', encoding='utf-8')
    s = make_session()
    metadata = s.load([str(path)])
    assert 'answer=42' not in json.dumps(metadata)
    out = s.exec("result = '42'\nFINAL_VAR('result', citations=[cite('source.txt', 3, 12)])")
    assert out['error'] is None
    assert out['final'] == '42'
    assert out['citations'] == [{'source': 'source.txt', 'start': 3, 'end': 12, 'quote': 'answer=42'}]
    invalid = s.exec("FINAL('42', citations=[{'source':'source.txt','start':3,'end':999}])")
    assert 'citation' in invalid['error'].lower()
    assert 'final' not in invalid


def test_cancel_stops_inflight_calls_not_just_waiter(make_session):
    started, stopped, cancel = threading.Event(), threading.Event(), threading.Event()

    async def subcall(prompt, max_tokens):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    s = make_session(subcall=subcall, cancelled=cancel.is_set,
                     settings={'timeout_s': 2, 'concurrency': 1})
    result = {}
    worker = threading.Thread(target=lambda: result.update(s.exec("llm_batch(['a','b','c'])")))
    worker.start()
    assert started.wait(2)
    cancel.set()
    worker.join(2)
    assert not worker.is_alive(), 'cancel must not wait for network timeout'
    assert stopped.is_set(), 'the in-flight completion itself must receive cancellation'
    assert 'cancel' in result['error'].lower()
    assert s.stats['subcalls'] == 1, 'queued undispatched prompts are not attempted calls'
    attempts = [json.loads(line) for line in s.log_path.read_text().splitlines()
                if json.loads(line)['event'] == 'subcall']
    assert len(attempts) == 1 and attempts[0]['status'] == 'cancelled'
    cancel.clear()
    assert not s.exec('1')['error']
    assert s.stats['subcalls'] == 1


def test_deadline_applies_inside_batch(make_session):
    stopped = threading.Event()

    async def subcall(prompt, max_tokens):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    s = make_session(subcall=subcall)
    assert not s.exec('1')['error']
    s.lim['timeout_s'] = 0.2
    t0 = time.monotonic()
    out = s.exec("llm_query('wait')")
    assert time.monotonic() - t0 < 1.5
    assert stopped.is_set()
    assert 'killed' in out['error']
    assert s.stats['subcalls'] == 1


def test_attempt_errors_count_once_and_order_is_stable(make_session):
    async def subcall(prompt, max_tokens):
        await asyncio.sleep(0.02 if prompt == 'slow' else 0)
        if prompt == 'bad':
            raise ValueError('test failure')
        return prompt, {'prompt_tokens': 2, 'completion_tokens': 1, 'cost_usd': 0.01}

    s = make_session(subcall=subcall, settings={'max_subcalls': 3})
    out = s.exec("answers=llm_batch(['slow','bad','fast']); print(answers)")
    assert out['subcalls_this_cell'] == 3
    assert "['slow', '[llm_query error: ValueError]', 'fast']" in out['stdout']
    assert s.stats['subcalls'] == 3
    assert s.stats['prompt_tokens'] == 4
    attempts = [json.loads(line) for line in s.log_path.read_text().splitlines()
                if json.loads(line)['event'] == 'subcall']
    assert len(attempts) == 3 and {a['attempt'] for a in attempts} == {1, 2, 3}
    assert sum(a['prompt_tokens'] for a in attempts) == 4
    assert 'budget exhausted' in s.exec("llm_query('fourth')")['error']


def test_no_cross_session_kernel_state(make_session):
    a, b = make_session(), make_session()
    a.exec('secret=42')
    assert a.proc.pid != b.exec('1').get('pid', -1)
    assert 'NameError' in b.exec('secret')['error']
    assert a.proc.pid != b.proc.pid


@pytest.mark.parametrize('setting,value', [
    ('max_subcalls', -1), ('max_subcalls', True), ('max_depth', 2),
    ('max_depth', -1), ('max_tokens', 0), ('max_tokens', 32001),
    ('timeout_s', 0), ('timeout_s', float('nan')), ('max_cost', -0.1),
    ('max_cost', float('inf')), ('concurrency', 0), ('max_prompt_chars', '500'),
    ('subcall_task', ''), ('subcall_task', ' task '),
])
def test_invalid_settings_fail_closed(make_session, setting, value):
    with pytest.raises(ValueError):
        make_session(settings={setting: value})


def test_env_fallback_warns_and_zero_is_preserved(monkeypatch):
    monkeypatch.setenv('RLM_MAX_DEPTH', '0')
    monkeypatch.setenv('RLM_SUBCALL_MAX_TOKENS', '17')
    with pytest.warns(DeprecationWarning):
        settings = rlm.limits()
    assert settings['max_depth'] == 0 and settings['max_tokens'] == 17


@pytest.mark.parametrize('span', [
    {'source': 'missing', 'start': 0, 'end': 1},
    {'source': 's.txt', 'start': -1, 'end': 1},
    {'source': 's.txt', 'start': 0, 'end': 0},
    {'source': 's.txt', 'start': True, 'end': 2},
    {'source': 's.txt', 'start': 0, 'end': 1.5},
    {'source': 's.txt', 'start': 0, 'end': 1, 'quote': 'wrong'},
])
def test_bad_citations_never_publish_final(make_session, tmp_path, span):
    source = tmp_path / 's.txt'
    source.write_text('original')
    s = make_session()
    s.load([str(source)])
    out = s.exec(f"FINAL('answer', citations=[{span!r}])")
    assert 'citation' in out['error'].lower()
    assert 'final' not in out


def test_original_sources_not_mutable_kernel_context(make_session, tmp_path):
    source = tmp_path / 's.txt'
    source.write_text('original')
    s = make_session()
    s.load([str(source)])
    out = s.exec("context = 'tampered'\nFINAL('answer', citations=[cite('s.txt', 0, 8)])")
    assert out['citations'][0]['quote'] == 'original'
    assert 'citation' in s.exec("FINAL('uncited')")['error']
    assert 'existing REPL variable' in s.exec("FINAL_VAR('missing')")['error']
    assert 'final' not in s.exec("FINAL('x', citations=[cite('s.txt',0,1)]); 1/0")


def test_zero_depth_and_token_override_cap(make_session):
    observed = []

    async def subcall(prompt, max_tokens):
        observed.append(max_tokens)
        return 'ok', {}

    s = make_session(subcall=subcall, settings={'max_depth': 0})
    assert 'max_depth=0' in s.exec("llm_query('x')")['error']
    assert s.stats['subcalls'] == 0
    s = make_session(subcall=subcall, settings={'max_tokens': 7})
    assert not s.exec("llm_query('x', max_tokens=900)")['error']
    assert observed == [7]
    for invalid in ['0', '-1', 'True', '1.2']:
        assert 'positive integer' in s.exec(f"llm_query('x', max_tokens={invalid})")['error']
    assert s.stats['subcalls'] == 1


@pytest.mark.parametrize('price', [None, 0.5])
def test_cost_cutoff_is_checked_before_queued_dispatch(make_session, price):
    async def subcall(prompt, max_tokens):
        return 'ok', {'cost_usd': price}

    s = make_session(subcall=subcall, settings={'max_cost': 0.5, 'concurrency': 1})
    out = s.exec("print(llm_batch(['a','b','c']))")
    assert s.stats['subcalls'] == 1
    assert out['stdout'].count('cost budget') == 2
    assert s.stats['cost_usd'] == (price or 0)
    assert s.stats['cost_unknown'] == (1 if price is None else 0)
    z = make_session(subcall=subcall, settings={'max_cost': 0})
    assert 'cost budget' in z.exec("llm_query('a')")['error']
    assert z.stats['subcalls'] == 0


def test_reload_clears_variables_but_not_attempts(make_session, tmp_path):
    async def subcall(prompt, max_tokens):
        return 'ok', {}

    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    a.write_text('first')
    b.write_text('second')
    s = make_session(subcall=subcall)
    s.load([str(a)])
    s.exec("answer='stale'; llm_query('hi')")
    s.load([str(b)])
    assert 'NameError' in s.exec('answer')['error']
    assert s.stats['subcalls'] == 1
    assert 'citation source' in s.exec("FINAL('x', citations=[cite('a.txt', 0, 1)])")['error']


def test_session_identity_has_no_sanitization_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(rlm, '_data_dir', lambda: tmp_path / 'profile-a')
    a = rlm.session_for(task_id='t/a', session_id='one')
    assert rlm.session_for(task_id='t/a', session_id='one') is a
    b = rlm.session_for(task_id='ta', session_id='one')
    c = rlm.session_for(task_id='t/a', session_id='two')
    monkeypatch.setattr(rlm, '_data_dir', lambda: tmp_path / 'profile-b')
    d = rlm.session_for(task_id='t/a', session_id='one')
    assert len({id(a), id(b), id(c), id(d)}) == 4
    assert len({a.log_path, b.log_path, c.log_path, d.log_path}) == 4
    rlm._shutdown()


def test_host_lane_uses_async_api_task_timeout_and_token_cap(make_session):
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='PONG', usage=SimpleNamespace(
            input_tokens=3, output_tokens=1, cost_usd=None))

    ctx = SimpleNamespace(llm=SimpleNamespace(acomplete=complete),
                          get_config=lambda key, default=None: default)
    s = make_session(ctx=ctx, settings={'max_tokens': 13, 'subcall_task': 'custom', 'timeout_s': 1})
    out = s.exec("llm_batch(['PING','PING'])")
    assert out['stdout'].strip() == "['PONG', 'PONG']"
    assert s.stats['subcalls'] == 2
    assert all(c['task'] == 'custom' and c['max_tokens'] == 13 and 0 < c['timeout'] <= 1 for c in calls)
    assert all(c['messages'][0]['role'] == 'system' for c in calls)


def test_cancel_preserves_completed_usage_and_skips_queued_work(make_session):
    second_started, cancel = threading.Event(), threading.Event()

    async def subcall(prompt, max_tokens):
        if prompt == 'first':
            return 'done', {'prompt_tokens': 5, 'completion_tokens': 2, 'cost_usd': 0.1}
        second_started.set()
        await asyncio.Event().wait()

    s = make_session(subcall=subcall, cancelled=cancel.is_set, settings={'concurrency': 1})
    result = {}
    worker = threading.Thread(target=lambda: result.update(s.exec("llm_batch(['first','second','third'])")))
    worker.start()
    assert second_started.wait(2)
    cancel.set()
    worker.join(2)
    assert not worker.is_alive()
    assert 'cancel' in result['error']
    assert s.stats['subcalls'] == 2
    assert s.stats['prompt_tokens'] == 5 and s.stats['completion_tokens'] == 2
    assert s.stats['cost_usd'] == 0.1
    attempts = [json.loads(line) for line in s.log_path.read_text().splitlines()
                if json.loads(line)['event'] == 'subcall']
    assert [a['status'] for a in attempts] == ['ok', 'cancelled']


def test_cancel_one_session_does_not_cancel_another(make_session):
    cancel, started, release = threading.Event(), threading.Event(), threading.Event()

    async def hanging(prompt, tokens):
        started.set()
        await asyncio.Event().wait()

    async def independent(prompt, tokens):
        while not release.is_set():
            await asyncio.sleep(0.01)
        return 'independent', {}

    a = make_session(subcall=hanging, cancelled=cancel.is_set)
    b = make_session(subcall=independent)
    results = [{}, {}]
    threads = [threading.Thread(target=lambda: results[0].update(a.exec("llm_query('a')"))),
               threading.Thread(target=lambda: results[1].update(b.exec("llm_query('b')")))]
    for thread in threads:
        thread.start()
    assert started.wait(2)
    cancel.set()
    release.set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert 'cancel' in results[0]['error']
    assert results[1]['error'] is None
    assert 'independent' in results[1]['stdout']
    assert a.stats['subcalls'] == b.stats['subcalls'] == 1


def test_opt_in_registration_and_cleanup(monkeypatch):
    tools, tasks, unload = [], [], []
    ctx = SimpleNamespace(
        get_config=lambda key, default=None: 'custom' if key == 'subcall_task' else default,
        register_tool=lambda name, toolset, *a, **kw: tools.append((name, toolset)),
        register_auxiliary_task=lambda name, **kw: tasks.append(name),
        register_command=lambda *a, **kw: None,
        register_hook=lambda *a, **kw: None,
        register_middleware=lambda *a, **kw: None,
        on_unload=unload.append,
    )
    monkeypatch.setattr(rlm, '_CTX', None)
    rlm.register(ctx)
    assert tools == [('rlm_load', 'rlm'), ('rlm_exec', 'rlm')]
    assert set(tasks) == {'rlm_subcall', 'custom'}
    assert unload == [rlm._shutdown]
    assert 'FINAL_VAR' in rlm.EXEC_SCHEMA['description']
    assert 'llm_batch' in rlm.LOAD_SCHEMA['description'] or 'rlm_exec' in rlm.LOAD_SCHEMA['description']
