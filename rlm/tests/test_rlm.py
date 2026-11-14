"""Plugin behavior tests with a fake sub-model.

Run python -m pytest rlm/tests. Add --rlm-sandbox for actual Linux isolation tests.

reuse-checked: tool-context-trim tests pattern -- same importlib layout; these cover the rlm plugin only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rlm_plugin", PLUGIN_DIR / "__init__.py")
rlm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rlm
spec.loader.exec_module(rlm)


async def fake_subcall(prompt, max_tokens):
    return f"LEN={len(prompt)}", {"prompt_tokens": len(prompt) // 4, "completion_tokens": 3}


@pytest.fixture()
def session(tmp_path, monkeypatch, request):
    s = rlm.Session("t", subcall=fake_subcall, log_dir=tmp_path / "runs",
                    settings={'max_subcalls': 5, 'timeout_s': 3,
                              'max_output_chars': 200, 'max_prompt_chars': 1000},
                    allow_unsandboxed=not request.config.getoption('--rlm-sandbox'))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha " * 100)
    (corpus / "b.txt").write_text("needle=42\n")
    (corpus / "bin.dat").write_bytes(b"\0\1\2")
    s.corpus_dir = corpus
    yield s
    s.kill()


def test_load_keeps_corpus_out_of_result_and_persists_state(session):
    info = session.load([str(session.corpus_dir)])
    assert info["files"] == 2 and info["skipped_binary"] == 1 and info["context_type"].startswith("dict")
    assert "alpha alpha" not in json.dumps(info)  # corpus text never enters the root context
    assert session.exec("x = sorted(context)\nx")["stdout"].strip() == "['a.txt', 'b.txt']"
    assert session.exec("import re; re.search(r'needle=(\\d+)', context['b.txt']).group(1)")["stdout"].strip() == "'42'"
    assert session.exec("x")["stdout"].strip() == "['a.txt', 'b.txt']"  # variables survive cells


def test_sandbox_has_no_network_no_host_files_no_env(session):
    if session.allow_unsandboxed:
        pytest.skip('requires --rlm-sandbox and Linux bubblewrap')
    session.load([str(session.corpus_dir)])
    out = session.exec(
        "import os, socket\n"
        "r = []\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); r.append('net')\n"
        "except OSError: pass\n"
        f"for p in {str([str(session.corpus_dir), str(session.log_path.parent)])}:\n"
        "    if os.path.exists(p): r.append(p)\n"
        "r.append(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k))\n"
        "print(r)")
    assert out["error"] is None
    assert out["stdout"].strip() == "[[]]"


def test_llm_batch_budget_and_prompt_cap(session):
    session.load([str(session.corpus_dir)])
    out = session.exec("r = llm_batch(['a'*10, 'b'*20]); r")
    assert out["stdout"].strip() == "['LEN=10', 'LEN=20']" and out["subcalls_this_cell"] == 2
    too_big = session.exec("llm_query('x'*5000)")
    assert "RLM_MAX_PROMPT_CHARS" in too_big["error"]
    over = session.exec("llm_batch(['q']*4)")  # 2 used + 4 > 5
    assert "budget exhausted" in over["error"]
    assert session.stats["subcalls"] == 2 and session.stats["prompt_tokens"] == 7


def test_depth0_disables_subcalls(session):
    s = rlm.Session("d0", subcall=fake_subcall, log_dir=session.log_path.parent,
                    settings={'max_subcalls': 0}, allow_unsandboxed=session.allow_unsandboxed)
    try:
        assert "budget exhausted" in s.exec("llm_query('hi')")["error"]
    finally:
        s.kill()


def test_timeout_kills_cell_and_reloads_context(session):
    session.load([str(session.corpus_dir)])
    session.exec("y = 1")
    out = session.exec("while True: pass")
    assert "killed" in out["error"] and session.stats["restarts"] == 1
    assert session.exec("len(context)")["stdout"].strip() == "2"  # corpus restored
    assert "NameError" in session.exec("y")["error"]               # honest: other state lost


def test_output_truncation_final_and_trajectory(session):
    session.load([str(session.corpus_dir)])
    out = session.exec("print('z'*1000)")
    assert len(out["stdout"]) < 400 and "truncated" in out["stdout"]
    fin = session.exec("FINAL('42', citations=[cite('b.txt', 7, 9)])")
    assert fin["final"] == "42"
    events = [json.loads(l)["event"] for l in session.log_path.read_text().splitlines()]
    assert events[0] == "load" and events.count("cell") == 2


def test_handlers_return_json_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(rlm, '_data_dir', lambda: tmp_path)
    assert "FileNotFoundError" in rlm._load_handler({"paths": [str(tmp_path / "nope")]}, task_id="h")
    rlm._shutdown()


def test_relative_paths_anchor_at_task_workspace_not_process_cwd(session, tmp_path, monkeypatch):
    # Regression: the benchmark's root model passed "corpus.txt" and got FileNotFoundError because
    # the oneshot process cwd (~) differs from the task workspace that read_file/terminal use.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    paths = ModuleType('tools.file_tools_paths')
    paths._resolve_base_dir = lambda task_id='default', **_: session.corpus_dir
    monkeypatch.setitem(sys.modules, 'tools', ModuleType('tools'))
    monkeypatch.setitem(sys.modules, 'tools.file_tools_paths', paths)
    out = session.load(["b.txt"], task_id="t")
    assert out["files"] == 1 and "error" not in out
