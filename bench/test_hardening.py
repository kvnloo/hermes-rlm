"""Staged-only regression checks for collection validity and logical child outcomes.

reuse-checked: test_paper_matrix.py fixtures, audit_results.py and staged plugin FINAL trajectories.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hermes_client
import paper_matrix as matrix
from paper_proxy import Ledger, ROOT, SUB, Router
import run_bench


CITATIONS = [{"source": "context.txt", "start": 0, "end": 5, "quote": "73521"}]


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        context = self.root / "context.txt"
        context.write_text("73521 input")
        self.task = dict(id="tiny", family="S-NIAH", size_tokens=8, question="Code?",
                         gold=["73521"], scoring="ruler", context_path=str(context),
                         context_sha256=matrix.digest(context), task_sha256="taskhash")
        template = self.root / "template"
        template.mkdir()
        plugin = self.root / "plugin"
        plugin.mkdir()
        (plugin / "__init__.py").write_text("")
        self.args = matrix.parser().parse_args([
            str(self.root / "manifest"), str(self.root / "out"), "--ops-dir", str(self.root / "ops"),
            "--home-template", str(template), "--plugin-source", str(plugin),
            "--released-source", str(self.root), "--released-python", sys.executable,
            "--fork-source", str(self.root), "--fork-python", sys.executable,
            "--nano-source", str(self.root)])
        self.ledger = Ledger(self.root / "ledger.db", cap=20)

    def collect(self, final="73521", prose="ANSWER: 73521", completed=True, child=None, recovered=False, broken=False, emit_summary=None):
        ledger = self.ledger
        observed = [{"role": "root", "tools": ["rlm_load", "rlm_exec"], "tool_calls": []}]
        class FakeRouter:
            def __init__(self, ledger, run, *a, **kw):
                self.ledger, self.run = ledger, run
                self.errors, self.blocked, self.observed = [], set(), observed
                self.url = "http://127.0.0.1:1"
            def __enter__(self):
                if child or recovered:
                    rid = ledger.reserve(self.run, SUB, "sub")
                    ledger.finish(rid, recovered if recovered else 500, 0.01, {})
                if recovered:
                    rid = ledger.reserve(self.run, SUB, "sub")
                    ledger.finish(rid, 200, 0.01, {"prompt_tokens": 3, "completion_tokens": 1})
                rid = ledger.reserve(self.run, ROOT, "root")
                ledger.finish(rid, 200, 0.01, {"prompt_tokens": 8, "completion_tokens": 2})
                return self
            def __exit__(self, *a):
                pass
            def logical_snapshot(self):
                return [{"id": "root", "role": "root", "status": 200, "pending": False}]
        def process(command, cwd, env, timeout):
            logs = Path(env["RLM_LOG_DIR"])
            logs.mkdir()
            events = []
            if child or recovered:
                events.append({"event": "subcall", "status": "ok" if recovered else child})
            events.append({"event": "cell", "error": None, "final": final,
                           "citations": CITATIONS if final is not None else None})
            (logs / "opaquehash.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
            summary = {"answer": prose, "library_error": False, "library_partial": False,
                       "completed": completed, "failed": False, "interrupted": False,
                       "turn_exit_reason": "completed" if completed else "max_iterations_reached(10/10)"}
            output = emit_summary(summary) if emit_summary else json.dumps(summary)
            return {"stdout": "broken" if broken else output, "stderr": "", "exit_code": 0,
                    "timed_out": False, "wall_s": 0.01}
        with patch.object(matrix, "Router", FakeRouter), patch.object(matrix, "run_process", side_effect=process), \
             patch.object(matrix, "hermes_stats", return_value={}):
            return matrix.run_one(self.task, "released-rlm", 0, self.args, {}, ledger)

    def test_invalid_adapter_keeps_logical_and_physical_outcomes(self):
        row = self.collect(broken=True)
        self.assertFalse(row["valid"])
        self.assertEqual(row["logical_call_outcomes"][0]["status"], 200)
        self.assertEqual(row["accounting"]["request_attempts"], 1)

    def test_missing_final_never_scores(self):
        with patch.object(matrix, "score_answer", side_effect=AssertionError("must not score")) as score:
            row = self.collect(final=None)
        self.assertFalse(row["valid"])
        self.assertIsNone(row["score"])
        score.assert_not_called()

    def test_host_final_not_contradictory_root_prose_is_scored(self):
        row = self.collect(prose="ANSWER: incorrect")
        self.assertTrue(row["valid"])
        self.assertEqual(row["answer"], "73521")
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["root_answer"], "ANSWER: incorrect")
        self.assertEqual(row["validated_final"]["citations"], CITATIONS)

    def test_library_summary_keeps_unicode_separators_in_answer(self):
        def emit(summary):
            with patch.object(hermes_client, "run", return_value=summary), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(hermes_client.main(
                    ["Code?", "--base-url", "http://unused.invalid", "--model", "fixture"]), 0)
            return out.getvalue()

        for separator in (" ", "\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=separator):
                prose = "The code is:" + separator + "73521"
                row = self.collect(prose=prose, emit_summary=emit)
                self.assertTrue(row["valid"], row.get("error"))
                self.assertEqual(row["root_answer"], prose)
                self.assertEqual(row["answer"], "73521")
                self.assertEqual(row["score"], 1)

    def test_incomplete_budget_exit_is_invalid(self):
        row = self.collect(completed=False)
        self.assertFalse(row["valid"])
        self.assertIsNone(row["score"])

    def test_child_failure_not_erased_by_root_success(self):
        for status in ("error", "cancelled"):
            row = self.collect(child=status)
            self.assertFalse(row["valid"])
            self.assertEqual(row["subcall_statuses"], {status: 1})

    def test_recovered_physical_error_and_plugin_ok_passes(self):
        for status in (400, 429):
            row = self.collect(recovered=status)
            self.assertTrue(row["valid"])
            self.assertTrue(row["accounting"]["provider_failure"])
            self.assertEqual(row["subcall_statuses"], {"ok": 1})
            self.assertEqual(row["sub_prompt_tokens"], 3)

    def test_final_resets_and_multiple_files_do_not_mix(self):
        logs = self.root / "logs"
        logs.mkdir()
        path = logs / "hash.jsonl"
        good = {"event": "cell", "error": None, "final": "73521", "citations": CITATIONS}
        for later in ({"event": "load"}, {"event": "cell", "error": "bad", "final": None},
                      {"event": "cell", "error": None, "final": None}):
            path.write_text(json.dumps(good) + "\n" + json.dumps(later) + "\n")
            self.assertIsNone(run_bench.rlm_usage(logs)["validated_final"])
        path.write_text(json.dumps(good) + "\n")
        self.assertEqual(run_bench.rlm_usage(logs)["validated_final"]["answer"], "73521")
        (logs / "other.jsonl").write_text(json.dumps(good) + "\n")
        self.assertIsNone(run_bench.rlm_usage(logs)["validated_final"])

    def test_old_final_without_quotes_is_invalid(self):
        path = self.root / "hash.jsonl"
        path.write_text(json.dumps({"event": "cell", "final": "73521", "citations": [
            {"source": "context.txt", "start": 0, "end": 5}], "error": None}) + "\n")
        self.assertIsNone(run_bench.rlm_usage(path)["validated_final"])

    def test_library_completion_metadata_preserves_counters(self):
        agent = SimpleNamespace(session_id="s", valid_tool_names={"rlm_exec"}, session_input_tokens=8,
                                session_output_tokens=2, session_api_calls=1)
        result = {"completed": False, "failed": True, "interrupted": True,
                  "turn_exit_reason": "max_iterations_reached", "final_response": "prose"}
        summary = hermes_client.result_summary(result, agent)
        for key in ("completed", "failed", "interrupted", "turn_exit_reason"):
            self.assertEqual(summary[key], result[key])
        self.assertEqual(summary["host_root_input_tokens"], 8)

    def test_legacy_invocations_do_not_mix_hashed_logs(self):
        seen = []
        def process(cmd, **kw):
            logs = Path(kw["env"]["RLM_LOG_DIR"])
            logs.mkdir(parents=True, exist_ok=True)
            seen.append(logs)
            (logs / "opaquehash.jsonl").write_text('{"event":"subcall","status":"ok","prompt_tokens":3}\n')
            return SimpleNamespace(stdout="ANSWER: x\nsession_id: same-session", stderr="")
        args = SimpleNamespace(provider="paper-root", model=ROOT, timeout=1)
        with patch.object(run_bench.subprocess, "run", side_effect=process), \
             patch.object(run_bench, "hermes_usage", return_value={}):
            rows = [run_bench.run_hermes("hermes-rlm", "prompt", self.root, args, self.root / "runs") for _ in range(2)]
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual([r["subcalls"] for r in rows], [1, 1])

    def test_native_logical_retry_recovers_only_same_call(self):
        original = urllib.request.urlopen
        statuses = iter((200, 500, 200, 500, 200))
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}
        def route(req, **kw):
            if req.full_url.startswith("http://127.0.0.1"):
                return original(req, **kw)
            status = next(statuses)
            if status != 200:
                raise urllib.error.HTTPError(req.full_url, status, "failed", {}, io.BytesIO())
            return Response(b'{"usage":{"prompt_tokens":1,"completion_tokens":1}}')
        with Router(self.ledger, "logical") as router, patch.dict(os.environ, NVIDIA_API_KEY="fake"), \
             patch("urllib.request.urlopen", side_effect=route):
            def request(session, request_id):
                req = urllib.request.Request(router.url + "/nano/v1/chat/completions", b'{"messages":[]}',
                    {"X-ACP-Agent-Session-ID": session, "X-ACP-Model-Request-ID": request_id})
                try:
                    with urllib.request.urlopen(req) as response:
                        response.read()
                except urllib.error.HTTPError as error:
                    error.close()
            request("parent", "root1")
            request("child", "child1")
            request("child", "child1")
            child = [r for r in router.logical_snapshot() if r["role"] == "sub"]
            self.assertEqual(len(child), 1)
            self.assertEqual(child[0]["status"], 200)
            self.assertTrue(matrix.child_outcomes_ok("nano-rlm", {
                "accounting": self.ledger.summary("logical"), "logical_call_outcomes": router.logical_snapshot()}))
            pending = router.logical_snapshot()
            next(c for c in pending if c["role"] == "sub").update(status=None, pending=True)
            self.assertFalse(matrix.child_outcomes_ok("nano-rlm", {"logical_call_outcomes": pending}))
            request("child", "child2")
            request("parent", "root2")
            child = {r["id"]: r for r in router.logical_snapshot() if r["role"] == "sub"}
            self.assertEqual(child["child2"]["status"], 500)
            self.assertEqual(child["child1"]["status"], 200)
            self.assertFalse(matrix.child_outcomes_ok("nano-rlm", {
                "accounting": self.ledger.summary("logical"), "logical_call_outcomes": router.logical_snapshot()}))
        self.assertEqual(self.ledger.summary("logical")["request_attempts"], 5)


if __name__ == "__main__":
    unittest.main()
