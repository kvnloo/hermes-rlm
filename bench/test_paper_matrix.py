"""Bounded single-file checks for paper matrix protocol and safety boundaries.

reuse-checked: existing run_bench scoring and paper_tasks worker tests; no API calls.
"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_matrix as matrix
from paper_proxy import Ledger, ROOT, SUB, Router, response_usage
from run_bench import rlm_usage


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger = Ledger(self.root / "ledger.db", cap=6)
        self.task = {"id": "tiny", "family": "S-NIAH", "size_tokens": 8,
                     "question": "What is the code?", "gold": ["73521"], "scoring": "ruler",
                     "metadata": {"forbidden": "GOLD_METADATA"}, "context_path": "context.txt"}

    def test_cap_persists_across_resume_and_provider_routes(self):
        for _ in range(6):
            self.ledger.reserve("run", ROOT, "root")
        resumed = Ledger(self.root / "ledger.db", cap=6)
        with self.assertRaisesRegex(RuntimeError, "provider_request_cap"):
            resumed.reserve("next", SUB, "sub")
        self.assertEqual(resumed.summary("run")["request_attempts"], 6)
        with self.assertRaises(ValueError):
            Ledger(self.root / "bad.db", 3000)

    def test_nano_session_routing(self):
        with Router(self.ledger, "r") as router:
            self.assertEqual(router.role("/nano/v1/chat/completions", "parent"), "root")
            self.assertEqual(router.role("/nano/v1/chat/completions", "child"), "sub")
            self.assertEqual(router.role("/nano/v1/chat/completions", "parent"), "root")
            with self.assertRaises(ValueError):
                router.role("/nano/v1/chat/completions", None)

    def test_three_backed_off_attempts_then_circuit(self):
        def unavailable(*a, **kw):
            raise urllib.error.HTTPError("https://unused", 429, "limited", {}, io.BytesIO(b"private"))
        with Router(self.ledger, "r", backoff=5) as router:
            with patch.dict(os.environ, NVIDIA_API_KEY="secret-not-persisted"), \
                 patch("urllib.request.urlopen", side_effect=unavailable) as request, \
                 patch("paper_proxy.time.sleep") as sleep:
                status, _, _ = router.forward({}, ROOT, "root")
                self.assertEqual(status, 429)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10])
                router.forward({}, SUB, "sub")
                self.assertEqual(request.call_count, 3)
        summary = self.ledger.summary("r")
        self.assertEqual(summary["request_attempts"], 3)
        self.assertNotIn("secret-not-persisted", json.dumps(summary))

    def test_real_http_proxy_rewrites_child_model(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}
        payload = b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":2}}'
        requests = []
        def upstream(req, **kw):
            requests.append(json.loads(req.data))
            return Response(payload)
        # Use the original opener for loopback, mock only the upstream hop.
        original = urllib.request.urlopen
        def route(req, **kw):
            return original(req, **kw) if req.full_url.startswith("http://127.0.0.1") else upstream(req, **kw)
        with Router(self.ledger, "r") as router, patch.dict(os.environ, NVIDIA_API_KEY="not-written"):
            with patch("urllib.request.urlopen", side_effect=route):
                for session in ("parent", "child"):
                    req = urllib.request.Request(router.url + "/nano/v1/chat/completions",
                        json.dumps({"model": ROOT, "messages": []}).encode(),
                        {"X-ACP-Agent-Session-ID": session, "X-ACP-Model-Request-ID": session + "-request",
                         "Content-Type": "application/json"})
                    with urllib.request.urlopen(req) as response:
                        self.assertEqual(response.status, 200)
        self.assertEqual([r["model"] for r in requests], [ROOT, SUB])
        summary = self.ledger.summary("r")
        self.assertEqual(summary["root_prompt_tokens"], 9)
        self.assertEqual(summary["sub_completion_tokens"], 2)

    def test_nim_thinking_policy_is_uniform_and_explicit(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}
        bodies = []
        def upstream(req, **kwargs):
            bodies.append(json.loads(req.data))
            return Response(b'{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}')
        with Router(self.ledger, "policy", disable_thinking=True) as router, \
             patch.dict(os.environ, NVIDIA_API_KEY="not-written"), \
             patch("urllib.request.urlopen", side_effect=upstream):
            router.forward({}, ROOT, "root")
            router.forward({}, SUB, "sub")
        self.assertEqual([b['chat_template_kwargs']['enable_thinking'] for b in bodies], [False, False])

    def test_hermes_prompt_requires_repl_final_not_prose(self):
        for arm in ('rlm-no-subcalls', 'released-rlm', 'fork-rlm'):
            prompt = matrix.task_prompt(self.task, arm)
            self.assertIn("FINAL_VAR('answer'", prompt)
            self.assertIn("cite('context.txt'", prompt)
            self.assertNotIn('End with ANSWER:', prompt)
        self.assertIn('End with ANSWER:', matrix.task_prompt(self.task, 'base-truncated'))
        self.assertIn('End with ANSWER:', matrix.task_prompt(self.task, 'nano-rlm'))

    def test_base_truncation_keeps_utf8_literally_before_character_cap(self):
        context = self.root / "context.txt"
        source = "é\r\n猫\r☀\n"
        context.write_bytes(source.encode("utf-8"))
        task = dict(self.task, context_path=str(context), context_sha256=matrix.digest(context),
                    task_sha256="hash")
        args = SimpleNamespace(ops_dir=self.root / "ops", root_model=ROOT, sub_model=SUB,
                               timeout=10, disable_thinking=False, max_tokens=64,
                               base_context_chars=4, scorer=matrix.HERE / "paper_tasks.py")
        bodies = []

        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

        def upstream(request, **kwargs):
            bodies.append(json.loads(request.data))
            return Response(b'{"choices":[{"message":{"content":"ANSWER: 73521"}}]}')

        with patch.dict(os.environ, NVIDIA_API_KEY="not-written"), \
             patch("urllib.request.urlopen", side_effect=upstream):
            row = matrix.run_one(task, "base-truncated", 0, args, {}, self.ledger)
        self.assertTrue(row["valid"], row.get("error"))
        self.assertEqual(bodies[0]["messages"][0]["content"],
                         "é\r\n猫\n\n" + matrix.task_prompt(task, "base-truncated"))
        self.assertEqual(row["truncation"], {"kind": "head_characters", "chars": 4,
                                            "original_chars": len(source), "truncated": True})

    def test_no_raw_context_or_gold_in_rlm_prompt(self):
        for arm in matrix.ARMS[1:]:
            prompt = matrix.task_prompt(self.task, arm)
            self.assertNotIn("73521", prompt)
            self.assertNotIn("GOLD_METADATA", prompt)
            self.assertNotIn("context_path", prompt)
        self.assertIn("max_depth=0", matrix.task_prompt(self.task, "rlm-no-subcalls"))

    def test_fresh_homes_have_only_safe_seed(self):
        template = self.root / "template"
        template.mkdir()
        (template / "SOUL.md").write_text("Answer carefully.")
        (template / "auth.json").write_text("secret")
        (template / "config.yaml").write_text("api_key: secret")
        plugin = self.root / "plugin"
        plugin.mkdir()
        (plugin / "__init__.py").write_text("")
        args = SimpleNamespace(max_subcalls=2, sub_max_tokens=64, cell_timeout=10,
                               max_context_chars=1000, root_model=ROOT, sub_model=SUB,
                               max_tokens=64, max_turns=4)
        with Router(self.ledger, "r") as router:
            for arm in ("fork-rlm", "rlm-no-subcalls"):
                home = self.root / arm
                cfg = matrix.fresh_home(home, template, plugin, arm, router, args)
                self.assertFalse((home / "auth.json").exists())
                self.assertEqual(cfg["_config_version"], 12)
                self.assertEqual(cfg["telemetry"]["shared_metrics"], {"enabled": False, "send": False})
                self.assertEqual((home / "SOUL.md").read_text(), "Answer carefully.")
                self.assertNotIn("secret", (home / "config.yaml").read_text())
                self.assertEqual(cfg["plugins"]["entries"]["rlm"]["settings"]["max_depth"],
                                 0 if arm == "rlm-no-subcalls" else 1)
                if arm == "fork-rlm":
                    self.assertEqual(cfg["tools"]["tool_search"], {"enabled": "on", "eager": ["rlm"]})
                with self.assertRaises(FileExistsError):
                    matrix.fresh_home(home, template, plugin, arm, router, args)

    def test_scoring_hook_uses_dataset_worker(self):
        self.assertEqual(matrix.score_answer(self.task, "ANSWER: 73521", matrix.HERE / "paper_tasks.py")["score"], 1)
        pair = dict(self.task, scoring="pairs_f1", gold=[[1, 2], [1, 3]])
        self.assertAlmostEqual(matrix.score_answer(pair, "ANSWER: [(2,1)]", matrix.HERE / "paper_tasks.py")["score"], 2/3)

    def test_scoring_hook_rejects_nonfinite_and_nonnumeric_grades(self):
        scorer = self.root / "custom_scorer.py"
        scorer.write_text("def score_task(task, answer):\n    return task['fixture_grade']\n")
        rejected = (float("nan"), float("inf"), 10 ** 400,
                    {"score": float("-inf")}, {"score": None}, {"score": "0.5"})
        for grade in rejected:
            with self.subTest(grade=grade), self.assertRaisesRegex(ValueError, "score"):
                matrix.score_answer(dict(self.task, fixture_grade=grade), "answer", scorer)

    def test_scoring_hook_preserves_valid_numeric_and_dict_grades(self):
        scorer = self.root / "custom_scorer.py"
        scorer.write_text("def score_task(task, answer):\n    return task['fixture_grade']\n")
        for grade, expected in ((0, {"score": 0.0}), (0.25, {"score": 0.25}),
                                (True, {"score": 1.0}), (-0.1, {"score": -0.1}),
                                ({"score": 2 ** 53 + 1, "note": "ok"}, {"score": 2 ** 53 + 1, "note": "ok"})):
            with self.subTest(grade=grade):
                result = matrix.score_answer(dict(self.task, fixture_grade=grade), "answer", scorer)
                self.assertEqual(result, expected)
                self.assertIs(type(result["score"]), type(expected["score"]))

    def test_interleaving_resume_invalid_is_terminal(self):
        task = dict(self.task, task_sha256="hash")
        scheduled = list(matrix.schedule([task], matrix.ARMS, 2))
        self.assertEqual([a for _, a, _ in scheduled[:5]], list(matrix.ARMS))
        self.assertEqual(scheduled[5][1], matrix.ARMS[1])
        key = matrix.row_key(task, matrix.ARMS[0], 0, {})
        out = self.root / "out.jsonl"
        out.write_text(json.dumps({"row_key": key, "valid": False, "score": None}) + "\n")
        self.assertEqual(matrix.completed(out), {key})
        self.assertNotEqual(key, matrix.row_key(task, matrix.ARMS[0], 0, {"changed": True}))

    def test_manifest_guards_and_hashes(self):
        context = self.root / "context.txt"
        context.write_text("unlabeled input")
        manifest = self.root / "tasks.jsonl"
        manifest.write_text(json.dumps(self.task) + "\n")
        task = matrix.load_tasks(manifest)[0]
        self.assertEqual(task["context_sha256"], matrix.digest(context))
        manifest.write_text(json.dumps(dict(self.task, context_path="gold.json")) + "\n")
        (self.root / "gold.json").write_text("secret")
        with self.assertRaises(ValueError):
            matrix.load_tasks(manifest)

    def test_manifest_checks_declared_context_bytes_before_scheduling(self):
        context = self.root / "context.txt"
        context.write_bytes(b"first\r\n")
        other_metadata = {"context_sha256": matrix.digest(context)}
        manifest = self.root / "tasks.jsonl"
        manifest.write_text(json.dumps(dict(self.task, metadata=other_metadata)) + "\n")
        self.assertEqual(matrix.load_tasks(manifest)[0]["context_sha256"],
                         other_metadata["context_sha256"])
        context.write_bytes(b"other\r\n")
        with self.assertRaisesRegex(ValueError, r"manifest line 1: context_sha256 .* context bytes"):
            matrix.load_tasks(manifest)

    def test_dry_run_rejects_partial_unknown_task_selection(self):
        (self.root / "context.txt").write_text("context", encoding="utf-8")
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(self.task) + "\n", encoding="utf-8")
        args = [str(manifest), str(self.root / "out.jsonl"),
                "--ops-dir", str(self.root / "ops"), "--home-template", str(self.root),
                "--released-source", str(self.root), "--released-python", sys.executable,
                "--fork-source", str(self.root), "--fork-python", sys.executable,
                "--nano-source", str(self.root), "--dry-run", "--arms", "base-truncated",
                "--repeats", "1"]
        stream = io.StringIO()
        with patch.object(matrix, "source_sha", return_value="fixture"), \
             patch.object(matrix, "run_one") as run, patch("sys.stdout", stream):
            with self.assertRaisesRegex(ValueError, "unknown --task-ids: missing"):
                matrix.main(args + ["--task-ids", "tiny,missing"])
            self.assertEqual(stream.getvalue(), "")
            self.assertEqual(matrix.main(args + ["--task-ids", "tiny"]), 0)
            run.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())["task_id"], "tiny")

    def test_hashed_logs_count_attempts_once_not_cell_totals(self):
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "opaque-hash.jsonl").write_text('\n'.join(json.dumps(x) for x in [
            {"event": "subcall", "status": "ok", "prompt_tokens": 8, "completion_tokens": 2, "ms": 9},
            {"event": "subcall", "status": "error"}, {"event": "cell", "subcalls": 2}]) + "\n{")
        stats = rlm_usage(logs)
        self.assertEqual(stats["subcalls"], 2)
        self.assertEqual(stats["cells"], 1)
        self.assertEqual(stats["sub_prompt_tokens"], 8)
        self.assertEqual(stats["malformed_log_lines"], 1)

    def test_sse_usage(self):
        usage = response_usage(b'data: {"choices":[]}\n\ndata: {"usage":{"prompt_tokens":8}}\n\ndata: [DONE]\n')
        self.assertEqual(usage["prompt_tokens"], 8)

    def test_sse_usage_without_optional_space_reaches_ledger(self):
        data = (b'data:{"choices":[]}\r\n\r\n'
                b'data:{"usage":{"prompt_tokens":8,"completion_tokens":3}}\r\n\r\n'
                b'data:[DONE]\r\n\r\n')

        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "text/event-stream"}

        with Router(self.ledger, "sse") as router, \
             patch.dict(os.environ, NVIDIA_API_KEY="offline-fixture"), \
             patch("urllib.request.urlopen", return_value=Response(data)) as forward:
            self.assertEqual(router.forward({"stream": True}, ROOT, "root"),
                             (200, data, "text/event-stream"))
            self.assertEqual(json.loads(forward.call_args.args[0].data)["stream_options"],
                             {"include_usage": True})
        summary = self.ledger.summary("sse")
        self.assertTrue(summary["usage_complete"])
        self.assertEqual(summary["request_attempts"], 1)
        self.assertEqual(summary["root_prompt_tokens"], 8)
        self.assertEqual(summary["root_completion_tokens"], 3)

    def test_hermes_row_end_to_end_adapter(self):
        context = self.root / "source.txt"
        context.write_text("73521 input only")
        task = dict(self.task, context_path=str(context), context_sha256=matrix.digest(context),
                    task_sha256="hash")
        template = self.root / "template"
        template.mkdir()
        plugin = self.root / "plugin"
        plugin.mkdir()
        (plugin / "__init__.py").write_text("")
        args = matrix.parser().parse_args([
            str(self.root / "manifest"), str(self.root / "out"),
            "--ops-dir", str(self.root / "ops"), "--home-template", str(template),
            "--plugin-source", str(plugin),
            "--released-source", str(self.root), "--released-python", sys.executable,
            "--fork-source", str(self.root), "--fork-python", sys.executable,
            "--nano-source", str(self.root)])
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}
        original = urllib.request.urlopen
        def route(req, **kw):
            if req.full_url.startswith("http://127.0.0.1"):
                return original(req, **kw)
            return Response(b'{"usage":{"prompt_tokens":8,"completion_tokens":2}}')
        def process(command, cwd, env, timeout):
            import yaml
            self.assertIn(str(matrix.HERE / "hermes_client.py"), command)
            self.assertNotIn("NVIDIA_API_KEY", env)
            self.assertEqual([p.name for p in cwd.iterdir()], ["context.txt"])
            cfg = yaml.safe_load((Path(env["HERMES_HOME"]) / "config.yaml").read_text())
            url = cfg["providers"]["paper-root"]["base_url"] + "/chat/completions"
            req = urllib.request.Request(url, json.dumps({"messages": [], "tools": [
                {"function": {"name": n}} for n in ("rlm_load", "rlm_exec")]}).encode())
            with urllib.request.urlopen(req) as response:
                response.read()
            logs = Path(env["RLM_LOG_DIR"])
            logs.mkdir()
            (logs / "hash.jsonl").write_text(json.dumps({"event": "cell", "final": "73521", "error": None,
                "citations": [{"source": "context.txt", "start": 0, "end": 5, "quote": "73521"}]}) + "\n")
            summary = {"answer": "ANSWER: 73521", "hermes_ingress": "library", "completed": True,
                       "failed": False, "interrupted": False, "turn_exit_reason": "completed",
                       "session_ids": ["library-session"], "discovery_traces": [], "library_error": False}
            return {"stdout": json.dumps(summary), "stderr": "", "exit_code": 0,
                    "timed_out": False, "wall_s": 0.01}
        with patch.dict(os.environ, NVIDIA_API_KEY="not-for-agent"), \
             patch("urllib.request.urlopen", side_effect=route), \
             patch.object(matrix, "run_process", side_effect=process), \
             patch.object(matrix, "hermes_stats", return_value={}):
            row = matrix.run_one(task, "rlm-no-subcalls", 0, args, {}, self.ledger)
        self.assertEqual(row.get("error"), None)
        self.assertTrue(row["valid"])
        self.assertEqual(row["answer"], "73521")
        self.assertEqual(row["root_answer"], "ANSWER: 73521")
        self.assertEqual(row["session_ids"], ["library-session"])
        self.assertEqual(row["hermes_ingress"], "library")
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["root_input_tokens"], 8)
        self.assertEqual(row["subcalls"], 0)
        with patch.object(matrix, "run_process", return_value={"stdout": "", "stderr": "failed",
                         "exit_code": 1, "timed_out": False, "wall_s": 0.01}), \
             patch.object(matrix, "hermes_stats", return_value={}):
            invalid = matrix.run_one(task, "released-rlm", 0, args, {}, self.ledger)
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["score"])
        self.assertIsNone(invalid["correct"])

    def test_orphan_inherited_output_does_not_break_timeout_collection(self):
        pidfile = self.root / "orphan.pid"
        code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)'],start_new_session=True); "
                f"open({str(pidfile)!r},'w').write(str(p.pid)); "
                "print('partial stdout',flush=True); print('partial stderr',file=sys.stderr,flush=True); time.sleep(30)")
        try:
            # A daemon can leave the parent tree before the next polling snapshot.
            with patch.object(matrix, "descendants", return_value=set()):
                result = matrix.run_process([sys.executable, "-c", code], self.root, os.environ.copy(), 0.3)
            self.assertTrue(result["timed_out"])
            self.assertIn("partial stdout", result["stdout"])
            self.assertIn("partial stderr", result["stderr"])
            self.assertLess(result["wall_s"], 1.5)
        finally:
            if pidfile.exists():
                import signal
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_timeout_kills_detached_descendant(self):
        pidfile = self.root / "child.pid"
        code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)'],start_new_session=True); "
                f"open({str(pidfile)!r},'w').write(str(p.pid)); time.sleep(30)")
        result = matrix.run_process([sys.executable, "-c", code], self.root, os.environ.copy(), 0.6)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["wall_s"], 3)
        child = int(pidfile.read_text())
        stat = Path(f"/proc/{child}/stat")
        self.assertTrue(not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")


if __name__ == "__main__":
    unittest.main()
