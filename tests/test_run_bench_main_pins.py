"""Pin run_bench.run_hermes parsing and main()'s provider-outage stop (no hermes binary, network or user home)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import make_corpus  # noqa: E402
import run_bench  # noqa: E402


def _cp(out="", err=""):
    return subprocess.CompletedProcess([], 0, out, err)


class RunHermes(unittest.TestCase):
    def _run(self, arm, proc, run_dir):
        args = SimpleNamespace(provider="prov", model="mod", timeout=5)
        with mock.patch.object(run_bench.subprocess, "run", return_value=proc) as r, \
                mock.patch.object(run_bench, "hermes_usage", return_value={"api_calls": 3}):
            row = run_bench.run_hermes(arm, "PROMPT", "/corp", args, run_dir)
        return row, r

    def test_answer_is_text_before_last_session_marker_and_session_id_is_last(self):
        with tempfile.TemporaryDirectory() as d:
            row, r = self._run("hermes-tools", _cp("the reply\nANSWER: 5\nsession_id: s1\nsession_id: s2\n", "err"), Path(d))
        self.assertEqual(row["answer"], "the reply\nANSWER: 5")
        self.assertEqual(row["session_id"], "s2")
        self.assertEqual(row["api_calls"], 3)
        self.assertNotIn("subcalls", row)  # only the hermes-rlm arm reads the rlm log
        cmd = r.call_args.args[0]
        self.assertEqual(cmd, ["hermes", "chat", "--oneshot", "-Q", "-t", "file,terminal,code_execution",
                               "--provider", "prov", "-m", "mod", "-q", "PROMPT"])
        kw = r.call_args.kwargs
        self.assertEqual((kw["cwd"], kw["timeout"]), ("/corp", 5))
        self.assertEqual(kw["env"]["RLM_LOG_DIR"], d)
        self.assertEqual(kw["env"]["PATH"], os.environ["PATH"])  # inherited, not replaced
        self.assertEqual(kw["env"]["HERMES_HOME"], str(run_bench.HERMES_HOME))

    def test_stderr_is_part_of_the_text_and_wall_is_monotonic_elapsed_to_2dp(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(run_bench.time, "monotonic", side_effect=[10.0, 11.23456]):
            row, _ = self._run("hermes-tools", _cp("out\n", "session_id: from-err\n"), Path(d))
        self.assertEqual((row["session_id"], row["wall_s"], row["answer"]), ("from-err", 1.23, "out"))

    def test_no_marker_gives_empty_session_id_and_answer_tail_is_capped(self):
        with tempfile.TemporaryDirectory() as d:
            row, _ = self._run("hermes-tools", _cp("h" * 500 + "t" * 2000), Path(d))
        self.assertEqual(row["session_id"], "")
        self.assertEqual(row["answer"], "t" * 2000)  # the tail is kept, not the head

    def test_rlm_arm_adds_usage_from_that_sessions_log(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sid.jsonl").write_text(
                json.dumps({"event": "subcall", "prompt_tokens": 4, "completion_tokens": 1}) + "\n"
                + json.dumps({"event": "cell"}) + "\n")
            row, r = self._run("hermes-rlm", _cp("a\nsession_id: sid\n"), Path(d))
        self.assertEqual((row["subcalls"], row["cells"], row["sub_prompt_tokens"]), (1, 1, 4))
        self.assertEqual(r.call_args.args[0][5], "rlm")


class MainOutage(unittest.TestCase):
    def _main(self, answers, arms="hermes-rlm,hermes-tools", questions=""):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d)
        corpus = d / "c"
        corpus.mkdir()
        (corpus / "corpus.txt").write_text("x")
        _, qs, meta = make_corpus.build(3000, 7)
        (d / "c.gold.json").write_text(json.dumps({"questions": qs, "meta": meta}))
        out = d / "out.jsonl"
        argv = ["run_bench", str(corpus), str(out), "--arms", arms, "--questions", questions]
        procs = [a if isinstance(a, Exception) else _cp(a + "\nsession_id: s\n") for a in answers]
        err = None
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(run_bench.subprocess, "run", side_effect=procs) as r, \
                mock.patch.object(run_bench, "hermes_usage", return_value={}):
            try:
                run_bench.main()
            except SystemExit as e:
                err = e
        rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        return rows, err, r.call_count, qs

    def test_provider_outage_writes_invalid_row_then_stops_the_run(self):
        for text in ("HTTP 429 too many", "provider is rate-limited", "I didn't produce a reply", "Agent didn't produce a reply"):
            rows, err, calls, _ = self._main([text, "ANSWER: late"])
            self.assertIsNotNone(err, text)
            self.assertIn("provider unavailable", str(err.code))
            self.assertEqual(calls, 1)  # no further arm/question runs after the outage
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["error"], rows[0]["correct"]), ("provider", None))

    def test_normal_answers_run_every_arm_and_question_and_are_scored(self):
        niah = next(q for q in make_corpus.build(3000, 7)[1] if q["id"] == "niah")
        rows, err, calls, _ = self._main([f"ANSWER: {niah['gold']}", "ANSWER: nope"], questions="niah")
        self.assertIsNone(err)
        self.assertEqual(calls, 2)
        self.assertEqual([(r["arm"], r["question"], r["correct"], r["error"] if "error" in r else None) for r in rows],
                         [("hermes-rlm", "niah", True, None), ("hermes-tools", "niah", False, None)])

    def test_timeout_row_is_an_error_not_provider_and_run_continues(self):
        rows, err, calls, _ = self._main([subprocess.TimeoutExpired("h", 900), "ANSWER: x"], questions="niah")
        self.assertIsNone(err)
        self.assertEqual(calls, 2)
        self.assertEqual((rows[0]["error"], rows[0]["wall_s"], rows[0]["answer"], rows[0]["correct"]),
                         ("timeout", 900, "", False))
        self.assertNotIn("error", rows[1])

    def test_marker_phrases_only_count_inside_the_answer(self):
        rows, err, calls, _ = self._main(["fine ANSWER: x", "was it rate limited? no"], questions="niah")
        self.assertIsNone(err)
        self.assertEqual(calls, 2)
        self.assertTrue(all("error" not in r for r in rows))


if __name__ == "__main__":
    unittest.main()
