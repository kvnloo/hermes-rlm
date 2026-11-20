"""Pin run_bench.main() row details the outage/repeat pins leave open: error guard, ts, corpus path, defaults, stdout summary."""
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import make_corpus  # noqa: E402
import run_bench  # noqa: E402


class MainRow(unittest.TestCase):
    def _run(self, *extra, runner):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d)
        corpus = d / "c"
        corpus.mkdir()
        (corpus / "corpus.txt").write_text("x")
        _, qs, meta = make_corpus.build(3000, 7)
        (d / "c.gold.json").write_text(json.dumps({"questions": qs[:1], "meta": meta}))
        out, buf, err = d / "out.jsonl", io.StringIO(), None
        with mock.patch.object(sys, "argv", ["run_bench", str(corpus), str(out), *extra]), \
                mock.patch.object(run_bench, "run_hermes", side_effect=runner) as h, \
                mock.patch.object(run_bench, "run_nano", side_effect=runner) as n, \
                contextlib.redirect_stdout(buf):
            try:
                run_bench.main()
            except SystemExit as e:
                err = e
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        return rows, err, buf.getvalue(), h, n, corpus

    def test_existing_error_is_not_overwritten_as_provider_outage(self):
        rows, err, _, _, _, _ = self._run("--arms", "nano-rlm", runner=lambda *a: {"answer": "HTTP 429", "error": "Traceback: rlm exited 1"})
        self.assertIsNone(err)  # run continues
        self.assertEqual(rows[0]["error"], "Traceback: rlm exited 1")  # not relabelled "provider"
        self.assertIs(rows[0]["correct"], False)

    def test_row_ts_is_wall_clock_now_and_prompt_reads_corpus_txt(self):
        before = time.time()
        with mock.patch.object(run_bench, "prompt_for", wraps=run_bench.prompt_for) as p:
            rows, _, _, _, _, corpus = self._run("--arms", "nano-rlm", runner=lambda *a: {"answer": "x"})
        self.assertTrue(before <= rows[0]["ts"] <= time.time())
        self.assertEqual(p.call_args.args[2], corpus.resolve() / "corpus.txt")

    def test_defaults_run_all_three_arms_on_kilo_free(self):
        rows, _, _, h, n, corpus = self._run(runner=lambda *a: {"answer": "x"})
        self.assertEqual([r["arm"] for r in rows], ["hermes-rlm", "hermes-tools", "nano-rlm"])
        self.assertEqual(h.call_args.args[3].provider, "kilo-free")
        self.assertEqual(h.call_args.args[4], corpus.resolve().parent / "runs")
        self.assertEqual(n.call_args.args[2].provider, "kilo-free")

    def test_stdout_is_one_summary_json_line_per_row_with_fixed_fields(self):
        _, _, out, _, _, _ = self._run("--arms", "nano-rlm", runner=lambda *a: {"answer": "x", "secret": 1})
        (line,) = out.splitlines()
        self.assertEqual(list(json.loads(line)), ["arm", "question", "correct", "wall_s",
                                                   "root_input_tokens", "subcalls", "nano_stop", "error"])

    def test_rerun_into_same_out_appends_rows_and_reuses_runs_dir(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d)
        (d / "c").mkdir()
        (d / "c" / "corpus.txt").write_text("x")
        _, qs, meta = make_corpus.build(3000, 7)
        (d / "c.gold.json").write_text(json.dumps({"questions": qs[:1], "meta": meta}))
        out = d / "out.jsonl"
        for _ in range(2):
            with mock.patch.object(sys, "argv", ["run_bench", str(d / "c"), str(out), "--arms", "nano-rlm"]), \
                    mock.patch.object(run_bench, "run_nano", return_value={"answer": "x"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_bench.main()
        self.assertEqual(len(out.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
