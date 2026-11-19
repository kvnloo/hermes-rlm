"""Pin bench/run_bench.py scoring, prompt, and usage readers against real make_corpus gold and real Session logs."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import make_corpus  # noqa: E402
import run_bench  # noqa: E402
import rlm  # noqa: E402


def _gold(seed=7):
    _, qs, meta = make_corpus.build(3000, seed)
    return {q["id"]: q for q in json.loads(json.dumps(qs))}, meta


class Score(unittest.TestCase):
    def test_gold_answers_score_correct_and_near_misses_do_not(self):
        for seed in range(1, 8):
            g, _ = _gold(seed)
            n, b, t = g["niah"], g["count_billing"], g["top_bug_user"]
            self.assertTrue(run_bench.score(n, f"ANSWER: {n['gold']}"))
            self.assertTrue(run_bench.score(n, "ANSWER: " + n["gold"].replace("-", " ")))
            self.assertFalse(run_bench.score(n, "ANSWER: " + n["gold"][:-1] + str((int(n["gold"][-1]) + 1) % 10)))
            self.assertTrue(run_bench.score(b, f"ANSWER: {b['gold']}"))
            self.assertFalse(run_bench.score(b, f"ANSWER: {b['gold'] + 1}"))
            self.assertTrue(run_bench.score(t, f"ANSWER: {t['gold']}"))
            user, count = t["gold"].split()
            self.assertFalse(run_bench.score(t, f"ANSWER: u999 {count}"))
            self.assertFalse(run_bench.score(t, f"ANSWER: {user} {int(count) + 1}"))
            self.assertFalse(run_bench.score(t, f"ANSWER: {user}"))

    def test_any_tied_user_is_accepted(self):
        q = {"id": "top_bug_user", "gold": "u010 5", "ties": ["u010", "u020"]}
        self.assertTrue(run_bench.score(q, "ANSWER: u020 5"))
        self.assertFalse(run_bench.score(q, "ANSWER: u030 5"))

    def test_thousands_separator_and_first_user_and_whole_word_count(self):
        self.assertTrue(run_bench.score({"id": "count_billing", "gold": 1042}, "ANSWER: 1,042"))
        t = {"id": "top_bug_user", "gold": "u010 5", "ties": ["u010"]}
        self.assertTrue(run_bench.score(t, "answer: U010 5 (u020 filed 9)"))  # case-insensitive, first user decides
        self.assertFalse(run_bench.score(t, "ANSWER: u020 5, then u010"))
        self.assertFalse(run_bench.score(t, "ANSWER: u010 15"))

    def test_top_bug_user_answer_without_a_user_is_false_not_a_crash(self):
        t = {"id": "top_bug_user", "gold": "u010 5", "ties": ["u010"]}
        for ans in ("ANSWER: 5", "ANSWER: none", "", None):
            self.assertFalse(run_bench.score(t, ans))

    def test_last_answer_line_wins_and_fallback_is_tail(self):
        q = {"id": "count_billing", "gold": 42}
        self.assertTrue(run_bench.score(q, "ANSWER: 7\nthinking\nANSWER: 42"))
        self.assertFalse(run_bench.score(q, "ANSWER: 42\nANSWER: 7"))
        self.assertTrue(run_bench.score(q, "x" * 400 + " 42"))
        self.assertFalse(run_bench.score(q, "42 " + "x" * 400))  # only the last 300 chars count
        self.assertTrue(run_bench.score(q, "42" + "x" * 298))  # exactly the last 300 chars
        self.assertFalse(run_bench.score(q, "42" + "x" * 299))
        self.assertFalse(run_bench.score(q, None))

    def test_count_does_not_match_inside_a_larger_number_and_unknown_id_is_false(self):
        q = {"id": "count_billing", "gold": 42}
        self.assertFalse(run_bench.score(q, "ANSWER: 142"))
        self.assertTrue(run_bench.score(q, "ANSWER: about 1,042 or 42"))
        self.assertFalse(run_bench.score({"id": "other", "gold": "x"}, "ANSWER: x"))


class Prompt(unittest.TestCase):
    def test_prompt_names_file_size_question_and_answer_line(self):
        g, meta = _gold()
        p = run_bench.prompt_for(g["niah"], meta, Path("/tmp/c/corpus.txt"))
        self.assertIn("/tmp/c/corpus.txt", p)
        self.assertIn(f"{meta['approx_tokens']:,} tokens ({meta['records']:,} records)", p)
        self.assertIn(g["niah"]["q"], p)
        self.assertTrue(p.endswith("'ANSWER: <answer>'."))
        self.assertNotIn(g["niah"]["gold"], p)


class Usage(unittest.TestCase):
    def test_rlm_usage_reads_a_real_session_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session("t", subcall=lambda p, n: ("ok", {"prompt_tokens": 11, "completion_tokens": 3}),
                            log_dir=Path(tmp))
            self.addCleanup(s.kill)
            s.exec("llm_batch(['a', 'b'])")
            s.exec("1")
            self.assertEqual(run_bench.rlm_usage(s.log_path),
                             {"subcalls": 2, "sub_prompt_tokens": 22, "sub_completion_tokens": 6, "cells": 2})

    def test_rlm_usage_counts_failed_subcall_rows_without_tokens(self):
        def sub(p, n):
            if p == "bad":
                raise RuntimeError("boom")
            return "ok", {"prompt_tokens": 5, "completion_tokens": 1}
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session("t", subcall=sub, log_dir=Path(tmp))
            self.addCleanup(s.kill)
            s.exec("llm_batch(['good', 'bad'])")
            self.assertEqual(run_bench.rlm_usage(s.log_path),
                             {"subcalls": 2, "sub_prompt_tokens": 5, "sub_completion_tokens": 1, "cells": 1})

    def test_rlm_usage_missing_log_is_zeroes(self):
        self.assertEqual(run_bench.rlm_usage(Path("/nonexistent/x.jsonl")),
                         {"subcalls": 0, "sub_prompt_tokens": 0, "sub_completion_tokens": 0, "cells": 0})

    def test_hermes_usage_maps_columns_and_missing_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = sqlite3.connect(Path(tmp) / "state.db")
            db.execute("create table sessions (id text, input_tokens int, output_tokens int, "
                       "cache_read_tokens int, api_call_count int, tool_call_count int)")
            db.execute("insert into sessions values ('s1', 1, 2, 3, 4, 5)")
            db.commit()
            db.close()
            old = run_bench.HERMES_HOME
            run_bench.HERMES_HOME = Path(tmp)
            try:
                self.assertEqual(run_bench.hermes_usage("s1"),
                                 {"root_input_tokens": 1, "root_output_tokens": 2, "root_cache_read_tokens": 3,
                                  "api_calls": 4, "tool_calls": 5})
                self.assertEqual(run_bench.hermes_usage("nope"), {})
            finally:
                run_bench.HERMES_HOME = old

    def test_hermes_usage_closes_its_sqlite_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = sqlite3.connect(Path(tmp) / "state.db")
            seed.execute("create table sessions (id text, input_tokens int, output_tokens int, "
                         "cache_read_tokens int, api_call_count int, tool_call_count int)")
            seed.execute("insert into sessions values ('s1', 1, 2, 3, 4, 5)")
            seed.commit()
            seed.close()
            opened, real = [], sqlite3.connect
            def spy(*a, **k):
                opened.append(real(*a, **k))
                return opened[-1]
            old_home, old_connect = run_bench.HERMES_HOME, run_bench.sqlite3.connect
            run_bench.HERMES_HOME, run_bench.sqlite3.connect = Path(tmp), spy
            try:
                self.assertTrue(run_bench.hermes_usage("s1"))  # hit path
                self.assertEqual(run_bench.hermes_usage("nope"), {})  # miss path
            finally:
                run_bench.HERMES_HOME, run_bench.sqlite3.connect = old_home, old_connect
            self.assertEqual(len(opened), 2)
            for c in opened:
                with self.assertRaises(sqlite3.ProgrammingError):
                    c.execute("select 1")


if __name__ == "__main__":
    unittest.main()
