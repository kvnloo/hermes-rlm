"""A malformed subcall result (bad usage values, non-dict usage, non-text) must not raise out of _serve_llm.

An escaped TypeError propagates out of Session.exec (it only catches TimeoutError/RuntimeError).
"""
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


def serve(result):
    with tempfile.TemporaryDirectory() as d:
        s = Session("t", subcall=lambda p, m: result, log_dir=Path(d))
        r = s._serve_llm({"id": 1, "prompts": ["a"]})
        rows = [json.loads(x) for x in s.log_path.read_text().splitlines()]
        return r, s.stats, [x for x in rows if x["event"] == "subcall"]


class ServeLlmMalformedResult(unittest.TestCase):
    def test_bad_usage_keeps_text_and_counts_nothing(self):
        for bad in ("bad", [1], {"prompt_tokens": "5"}, {"prompt_tokens": float("nan")},
                    {"cost_usd": Decimal("0.1")}, {"ms": 1, "event": "x"}):
            with self.subTest(usage=bad):
                r, stats, rows = serve(("txt", bad))
                self.assertEqual(r["results"], ["txt"])
                self.assertEqual((stats["prompt_tokens"], stats["completion_tokens"]), (0, 0))
                self.assertEqual(len(rows), 1)

    def test_good_numbers_survive_and_foreign_keys_are_dropped(self):
        r, stats, rows = serve(("t", {"prompt_tokens": 3, "completion_tokens": 4, "cost_usd": 0.5, "ms": 99}))
        self.assertEqual((stats["prompt_tokens"], stats["completion_tokens"]), (3, 4))
        self.assertEqual((rows[0]["prompt_tokens"], rows[0]["cost_usd"]), (3, 0.5))
        self.assertNotEqual(rows[0]["ms"], 99)

    def test_bool_usage_is_not_a_number(self):
        # bool is an int subclass: True would be charged as 1 token / logged as `true`
        r, stats, rows = serve(("txt", {"prompt_tokens": True, "completion_tokens": False, "cost_usd": True}))
        self.assertEqual(r["results"], ["txt"])
        self.assertEqual((stats["prompt_tokens"], stats["completion_tokens"]), (0, 0))
        for k in ("prompt_tokens", "completion_tokens", "cost_usd"):
            self.assertNotIn(k, rows[0])

    def test_non_finite_usage_never_reaches_stats_or_log(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=bad):
                r, stats, rows = serve(("txt", {"prompt_tokens": bad, "completion_tokens": bad, "cost_usd": bad}))
                self.assertEqual(r["results"], ["txt"])
                self.assertEqual((stats["prompt_tokens"], stats["completion_tokens"]), (0, 0))
                for k in ("prompt_tokens", "completion_tokens", "cost_usd"):
                    self.assertNotIn(k, rows[0])

    def test_negative_usage_never_reaches_stats_or_log_but_zero_does(self):
        # a negative count would decrement the running totals / be summed by log readers
        r, stats, rows = serve(("txt", {"prompt_tokens": -500, "completion_tokens": -7, "cost_usd": -1.0}))
        self.assertEqual(r["results"], ["txt"])
        self.assertEqual((stats["prompt_tokens"], stats["completion_tokens"]), (0, 0))
        for k in ("prompt_tokens", "completion_tokens", "cost_usd"):
            self.assertNotIn(k, rows[0])
        _, _, rows = serve(("txt", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}))
        self.assertEqual((rows[0]["prompt_tokens"], rows[0]["cost_usd"]), (0, 0.0))

    def test_non_text_result_becomes_error_slice(self):
        r, _, _ = serve((5, {"prompt_tokens": 1}))
        self.assertTrue(r["results"][0].startswith("[llm_query error: TypeError"))


if __name__ == "__main__":
    unittest.main()
