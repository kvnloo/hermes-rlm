"""A non-numeric or non-positive max_tokens from model code is refused as an llm_result error, not raised or charged."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmBadMaxTokens(unittest.TestCase):
    def test_bad_max_tokens_is_refused_uncharged(self):
        for bad in ("abc", [1], float("inf"), -5, "-1", 0.5):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as d:
                calls = []
                s = Session("t", subcall=lambda p, m: calls.append(m) or ("ok", {}), log_dir=Path(d))
                r = s._serve_llm({"id": 7, "prompts": ["a"], "max_tokens": bad})
                self.assertEqual((r["type"], r["id"]), ("llm_result", 7))
                self.assertIn("max_tokens", r["error"])
                self.assertEqual((calls, s.stats["subcalls"]), ([], 0))

    def test_valid_max_tokens_still_reaches_subcall_capped(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            s = Session("t", subcall=lambda p, m: calls.append(m) or ("ok", {}), log_dir=Path(d))
            s._serve_llm({"id": 1, "prompts": ["a"], "max_tokens": "50"})
            s._serve_llm({"id": 2, "prompts": ["a"], "max_tokens": 10**9})
            s._serve_llm({"id": 3, "prompts": ["a"], "max_tokens": 1})
        self.assertEqual(calls, [50, 32_000, 1])


if __name__ == "__main__":
    unittest.main()
