"""_serve_llm treats a subcall returning usage=None as empty usage (defensive; no observed failure)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmNoneUsage(unittest.TestCase):
    def test_none_usage_counts_as_zero_and_batch_survives(self):
        usages = iter([None, {"prompt_tokens": 4, "completion_tokens": 1}])
        with tempfile.TemporaryDirectory() as d:
            s = Session("t", subcall=lambda p, m: ("ok", next(usages)), log_dir=Path(d))
            r = s._serve_llm({"id": 1, "prompts": ["a", "b"]})
        self.assertEqual(r["results"], ["ok", "ok"])
        self.assertEqual((s.stats["prompt_tokens"], s.stats["completion_tokens"]), (4, 1))


if __name__ == "__main__":
    unittest.main()
