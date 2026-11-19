"""_serve_llm: a subcall returning text=None yields '' for that slice and does not sink the batch."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmNoneText(unittest.TestCase):
    def test_none_text_becomes_empty_string_and_batch_survives(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session("t", subcall=lambda p, m: (None if p == "a" else "ok", {}), log_dir=Path(d))
            r = s._serve_llm({"id": 1, "prompts": ["a", "b"]})
        self.assertEqual(r["results"], ["", "ok"])


if __name__ == "__main__":
    unittest.main()
