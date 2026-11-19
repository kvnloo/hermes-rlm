"""default_subcall keeps the model's text when the host lane reports usage=None."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class DefaultSubcallNoneUsage(unittest.TestCase):
    def setUp(self):
        self._ctx = rlm._CTX
        rlm._CTX = SimpleNamespace(llm=SimpleNamespace(
            complete=lambda **kw: SimpleNamespace(text="ok", usage=None)))

    def tearDown(self):
        rlm._CTX = self._ctx

    def test_text_survives_and_usage_is_empty(self):
        self.assertEqual(rlm.default_subcall("p", 8), ("ok", {}))

    def test_none_text_with_none_usage_is_empty_string(self):
        rlm._CTX = SimpleNamespace(llm=SimpleNamespace(
            complete=lambda **kw: SimpleNamespace(text=None, usage=None)))
        self.assertEqual(rlm.default_subcall("p", 8), ("", {}))

    def test_serve_llm_returns_text_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            s = rlm.Session("t", log_dir=Path(d))
            r = s._serve_llm({"id": 1, "prompts": ["a"]})
        self.assertEqual(r["results"], ["ok"])
        self.assertEqual((s.stats["prompt_tokens"], s.stats["completion_tokens"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
