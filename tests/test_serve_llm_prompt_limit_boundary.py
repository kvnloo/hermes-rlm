"""A prompt of exactly RLM_MAX_PROMPT_CHARS is served; one char more is refused uncharged."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmPromptLimitBoundary(unittest.TestCase):
    def test_limit_is_inclusive(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            s = Session("t", subcall=lambda p, m: calls.append(len(p)) or ("ok", {}), log_dir=Path(d))
            n = s.lim["max_prompt_chars"]
            at = s._serve_llm({"id": 1, "prompts": ["x" * n]})
            over = s._serve_llm({"id": 2, "prompts": ["x" * (n + 1)]})
        self.assertEqual(at["results"], ["ok"])
        self.assertIn("exceeds RLM_MAX_PROMPT_CHARS", over["error"])
        self.assertEqual((calls, s.stats["subcalls"]), ([n], 1))


if __name__ == "__main__":
    unittest.main()
