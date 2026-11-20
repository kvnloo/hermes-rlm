"""llm_batch("text") must not iterate the string into one sub-call per character."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class LlmBatchBareString(unittest.TestCase):
    def run_cell(self, code):
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"RLM_ALLOW_UNSANDBOXED": "1"}), \
                mock.patch("shutil.which", return_value=None):
            s = Session("t", subcall=lambda p, m: calls.append(p) or ("ok", {}), log_dir=Path(d))
            try:
                r = s.exec(code)
            finally:
                s.kill()
        return r, calls, s.stats["subcalls"]

    def test_bare_string_is_refused_before_any_subcall(self):
        for lit in ("'abc'", "b'abc'"):
            with self.subTest(lit=lit):
                r, calls, charged = self.run_cell(f"llm_batch({lit})")
                self.assertEqual((calls, charged), ([], 0))
                self.assertIn("TypeError", r["error"])
                self.assertIn("llm_query", r["error"])

    def test_list_and_llm_query_still_work(self):
        r, calls, charged = self.run_cell("print(llm_batch(['a', 'b']), llm_query('c'))")
        self.assertEqual((calls, charged), (["a", "b", "c"], 3))
        self.assertEqual(r["stdout"].strip(), "['ok', 'ok'] ok")


if __name__ == "__main__":
    unittest.main()
