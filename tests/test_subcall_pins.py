"""Pin Session._serve_llm: max_tokens request default and 32k cap, and the subcall log row's timing."""
import json
import time
import tempfile
import unittest
from pathlib import Path
import rlm


class SubcallPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.seen = []
        self.s = rlm.Session('t', subcall=self.sub, log_dir=Path(tmp.name))

    def sub(self, prompt, max_tokens):
        self.seen.append(max_tokens)
        time.sleep(0.03)
        return 'ok', {}

    def serve(self, **req):
        return self.s._serve_llm({'id': 1, 'prompts': ['p'], **req})

    def test_max_tokens_default_request_and_cap(self):
        self.s.lim = {**self.s.lim, 'subcall_max_tokens': 123}
        self.serve()
        self.serve(max_tokens=None)  # what kernel.llm_query sends when the caller gives none
        self.serve(max_tokens=0)
        self.serve(max_tokens=500)
        self.serve(max_tokens=32_000)
        self.serve(max_tokens=32_001)
        self.assertEqual(self.seen, [123, 123, 123, 500, 32_000, 32_000])

    def test_subcall_log_row_has_elapsed_ms(self):
        self.serve()
        row = json.loads(self.s.log_path.read_text().splitlines()[-1])
        self.assertEqual((row['event'], row['prompt_chars'], row['output_chars']), ('subcall', 1, 2))
        self.assertIsInstance(row['ms'], int)
        self.assertGreaterEqual(row['ms'], 20)
        self.assertLess(row['ms'], 5000)  # elapsed, not a raw monotonic clock


if __name__ == "__main__":
    unittest.main()
