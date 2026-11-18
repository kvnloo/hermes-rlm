"""RLM_CONCURRENCY=0 must not crash the sub-call pool: workers are floored at 1."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class ConcurrencyFloorTests(unittest.TestCase):
    def test_zero_concurrency_still_serves_batch_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('t', subcall=lambda p, n: (p.upper(), {}), log_dir=Path(root))
            with mock.patch.dict('os.environ', {'RLM_CONCURRENCY': '0'}):
                s.lim = rlm.limits()
            self.assertEqual(s.lim['concurrency'], 0)
            r = s._serve_llm({'id': 1, 'prompts': ['a', 'b', 'c']})
            self.assertEqual(r['results'], ['A', 'B', 'C'])


if __name__ == '__main__':
    unittest.main()
