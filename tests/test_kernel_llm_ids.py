"""Pin the kernel's llm request wire ids: sequential from 1, one per llm_batch call, not per prompt."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class KernelLlmIds(unittest.TestCase):
    def test_request_ids_count_batch_calls_from_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session('t', subcall=lambda p, n: ('r', {}), log_dir=Path(tmp))
            self.addCleanup(s.kill)
            ids = []
            real = s._serve_llm
            s._serve_llm = lambda req: (ids.append(req['id']), real(req))[1]
            s.exec("llm_query('a')")
            s.exec("llm_batch(['b', 'c', 'd'])")
            s.exec("llm_query('e')")
        self.assertEqual(ids, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
