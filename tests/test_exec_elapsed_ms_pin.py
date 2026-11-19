"""Pin Session.exec's reported `ms` as elapsed time (not a raw monotonic clock reading)."""
import tempfile
import unittest
from pathlib import Path
import rlm


class ExecElapsedMs(unittest.TestCase):
    def test_ms_is_elapsed_for_a_trivial_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(tmp))
            self.addCleanup(s.kill)
            ms = s.exec('1+1')['ms']
        self.assertIsInstance(ms, int)
        self.assertTrue(0 <= ms < 30000, ms)


if __name__ == "__main__":
    unittest.main()
