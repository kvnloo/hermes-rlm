"""Pin Session.exec output truncation at max_output_chars."""
from pathlib import Path
import tempfile
import unittest
import rlm


class OutputTruncation(unittest.TestCase):
    def run_print(self, n, cap=10):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(tmp))
            self.addCleanup(s.kill)
            s.lim = {**s.lim, 'max_output_chars': cap}
            return s.exec(f"print('x' * {n - 1})")['stdout']  # print adds a newline

    def test_output_at_cap_is_untouched(self):
        self.assertEqual(self.run_print(10), 'x' * 9 + '\n')

    def test_output_over_cap_is_cut_and_counts_the_rest(self):
        out = self.run_print(15)
        self.assertTrue(out.startswith('x' * 10 + '\n... [5 more chars truncated;'), out)
        self.assertTrue(out.endswith('keep large values in variables and print summaries]'), out)
