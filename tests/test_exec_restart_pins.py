"""Pin Session.exec restart paths: cell timeout and kernel crash both restart and reload context."""
from pathlib import Path
import tempfile
import unittest
import rlm


class ExecRestartPins(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(self.tmp.name))
        self.addCleanup(self.s.kill)
        self.s.context = 'corpus'
        self.s.exec('keep = 7')

    def assert_restarted(self, r, msg):
        self.assertIn(msg, r['error'])
        self.assertEqual(self.s.stats['restarts'], 1)
        self.assertIn('NameError', self.s.exec('keep')['error'])
        self.assertEqual(self.s.exec('context')['stdout'].strip(), "'corpus'")

    def test_timeout_kills_and_restarts_with_context(self):
        self.s.lim = {**self.s.lim, 'cell_timeout_s': 1}
        r = self.s.exec('import time; time.sleep(5)')
        self.s.lim = {**self.s.lim, 'cell_timeout_s': 60}
        self.assert_restarted(r, 'exceeded 1s and was killed')

    def test_kernel_crash_restarts_with_context(self):
        r = self.s.exec('import os; os._exit(1)')
        self.assert_restarted(r, 'sandbox kernel exited')


class SubcallBudgetBoundary(unittest.TestCase):
    def test_budget_allows_exactly_max_subcalls_then_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session('t', subcall=lambda p, n: ('ok', {}), log_dir=Path(tmp))
            self.addCleanup(s.kill)
            s.lim = {**s.lim, 'max_subcalls': 2}
            s.exec('a = llm_batch(["x", "y"])')
            self.assertIsNone(s.exec('print(a)')['error'])
            self.assertEqual(s.stats['subcalls'], 2)
            self.assertIn('budget exhausted', s.exec('llm_query("z")')['error'])
