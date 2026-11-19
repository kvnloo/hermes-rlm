"""Pin kernel.main: error reporting, FINAL per-cell reset, SystemExit from model code, stderr capture."""
from pathlib import Path
import tempfile
import unittest
import rlm


class KernelMainPins(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(self.tmp.name))
        self.addCleanup(self.s.kill)

    def test_exception_reports_traceback_and_keeps_namespace(self):
        self.s.exec('keep = 7')
        r = self.s.exec('1 / 0')
        self.assertIn('ZeroDivisionError', r['error'])
        self.assertEqual(self.s.exec('keep')['stdout'].strip(), '7')
        self.assertEqual(self.s.stats['restarts'], 0)

    def test_systemexit_is_reported_without_killing_kernel(self):
        self.s.exec('keep = 7')
        r = self.s.exec('raise SystemExit(3)')
        self.assertIn('SystemExit', r['error'])
        self.assertEqual(self.s.stats['restarts'], 0)
        self.assertEqual(self.s.exec('keep')['stdout'].strip(), '7')

    def test_stderr_is_captured_into_stdout(self):
        r = self.s.exec('import sys; sys.stderr.write("oops")')
        self.assertIn('oops', r['stdout'])

    def test_final_is_per_cell(self):
        r = self.s.exec('FINAL(5)')
        self.assertEqual(r.get('final'), '5')
        self.assertIsNone(self.s.exec('1').get('final'))

    def test_error_is_per_cell(self):
        self.assertIn('ZeroDivisionError', self.s.exec('1 / 0')['error'])
        self.assertIsNone(self.s.exec('1').get('error'))
