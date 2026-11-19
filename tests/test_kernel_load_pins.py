"""Pin kernel.main 'load': context becomes visible, the load reply is consumed."""
from pathlib import Path
import tempfile
import unittest
import rlm


class KernelLoadPins(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(self.tmp.name))
        self.addCleanup(self.s.kill)

    def test_context_visible_and_next_exec_in_sync(self):
        self.s.context = 'corpus-text'
        self.assertEqual(self.s.exec('context')['stdout'].strip(), "'corpus-text'")
        self.assertEqual(self.s.exec('1 + 1')['stdout'].strip(), '2')
