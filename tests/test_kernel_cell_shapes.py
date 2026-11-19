"""Pin kernel._run: cells with no trailing expression (empty, comment-only, assignment) must run clean."""
from pathlib import Path
import tempfile
import unittest
import rlm


class CellShapeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(self.tmp.name))
        self.addCleanup(self.s.kill)

    def test_empty_and_comment_only_cells_succeed_silently(self):
        for code in ('', '# nothing'):
            r = self.s.exec(code)
            self.assertEqual((r['error'], r['stdout']), (None, ''), code)

    def test_assignment_cell_executes_and_prints_nothing_then_expression_echoes(self):
        r = self.s.exec('x = 41')
        self.assertEqual((r['error'], r['stdout']), (None, ''))
        r = self.s.exec('x + 1')
        self.assertEqual((r['error'], r['stdout'].strip()), (None, '42'))

    def test_only_trailing_expression_is_echoed_as_repr(self):
        r = self.s.exec("1\n'a'")
        self.assertEqual((r['error'], r['stdout'].strip()), (None, "'a'"))
        r = self.s.exec('y = 2\ny')
        self.assertEqual((r['error'], r['stdout'].strip()), (None, '2'))
        r = self.s.exec('None')
        self.assertEqual((r['error'], r['stdout']), (None, ''))


if __name__ == '__main__':
    unittest.main()
