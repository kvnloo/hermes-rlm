"""Output-cap boundary, cell counter, and .git exclusion in directory corpora."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class OutputCellsGitTests(unittest.TestCase):
    def _exec(self, session, stdout):
        session.proc = mock.Mock()
        session.proc.poll.return_value = None
        reply = {'type': 'done', 'stdout': stdout, 'error': None}
        with mock.patch.object(session, '_request', return_value=reply):
            return session.exec('x')

    def test_output_cap_is_inclusive_and_counts_cells(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('test', log_dir=Path(root))
            session.lim['max_output_chars'] = 5
            self.assertEqual(self._exec(session, 'abcde')['stdout'], 'abcde')
            shown = self._exec(session, 'abcdef')['stdout']
            self.assertTrue(shown.startswith('abcde\n'))
            self.assertIn('1 more chars truncated', shown)
            self.assertEqual(session.stats['cells'], 2)

    def test_directory_corpus_skips_git_dirs_at_any_depth(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / 'a.txt').write_text('keep', encoding='utf-8')
            for rel in ('.git/HEAD', 'sub/.git/x'):
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_text('secret', encoding='utf-8')
            self.assertEqual(rlm.load_corpus([root], 100)[0], {'a.txt': 'keep'})


if __name__ == '__main__':
    unittest.main()
