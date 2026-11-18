"""Resolved paths can still collide in the returned lexical source-key map."""
from pathlib import Path
import tempfile
import unittest
import rlm


class CorpusSourceKeyTests(unittest.TestCase):
    def test_replacement_budget_tracks_retained_value_not_all_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root, other = Path(temp) / 'root', Path(temp) / 'other'
            root.mkdir()
            (other / 'sub').mkdir(parents=True)
            (root / 'z.txt').write_text('old')
            (other / 'z.txt').write_text('new!')
            (root / 'link').symlink_to(other / 'sub', target_is_directory=True)
            paths = [str(root / 'z.txt'), str(root / 'link' / '..' / 'z.txt')]
            self.assertEqual(rlm.load_corpus(paths, 4), ({'z.txt': 'new!'}, 0))
            self.assertEqual(rlm.load_corpus(paths[::-1], 4), ({'z.txt': 'old'}, 0))
            with self.assertRaisesRegex(ValueError, 'RLM_MAX_CORPUS_CHARS=3'):
                rlm.load_corpus(paths, 3)

    def test_distinct_source_keys_still_charge_both_values(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'a.txt', Path(temp) / 'b.txt'
            first.write_text('old')
            second.write_text('new!')
            paths = [str(first), str(second)]
            self.assertEqual(rlm.load_corpus(paths, 7), ({'a.txt': 'old', 'b.txt': 'new!'}, 0))
            with self.assertRaisesRegex(ValueError, 'RLM_MAX_CORPUS_CHARS=6'):
                rlm.load_corpus(paths, 6)
