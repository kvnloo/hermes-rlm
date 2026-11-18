"""Overlapping path inputs must not spend the corpus budget twice."""
from pathlib import Path
import tempfile
import unittest

import rlm


class CorpusDedupTests(unittest.TestCase):
    def test_overlapping_files_directory_and_glob_have_one_budget(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            first, second = directory / 'a.txt', directory / 'b.txt'
            first.write_text('abcd', encoding='utf-8')
            second.write_text('xy', encoding='utf-8')
            for paths in ([str(first), str(first), str(second)],
                          [root, str(first)], [root, str(directory / '*.txt')]):
                with self.subTest(paths=paths):
                    corpus, skipped = rlm.load_corpus(paths, 6)
                    self.assertEqual(corpus, {'a.txt': 'abcd', 'b.txt': 'xy'})
                    self.assertEqual(skipped, 0)
                    # Deduplication must not weaken the real unique corpus limit.
                    with self.assertRaisesRegex(ValueError, 'RLM_MAX_CORPUS_CHARS=5'):
                        rlm.load_corpus(paths, 5)

    def test_alternate_spellings_of_one_file_count_once(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'sub').mkdir()
            real = directory / 'a.txt'
            real.write_text('abcd', encoding='utf-8')
            (directory / 'link.txt').symlink_to(real)
            for alias in (str(directory / 'sub' / '..' / 'a.txt'), str(directory / 'link.txt')):
                with self.subTest(alias=alias):
                    corpus, _ = rlm.load_corpus([str(real), alias], 4)
                    self.assertEqual(corpus, {'a.txt': 'abcd'})

    def test_in_tree_symlink_to_file_is_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'a.txt').write_text('abcd', encoding='utf-8')
            (directory / 'link.txt').symlink_to(directory / 'a.txt')
            self.assertEqual(rlm.load_corpus([root], 4)[0], {'a.txt': 'abcd'})

    def test_duplicate_binary_is_counted_once(self):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / 'binary.dat'
            binary.write_bytes(b'\0binary')
            self.assertEqual(rlm.load_corpus([root, str(binary)], 0), ({}, 1))


if __name__ == '__main__':
    unittest.main()
