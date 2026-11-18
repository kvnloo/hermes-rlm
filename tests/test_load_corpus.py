"""load_corpus(): file/dir/glob expansion, relative keys, binary and .git skipping, budget, missing path."""
import tempfile
import unittest
from pathlib import Path

import rlm


def _tree(root: Path) -> None:
    (root / "src" / "sub").mkdir(parents=True)
    (root / "src" / ".git").mkdir()
    (root / "src" / "a.txt").write_text("alpha")
    (root / "src" / "sub" / "b.txt").write_text("beta")
    (root / "src" / ".git" / "cfg").write_text("secret")
    (root / "src" / "bin.dat").write_bytes(b"ab\0cd")


class LoadCorpusTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _tree(self.root)

    def test_directory_keys_are_relative_and_binary_and_git_are_skipped(self):
        corpus, skipped = rlm.load_corpus(str(self.root / "src"), 1000)
        self.assertEqual(corpus, {"a.txt": "alpha", "sub/b.txt": "beta"})
        self.assertEqual(skipped, 1)

    def test_glob_and_list_of_paths(self):
        corpus, _ = rlm.load_corpus([str(self.root / "src" / "**" / "*.txt")], 1000)
        self.assertEqual(corpus, {"a.txt": "alpha", "sub/b.txt": "beta"})
        corpus, _ = rlm.load_corpus([str(self.root / "src" / "a.txt"), str(self.root / "src" / "sub" / "b.txt")], 1000)
        self.assertEqual(set(corpus.values()), {"alpha", "beta"})

    def test_single_file_keyed_by_basename(self):
        corpus, skipped = rlm.load_corpus(str(self.root / "src" / "a.txt"), 1000)
        self.assertEqual((corpus, skipped), ({"a.txt": "alpha"}, 0))

    def test_budget_is_total_chars_and_exact_fit_passes(self):
        paths = [str(self.root / "src" / "a.txt"), str(self.root / "src" / "sub" / "b.txt")]
        self.assertEqual(len(rlm.load_corpus(paths, 9)[0]), 2)
        with self.assertRaisesRegex(ValueError, "RLM_MAX_CORPUS_CHARS=8"):
            rlm.load_corpus(paths, 8)

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            rlm.load_corpus(str(self.root / "nope.txt"), 1000)

    def test_budget_counts_characters_not_bytes(self):
        (self.root / "u.txt").write_text("\u00e9\u00e9\u00e9", encoding="utf-8")  # 3 chars, 6 bytes
        self.assertEqual(rlm.load_corpus(str(self.root / "u.txt"), 3)[0], {"u.txt": "\u00e9\u00e9\u00e9"})

    def test_invalid_utf8_is_replaced_not_raised(self):
        (self.root / "bad.txt").write_bytes(b"x\xffy")
        corpus, _ = rlm.load_corpus(str(self.root / "bad.txt"), 1000)
        self.assertEqual(corpus, {"bad.txt": "x�y"})


if __name__ == "__main__":
    unittest.main()
