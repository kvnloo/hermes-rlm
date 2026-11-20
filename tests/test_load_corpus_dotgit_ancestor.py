"""load_corpus skips .git inside the corpus dir, not an ancestor named .git above it."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class LoadCorpusDotGit(unittest.TestCase):
    def test_dir_under_dotgit_ancestor_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            corpus_dir = Path(d) / ".git" / "docs"
            corpus_dir.mkdir(parents=True)
            (corpus_dir / "a.txt").write_text("hello")
            corpus, _ = rlm.load_corpus(str(corpus_dir), 100)
            self.assertEqual(corpus, {"a.txt": "hello"})

    def test_corpus_dir_named_dotgit_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            corpus_dir = Path(d) / ".git"
            corpus_dir.mkdir()
            (corpus_dir / "a.txt").write_text("hello")
            corpus, _ = rlm.load_corpus(str(corpus_dir), 100)
            self.assertEqual(corpus, {"a.txt": "hello"})

    def test_dotgit_inside_dir_is_still_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".git").mkdir()
            (Path(d) / ".git" / "config").write_text("x")
            (Path(d) / "a.txt").write_text("hello")
            corpus, _ = rlm.load_corpus(d, 100)
            self.assertEqual(corpus, {"a.txt": "hello"})

    def test_nested_and_deep_dotgit_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in (".git/objects/aa/x", "sub/.git/HEAD", "sub/.git/objects/bb/y"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("x")
            (root / "sub" / "b.txt").write_text("hi")
            corpus, _ = rlm.load_corpus(d, 100)
            self.assertEqual(corpus, {"b.txt": "hi"})


if __name__ == "__main__":
    unittest.main()
