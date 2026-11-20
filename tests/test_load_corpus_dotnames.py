"""load_corpus skips only a path component named exactly `.git`; look-alike names (.github, .gitignore) are real corpus files."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class LoadCorpusDotNames(unittest.TestCase):
    def test_only_exact_dot_git_component_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in (".git/config", ".github/ci.yml", ".gitignore", "x.git/y.txt", "a.txt"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(rel)
            corpus, skipped = rlm.load_corpus(d, 1000)
            self.assertEqual(
                sorted(corpus), [".github/ci.yml", ".gitignore", "a.txt", "x.git/y.txt"]
            )
            self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
