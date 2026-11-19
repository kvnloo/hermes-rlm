"""load_corpus counts each file once: `dir/**` matches the dir and its files, overlapping paths repeat files."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class LoadCorpusDedupe(unittest.TestCase):
    def test_recursive_glob_counts_each_file_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "s").mkdir()
            (root / "s" / "a.txt").write_text("hello")
            (root / "b.txt").write_text("hi")
            (root / "bin").write_bytes(b"\0x")
            corpus, skipped = rlm.load_corpus(f"{d}/**", 7)  # exactly 5 + 2 chars
            self.assertEqual(corpus, {"s/a.txt": "hello", "b.txt": "hi"})
            self.assertEqual(list(corpus), ["b.txt", "s/a.txt"])
            self.assertEqual(skipped, 1)

    def test_overlapping_paths_count_once(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "y.txt").write_text("hello")
            corpus, _ = rlm.load_corpus([f"{d}/y.txt", d], 5)
            self.assertEqual(corpus, {"y.txt": "hello"})

    def test_dedupe_keeps_first_seen_order_and_counts_binary_once(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("A")
            (Path(d) / "z.txt").write_text("Z")
            (Path(d) / "bin").write_bytes(b"\0x")
            corpus, skipped = rlm.load_corpus([f"{d}/z.txt", f"{d}/bin", d], 100)
            self.assertEqual(list(corpus), ["z.txt", "a.txt"])  # first-seen, not sorted
            self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
