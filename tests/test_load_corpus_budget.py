"""load_corpus budget: counts decoded characters (not bytes), is inclusive at the limit, and ignores skipped binaries."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


def load(files: dict[str, bytes], max_chars: int):
    with tempfile.TemporaryDirectory() as d:
        for name, data in files.items():
            (Path(d) / name).write_bytes(data)
        return rlm.load_corpus(d, max_chars)


class LoadCorpusBudget(unittest.TestCase):
    def test_limit_is_inclusive_and_one_over_raises(self):
        self.assertEqual(load({"t": b"12345"}, 5)[0], {"t": "12345"})
        with self.assertRaisesRegex(ValueError, "RLM_MAX_CORPUS_CHARS=4"):
            load({"t": b"12345"}, 4)

    def test_counts_characters_not_bytes(self):
        self.assertEqual(load({"t": "éééé".encode()}, 4)[0], {"t": "éééé"})  # 8 bytes, 4 chars

    def test_skipped_binary_does_not_count(self):
        corpus, skipped = load({"bin": b"\0" * 100, "t": b"12345"}, 5)
        self.assertEqual((corpus, skipped), ({"t": "12345"}, 1))

    def test_budget_is_cumulative_across_files(self):
        with self.assertRaises(ValueError):
            load({"a": b"123", "b": b"456"}, 5)


if __name__ == "__main__":
    unittest.main()
