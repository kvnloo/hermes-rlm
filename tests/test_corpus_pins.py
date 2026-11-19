"""Pin load_corpus's NUL-sniff window and key order, and _task_path's relative-path anchoring."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
import rlm


class CorpusPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_nul_sniff_covers_first_8192_bytes_only(self):
        (self.root / "in.txt").write_bytes(b"a" * 8191 + b"\0")
        (self.root / "out.txt").write_bytes(b"a" * 8192 + b"\0")
        corpus, skipped = rlm.load_corpus(str(self.root), 10**6)
        self.assertEqual((list(corpus), skipped), (["out.txt"], 1))

    def test_directory_keys_are_sorted(self):
        for n in ("b", "c", "a"):
            (self.root / f"{n}.txt").write_text(n)
        self.assertEqual(list(rlm.load_corpus(str(self.root), 10)[0]), ["a.txt", "b.txt", "c.txt"])


class TaskPathPins(unittest.TestCase):
    def host(self, calls):
        mod = types.SimpleNamespace(_resolve_base_dir=lambda t: calls.append(t) or "/ws")
        return mock.patch.dict(sys.modules, {"tools": types.SimpleNamespace(), "tools.file_tools_paths": mod})

    def test_relative_path_is_anchored_at_task_workspace(self):
        calls = []
        with self.host(calls):
            self.assertEqual(rlm._task_path("a/b", "t1"), "/ws/a/b")
            self.assertEqual(rlm._task_path("a", ""), "/ws/a")
        self.assertEqual(calls, ["t1", "default"])

    def test_relative_path_without_host_is_made_absolute(self):
        with mock.patch.dict(sys.modules, {"tools.file_tools_paths": None}):
            self.assertEqual(rlm._task_path("a/b", "t"), os.path.abspath("a/b"))
            self.assertTrue(os.path.isabs(rlm._task_path("a/b", "t")))


if __name__ == "__main__":
    unittest.main()
