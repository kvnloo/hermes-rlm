"""Pin Session.kill releasing the kernel's pipe handles (no ResourceWarning leak)."""
import gc
import tempfile
import unittest
import warnings
from pathlib import Path
import rlm


class SessionKillClosesPipes(unittest.TestCase):
    def session(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = rlm.Session("k", subcall=lambda p, n: ("", {}), log_dir=Path(tmp.name))
        s._spawn()
        self.addCleanup(s.kill)
        return s

    def test_kill_closes_stdin_and_stdout(self):
        s = self.session()
        proc = s.proc
        s.kill()
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)
        self.assertIsNone(s.proc)

    def test_kill_of_already_exited_kernel_still_closes_pipes(self):
        s = self.session()
        proc = s.proc
        proc.kill()
        proc.wait()
        s.kill()
        self.assertTrue(proc.stdin.closed and proc.stdout.closed)

    def test_kill_emits_no_resource_warning(self):
        s = self.session()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.kill()
            gc.collect()
        self.assertEqual([x for x in w if issubclass(x.category, ResourceWarning)], [])

    def test_kill_survives_broken_pipe_and_still_closes_stdout(self):
        s = self.session()
        proc = s.proc
        proc.kill()
        proc.wait()
        with self.assertRaises(OSError):
            s._send({"a": 1})  # leaves unflushable bytes in stdin's buffer
        s.kill()
        self.assertTrue(proc.stdout.closed and proc.stdin.closed)
        self.assertIsNone(s.proc)

    def test_exec_respawn_of_dead_kernel_releases_old_pipes(self):
        s = self.session()
        old = s.proc
        old.kill()
        old.wait()
        s.exec("1")
        self.assertIsNot(s.proc, old)
        self.assertTrue(old.stdin.closed and old.stdout.closed)

    def test_kill_without_process_is_noop(self):
        s = self.session()
        s.kill()
        s.kill()


if __name__ == "__main__":
    unittest.main()
