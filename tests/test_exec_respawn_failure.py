"""A failing respawn inside exec is a reported restart that leaves no half-loaded kernel, not an escaping exception."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402

JUNK_ON_LOAD = "import sys,time; sys.stdin.readline(); print('not json', flush=True); time.sleep(60)"


class ExecRespawnFailure(unittest.TestCase):
    def run_exec(self, **patch):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(rlm, "sandbox_argv", return_value=[sys.executable, "-I", "-c", JUNK_ON_LOAD]):
            s = rlm.Session("t", log_dir=Path(d))
            s.context = "corpus"  # a respawn must reload it
            with mock.patch.object(rlm.subprocess, "Popen", **patch) if patch else mock.MagicMock():
                r = s.exec("1+1")
            return s, r

    def test_bad_reload_reply_drops_the_kernel(self):
        s, r = self.run_exec()
        self.assertIn("kernel restarted", r["error"])
        self.assertIsNone(s.proc)  # not left alive without its context
        self.assertEqual((s.stats["restarts"], s.stats["cells"]), (1, 1))

    def test_popen_oserror_is_reported(self):
        s, r = self.run_exec(side_effect=OSError("no bwrap"))
        self.assertIn("no bwrap", r["error"])
        self.assertIsNone(s.proc)
        self.assertEqual((s.stats["restarts"], s.stats["cells"]), (1, 1))

    def test_load_timeout_names_itself(self):
        with tempfile.TemporaryDirectory() as d:
            s = rlm.Session("t", log_dir=Path(d))
            with mock.patch.object(rlm.Session, "_request", side_effect=TimeoutError), mock.patch.object(rlm.subprocess, "Popen"):
                s.context = "corpus"
                r = s.exec("1+1")
            self.assertIn("TimeoutError", r["error"])
            self.assertNotIn("cell exceeded", r["error"])

    def test_spawn_failure_leaves_no_kernel_for_load(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(rlm, "sandbox_argv", return_value=[sys.executable, "-I", "-c", JUNK_ON_LOAD]):
            s = rlm.Session("t", log_dir=Path(d))
            s.context = "corpus"
            with self.assertRaises(RuntimeError):
                s._spawn()
            self.assertIsNone(s.proc)


if __name__ == "__main__":
    unittest.main()
