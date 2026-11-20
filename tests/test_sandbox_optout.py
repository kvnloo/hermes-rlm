"""Pins the explicit allow_unsandboxed opt-out that replaced the RLM_ALLOW_UNSANDBOXED env bypass."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class SandboxOptOutTests(unittest.TestCase):
    def test_opt_out_returns_plain_interpreter_argv(self):
        self.assertEqual(rlm.sandbox_argv(allow_unsandboxed=True), [sys.executable, "-I", str(rlm.KERNEL)])

    def test_default_refuses_without_bwrap_even_with_env_var(self):
        with mock.patch.dict(os.environ, {"RLM_ALLOW_UNSANDBOXED": "1"}), \
                mock.patch.object(rlm.shutil, "which", return_value=None), \
                mock.patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "bubblewrap"):
                rlm.sandbox_argv()

    def test_default_uses_bwrap_when_present(self):
        with mock.patch.object(rlm.shutil, "which", return_value="/usr/bin/bwrap"), \
                mock.patch.object(Path, "exists", return_value=True):
            self.assertEqual(rlm.sandbox_argv()[0], "/usr/bin/bwrap")

    def spawned_argv(self, **kw):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(rlm.subprocess, "Popen") as popen, \
                mock.patch.object(rlm.os, "set_blocking"), \
                mock.patch.object(rlm.shutil, "which", return_value="/usr/bin/bwrap"), \
                mock.patch.object(Path, "exists", return_value=True):
            rlm.Session("k", log_dir=Path(d), **kw)._spawn()
            return popen.call_args[0][0]

    def test_session_passes_opt_out_through_and_defaults_to_sandbox(self):
        self.assertEqual(self.spawned_argv(allow_unsandboxed=True), [sys.executable, "-I", str(rlm.KERNEL)])
        self.assertEqual(self.spawned_argv()[0], "/usr/bin/bwrap")


if __name__ == "__main__":
    unittest.main()
