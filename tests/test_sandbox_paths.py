"""Pins sandbox_argv, _task_path and _data_dir; host modules and the machine are faked, never relied on."""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import rlm


def fake(**attrs):
    return types.SimpleNamespace(**attrs)


class SandboxArgvTests(unittest.TestCase):
    def argv(self, bwrap="/usr/bin/bwrap", py_exists=True, env=None):
        with mock.patch.dict(os.environ), mock.patch.object(rlm.shutil, "which", return_value=bwrap), \
                mock.patch.object(Path, "exists", return_value=py_exists):
            os.environ.pop("RLM_ALLOW_UNSANDBOXED", None)
            os.environ.update(env or {})
            return rlm.sandbox_argv()

    def test_bwrap_argv_is_the_isolation_boundary(self):
        k = str(rlm.KERNEL)
        self.assertEqual(self.argv(), [
            "/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", k, "/rlm/kernel.py", "--chdir", "/tmp",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "/usr/bin/python3", "-I", "/rlm/kernel.py"])

    def test_unsandboxed_only_when_flag_is_exactly_1(self):
        self.assertEqual(self.argv(bwrap=None, env={"RLM_ALLOW_UNSANDBOXED": "1"}),
                         [sys.executable, "-I", str(rlm.KERNEL)])
        for v in ("0", "true", ""):
            with self.subTest(v=v), self.assertRaisesRegex(RuntimeError, "bubblewrap"):
                self.argv(bwrap=None, env={"RLM_ALLOW_UNSANDBOXED": v})

    def test_missing_bwrap_or_python_never_returns_bwrap_argv(self):
        with self.assertRaises(RuntimeError):
            self.argv(bwrap=None)
        with self.assertRaises(RuntimeError):
            self.argv(py_exists=False)
        self.assertEqual(self.argv(py_exists=False, env={"RLM_ALLOW_UNSANDBOXED": "1"})[0], sys.executable)


class TaskPathTests(unittest.TestCase):
    def mods(self, resolver):
        return {"tools": fake(), "tools.file_tools_paths": fake(_resolve_base_dir=resolver)}

    def test_absolute_passes_through_and_tilde_expands(self):
        with mock.patch.dict(os.environ, {"HOME": "/home/x"}):
            self.assertEqual(rlm._task_path("/a/b", "t"), "/a/b")
            self.assertEqual(rlm._task_path("~/c", "t"), "/home/x/c")

    def test_relative_anchors_at_task_workspace(self):
        seen = []
        with mock.patch.dict(sys.modules, self.mods(lambda t: seen.append(t) or "/ws")):
            self.assertEqual(rlm._task_path("d/e.txt", "t1"), "/ws/d/e.txt")
            self.assertEqual(rlm._task_path("f", ""), "/ws/f")
        self.assertEqual(seen, ["t1", "default"])

    def test_falls_back_to_cwd_when_host_missing_or_raising(self):
        def boom(_t):
            raise RuntimeError
        for mods in ({"tools.file_tools_paths": None}, self.mods(boom)):
            with self.subTest(), mock.patch.dict(sys.modules, mods):
                self.assertEqual(rlm._task_path("g", "t"), os.path.abspath("g"))


class DataDirTests(unittest.TestCase):
    def test_prefers_plugin_storage(self):
        calls = []
        mods = {"plugins": fake(), "plugins.plugin_storage": fake(plugin_data_dir=lambda n: calls.append(n) or Path("/ps"))}
        with mock.patch.dict(sys.modules, mods):
            self.assertEqual(rlm._data_dir(), Path("/ps"))
        self.assertEqual(calls, ["rlm"])

    def test_falls_back_to_hermes_home(self):
        mods = {"plugins.plugin_storage": None, "hermes_constants": fake(get_hermes_home=lambda: Path("/hh"))}
        with mock.patch.dict(sys.modules, mods):
            self.assertEqual(rlm._data_dir(), Path("/hh/plugin-data/rlm"))


if __name__ == "__main__":
    unittest.main()
