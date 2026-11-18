"""Pins where Session puts its log directory: explicit arg > RLM_LOG_DIR > _data_dir()/runs.
Asserts on the directory only: the file name and the RLM_LOG_DIR deprecation warning differ across sibling PRs."""
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
import rlm


class SessionLogDirTests(unittest.TestCase):
    def session(self, env=None, **kw):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ.pop("RLM_LOG_DIR", None)
            os.environ.update(env(d) if env else {})
            with mock.patch.object(rlm, "_data_dir", return_value=Path(d) / "data"):
                parent = rlm.Session("k", **{k: v(d) for k, v in kw.items()}).log_path.parent
                return Path(d), parent, parent.is_dir()

    def test_explicit_log_dir_beats_env(self):
        d = self.session(env=lambda d: {"RLM_LOG_DIR": f"{d}/env"}, log_dir=lambda d: Path(d) / "arg")
        self.assertEqual(d[1], d[0] / "arg")
        self.assertTrue(d[2])

    def test_env_used_when_no_arg_and_created_with_parents(self):
        d, parent, made = self.session(env=lambda d: {"RLM_LOG_DIR": f"{d}/a/b/c"})
        self.assertEqual(parent, d / "a/b/c")
        self.assertTrue(made)

    def test_unset_or_empty_env_falls_back_to_data_dir_runs(self):
        for env in (None, lambda d: {"RLM_LOG_DIR": ""}):
            d, parent, made = self.session(env=env)
            self.assertEqual(parent, d / "data" / "runs")
            self.assertTrue(made)


if __name__ == "__main__":
    unittest.main()
