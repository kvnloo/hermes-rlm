"""Pins the RLM_LOG_DIR deprecation warning: only when the env var is non-empty and no log_dir is passed."""
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
import rlm


class LogDirDeprecationTests(unittest.TestCase):
    def warnings_for(self, env_value, **kw):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            os.environ.pop("RLM_LOG_DIR", None)
            if env_value is not None:
                os.environ["RLM_LOG_DIR"] = env_value.format(d=d)
            if "log_dir" in kw:
                kw["log_dir"] = Path(d) / "arg"
            with mock.patch.object(rlm, "_data_dir", return_value=Path(d) / "data"):
                rlm.Session("k", **kw)
        return [w for w in caught if "RLM_LOG_DIR" in str(w.message)]

    def test_env_without_log_dir_warns_once(self):
        got = self.warnings_for("{d}/env")
        self.assertEqual(len(got), 1)
        self.assertTrue(issubclass(got[0].category, DeprecationWarning))
        self.assertIn("RLM_LOG_DIR is deprecated", str(got[0].message))
        self.assertEqual(got[0].filename, __file__)

    def test_explicit_log_dir_does_not_warn(self):
        self.assertEqual(self.warnings_for("{d}/env", log_dir=True), [])

    def test_unset_or_empty_env_does_not_warn(self):
        self.assertEqual(self.warnings_for(None), [])
        self.assertEqual(self.warnings_for(""), [])


if __name__ == "__main__":
    unittest.main()
