"""README "Writes" disclosure: trajectory log location and RLM_LOG_DIR override."""
import os
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class LogPathTests(unittest.TestCase):
    def test_default_and_override_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, override = Path(tmp, "data"), Path(tmp, "override")
            with mock.patch.object(rlm, "_data_dir", return_value=data):
                with mock.patch.dict(os.environ, {"RLM_LOG_DIR": ""}):
                    self.assertEqual(rlm.Session("s1").log_path, data / "runs" / (hashlib.sha256(b"s1").hexdigest() + ".jsonl"))
                    self.assertEqual(rlm.Session("").log_path, data / "runs" / (hashlib.sha256(b"").hexdigest() + ".jsonl"))
                with mock.patch.dict(os.environ, {"RLM_LOG_DIR": str(override)}):
                    self.assertEqual(rlm.Session("s2").log_path, override / (hashlib.sha256(b"s2").hexdigest() + ".jsonl"))
            self.assertTrue((data / "runs").is_dir() and override.is_dir())


if __name__ == "__main__":
    unittest.main()
