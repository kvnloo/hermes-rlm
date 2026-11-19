"""Pin Session log file naming: an empty key falls back to default.jsonl, a real key names its own file."""
import tempfile
import unittest
from pathlib import Path
import rlm


class SessionDefaultLogName(unittest.TestCase):
    def path_for(self, key):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return rlm.Session(key, subcall=lambda p, n: ('', {}), log_dir=Path(tmp.name)).log_path

    def test_empty_key_logs_to_default_jsonl(self):
        self.assertEqual(self.path_for('').name, 'default.jsonl')

    def test_key_names_its_own_log(self):
        self.assertEqual(self.path_for('abc').name, 'abc.jsonl')


if __name__ == "__main__":
    unittest.main()
