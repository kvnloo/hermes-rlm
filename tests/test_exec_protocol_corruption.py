"""A malformed protocol line from the kernel is a kernel restart, not a JSONDecodeError out of exec."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402

FAKES = {
    "garbage_line_kernel_alive": "import sys,time; sys.stdin.readline(); print('not json', flush=True); time.sleep(60)",
    "truncated_line_then_exit": "import sys; sys.stdin.readline(); sys.stdout.write('{\"type\": \"do'); sys.stdout.flush()",
}


class ExecProtocolCorruption(unittest.TestCase):
    def test_malformed_line_is_a_restart(self):
        for name, src in FAKES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                with mock.patch.object(rlm, "sandbox_argv", return_value=[sys.executable, "-I", "-c", src]):
                    s = rlm.Session("t", subcall=lambda p, m: ("x", {}), log_dir=Path(d))
                    r = s.exec("1+1")
                    self.assertIn("kernel restarted", r["error"])
                    self.assertNotIn("not json", r["error"])
                    self.assertEqual((s.stats["restarts"], s.stats["cells"]), (1, 1))
                    self.assertIsNone(s.proc)
                    rows = [json.loads(x) for x in s.log_path.read_text().splitlines()]
                    self.assertEqual(rows[-1]["event"], "cell")


if __name__ == "__main__":
    unittest.main()
