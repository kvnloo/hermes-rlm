"""A cell writing straight to fd 1 (os.write / sys.__stdout__) must not corrupt the kernel protocol."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class KernelFd1Isolated(unittest.TestCase):
    def test_raw_fd1_writes_are_dropped_and_kernel_survives(self):
        with tempfile.TemporaryDirectory() as d:
            argv = [sys.executable, "-I", str(rlm.KERNEL)]
            with mock.patch.object(rlm, "sandbox_argv", return_value=argv):
                s = rlm.Session("t", subcall=lambda p, m: ("x", {}), log_dir=Path(d))
                f = Path(d) / "c.txt"
                f.write_text("hi")
                s.load([str(f)])
                for code in ("import os; os.write(1, b'junk\\n')", "import sys; sys.__stdout__.write('junk\\n'); sys.__stdout__.flush()"):
                    r = s.exec(code)
                    self.assertIsNone(r["error"], code)
                self.assertEqual(s.exec("1+1")["stdout"], "2\n")
                self.assertEqual(s.stats["restarts"], 0)
                s.kill()


if __name__ == "__main__":
    unittest.main()
