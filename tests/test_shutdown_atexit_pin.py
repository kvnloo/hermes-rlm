"""Pin that rlm._shutdown is registered with atexit: a real interpreter exit must kill live sessions."""
import subprocess
import sys
import unittest
from pathlib import Path

import rlm

ROOT = Path(rlm.__file__).resolve().parent.parent
CODE = ("import rlm\n"
        "class S:\n"
        "    def kill(self): print('KILLED')\n"
        "rlm._sessions['x'] = S()\n")


class ShutdownAtexitPin(unittest.TestCase):
    def test_interpreter_exit_kills_sessions(self):
        r = subprocess.run([sys.executable, "-B", "-c", CODE], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["KILLED"])


if __name__ == "__main__":
    unittest.main()
