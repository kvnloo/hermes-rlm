"""Two protocol lines arriving in one pipe read must both be seen: _read may not strand the second behind the first."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402

KERNEL = r"""
import json, sys, time
req = json.loads(sys.stdin.readline())
llm = {"type": "llm", "id": 1, "prompts": ["p"]}
done = {"type": "done", "stdout": "ok", "tok": req["tok"]}
sys.stdout.write(json.dumps(llm) + "\n" + json.dumps(done) + "\n")  # one write: both lines land in one read
sys.stdout.flush()
time.sleep(60)
"""


class ReadBufferedLines(unittest.TestCase):
    def test_second_line_of_one_read_is_not_stranded(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rlm, "sandbox_argv", return_value=[sys.executable, "-I", "-c", KERNEL]), \
                mock.patch.dict("os.environ", {"RLM_CELL_TIMEOUT_S": "5"}):
            s = rlm.Session("t", subcall=lambda p, m: ("x", {}), log_dir=Path(d))
            t0 = time.monotonic()
            try:
                r = s.exec("1")
            finally:
                s.kill()
            self.assertLess(time.monotonic() - t0, 3)  # not a cell timeout
            self.assertEqual((r["error"], r["stdout"], r["subcalls_this_cell"]), (None, "ok", 1))
            self.assertEqual(s.stats["restarts"], 0)


if __name__ == "__main__":
    unittest.main()
