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

RESIDUE_KERNEL = r"""
import json, sys, time
req = json.loads(sys.stdin.readline())
sys.stdout.write(json.dumps({"type": "done", "stdout": "A", "tok": req["tok"]}) + "\n" + '{"type": "do')  # torn residue
sys.stdout.flush()
time.sleep(60)
"""
CLEAN_KERNEL = r"""
import json, sys
req = json.loads(sys.stdin.readline())
sys.stdout.write(json.dumps({"type": "done", "stdout": "B", "tok": req["tok"]}) + "\n")
sys.stdout.flush()
sys.stdin.read()
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

    def test_respawn_drops_bytes_left_by_the_previous_kernel(self):
        argv = [[sys.executable, "-I", "-c", k] for k in (RESIDUE_KERNEL, CLEAN_KERNEL)]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rlm, "sandbox_argv", side_effect=argv), \
                mock.patch.dict("os.environ", {"RLM_CELL_TIMEOUT_S": "5"}):
            s = rlm.Session("t", log_dir=Path(d))
            try:
                first = s.exec("1")
                s.kill()
                second = s.exec("2")
            finally:
                s.kill()
            self.assertEqual((first["error"], first["stdout"]), (None, "A"))
            self.assertEqual((second["error"], second["stdout"]), (None, "B"))

    def test_buffered_line_does_not_outlive_the_deadline(self):
        s = rlm.Session("t", log_dir=Path(tempfile.gettempdir()))
        s.proc = mock.Mock()
        s._rbuf = bytearray(b'{"type": "done"}\n')
        with self.assertRaises(TimeoutError):
            s._read(time.monotonic())


if __name__ == "__main__":
    unittest.main()
