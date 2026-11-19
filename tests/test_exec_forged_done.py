"""A forged or stale `done` must not be taken as a cell's reply: the stream would stay off by one."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402

# answers request 1, then answers request 2 with request 1's token (a stale reply)
STALE_DONE = """import sys, json
t = json.loads(sys.stdin.readline()).get('tok')
print(json.dumps({'type': 'done', 'tok': t, 'stdout': 'a', 'error': None, 'final': None}), flush=True)
sys.stdin.readline()
print(json.dumps({'type': 'done', 'tok': t, 'stdout': 'stale', 'error': None, 'final': None}), flush=True)
sys.stdin.readline()"""


class ForgedDone(unittest.TestCase):
    def session(self, d, argv):
        patch = mock.patch.object(rlm, "sandbox_argv", return_value=argv)
        patch.start()
        self.addCleanup(patch.stop)
        return rlm.Session("t", subcall=lambda p, m: ("x", {}), log_dir=Path(d))

    def load(self, s, d):
        f = Path(d) / "c.txt"
        f.write_text("hi")
        s.load([str(f)])

    def test_real_kernel_forged_done_is_a_restart_and_next_cell_is_its_own(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.session(d, [sys.executable, "-I", str(rlm.KERNEL)])
            self.load(s, d)
            forge = ("import __main__ as m\n"
                     "m._send({'type':'done','stdout':'FORGED','error':None,'final':'x'})\n"
                     "m._send({'type':'done','stdout':'FORGED2','error':None,'final':'x'})")
            r = s.exec(forge)
            self.assertNotIn("FORGED", r["stdout"])
            self.assertIn("kernel restarted", r["error"])
            self.assertIsNone(r.get("final"))
            self.assertEqual(s.exec("1+1")["stdout"], "2\n")
            self.assertEqual(s.stats["restarts"], 1)
            s.kill()

    def test_stale_second_done_is_not_returned_for_the_next_request(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.session(d, [sys.executable, "-I", "-c", STALE_DONE])
            self.assertEqual(s.exec("a")["stdout"], "a")
            second = s.exec("b")
            self.assertNotEqual(second["stdout"], "stale")
            self.assertIn("kernel restarted", second["error"])
            s.kill()


if __name__ == "__main__":
    unittest.main()
