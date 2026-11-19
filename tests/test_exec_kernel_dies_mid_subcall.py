"""exec reports a kernel that dies while a sub-call is in flight (reply write hits a broken pipe)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402


class ExecKernelDiesMidSubcall(unittest.TestCase):
    def test_broken_pipe_is_a_restart_not_an_exception(self):
        with tempfile.TemporaryDirectory() as d:
            holder = {}

            def subcall(prompt, max_tokens):
                holder["s"].proc.kill()  # kernel dies before the host writes llm_result
                holder["s"].proc.wait()
                return "ok", {}

            argv = [sys.executable, "-I", str(rlm.KERNEL)]
            with mock.patch.object(rlm, "sandbox_argv", return_value=argv):
                s = holder["s"] = rlm.Session("t", subcall=subcall, log_dir=Path(d))
                f = Path(d) / "c.txt"
                f.write_text("hi")
                s.load([str(f)])
                r = s.exec("llm_query('x')")
                self.assertIn("kernel restarted", r["error"])
                self.assertEqual((s.stats["restarts"], s.stats["cells"]), (1, 1))
                rows = [json.loads(x) for x in s.log_path.read_text().splitlines()]
                self.assertEqual([x["event"] for x in rows][-1], "cell")
                self.assertEqual(s.exec("context")["stdout"], "'hi'\n")
                s.kill()


if __name__ == "__main__":
    unittest.main()
