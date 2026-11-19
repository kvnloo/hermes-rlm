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
    "non_dict_json": "import sys,time; sys.stdin.readline(); print('5', flush=True); time.sleep(60)",
    "dict_without_type": "import sys,time; sys.stdin.readline(); print('{}', flush=True); time.sleep(60)",
    "llm_without_prompts": "import sys,time; sys.stdin.readline(); print('{\"type\":\"llm\",\"id\":1}', flush=True); time.sleep(60)",
    "llm_string_prompts": "import sys,time; sys.stdin.readline(); print('{\"type\":\"llm\",\"id\":1,\"prompts\":\"ab\"}', flush=True); time.sleep(60)",
    "llm_non_str_prompt": "import sys,time; sys.stdin.readline(); print('{\"type\":\"llm\",\"id\":1,\"prompts\":[1]}', flush=True); time.sleep(60)",
    "done_non_str_stdout": "import sys,time; sys.stdin.readline(); print('{\"type\":\"done\",\"stdout\":5}', flush=True); time.sleep(60)",
    "llm_without_id": "import sys,time; sys.stdin.readline(); print('{\"type\":\"llm\",\"prompts\":[\"x\"]}', flush=True); time.sleep(60)",
}


class ExecProtocolCorruption(unittest.TestCase):
    def test_malformed_line_is_a_restart(self):
        for name, src in FAKES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                with mock.patch.object(rlm, "sandbox_argv", return_value=[sys.executable, "-I", "-c", src]):
                    calls = []
                    s = rlm.Session("t", subcall=lambda p, m: calls.append(p) or ("x", {}), log_dir=Path(d))
                    r = s.exec("1+1")
                    self.assertIn("kernel restarted", r["error"])
                    self.assertNotIn("not json", r["error"])
                    self.assertEqual((s.stats["restarts"], s.stats["cells"]), (1, 1))
                    self.assertIsNone(s.proc)
                    self.assertEqual((calls, s.stats["subcalls"]), ([], 0))  # nothing served or charged
                    rows = [json.loads(x) for x in s.log_path.read_text().splitlines()]
                    self.assertEqual(rows[-1]["event"], "cell")


if __name__ == "__main__":
    unittest.main()
