"""A sub-call raising a BaseException (e.g. asyncio.CancelledError from a timed-out ctx.llm) must
come back to the model as that slice's error text, not strand the kernel waiting for a reply."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


def raiser(exc):
    def subcall(prompt, max_tokens):
        raise exc
    return subcall


class SubcallBaseException(unittest.TestCase):
    def run_cells(self, exc, *cells):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"RLM_ALLOW_UNSANDBOXED": "1"}), \
                mock.patch("shutil.which", return_value=None):
            s = Session("t", subcall=raiser(exc), log_dir=Path(d))
            try:
                return [s.exec(c) for c in cells], s.stats["restarts"]
            finally:
                s.kill()

    def test_baseexception_subcall_is_a_slice_error_and_kernel_stays_in_sync(self):
        for exc in (asyncio.CancelledError("boom"), SystemExit("boom")):
            with self.subTest(exc=type(exc).__name__):
                (r1, r2), restarts = self.run_cells(exc, "x = 7\nprint(llm_batch(['a', 'b']))", "x + 1")
                self.assertIsNone(r1["error"])
                self.assertEqual(r1["stdout"].count(f"{type(exc).__name__}: boom"), 2)
                self.assertEqual((r2["error"], r2["stdout"].strip(), restarts), (None, "8", 0))


if __name__ == "__main__":
    unittest.main()
