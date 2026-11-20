"""Pins that _batch accounts a SystemExit sub-call (status=error, cost_unknown) before propagating."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
import rlm


class BatchSystemExitTests(unittest.TestCase):
    def test_systemexit_is_accounted_and_propagates(self):
        async def subcall(prompt, max_tokens):
            raise SystemExit("boom")

        with tempfile.TemporaryDirectory() as d:
            s = rlm.Session("k", log_dir=Path(d), allow_unsandboxed=True, subcall=subcall)
            self.addCleanup(s.kill)
            with self.assertRaises(SystemExit):
                asyncio.run(s._batch(["a"], 8, time.monotonic() + 5))
            rows = [json.loads(line) for line in s.log_path.read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows if r["event"] == "subcall"], ["error"])
            self.assertEqual(s.stats["cost_unknown"], 1)


if __name__ == "__main__":
    unittest.main()
