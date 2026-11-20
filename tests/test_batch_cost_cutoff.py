"""Pins that _batch stops dispatching queued sub-calls once max_cost is spent or a price is unknown."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
import rlm


class BatchCostCutoffTests(unittest.TestCase):
    def _run(self, price):
        async def subcall(prompt, max_tokens):
            return "ok", {"cost_usd": price}

        with tempfile.TemporaryDirectory() as d:
            s = rlm.Session("k", log_dir=Path(d), allow_unsandboxed=True, subcall=subcall,
                            settings={"max_cost": 0.5, "concurrency": 1})
            self.addCleanup(s.kill)
            out = asyncio.run(s._batch(["a", "b", "c"], 8, time.monotonic() + 5))
        return s, out

    def test_known_price_at_cap_refuses_queued_calls(self):
        s, out = self._run(0.5)
        self.assertEqual(s.stats["subcalls"], 1)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "ok")
        self.assertTrue(all("cost budget exhausted" in o for o in out[1:]))
        self.assertEqual(s.stats["cost_unknown"], 0)

    def test_unknown_price_refuses_queued_calls(self):
        s, out = self._run(None)
        self.assertEqual(s.stats["subcalls"], 1)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "ok")
        self.assertTrue(all("cost budget unavailable" in o for o in out[1:]))
        self.assertEqual(s.stats["cost_unknown"], 1)


if __name__ == "__main__":
    unittest.main()
