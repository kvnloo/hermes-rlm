"""Pins that _batch stops dispatching queued sub-calls once max_cost is spent or a price is unknown."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
import rlm


class BatchCostCutoffTests(unittest.TestCase):
    def _run(self, price, max_cost=0.5):
        async def subcall(prompt, max_tokens):
            if isinstance(price, Exception):
                raise price
            return "ok", {"cost_usd": price}

        with tempfile.TemporaryDirectory() as d:
            s = rlm.Session("k", log_dir=Path(d), allow_unsandboxed=True, subcall=subcall,
                            settings={"max_cost": max_cost, "concurrency": 1})
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

    def test_prices_accumulate_below_cap(self):
        s, out = self._run(0.25)
        self.assertEqual(s.stats["subcalls"], 2)
        self.assertEqual(s.stats["cost_usd"], 0.5)
        self.assertEqual(out[:2], ["ok", "ok"])
        self.assertIn("cost budget exhausted", out[2])

    def test_unusable_price_counts_as_unknown(self):
        for price in (float("nan"), float("inf"), -1, True):
            with self.subTest(price=price):
                s, out = self._run(price)
                self.assertEqual(s.stats["cost_usd"], 0)
                self.assertEqual(s.stats["cost_unknown"], 1)
                self.assertTrue(all("cost budget unavailable" in o for o in out[1:]))

    def test_zero_price_is_known_and_free(self):
        s, out = self._run(0)
        self.assertEqual(out, ["ok"] * 3)
        self.assertEqual(s.stats["cost_unknown"], 0)

    def test_raising_subcall_counts_as_unknown_price(self):
        s, out = self._run(RuntimeError("boom"))
        self.assertEqual(s.stats["subcalls"], 1)
        self.assertEqual(s.stats["cost_unknown"], 1)
        self.assertEqual(out[0], "[llm_query error: RuntimeError]")
        self.assertTrue(all("cost budget unavailable" in o for o in out[1:]))

    def test_no_cap_never_refuses(self):
        s, out = self._run(None, max_cost=None)
        self.assertEqual(out, ["ok"] * 3)
        self.assertEqual(s.stats["subcalls"], 3)

    def test_unknown_price_refuses_queued_calls(self):
        s, out = self._run(None)
        self.assertEqual(s.stats["subcalls"], 1)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "ok")
        self.assertTrue(all("cost budget unavailable" in o for o in out[1:]))
        self.assertEqual(s.stats["cost_unknown"], 1)


if __name__ == "__main__":
    unittest.main()
