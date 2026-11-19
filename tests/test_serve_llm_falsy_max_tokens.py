"""A missing/None/0 max_tokens means 'unset' (kernel.llm_batch sends None by default): the default cap is used and the call is charged."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmFalsyMaxTokens(unittest.TestCase):
    def test_unset_max_tokens_uses_default_and_is_charged(self):
        for unset in (None, 0, "omitted"):
            with self.subTest(unset=unset), tempfile.TemporaryDirectory() as d:
                calls = []
                s = Session("t", subcall=lambda p, m: calls.append(m) or ("ok", {}), log_dir=Path(d))
                s.lim = {**s.lim, "subcall_max_tokens": 123}
                req = {"id": 1, "prompts": ["a"]}
                if unset != "omitted":
                    req["max_tokens"] = unset
                r = s._serve_llm(req)
                self.assertEqual(r["results"], ["ok"])
                self.assertEqual((calls, s.stats["subcalls"]), ([123], 1))


if __name__ == "__main__":
    unittest.main()
