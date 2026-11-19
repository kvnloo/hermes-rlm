"""_serve_llm normalises whole-number float token counts (4.0) to int in stats and log rows."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlm import Session  # noqa: E402


class ServeLlmWholeFloatTokens(unittest.TestCase):
    def test_whole_float_tokens_are_ints_in_stats_and_log(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session("t", subcall=lambda p, m: ("ok", {"prompt_tokens": 4.0, "completion_tokens": 2.0}),
                        log_dir=Path(d))
            s._serve_llm({"id": 1, "prompts": ["a"]})
            rows = [json.loads(l) for f in Path(d).glob("*.jsonl") for l in f.read_text().splitlines()]
        row = next(r for r in rows if r["event"] == "subcall")
        for v in (s.stats["prompt_tokens"], s.stats["completion_tokens"], row["prompt_tokens"], row["completion_tokens"]):
            self.assertIs(type(v), int)
        self.assertEqual((s.stats["prompt_tokens"], s.stats["completion_tokens"]), (4, 2))


if __name__ == "__main__":
    unittest.main()
