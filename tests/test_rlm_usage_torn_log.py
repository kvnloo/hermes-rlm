"""rlm_usage skips an unparseable log line instead of aborting the bench (defensive; no observed failure)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import run_bench  # noqa: E402


class RlmUsageTornLog(unittest.TestCase):
    def _usage(self, text):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "s.jsonl"
            log.write_text(text)
            return run_bench.rlm_usage(log)

    def test_torn_tail_is_skipped_and_complete_rows_still_counted(self):
        good = '{"event": "subcall", "prompt_tokens": 5, "completion_tokens": 2}\n{"event": "cell"}\n'
        self.assertEqual(self._usage(good + '{"event": "subc'),
                         {"subcalls": 1, "sub_prompt_tokens": 5, "sub_completion_tokens": 2, "cells": 1})


if __name__ == "__main__":
    unittest.main()
