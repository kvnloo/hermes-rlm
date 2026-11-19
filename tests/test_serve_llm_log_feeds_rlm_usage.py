"""The log _serve_llm writes (None/null/missing usage, raising subcall) is what run_bench.rlm_usage reads back."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import run_bench  # noqa: E402
from rlm import Session  # noqa: E402

USAGE = {"none": None, "null": {"prompt_tokens": None, "completion_tokens": None},
         "real": {"prompt_tokens": 4, "completion_tokens": 1}}


def _subcall(prompt, max_tokens):
    if prompt == "boom":
        raise RuntimeError("x")
    return "ok", USAGE[prompt]


class ServeLlmLogFeedsRlmUsage(unittest.TestCase):
    def test_real_writer_log_round_trips_through_rlm_usage(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session("t", subcall=_subcall, log_dir=Path(d))
            s._serve_llm({"id": 1, "prompts": ["none", "null", "real", "boom"]})
            got = run_bench.rlm_usage(s.log_path)
        self.assertEqual(got, {"subcalls": 4, "sub_prompt_tokens": 4, "sub_completion_tokens": 1, "cells": 0})
        self.assertEqual((s.stats["prompt_tokens"], s.stats["completion_tokens"]), (4, 1))


if __name__ == "__main__":
    unittest.main()
