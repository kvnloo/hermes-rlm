"""limits() defaults and per-key env overrides (invalid-value handling is deliberately not pinned)."""
import os
import unittest
from unittest import mock
import rlm

DEFAULTS = {"max_subcalls": 200, "max_prompt_chars": 400_000, "max_output_chars": 6_000,
            "cell_timeout_s": 600, "concurrency": 8, "subcall_max_tokens": 4_000,
            "max_corpus_chars": 400_000_000}
ENV = {"max_subcalls": "RLM_MAX_SUBCALLS", "max_prompt_chars": "RLM_MAX_PROMPT_CHARS",
       "max_output_chars": "RLM_MAX_OUTPUT_CHARS", "cell_timeout_s": "RLM_CELL_TIMEOUT_S",
       "concurrency": "RLM_CONCURRENCY", "subcall_max_tokens": "RLM_SUBCALL_MAX_TOKENS",
       "max_corpus_chars": "RLM_MAX_CORPUS_CHARS"}


class LimitsTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ENV.values():
            os.environ.pop(name, None)

    def test_defaults_when_env_unset(self):
        self.assertEqual(rlm.limits(), DEFAULTS)

    def test_each_key_reads_its_own_env_var(self):
        for key, name in ENV.items():
            with self.subTest(key=key), mock.patch.dict(os.environ, {name: "7"}):
                self.assertEqual(rlm.limits(), {**DEFAULTS, key: 7})

    def test_zero_is_kept_not_replaced_by_default(self):
        with mock.patch.dict(os.environ, {"RLM_MAX_SUBCALLS": "0"}):
            self.assertEqual(rlm.limits()["max_subcalls"], 0)


if __name__ == '__main__':
    unittest.main()
