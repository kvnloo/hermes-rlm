"""limits()/_env_int: every limit's default, env override (zero and padded ints), garbage/empty -> default."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rlm  # noqa: E402

TABLE = [  # (env name, limits() key, default)
    ("RLM_MAX_SUBCALLS", "max_subcalls", 200),
    ("RLM_MAX_PROMPT_CHARS", "max_prompt_chars", 400_000),
    ("RLM_MAX_OUTPUT_CHARS", "max_output_chars", 6_000),
    ("RLM_CELL_TIMEOUT_S", "cell_timeout_s", 600),
    ("RLM_CONCURRENCY", "concurrency", 8),
    ("RLM_SUBCALL_MAX_TOKENS", "subcall_max_tokens", 4_000),
    ("RLM_MAX_CORPUS_CHARS", "max_corpus_chars", 400_000_000),
]


def limits_with(**env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RLM_")}
    with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
        return rlm.limits()


class LimitsEnv(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(limits_with(), {key: d for _, key, d in TABLE})

    def test_each_env_name_drives_only_its_own_key(self):
        for i, (name, key, d) in enumerate(TABLE):
            got = limits_with(**{name: str(d + 7 + i)})
            self.assertEqual(got[key], d + 7 + i, name)
            self.assertEqual({k: v for k, v in got.items() if k != key},
                             {k: dd for _, k, dd in TABLE if k != key}, name)

    def test_zero_and_padded_ints_are_honoured(self):
        got = limits_with(RLM_MAX_SUBCALLS="0", RLM_CONCURRENCY=" 3 ")
        self.assertEqual((got["max_subcalls"], got["concurrency"]), (0, 3))

    def test_garbage_and_empty_fall_back_to_default(self):
        for bad in ("lots", "", "1.5"):
            self.assertEqual(limits_with(**{n: bad for n, _, _ in TABLE}), {key: d for _, key, d in TABLE}, bad)


if __name__ == "__main__":
    unittest.main()
