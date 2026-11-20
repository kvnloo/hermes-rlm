import os
import unittest
import warnings
from unittest import mock

from rlm import settings


def env(**kw):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('RLM_')}
    return mock.patch.dict(os.environ, {**clean, **kw}, clear=True)


class Ctx:
    def __init__(self, **cfg):
        self.cfg = cfg

    def get_config(self, key, default):
        return self.cfg.get(key, default)


class ValidateTests(unittest.TestCase):
    def test_accepted_boundaries(self):
        for s in ({'max_cost': None}, {'max_cost': 0}, {'max_subcalls': 0}, {'max_depth': 0},
                  {'max_depth': 1}, {'max_tokens': 32_000}, {'max_tokens': 1}, {'concurrency': 1}):
            with self.subTest(s=s):
                settings.validate(s)

    def test_rejected(self):
        for s in ({'max_cost': -1}, {'timeout_s': 0}, {'timeout_s': -1}, {'concurrency': 0},
                  {'max_tokens': 0}, {'max_depth': 2}, {'max_tokens': 32_001}, {'max_subcalls': -1},
                  {'max_subcalls': 1.5}, {'max_tokens': True}, {'max_cost': True},
                  {'timeout_s': float('inf')}, {'max_cost': float('nan')}, {'max_tokens': 'x'},
                  {'bogus': 1}, {'subcall_task': ''}, {'subcall_task': ' a'}, {'subcall_task': 3},
                  {'max_prompt_chars': 0}, {'timeout_s': None}, {'max_tokens': None}):
            with self.subTest(s=s), self.assertRaises(ValueError):
                settings.validate(s)

    def test_merges_over_defaults(self):
        self.assertEqual(settings.validate({'max_cost': 0.25}), {**settings.DEFAULTS, 'max_cost': 0.25})

    def test_float_timeout_and_cost_allowed(self):
        r = settings.validate({'timeout_s': 0.5, 'max_cost': 0.25})
        self.assertEqual((r['timeout_s'], r['max_cost']), (0.5, 0.25))


class LimitsTests(unittest.TestCase):
    def test_defaults_without_env_or_ctx(self):
        with env(), warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(settings.limits(), {
                'max_subcalls': 200, 'max_depth': 1, 'max_tokens': 4000, 'timeout_s': 600.0,
                'max_cost': None, 'subcall_task': 'rlm_subcall', 'max_prompt_chars': 400_000,
                'max_output_chars': 6000, 'concurrency': 8, 'max_corpus_chars': 400_000_000})

    def test_config_beats_env_without_warning(self):
        with env(RLM_MAX_TOKENS='9'), warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(settings.limits(Ctx(max_tokens=5))['max_tokens'], 5)

    def test_env_aliases_warn_and_apply(self):
        for name, key, raw, want in (
                ('RLM_MAX_TOKENS', 'max_tokens', '11', 11), ('RLM_SUBCALL_MAX_TOKENS', 'max_tokens', '12', 12),
                ('RLM_TIMEOUT_S', 'timeout_s', '1.5', 1.5), ('RLM_CELL_TIMEOUT_S', 'timeout_s', '2', 2.0),
                ('RLM_MAX_COST', 'max_cost', '0.5', 0.5), ('RLM_SUBCALL_TASK', 'subcall_task', 'x', 'x'),
                ('RLM_MAX_SUBCALLS', 'max_subcalls', '3', 3)):
            with self.subTest(name=name), env(**{name: raw}), self.assertWarns(DeprecationWarning):
                self.assertEqual(settings.limits()[key], want)

    def test_first_env_alias_wins(self):
        with env(RLM_MAX_TOKENS='7', RLM_SUBCALL_MAX_TOKENS='8'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(settings.limits()['max_tokens'], 7)

    def test_bad_env_value_raises(self):
        with env(RLM_MAX_TOKENS='abc'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaisesRegex(ValueError, 'RLM_MAX_TOKENS'):
                settings.limits()

    def test_env_value_is_validated(self):
        with env(RLM_MAX_DEPTH='5'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                settings.limits()


if __name__ == '__main__':
    unittest.main()
