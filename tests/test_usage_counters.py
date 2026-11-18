"""/rlm reports these counters: sub-call usage must accumulate per prompt, and a refusal must be logged."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class UsageCounterTests(unittest.TestCase):
    def test_usage_accumulates_and_refusal_is_logged(self):
        usage = {'aa': {'prompt_tokens': 3, 'completion_tokens': 5},
                 'bbbb': {'prompt_tokens': 7, 'completion_tokens': 11}}
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('t', subcall=lambda p, n: ('ok', usage[p]), log_dir=Path(root))
            with mock.patch.dict('os.environ', {'RLM_MAX_SUBCALLS': '2'}):
                s.lim = rlm.limits()
            with mock.patch.object(s, 'log') as log:
                r = s._serve_llm({'id': 1, 'prompts': ['aa', 'bbbb']})
                self.assertEqual(r['results'], ['ok', 'ok'])
                self.assertEqual((s.stats['subcalls'], s.stats['subcall_prompt_chars'],
                                  s.stats['prompt_tokens'], s.stats['completion_tokens']), (2, 6, 10, 16))
                self.assertIn('budget', s._serve_llm({'id': 2, 'prompts': ['aa']})['error'])
            self.assertEqual(log.call_args.args, ('budget_refused',))
            self.assertEqual((log.call_args.kwargs['kind'], log.call_args.kwargs['requested']), ('subcalls', 1))
            self.assertEqual(s.stats['prompt_tokens'], 10)


if __name__ == '__main__':
    unittest.main()
