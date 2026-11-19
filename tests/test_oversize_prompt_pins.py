"""Pin Session._serve_llm's oversize-prompt refusal: exact error, no sub-call, no log row, no stats bump; budget check wins; empty batch is fine."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class OversizePromptPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.calls = []
        self.s = rlm.Session('t', subcall=self.sub, log_dir=Path(tmp.name))
        self.s.lim = {**self.s.lim, 'max_prompt_chars': 5, 'max_subcalls': 10}

    def sub(self, prompt, max_tokens):
        self.calls.append(prompt)
        return 'ok', {}

    def rows(self):
        return [json.loads(l) for l in self.s.log_path.read_text().splitlines()] if self.s.log_path.exists() else []

    def test_refusal_message_and_no_side_effects(self):
        self.s.stats.update(subcalls=2, subcall_prompt_chars=3, prompt_tokens=4, completion_tokens=5)
        before = dict(self.s.stats)
        reply = self.s._serve_llm({'id': 9, 'prompts': ['a', 'b' * 6, 'c']})
        self.assertEqual(reply, {'type': 'llm_result', 'id': 9, 'error':
                                 'prompt of 6 chars exceeds RLM_MAX_PROMPT_CHARS=5; split the slice into smaller chunks'})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.s.stats, before)
        self.assertEqual(self.rows(), [])

    def test_budget_refusal_takes_precedence_over_size(self):
        self.s.stats['subcalls'] = 10
        reply = self.s._serve_llm({'id': 1, 'prompts': ['x' * 6]})
        self.assertIn('sub-call budget exhausted', reply['error'])

    def test_empty_batch_returns_empty_results(self):
        self.assertEqual(self.s._serve_llm({'id': 2, 'prompts': []}),
                         {'type': 'llm_result', 'id': 2, 'results': []})


if __name__ == "__main__":
    unittest.main()
