"""Pin Session._serve_llm's budget_refused log row: kind, requested count, pre-call stats; no sub-call; limit is inclusive."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class BudgetRefusedPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.calls = []
        self.s = rlm.Session('t', subcall=self.sub, log_dir=Path(tmp.name))
        self.s.lim = {**self.s.lim, 'max_subcalls': 10}

    def sub(self, prompt, max_tokens):
        self.calls.append(prompt)
        return 'ok', {}

    def rows(self):
        return [json.loads(l) for l in self.s.log_path.read_text().splitlines()] if self.s.log_path.exists() else []

    def test_refusal_logs_requested_and_untouched_stats(self):
        self.s.stats.update(subcalls=7, subcall_prompt_chars=11, prompt_tokens=13, completion_tokens=17)
        before = dict(self.s.stats)
        reply = self.s._serve_llm({'id': 5, 'prompts': ['a', 'b', 'c', 'd']})
        self.assertEqual(reply['id'], 5)
        self.assertIn('sub-call budget exhausted (7/10 used)', reply['error'])
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        row.pop('ts')
        self.assertEqual(row, {'event': 'budget_refused', 'kind': 'subcalls', 'requested': 4, **before})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.s.stats, before)

    def test_exactly_at_limit_is_not_refused(self):
        self.s.stats['subcalls'] = 7
        reply = self.s._serve_llm({'id': 1, 'prompts': ['a', 'b', 'c']})
        self.assertNotIn('error', reply)
        self.assertEqual(reply['results'], ['ok'] * 3)
        self.assertNotIn('budget_refused', [r['event'] for r in self.rows()])
        self.assertEqual(self.s.stats['subcalls'], 10)


if __name__ == "__main__":
    unittest.main()
