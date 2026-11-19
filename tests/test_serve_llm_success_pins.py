"""Pin Session._serve_llm success path: result order, failed-slice isolation, stats accumulation, subcall log usage fields."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class ServeLlmSuccessPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.s = rlm.Session('t', subcall=self.sub, log_dir=Path(tmp.name))

    def sub(self, prompt, max_tokens):
        if prompt == 'bad':
            raise ValueError('boom')
        return prompt.upper(), {'prompt_tokens': 2, 'completion_tokens': 3}

    def test_results_in_prompt_order_and_failed_slice_does_not_sink_batch(self):
        reply = self.s._serve_llm({'id': 4, 'prompts': ['a', 'bad', 'ccc']})
        self.assertEqual(reply, {'type': 'llm_result', 'id': 4,
                                 'results': ['A', '[llm_query error: ValueError: boom]', 'CCC']})

    def test_stats_accumulate_per_prompt_and_failed_slice_adds_no_tokens(self):
        self.s.stats.update(subcalls=1, subcall_prompt_chars=10, prompt_tokens=5, completion_tokens=7)
        self.s._serve_llm({'id': 1, 'prompts': ['a', 'bad', 'ccc']})
        self.assertEqual(self.s.stats, {'subcalls': 4, 'subcall_prompt_chars': 17, 'prompt_tokens': 9,
                                        'completion_tokens': 13, 'cells': 0, 'restarts': 0})

    def test_subcall_log_row_carries_usage_fields(self):
        self.s._serve_llm({'id': 1, 'prompts': ['ab']})
        row = json.loads(self.s.log_path.read_text().splitlines()[-1])
        self.assertEqual((row['event'], row['prompt_chars'], row['output_chars'], row['prompt_tokens'],
                          row['completion_tokens']), ('subcall', 2, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
