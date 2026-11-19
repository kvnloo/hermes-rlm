"""Pin Session.load's return dict and context, the prompt-size boundary, and subcall prompt-char stats."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class LoadResultPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.s = rlm.Session('t', subcall=lambda p, n: ('ok', {}), log_dir=self.tmp / 'log')
        self.addCleanup(self.s.kill)

    def test_single_file_loads_as_str(self):
        f = self.tmp / 'a.txt'
        f.write_text('x' * 10)
        r = self.s.load([str(f)])
        self.assertEqual((r['loaded'], r['context_type'], r['files'], r['chars'], r['approx_tokens']),
                         (True, 'str', 1, 10, 2))
        self.assertEqual((r['skipped_binary'], r['sample_keys']), (0, None))
        self.assertEqual(self.s.exec('print(type(context).__name__, len(context))')['stdout'].strip(), 'str 10')

    def test_dir_loads_as_dict_with_binary_skipped_and_sample_keys_capped(self):
        d = self.tmp / 'c'
        d.mkdir()
        for i in range(21):
            (d / f'f{i:02}.txt').write_text('abcd')
        (d / 'z.bin').write_bytes(b'\0\1')
        r = self.s.load([str(d)])
        self.assertEqual((r['context_type'], r['files'], r['chars'], r['skipped_binary']),
                         ('dict[path -> text]', 21, 84, 1))
        self.assertEqual(r['sample_keys'], [f'f{i:02}.txt' for i in range(20)])
        self.assertEqual(self.s.exec('print(type(context).__name__, len(context))')['stdout'].strip(), 'dict 21')
        row = [r for r in map(json.loads, self.s.log_path.read_text().splitlines()) if r['event'] == 'load'][0]
        self.assertEqual((row['event'], row['files'], row['chars'], row['skipped_binary']), ('load', 21, 84, 1))
        self.assertEqual(row['paths'], [str(d)])

    def test_load_replaces_previous_kernel_state(self):
        f = self.tmp / 'a.txt'
        f.write_text('x')
        self.s.exec('v = 1')
        old = self.s.proc
        self.s.load([str(f)])
        self.assertIsNotNone(old.poll())
        self.assertIn('NameError', self.s.exec('print(v)')['error'])

    def test_prompt_exactly_at_limit_is_allowed_and_over_is_refused(self):
        self.s.lim = {**self.s.lim, 'max_prompt_chars': 5}
        self.assertIsNone(self.s.exec('llm_query("a" * 5)')['error'])
        self.assertIn('exceeds RLM_MAX_PROMPT_CHARS=5', self.s.exec('llm_query("a" * 6)')['error'])
        self.assertIn('exceeds', self.s.exec('llm_batch(["a", "b" * 6])')['error'])

    def test_empty_final_answer_is_still_returned(self):
        self.assertEqual(self.s.exec('FINAL("")')['final'], '')

    def test_subcall_prompt_chars_accumulate(self):
        self.s.exec('llm_batch(["ab", "cde"])')
        self.assertEqual(self.s.stats['subcall_prompt_chars'], 5)


if __name__ == '__main__':
    unittest.main()
