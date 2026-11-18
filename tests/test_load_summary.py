"""Pin load()'s summary shape and the 8192-byte binary sniff window."""
from pathlib import Path
import tempfile
import unittest
import rlm


class LoadSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def load(self, names_texts):
        for name, text in names_texts:
            (self.root / name).write_text(text)
        s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=self.root / 'logs')
        self.addCleanup(s._teardown)
        return s.load([str(self.root / n) for n, _ in names_texts])

    def test_single_file_has_no_sample_keys_and_approx_tokens_is_floor_div_4(self):
        r = self.load([('a.txt', 'x' * 22)])
        self.assertEqual((r['chars'], r['approx_tokens'], r['sample_keys']), (22, 5, None))
        self.assertEqual(r['context_type'], 'str')

    def test_two_files_list_both_keys(self):
        r = self.load([('a.txt', 'x'), ('b.txt', 'y')])
        self.assertEqual(r['sample_keys'], ['a.txt', 'b.txt'])

    def test_sample_keys_capped_at_first_20(self):
        names = [f'f{i:02d}.txt' for i in range(21)]
        r = self.load([(n, 'x') for n in names])
        self.assertEqual((r['files'], r['sample_keys']), (21, names[:20]))

    def test_nul_inside_8192_byte_window_is_skipped_but_byte_8192_is_not(self):
        (self.root / 'in.bin').write_bytes(b'a' * 8191 + b'\0')
        (self.root / 'out.txt').write_bytes(b'a' * 8192 + b'\0')
        corpus, skipped = rlm.load_corpus([str(self.root / 'in.bin'), str(self.root / 'out.txt')], 10**6)
        self.assertEqual((list(corpus), skipped), (['out.txt'], 1))


if __name__ == '__main__':
    unittest.main()
