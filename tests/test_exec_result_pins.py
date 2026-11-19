"""Pin Session.exec result accounting: per-cell subcall delta, budget block, cells counter, cell log row."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class ExecResultPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.s = rlm.Session('t', subcall=lambda p, n: ('ok', {}), log_dir=Path(tmp.name))
        self.addCleanup(self.s.kill)
        self.s.lim = {**self.s.lim, 'max_subcalls': 5}

    def test_budget_and_per_cell_subcall_delta(self):
        self.s.exec('llm_batch(["a", "b"])')
        r = self.s.exec('llm_query("c")')
        self.assertEqual(r['subcalls_this_cell'], 1)
        self.assertEqual(r['budget'], {'subcalls_used': 3, 'subcalls_max': 5})
        self.assertEqual(self.s.exec('pass')['subcalls_this_cell'], 0)

    def test_cells_counter_and_cell_log_row(self):
        self.s.exec('llm_batch(["a", "b"])')
        self.s.exec('print("hi")')
        self.assertEqual(self.s.stats['cells'], 2)
        rows = [json.loads(x) for x in self.s.log_path.read_text().splitlines()]
        cells = [x for x in rows if x['event'] == 'cell']
        self.assertEqual([c['code'] for c in cells], ['llm_batch(["a", "b"])', 'print("hi")'])
        self.assertEqual([c['subcalls'] for c in cells], [2, 0])
        self.assertEqual(cells[1]['stdout_chars'], 3)


if __name__ == '__main__':
    unittest.main()
