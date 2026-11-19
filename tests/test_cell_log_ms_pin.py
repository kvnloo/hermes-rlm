"""Pin that the 'cell' log row records the same elapsed ms that exec returns."""
import json
import tempfile
import unittest
from pathlib import Path
import rlm


class CellLogMs(unittest.TestCase):
    def test_cell_row_ms_equals_result_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = rlm.Session('t', subcall=lambda p, n: ('', {}), log_dir=Path(tmp))
            self.addCleanup(s.kill)
            ms = s.exec('import time; time.sleep(0.3)')['ms']
            rows = [json.loads(x) for x in s.log_path.read_text().splitlines()]
        cell = [r for r in rows if r['event'] == 'cell']
        self.assertEqual(len(cell), 1)
        self.assertEqual(cell[0]['ms'], ms)
        self.assertGreaterEqual(ms, 250)


if __name__ == "__main__":
    unittest.main()
