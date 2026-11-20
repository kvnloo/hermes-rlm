import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit_results as audit


class UnknownArmReportTests(unittest.TestCase):
    def test_unknown_arm_is_reported_invalid_without_aborting_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'rows.jsonl'
            tasks = root / 'tasks.jsonl'
            out = root / 'audited'
            rows = [dict(task_id='t', arm=arm, valid=True, score=1.0)
                    for arm in ('base-truncated', 'unrecognized-arm')]
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            original = raw.read_bytes()
            tasks.write_text(json.dumps(dict(id='t', gold='A', scoring='choice')) + '\n')
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = audit.main(['--raw', str(raw), '--ops-dir', str(root),
                                     '--tasks', str(tasks), '--out-dir', str(out)])
            self.assertEqual(result, 0)
            stats = json.loads(stream.getvalue())['rows.jsonl']
            self.assertEqual(stats['rows'], 2)
            self.assertEqual(stats['collection_valid'], 2)
            self.assertEqual(stats['audited_valid'], 1)
            self.assertEqual(stats['adherence'], {'not_applicable_base': 1, 'unknown': 1})
            self.assertEqual(stats['reasons'], {'unknown_arm': 1})
            checked = [json.loads(line) for line in (out / raw.name).read_text().splitlines()]
            self.assertTrue(checked[0]['valid'])
            self.assertFalse(checked[1]['valid'])
            self.assertIsNone(checked[1]['score'])
            self.assertTrue(checked[1]['collection_valid'])
            self.assertEqual(raw.read_bytes(), original)
