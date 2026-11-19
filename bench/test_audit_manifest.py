"""Offline CLI fingerprint checks apply to base and reference arms too."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit_results as audit


class ManifestAuditTests(unittest.TestCase):
    def check_case(self, arm, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = dict(id='t', gold='A', scoring='choice', context_path='absent-context.txt')
            frozen = dict(task, context_path=str(root / task['context_path']),
                          context_sha256='recorded-context-hash')
            row = dict(task_id='t', arm=arm, valid=True, score=1.0, correct=True,
                       status='valid', accounting={'requests': []},
                       context_sha256=frozen['context_sha256'])
            if mode != 'legacy':
                row['task_sha256'] = audit.json_sha(frozen)
            if mode == 'changed':
                task['gold'] = 'B'
            tasks = root / 'tasks.jsonl'
            tasks.write_text('' if mode in ('missing', 'legacy') else json.dumps(task) + '\n')
            raw = root / 'rows.jsonl'
            raw.write_text(json.dumps(row) + '\n')
            original = raw.read_bytes()
            out = root / 'audit'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(audit.main(['--raw', str(raw), '--ops-dir', str(root),
                                            '--tasks', str(tasks), '--out-dir', str(out)]), 0)
            checked = json.loads((out / raw.name).read_text())
            expected = mode in ('matching', 'legacy')
            self.assertEqual(checked['valid'], expected)
            self.assertEqual(checked['score'], 1.0 if expected else None)
            self.assertEqual(checked['audit_reasons'], [] if expected else ['missing_or_mismatched_task'])
            self.assertEqual(json.loads(stdout.getvalue())['rows.jsonl']['audited_valid'], int(expected))
            self.assertTrue(checked['collection_valid'])
            self.assertEqual(checked['collection_score'], 1.0)
            self.assertEqual(raw.read_bytes(), original)
            self.assertFalse((root / task['context_path']).exists())

    def test_declared_manifest_identity_for_base_and_nano(self):
        for arm in ('base-truncated', 'nano-rlm'):
            for mode in ('matching', 'changed', 'missing', 'legacy'):
                with self.subTest(arm=arm, mode=mode):
                    self.check_case(arm, mode)
