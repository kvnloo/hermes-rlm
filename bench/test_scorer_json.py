"""A custom grade must not make the matrix lose a completed attempt."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class OfflineRouter:
    errors = ()
    observed = ()
    blocked = False
    calls = 0

    def __init__(self, ledger, run, *args, **kwargs):
        self.ledger, self.run = ledger, run

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def logical_snapshot(self):
        return []

    def forward(self, body, model, role):
        type(self).calls += 1
        attempt = self.ledger.reserve(self.run, model, role)
        self.ledger.finish(attempt, 200, 0, {'prompt_tokens': 1, 'completion_tokens': 1})
        return 200, b'{"choices":[{"message":{"content":"label"}}]}', {}


class ScorerJsonTests(unittest.TestCase):
    def test_custom_grade_is_persisted_or_terminal_invalid_without_rerun(self):
        for metadata, expected in [("{'label'}", None), ("['étude']", ['étude']),
                                   ("chr(0xd800)", None)]:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'context.txt').write_text('Synthetic source.')
                manifest = root / 'tasks.jsonl'
                manifest.write_text(json.dumps(dict(id='t', family='fixture', size_tokens=4,
                    context_path='context.txt', question='Which label?', gold=['label'], scoring='ruler')) + '\n')
                scorer = root / 'scorer.py'
                scorer.write_text(f"def score_task(task, answer): return {{'score': 1.0, 'matched': {metadata}}}\n",
                                  encoding='utf-8')
                out = root / 'out.jsonl'
                args = [str(manifest), str(out), '--ops-dir', str(root / 'ops'),
                    '--home-template', str(root), '--released-source', str(root),
                    '--released-python', sys.executable, '--fork-source', str(root),
                    '--fork-python', sys.executable, '--nano-source', str(root),
                    '--arms', 'base-truncated', '--scorer', str(scorer), '--repeats', '1']
                OfflineRouter.calls = 0
                with patch.object(matrix, 'source_sha', return_value='fixture'), \
                        patch.object(matrix, 'Router', OfflineRouter), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(matrix.main(args), 0)
                    recorded = out.read_bytes()
                    self.assertEqual(matrix.main(args), 0)
                self.assertEqual(OfflineRouter.calls, 1)
                self.assertEqual(out.read_bytes(), recorded)
                row = json.loads(recorded)
                self.assertEqual(row['accounting']['request_attempts'], 1)
                if expected is None:
                    self.assertFalse(row['valid'])
                    self.assertEqual(row['status'], 'invalid')
                    self.assertEqual(row['error'], 'InvalidScorerResult')
                    self.assertIsNone(row['score'])
                    self.assertIsNone(row['correct'])
                else:
                    self.assertTrue(row['valid'])
                    self.assertEqual(row['grading'], {'score': 1.0, 'matched': expected})
