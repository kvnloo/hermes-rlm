"""Offline resume preserves complete final rows even without their newline."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class ResumeNewlineTests(unittest.TestCase):
    def test_resume_appends_separate_json_record(self):
        for ending in ('', '\n', '\r\n', '\r'):
            with self.subTest(ending=repr(ending)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'context.txt').write_text('context')
                manifest = root / 'manifest.jsonl'
                task = dict(id='t', family='S-NIAH', size_tokens=8, question='Q',
                            gold='A', scoring='choice', context_path='context.txt')
                manifest.write_text(json.dumps(task) + '\n')
                out = root / 'out.jsonl'
                args = [str(manifest), str(out), '--ops-dir', str(root / 'ops'),
                        '--home-template', str(root), '--released-source', str(root),
                        '--released-python', sys.executable, '--fork-source', str(root),
                        '--fork-python', sys.executable, '--nano-source', str(root),
                        '--arms', 'base-truncated', '--repeats', '2']
                scheduled = io.StringIO()
                with patch.object(matrix, 'source_sha', return_value='fixture'), \
                        patch.object(matrix, 'run_one') as run, contextlib.redirect_stdout(scheduled):
                    matrix.main(args + ['--dry-run'])
                    run.assert_not_called()
                rows = [json.loads(line) for line in scheduled.getvalue().splitlines()]
                rows[0]['answer'] = 'café 🚀'
                original = (json.dumps(rows[0], ensure_ascii=False) + ending).encode()
                out.write_bytes(original)
                with patch.object(matrix, 'source_sha', return_value='fixture'), \
                        patch.object(matrix, 'run_one') as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(matrix.main(args + ['--dry-run']), 0)
                    run.assert_not_called()
                self.assertEqual(out.read_bytes(), original)
                with patch.object(matrix, 'source_sha', return_value='fixture'), \
                        patch.object(matrix, 'run_one', return_value=rows[1]) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(matrix.main(args), 0)
                    run.assert_called_once()
                self.assertTrue(out.read_bytes().startswith(original))
                self.assertEqual(out.read_bytes()[len(original):],
                                 (b'\n' if not ending else b'') +
                                 (json.dumps(rows[1], ensure_ascii=False) + '\n').encode())
                actual = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
                self.assertEqual(actual, rows)
                self.assertEqual(matrix.completed(out), {r['row_key'] for r in rows})
