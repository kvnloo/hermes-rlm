"""Missing scorer must be refused before benchmark runs can consume requests."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class ScorerPreflightTests(unittest.TestCase):
    def test_missing_or_directory_scorer_rejected_before_run_state(self):
        for kind in ('missing', 'directory'):
            for dry_run in (False, True):
                with self.subTest(kind=kind, dry_run=dry_run), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    scorer = root / 'scorer.py'
                    if kind == 'directory':
                        scorer.mkdir()
                    args = self.args(root) + ['--scorer', str(scorer)]
                    if dry_run:
                        args += ['--dry-run']
                    with patch.object(matrix, 'load_tasks', return_value=[dict(id='t', task_sha256='fixture')]) as load, \
                            patch.object(matrix, 'source_sha', return_value='fixture'), \
                            patch.object(matrix, 'run_one', return_value={}) as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, 'scorer must be a regular file'):
                            matrix.main(args)
                        load.assert_not_called()
                        run.assert_not_called()
                    self.assertFalse((root / 'ops').exists())
                    self.assertFalse((root / 'out.jsonl.lock').exists())
                    self.assertFalse((root / 'out.jsonl').exists())

    def args(self, root):
        return [str(root / 'manifest.jsonl'), str(root / 'out.jsonl'),
                '--ops-dir', str(root / 'ops'), '--home-template', str(root),
                '--released-source', str(root), '--released-python', sys.executable,
                '--fork-source', str(root), '--fork-python', sys.executable,
                '--nano-source', str(root), '--arms', 'base-truncated', '--repeats', '1']

    def test_custom_and_default_scorers_dry_run_without_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / 'custom.py'
            custom.write_text('raise AssertionError("must not import during preflight")\n')
            task = dict(id='t', task_sha256='fixture')
            for options in ([], ['--scorer', str(custom)]):
                with self.subTest(options=options), patch.object(matrix, 'load_tasks', return_value=[task]), \
                        patch.object(matrix, 'source_sha', return_value='fixture'), \
                        patch.object(matrix, 'run_one') as run, contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(matrix.main(self.args(root) + options + ['--dry-run']), 0)
                    self.assertEqual(json.loads(output.getvalue())['task_id'], 't')
                    run.assert_not_called()
