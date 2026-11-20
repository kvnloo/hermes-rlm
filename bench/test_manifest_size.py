"""Nominal sizes must support meaningful numeric task ordering."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class ManifestSizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'context.txt').write_text('context')
        self.manifest = self.root / 'manifest.jsonl'

    def write(self, sizes):
        self.manifest.write_text(''.join(json.dumps(dict(
            id=str(i), family='S-NIAH', size_tokens=size, question='Q',
            gold='A', scoring='choice', context_path='context.txt')) + '\n'
            for i, size in enumerate(sizes)))

    def test_invalid_sizes_rejected(self):
        for size in (float('nan'), float('inf'), -float('inf'), -1, -0.5, True, False):
            with self.subTest(size=size):
                self.write([size])
                with self.assertRaisesRegex(ValueError, 'size_tokens'):
                    matrix.load_tasks(self.manifest)

    def test_nonnegative_finite_sizes_preserved(self):
        sizes = [0, 0.25, 8, 10 ** 400]
        self.write(sizes)
        self.assertEqual([t['size_tokens'] for t in matrix.load_tasks(self.manifest)], sizes)

    def test_smallest_selection_rejects_nan_instead_of_selecting_it(self):
        args = [str(self.manifest), str(self.root / 'out.jsonl'),
                '--ops-dir', str(self.root / 'ops'), '--home-template', str(self.root),
                '--released-source', str(self.root), '--released-python', sys.executable,
                '--fork-source', str(self.root), '--fork-python', sys.executable,
                '--nano-source', str(self.root), '--arms', 'base-truncated',
                '--repeats', '1', '--smallest-sniah', '--dry-run']
        with patch.object(matrix, 'source_sha', return_value='fixture'), \
                patch.object(matrix, 'run_one') as run:
            self.write([float('nan'), 1])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                with self.assertRaisesRegex(ValueError, 'size_tokens'):
                    matrix.main(args)
            self.assertEqual(stream.getvalue(), '')
            self.assertFalse((self.root / 'ops/request-ledger.sqlite3').exists())
            self.write([8, 0.25, 1])
            with contextlib.redirect_stdout(stream):
                self.assertEqual(matrix.main(args), 0)
            self.assertEqual(json.loads(stream.getvalue())['task_id'], '1')
            run.assert_not_called()
