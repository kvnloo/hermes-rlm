"""Task limits must never silently expand the requested benchmark workload."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class TaskLimitTests(unittest.TestCase):
    def invoke(self, limit):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'context.txt').write_text('context')
            tasks = [dict(id=str(i), family='S-NIAH', size_tokens=8, question='Q',
                          gold='A', scoring='choice', context_path='context.txt') for i in range(3)]
            manifest = root / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
            args = [str(manifest), str(root / 'out.jsonl'), '--ops-dir', str(root / 'ops'),
                    '--home-template', str(root), '--released-source', str(root),
                    '--released-python', sys.executable, '--fork-source', str(root),
                    '--fork-python', sys.executable, '--nano-source', str(root),
                    '--dry-run', '--arms', 'base-truncated', '--repeats', '1']
            if limit is not None:
                args += ['--limit-tasks', str(limit)]
            stream = io.StringIO()
            with patch.object(matrix, 'source_sha', return_value='fixture'), \
                    patch.object(matrix, 'run_one') as run, contextlib.redirect_stdout(stream):
                try:
                    matrix.main(args)
                except ValueError:
                    self.assertFalse((root / 'ops').exists())
                    self.assertFalse((root / 'out.jsonl.lock').exists())
                    self.assertFalse((root / 'out.jsonl').exists())
                    self.assertEqual(stream.getvalue(), '')
                    raise
                finally:
                    run.assert_not_called()
            return [json.loads(line)['task_id'] for line in stream.getvalue().splitlines()]

    def test_nonpositive_limit_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, 'limit-tasks must be positive'):
                    self.invoke(limit)

    def test_positive_and_omitted_limits_preserve_manifest_order(self):
        self.assertEqual(self.invoke(1), ['0'])
        self.assertEqual(self.invoke(2), ['0', '1'])
        self.assertEqual(self.invoke(10), ['0', '1', '2'])
        self.assertEqual(self.invoke(None), ['0', '1', '2'])
