"""The copied SOUL identity must participate in benchmark resume identity."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class SoulResumeTests(unittest.TestCase):
    def test_changed_added_and_removed_soul_reschedule_but_other_template_files_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'template'
            template.mkdir()
            context = root / 'context.txt'
            context.write_text('A short context.')
            manifest = root / 'tasks.jsonl'
            manifest.write_text(json.dumps(dict(id='t', family='S-NIAH', size_tokens=4,
                context_path=str(context), question='Which letter?', gold=['A'], scoring='ruler')) + '\n')
            out = root / 'out.jsonl'
            args = [str(manifest), str(out), '--ops-dir', str(root / 'ops'),
                '--home-template', str(template), '--released-source', str(root),
                '--released-python', sys.executable, '--fork-source', str(root),
                '--fork-python', sys.executable, '--nano-source', str(root),
                '--arms', 'released-rlm', '--repeats', '1', '--dry-run']

            def pending():
                with patch.object(matrix, 'source_sha', return_value='fixed-sha'), \
                        patch.object(matrix, 'run_one') as run, \
                        patch.object(matrix, 'row_key', wraps=matrix.row_key) as key, \
                        contextlib.redirect_stdout(io.StringIO()) as captured:
                    self.assertEqual(matrix.main(args), 0)
                    run.assert_not_called()
                    protocol = key.call_args.args[3]
                    soul = template / 'SOUL.md'
                    if soul.is_file():
                        self.assertEqual(protocol['home_soul_sha256'], matrix.digest(soul))
                    else:
                        self.assertNotIn('home_soul_sha256', protocol)
                return [json.loads(line) for line in captured.getvalue().splitlines()]

            absent = pending()[0]['row_key']
            soul = template / 'SOUL.md'
            soul.write_text('Answer carefully.\n')
            original = pending()[0]['row_key']
            self.assertNotEqual(absent, original)
            out.write_text(json.dumps(dict(row_key=original)) + '\n')
            frozen = out.read_bytes()
            self.assertEqual(pending(), [])
            (template / 'unrelated.txt').write_text('Not copied into benchmark home.')
            self.assertEqual(pending(), [])
            soul.write_text('Answer concisely.\n')
            changed = pending()[0]['row_key']
            self.assertNotEqual(original, changed)
            soul.unlink()
            self.assertEqual(pending()[0]['row_key'], absent)
            self.assertEqual(out.read_bytes(), frozen)
            out.write_text(json.dumps(dict(row_key=absent)) + '\n')
            self.assertEqual(pending(), [])
