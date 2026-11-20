"""The copied plugin manifest must participate in benchmark resume identity."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class PluginManifestResumeTests(unittest.TestCase):
    def test_manifest_changes_reschedule_and_identical_plugin_reuses_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'template'
            template.mkdir()
            plugin = root / 'plugin'
            plugin.mkdir()
            (plugin / '__init__.py').write_text('def register(ctx): pass\n')
            context = root / 'context.txt'
            context.write_text('A short context.')
            manifest = root / 'tasks.jsonl'
            manifest.write_text(json.dumps(dict(id='t', family='S-NIAH', size_tokens=4,
                context_path=str(context), question='Which letter?', gold=['A'], scoring='ruler')) + '\n')
            out = root / 'out.jsonl'
            args = [str(manifest), str(out), '--ops-dir', str(root / 'ops'),
                '--plugin-source', str(plugin), '--home-template', str(template), '--released-source', str(root),
                '--released-python', sys.executable, '--fork-source', str(root),
                '--fork-python', sys.executable, '--nano-source', str(root),
                '--arms', 'released-rlm', '--repeats', '1', '--dry-run']

            def pending():
                with patch.object(matrix, 'source_sha', return_value='fixed-sha'), \
                        patch.object(matrix, 'run_one') as run, \
                        contextlib.redirect_stdout(io.StringIO()) as captured:
                    self.assertEqual(matrix.main(args), 0)
                    run.assert_not_called()
                return [json.loads(line) for line in captured.getvalue().splitlines()]

            absent = pending()[0]['row_key']
            plugin_manifest = plugin / 'plugin.yaml'
            plugin_manifest.write_text('name: rlm\nversion: 0.1.0\n')
            original = pending()[0]['row_key']
            self.assertNotEqual(absent, original)
            out.write_text(json.dumps(dict(row_key=original)) + '\n')
            frozen = out.read_bytes()
            self.assertEqual(pending(), [])
            (template / 'unrelated.txt').write_text('Not copied into benchmark home.')
            self.assertEqual(pending(), [])
            plugin_manifest.write_text('name: rlm\nversion: 0.2.0\n')
            changed = pending()[0]['row_key']
            self.assertNotEqual(original, changed)
            plugin_manifest.unlink()
            self.assertEqual(pending()[0]['row_key'], absent)
            self.assertEqual(out.read_bytes(), frozen)
            out.write_text(json.dumps(dict(row_key=absent)) + '\n')
            self.assertEqual(pending(), [])
