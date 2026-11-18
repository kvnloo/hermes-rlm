"""Host corpus paths must never silently fall back to a gateway's cwd."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import rlm


class TaskPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gateway = self.root / 'gateway'
        self.workspace = self.root / 'workspace'
        self.gateway.mkdir()
        self.workspace.mkdir()
        (self.gateway / 'corpus.txt').write_text('WRONG GATEWAY CONTENT')
        (self.workspace / 'corpus.txt').write_text('task content')
        previous = os.getcwd()
        os.chdir(self.gateway)
        self.addCleanup(os.chdir, previous)
        # Simulate the original private resolver failing. The fixed implementation
        # must not import it at all; the supported context API supplies the anchor.
        self.legacy = mock.Mock(side_effect=RuntimeError('resolver unavailable'))
        modules = {'tools': SimpleNamespace(), 'tools.file_tools_paths':
                   SimpleNamespace(_resolve_base_dir=self.legacy)}
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)

    def context(self, resolver):
        patch = mock.patch.object(rlm, '_CTX', SimpleNamespace(session_cwd=resolver))
        patch.start()
        self.addCleanup(patch.stop)

    def test_resolver_error_never_reads_gateway_file(self):
        self.context(mock.Mock(side_effect=RuntimeError('resolver unavailable')))
        with self.assertRaisesRegex(RuntimeError, 'session workspace'):
            rlm.load_corpus(['corpus.txt'], 100, 'task-a')

    def test_missing_or_relative_workspace_is_refused(self):
        for value in ('', '.', 'relative', None):
            with self.subTest(value=value), mock.patch.object(
                    rlm, '_CTX', SimpleNamespace(session_cwd=lambda: value)):
                with self.assertRaisesRegex(RuntimeError, 'session workspace'):
                    rlm.load_corpus(['corpus.txt'], 100, 'task-a')

    def test_unregistered_context_is_refused(self):
        with mock.patch.object(rlm, '_CTX', None):
            with self.assertRaisesRegex(RuntimeError, 'session workspace'):
                rlm.load_corpus(['corpus.txt'], 100, 'task-a')

    def test_supported_context_anchors_files_dirs_and_globs(self):
        resolver = mock.Mock(return_value=str(self.workspace))
        self.context(resolver)
        for raw in ('corpus.txt', '.', '*.txt'):
            with self.subTest(raw=raw):
                corpus, skipped = rlm.load_corpus([raw], 100, 'task-a')
                self.assertEqual(corpus, {'corpus.txt': 'task content'})
                self.assertEqual(skipped, 0)
        self.assertEqual(resolver.call_count, 3)
        self.legacy.assert_not_called()

    def test_load_handler_returns_json_error_on_unresolvable_workspace(self):
        self.context(mock.Mock(side_effect=RuntimeError('resolver unavailable')))
        session = SimpleNamespace(load=lambda paths, task_id: rlm.load_corpus(paths, 100, task_id))
        with mock.patch.object(rlm, 'session_for', return_value=session):
            out = json.loads(rlm._load_handler({'paths': ['corpus.txt']}, task_id='task-a'))
        self.assertIn('session workspace', out['error'])

    def test_explicit_absolute_path_needs_no_context(self):
        with mock.patch.object(rlm, '_CTX', None):
            corpus, _ = rlm.load_corpus([str(self.workspace / 'corpus.txt')], 100)
        self.assertEqual(corpus, {'corpus.txt': 'task content'})
        self.legacy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
