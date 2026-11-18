"""Malformed tool calls must not replace an existing corpus or kernel."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm

class LoadAdmissionTests(unittest.TestCase):
    def test_invalid_paths_refused_before_session_lookup_or_mutation(self):
        invalid = [None, False, 0, {}, '', 'file.txt', [None], [False], [1], [{}], [['file.txt']], ['ok', None]]
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('existing', log_dir=Path(root))
            corpus = {'original.txt': 'retain this corpus'}
            session.context = corpus
            for args in [{}] + [{'paths': value} for value in invalid]:
                with self.subTest(args=args), mock.patch.object(rlm, 'session_for', return_value=session) as lookup, mock.patch.object(session, '_teardown') as teardown, mock.patch.object(session, '_spawn') as spawn:
                    result = json.loads(rlm._load_handler(args, task_id='t', session_id='s'))
                    self.assertIn('paths must be an array of strings', result.get('error', ''))
                    lookup.assert_not_called()
                    teardown.assert_not_called()
                    spawn.assert_not_called()
                    self.assertIs(session.context, corpus)

    def test_explicit_empty_list_preserves_existing_reset_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('existing', log_dir=Path(root))
            session.context = 'old corpus'
            with mock.patch.object(rlm, 'session_for', return_value=session), mock.patch.object(session, '_teardown') as teardown, mock.patch.object(session, '_spawn') as spawn:
                result = json.loads(rlm._load_handler({'paths': []}, task_id='t', session_id='s'))
            self.assertTrue(result['loaded'])
            self.assertEqual(result['files'], 0)
            self.assertEqual(session.context, {})
            teardown.assert_called_once()
            spawn.assert_called_once()

    def test_string_list_and_direct_loader_string_remain_supported(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / 'corpus.txt'
            file.write_text('valid text')
            session = rlm.Session('existing', log_dir=Path(root))
            with mock.patch.object(rlm, 'session_for', return_value=session), mock.patch.object(session, '_spawn'):
                result = json.loads(rlm._load_handler({'paths': [str(file)]}, task_id='t', session_id='s'))
            self.assertTrue(result['loaded'])
            self.assertEqual(session.context, 'valid text')
            self.assertEqual(rlm.load_corpus(str(file), 100), ({'corpus.txt': 'valid text'}, 0))

if __name__ == '__main__':
    unittest.main()
