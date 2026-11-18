"""A failed transport exchange must never reuse its desynchronized child."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rlm


class ExecCleanupTests(unittest.TestCase):
    def test_unexpected_request_failure_discards_child(self):
        with tempfile.TemporaryDirectory() as root:
            for error in (json.JSONDecodeError('invalid reply', '!', 0),
                          UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad byte'),
                          KeyboardInterrupt('interrupted')):
                with self.subTest(error=type(error).__name__):
                    session = rlm.Session('test', log_dir=Path(root))
                    proc = session.proc = mock.Mock()
                    proc.poll.return_value = None
                    with mock.patch.object(session, '_request', side_effect=error):
                        with self.assertRaises(type(error)) as caught:
                            session.exec('1 + 1')
                    self.assertIs(caught.exception, error)
                    proc.kill.assert_called_once_with()
                    proc.wait.assert_called_once_with()
                    self.assertIsNone(session.proc)
                    self.assertEqual(session.stats['restarts'], 1)

    def test_malformed_reply_shape_discards_child(self):
        with tempfile.TemporaryDirectory() as root:
            for reply, error in (([], TypeError), ('x', TypeError), ({}, KeyError)):
                with self.subTest(reply=reply):
                    session = rlm.Session('test', log_dir=Path(root))
                    proc = session.proc = mock.Mock()
                    proc.poll.return_value = None
                    with mock.patch.object(session, '_send'), \
                            mock.patch.object(session, '_read', return_value=reply):
                        with self.assertRaises(error):
                            session.exec('1 + 1')
                    proc.kill.assert_called_once_with()
                    self.assertIsNone(session.proc)
                    self.assertEqual(session.stats['restarts'], 1)

    def test_kernel_reported_cell_error_does_not_discard_healthy_child(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('test', log_dir=Path(root))
            proc = session.proc = mock.Mock()
            proc.poll.return_value = None
            with mock.patch.object(session, '_request', return_value={
                    'type': 'done', 'stdout': '', 'error': 'NameError: missing'}):
                result = session.exec('missing')
            self.assertEqual(result['error'], 'NameError: missing')
            self.assertIs(session.proc, proc)
            proc.kill.assert_not_called()
            self.assertEqual(session.stats['restarts'], 0)
