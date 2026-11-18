"""Failed initial context handshakes must not leave a live child attached."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rlm


class SpawnCleanupTests(unittest.TestCase):
    def test_failed_handshake_kills_child_and_preserves_exception(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('test', log_dir=Path(root))
            session.context = 'retained corpus'
            for error in (TimeoutError('startup timeout'), ValueError('invalid JSON'),
                          KeyboardInterrupt('cancelled')):
                with self.subTest(error=type(error).__name__):
                    proc = mock.Mock()
                    proc.poll.return_value = None
                    with mock.patch.object(rlm, 'sandbox_argv', return_value=['synthetic']), \
                         mock.patch.object(rlm.subprocess, 'Popen', return_value=proc), \
                         mock.patch.object(session, '_request', side_effect=error):
                        with self.assertRaises(type(error)) as caught:
                            session._spawn()
                    self.assertIs(caught.exception, error)
                    proc.kill.assert_called_once_with()
                    proc.wait.assert_called_once_with()
                    self.assertIsNone(session.proc)
                    self.assertEqual(session.context, 'retained corpus')

    def test_successful_handshake_keeps_child(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('test', log_dir=Path(root))
            session.context = 'corpus'
            proc = mock.Mock()
            with mock.patch.object(rlm, 'sandbox_argv', return_value=['synthetic']), \
                 mock.patch.object(rlm.subprocess, 'Popen', return_value=proc), \
                 mock.patch.object(session, '_request', return_value={'type': 'done'}):
                session._spawn()
            self.assertIs(session.proc, proc)
            proc.kill.assert_not_called()
