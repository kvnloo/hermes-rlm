"""A deadline expiring during unload must not promise a restart."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm

class ShutdownTimeoutTests(unittest.TestCase):
    def test_timeout_after_shutdown_reports_closed_without_restart(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('test', log_dir=Path(root))
            s.proc = mock.Mock()
            s.proc.poll.return_value = None
            def request(*args):
                rlm._shutdown()
                raise TimeoutError
            with mock.patch.object(rlm, '_sessions', {'test': s}), \
                 mock.patch.object(s, '_request', side_effect=request):
                result = s.exec('fixture')
            self.assertIn('session closed', result['error'])
            self.assertNotIn('restarted', result['error'])
            self.assertEqual(s.stats['restarts'], 0)
            self.assertIsNone(s.proc)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                s.exec('next')

    def test_ordinary_timeout_keeps_restart_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('test', log_dir=Path(root))
            s.proc = mock.Mock()
            s.proc.poll.return_value = None
            with mock.patch.object(s, '_request', side_effect=TimeoutError):
                result = s.exec('fixture')
            self.assertIn('kernel restarted', result['error'])
            self.assertNotIn('session closed', result['error'])
            self.assertEqual(s.stats['restarts'], 1)
            self.assertIsNone(s.proc)
            with mock.patch.object(s, '_spawn') as spawn, mock.patch.object(s, '_request', return_value={'stdout': ''}):
                s.exec('next')
                spawn.assert_called_once()

if __name__ == '__main__': unittest.main()
