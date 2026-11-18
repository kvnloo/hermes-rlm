"""Unloaded Session references may not resurrect detached kernels."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm

class RetiredSessionTests(unittest.TestCase):
    def test_retained_reference_refuses_exec_and_load(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('old', log_dir=Path(root))
            with mock.patch.object(rlm, '_sessions', {'old': s}):
                rlm._shutdown()
                self.assertEqual(rlm._sessions, {})
                with mock.patch.object(s, '_spawn') as spawn, mock.patch.object(rlm, 'load_corpus') as read:
                    with self.assertRaisesRegex(RuntimeError, 'closed'):
                        s.exec('pass')
                    with self.assertRaisesRegex(RuntimeError, 'closed'):
                        s.load(['/not/read'])
                    spawn.assert_not_called()
                    read.assert_not_called()

    def test_shutdown_during_spawn_disposes_new_child(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('old', log_dir=Path(root))
            proc = mock.Mock()
            proc.poll.return_value = None
            def create(*a, **kw):
                rlm._shutdown()  # child not published to s.proc yet
                return proc
            with mock.patch.object(rlm, '_sessions', {'old': s}), \
                 mock.patch.object(rlm, 'sandbox_argv', return_value=['fixture']), \
                 mock.patch.object(rlm.subprocess, 'Popen', side_effect=create), s.lock:
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    s._spawn()
                proc.kill.assert_called_once()
                proc.wait.assert_called_once()
                self.assertIsNone(s.proc)

    def test_shutdown_during_corpus_read_refuses_spawn(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('old', log_dir=Path(root))
            def read(*a, **kw):
                rlm._shutdown()
                return {'file': 'text'}, 0
            with mock.patch.object(rlm, '_sessions', {'old': s}), \
                 mock.patch.object(rlm, 'load_corpus', side_effect=read), \
                 mock.patch.object(s, '_spawn') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    s.load(['fixture'])
                spawn.assert_not_called()

    def test_normal_kill_is_restartable(self):
        with tempfile.TemporaryDirectory() as root:
            s = rlm.Session('alive', log_dir=Path(root))
            s.kill()
            with mock.patch.object(s, '_spawn') as spawn, mock.patch.object(s, '_request', return_value={'stdout': ''}):
                s.exec('pass')
                spawn.assert_called_once()

if __name__ == '__main__': unittest.main()
