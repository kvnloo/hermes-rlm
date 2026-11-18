"""A session retired between exec's closed check and _spawn must not start a child at all."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rlm


class SpawnClosedGuardTests(unittest.TestCase):
    def test_shutdown_before_popen_spawns_nothing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=Path(tmp.name) / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': tmp.name})):
            p.start()
            self.addCleanup(p.stop)
        s = rlm.session_for(task_id='t', session_id='pre-popen')
        spawn = s._spawn
        s._spawn = lambda: (rlm._shutdown(), spawn())
        with mock.patch.object(rlm.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                s.exec('1')
        popen.assert_not_called()
        self.assertIsNone(s.proc)


if __name__ == '__main__':
    unittest.main()
