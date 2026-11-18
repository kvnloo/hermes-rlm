"""kill() must close the kernel's stdin/stdout pipes, not leave them to GC (fd leak per restart/unload)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rlm


class KillClosesPipesTests(unittest.TestCase):
    def _session(self, sid):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=Path(tmp.name) / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': tmp.name})):
            p.start()
            self.addCleanup(p.stop)
        s = rlm.session_for(task_id='t', session_id=sid)
        s.exec('1')
        return s

    def test_kill_closes_both_pipes_of_live_kernel(self):
        s = self._session('live')
        proc = s.proc
        self.assertIsNotNone(proc)
        s.kill()
        self.assertIsNone(s.proc)
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)
        s.kill()  # idempotent

    def test_kill_closes_pipes_of_already_exited_kernel(self):
        s = self._session('dead')
        proc = s.proc
        proc.kill()
        proc.wait()
        s.kill()
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)


if __name__ == '__main__':
    unittest.main()
