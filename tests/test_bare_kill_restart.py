"""A bare kill() mid-exec is a real kernel restart and the next cell respawns; only _shutdown retires a session."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import rlm


class BareKillIsRestartTests(unittest.TestCase):
    def test_bare_kill_during_exec_counts_restart_and_stays_restartable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=Path(tmp.name) / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': tmp.name})):
            p.start()
            self.addCleanup(p.stop)
        s = rlm.session_for(task_id='t', session_id='direct-kill')
        self.addCleanup(s.kill)
        s.exec('1')
        waiting, real_select, out = threading.Event(), rlm.select.select, {}
        def select(r, w, x, t):
            if r:
                waiting.set()
            return real_select(r, w, x, t)
        def run():
            try:
                out['reply'] = s.exec('import time; time.sleep(60)')
            except BaseException as exc:
                out['exc'] = exc
        with mock.patch.object(rlm.select, 'select', select):
            t = threading.Thread(target=run)
            t.start()
            self.assertTrue(waiting.wait(10))
            s.kill()
            t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(s.stats['restarts'], 1)
        self.assertIn('restarted', out['reply']['error'])
        self.assertEqual(s.exec('1 + 1').get('error'), None)


if __name__ == '__main__':
    unittest.main()
