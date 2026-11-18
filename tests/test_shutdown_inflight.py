"""_shutdown during an in-flight exec must not crash the exec or close its pipes under it."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import rlm


class ShutdownInflightTests(unittest.TestCase):
    def test_shutdown_during_exec_returns_error_reply(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=Path(tmp.name) / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': tmp.name})):
            p.start()
            self.addCleanup(p.stop)
        s = rlm.session_for(task_id='t', session_id='inflight')
        s.exec('1')
        proc = s.proc
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
            rlm._shutdown()
            t.join(10)
        self.assertFalse(t.is_alive())
        self.assertNotIn('exc', out)
        self.assertIn('kernel exited', out['reply']['error'])
        self.assertIsNone(s.proc)
        self.assertEqual(s.stats['restarts'], 0)
        self.assertTrue(proc.stdin.closed and proc.stdout.closed)

    def test_shutdown_during_spawn_leaves_no_child(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=Path(tmp.name) / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': tmp.name})):
            p.start()
            self.addCleanup(p.stop)
        s = rlm.session_for(task_id='t', session_id='spawning')
        inside, gate, spawn = threading.Event(), threading.Event(), s._spawn
        def slow_spawn():
            inside.set()
            gate.wait(10)
            spawn()
        s._spawn = slow_spawn
        out = {}
        def run():
            try:
                out['reply'] = s.exec('1')
            except BaseException as exc:
                out['exc'] = exc
        t = threading.Thread(target=run)
        t.start()
        self.addCleanup(lambda: s.proc and s.proc.kill())
        self.assertTrue(inside.wait(10))
        rlm._shutdown()
        gate.set()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertIsInstance(out.get('exc'), RuntimeError)
        self.assertIn('closed', str(out['exc']))
        self.assertIsNone(s.proc)


if __name__ == '__main__':
    unittest.main()
