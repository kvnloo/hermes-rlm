"""Trusted pipe fixtures for write-side request deadline enforcement."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import rlm


class WriterDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = rlm.Session('writer', log_dir=Path(self.tmp.name))
        self.addCleanup(self.session.kill)

    def child(self, code):
        proc = subprocess.Popen([sys.executable, '-u', '-c', code],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.session.proc = proc
        def cleanup():
            if proc.poll() is None: proc.kill()
            proc.wait(timeout=3)
            proc.stdin.close()
            proc.stdout.close()
        self.addCleanup(cleanup)
        return proc

    def test_full_pipe_cannot_delay_exec_timeout(self):
        proc = self.child("import sys,time; time.sleep(.8); sys.stdin.readline(); print('{\"type\":\"done\"}')")
        self.session.lim['cell_timeout_s'] = .1
        start = time.monotonic()
        result = self.session.exec('x' * 1000000)
        self.assertLess(time.monotonic() - start, .5)
        self.assertIn('exceeded', result['error'])
        self.assertIsNone(self.session.proc)
        self.assertIsNotNone(proc.poll())

    def test_large_unicode_frame_round_trips(self):
        self.child("import sys,json; msg=json.loads(sys.stdin.readline()); print(json.dumps({'type':'done','text':msg['text']}))")
        value = '雪🙂' * 20000
        reply = self.session._request({'text': value}, time.monotonic() + 3)
        self.assertEqual(reply['text'], value)

    def test_expired_deadline_never_writes(self):
        self.child('import time; time.sleep(2)')
        with mock.patch.object(rlm.os, 'write') as write:
            with self.assertRaises(TimeoutError):
                self.session._send({'x': 1}, time.monotonic() - 1)
        write.assert_not_called()

    def test_zero_progress_write_refuses(self):
        self.child('import time; time.sleep(2)')
        with mock.patch.object(rlm.os, 'write', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'write'):
                self.session._send({'x': 1}, time.monotonic() + 1)

    def test_subcall_reply_uses_original_deadline(self):
        deadline = time.monotonic() + 1
        with mock.patch.object(self.session, '_send') as send, \
             mock.patch.object(self.session, '_read', side_effect=[{'type':'llm'}, {'type':'done'}]), \
             mock.patch.object(self.session, '_serve_llm', return_value={'type':'llm_result'}):
            self.session._request({'type':'exec'}, deadline)
        self.assertEqual(send.call_args_list, [mock.call({'type':'exec'}, deadline),
                                               mock.call({'type':'llm_result'}, deadline)])
