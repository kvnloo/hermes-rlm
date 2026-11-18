"""Trusted pipe fixtures: partial frames must not bypass the read deadline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import rlm


class ReaderDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = rlm.Session('reader', log_dir=Path(self.tmp.name))
        self.addCleanup(self.session.kill)

    def child(self, code):
        proc = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True)
        self.session.proc = proc
        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
            proc.stdin.close()
            proc.stdout.close()
        self.addCleanup(cleanup)
        return proc

    def test_partial_line_obeys_deadline_and_exec_cleans_up(self):
        proc = self.child("import sys,time; sys.stdin.readline(); sys.stdout.write('{'); sys.stdout.flush(); time.sleep(.5); sys.stdout.write('\"type\":\"done\"}\\n'); sys.stdout.flush()")
        self.session.lim['cell_timeout_s'] = .1
        result = self.session.exec('fixture')
        self.assertIn('exceeded', result['error'] or '')
        self.assertIsNone(self.session.proc)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(self.session.stats['restarts'], 1)

    def test_multiple_frames_preserve_buffer_and_expired_deadline_refuses(self):
        self.child("import os,time; os.write(1,b'{\"type\":\"done\",\"n\":1}\\n{\"type\":\"done\",\"n\":2}\\n'); time.sleep(.5)")
        self.assertEqual(self.session._read(time.monotonic()+2)['n'], 1)
        with self.assertRaises(TimeoutError):
            self.session._read(time.monotonic()-1)
        self.assertEqual(self.session._read(time.monotonic()+.2)['n'], 2)

    def test_utf8_split_across_chunks(self):
        payload = json.dumps({'type':'done','text':'雪'}, ensure_ascii=False).encode()+b'\n'
        at = payload.index('雪'.encode())+1
        self.child('import os,time; os.write(1,'+repr(payload[:at])+'); time.sleep(.03); os.write(1,'+repr(payload[at:])+')')
        self.assertEqual(self.session._read(time.monotonic()+2)['text'], '雪')

    def test_partial_eof_refuses_incomplete_frame(self):
        self.child("import os; os.write(1,b'{')")
        with self.assertRaisesRegex(RuntimeError, 'kernel exited'):
            self.session._read(time.monotonic()+2)

    def test_spawn_and_kill_clear_old_buffer(self):
        self.session._read_buffer = b'{"type":"done","stale":true}\n'
        argv = [sys.executable, '-c', 'import os; os.write(1,b\'{"type":"done","fresh":true}\\n\')']
        with mock.patch.object(rlm, 'sandbox_argv', return_value=argv):
            self.session._spawn()
        proc = self.session.proc
        self.addCleanup(proc.stdin.close)
        self.addCleanup(proc.stdout.close)
        self.assertTrue(self.session._read(time.monotonic()+2)['fresh'])
        self.session._read_buffer = b'partial'
        self.session.kill()
        self.assertEqual(self.session._read_buffer, b'')
