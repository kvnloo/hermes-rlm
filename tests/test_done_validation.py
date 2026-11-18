"""Malformed completion fields must take the guarded protocol cleanup path."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import rlm


class DoneValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def session(self):
        return rlm.Session('done', log_dir=Path(self.tmp.name))

    def test_invalid_fields_discard_child_before_formatting_or_receipt(self):
        for field in ('stdout', 'error', 'final'):
            for value in (1, False, [], {}):
                with self.subTest(field=field, value=value):
                    s = self.session()
                    proc = s.proc = mock.Mock()
                    proc.poll.return_value = None
                    with mock.patch.object(s, '_send'), mock.patch.object(s, '_read', return_value={
                            'type': 'done', field: value}):
                        result = s.exec('fixture')
                    self.assertIn('invalid sandbox done field: ' + field, result['error'] or '')
                    self.assertIsNone(s.proc)
                    proc.kill.assert_called_once_with()
                    proc.wait.assert_called_once_with()
                    self.assertEqual(s.stats['restarts'], 1)
                    self.assertNotIn('final', result)

    def test_optional_fields_and_valid_kernel_error_preserve_child(self):
        for fields in ({}, {'stdout': None, 'error': None, 'final': None},
                       {'stdout': '', 'error': '', 'final': ''},
                       {'stdout': 'ok', 'error': 'NameError: missing', 'final': None},
                       {'stdout': '雪', 'error': None, 'final': 'answer', 'extra': {'allowed': True}}):
            with self.subTest(fields=fields):
                s = self.session()
                proc = s.proc = mock.Mock()
                proc.poll.return_value = None
                with mock.patch.object(s, '_send'), mock.patch.object(s, '_read', return_value={
                        'type': 'done', **fields}):
                    result = s.exec('fixture')
                self.assertEqual(result['stdout'], fields.get('stdout') or '')
                self.assertEqual(result['error'], fields.get('error'))
                self.assertIs(s.proc, proc)
                proc.kill.assert_not_called()
                self.assertEqual(s.stats['restarts'], 0)

    def test_real_malformed_reply_cleans_up_exec(self):
        s = self.session()
        proc = s.proc = subprocess.Popen([sys.executable, '-u', '-c',
            "import sys,time; sys.stdin.readline(); print('{\"type\":\"done\",\"stdout\":1}'); time.sleep(5)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            result = s.exec('fixture')
            self.assertIn('invalid sandbox done field: stdout', result['error'])
            self.assertIsNone(s.proc)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None: proc.kill()
            proc.wait(timeout=3)
            proc.stdin.close()
            proc.stdout.close()

    def test_invalid_load_handshake_cleans_up_spawned_child(self):
        s = self.session()
        s.context = 'retained corpus'
        children = []
        real_popen = subprocess.Popen
        def popen(*args, **kwargs):
            p = real_popen(*args, **kwargs)
            children.append(p)
            return p
        argv = [sys.executable, '-u', '-c',
                "import sys,time; sys.stdin.readline(); print('{\"type\":\"done\",\"error\":[]}'); time.sleep(5)"]
        try:
            with mock.patch.object(rlm, 'sandbox_argv', return_value=argv), mock.patch.object(rlm.subprocess, 'Popen', side_effect=popen):
                with self.assertRaisesRegex(RuntimeError, 'invalid sandbox done field: error'):
                    s._spawn()
            self.assertIsNone(s.proc)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual(s.context, 'retained corpus')
        finally:
            for p in children:
                if p.poll() is None: p.kill()
                p.wait(timeout=3)
                p.stdin.close()
                p.stdout.close()
