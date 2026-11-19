"""Pin Session._send (one newline-terminated JSON line, flushed) and Session._request (serve LLM rounds until 'done')."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class SendRequestPins(unittest.TestCase):
    def session(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return rlm.Session('t', log_dir=Path(tmp.name))

    def test_send_writes_one_json_line_then_flushes(self):
        s = self.session()
        s.proc = mock.Mock()
        msg = {"type": "load", "context": "a\nb"}
        s._send(msg)
        calls = s.proc.stdin.mock_calls
        self.assertEqual([c[0] for c in calls], ['write', 'flush'])
        line = calls[0].args[0]
        self.assertTrue(line.endswith('\n'))
        self.assertEqual(line.count('\n'), 1)
        self.assertEqual(json.loads(line), msg)

    def request(self, replies):
        s = self.session()
        parent = mock.Mock()
        parent._read.side_effect = replies + [AssertionError('read past done')]
        parent._serve_llm.side_effect = lambda r: ('served', r['id'])
        for n in ('_send', '_read', '_serve_llm'):
            setattr(s, n, getattr(parent, n))
        msg, deadline = {'type': 'exec'}, object()
        return s._request(msg, deadline), parent, msg, deadline

    def test_immediate_done(self):
        done = {'type': 'done', 'stdout': 'x'}
        got, p, msg, dl = self.request([done])
        self.assertIs(got, done)
        self.assertEqual(p.mock_calls, [mock.call._send(msg), mock.call._read(dl)])

    def test_serves_each_llm_round_then_returns_done(self):
        r1, r2 = {'type': 'llm', 'id': 1}, {'type': 'llm', 'id': 2}
        done = {'type': 'done'}
        got, p, msg, dl = self.request([r1, r2, done])
        self.assertIs(got, done)
        self.assertEqual(p.mock_calls, [
            mock.call._send(msg), mock.call._read(dl),
            mock.call._serve_llm(r1), mock.call._send(('served', 1)), mock.call._read(dl),
            mock.call._serve_llm(r2), mock.call._send(('served', 2)), mock.call._read(dl)])


if __name__ == "__main__":
    unittest.main()
