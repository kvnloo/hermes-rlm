"""Pin Session._spawn: Popen argv/pipes/mode, the context load gets a 120s deadline from now, and only when a context exists."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class SpawnDeadlinePins(unittest.TestCase):
    def session(self, context):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = rlm.Session('t', log_dir=Path(tmp.name))
        s.context = context
        return s

    def spawn(self, s):
        clock = mock.Mock(monotonic=mock.Mock(return_value=1000.0))
        with mock.patch.object(rlm, 'time', clock), \
                mock.patch.object(rlm, 'sandbox_argv', return_value=['argv', 'x']), \
                mock.patch.object(rlm.subprocess, 'Popen') as popen, \
                mock.patch.object(s, '_request') as request:
            s._spawn()
        return popen, request

    def test_load_deadline_is_120s_from_now(self):
        popen, request = self.spawn(self.session({'k': 'v'}))
        popen.assert_called_once()
        request.assert_called_once_with({'type': 'load', 'context': {'k': 'v'}}, 1120.0)

    def test_popen_gets_sandbox_argv_pipes_devnull_stderr_and_line_text_mode(self):
        s = self.session(None)
        popen, _ = self.spawn(s)
        popen.assert_called_once_with(['argv', 'x'], stdin=rlm.subprocess.PIPE, stdout=rlm.subprocess.PIPE,
                                      stderr=rlm.subprocess.DEVNULL, text=True, bufsize=1)
        self.assertIs(s.proc, popen.return_value)

    def test_falsy_but_set_context_still_loads(self):
        _, request = self.spawn(self.session(''))
        request.assert_called_once_with({'type': 'load', 'context': ''}, 1120.0)

    def test_no_context_no_load(self):
        popen, request = self.spawn(self.session(None))
        popen.assert_called_once()
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
