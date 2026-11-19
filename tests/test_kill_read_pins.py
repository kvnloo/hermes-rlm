"""Pin Session.kill (only kills a live proc, waits, clears it) and Session._read (deadline boundary, 1s select cap)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import rlm


class KillReadPins(unittest.TestCase):
    def session(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return rlm.Session('t', log_dir=Path(tmp.name))

    def test_kill_live_proc_kills_waits_and_clears(self):
        s = self.session()
        proc = s.proc = mock.Mock(**{'poll.return_value': None})
        s.kill()
        self.assertEqual(proc.mock_calls, [mock.call.poll(), mock.call.kill(), mock.call.wait()])
        self.assertIsNone(s.proc)

    def test_kill_exited_proc_not_killed_but_cleared(self):
        s = self.session()
        proc = s.proc = mock.Mock(**{'poll.return_value': 0})
        s.kill()
        proc.kill.assert_not_called()
        proc.wait.assert_not_called()
        self.assertIsNone(s.proc)

    def read(self, left):
        s = self.session()
        s.proc = mock.Mock()
        s.proc.stdout.readline.return_value = '{"type": "done"}\n'
        clock = mock.Mock(monotonic=mock.Mock(return_value=100.0))
        with mock.patch.object(rlm, 'time', clock), \
                mock.patch.object(rlm.select, 'select', return_value=([s.proc.stdout], [], [])) as sel:
            return s._read(100.0 + left), sel

    def test_deadline_reached_raises_without_polling(self):
        s = self.session()
        s.proc = mock.Mock()
        clock = mock.Mock(monotonic=mock.Mock(return_value=100.0))
        with mock.patch.object(rlm, 'time', clock), \
                mock.patch.object(rlm.select, 'select') as sel:
            with self.assertRaises(TimeoutError):
                s._read(100.0)
        sel.assert_not_called()

    def test_select_wait_capped_at_one_second(self):
        reply, sel = self.read(50.0)
        self.assertEqual(reply, {'type': 'done'})
        self.assertEqual(sel.call_args.args[3], 1.0)

    def test_select_wait_uses_remaining_time_when_under_cap(self):
        _, sel = self.read(0.25)
        self.assertEqual(sel.call_args.args[3], 0.25)


if __name__ == "__main__":
    unittest.main()
