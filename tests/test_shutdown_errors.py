"""A failed session must not prevent disposal of the remaining sessions."""
import unittest
from types import SimpleNamespace
from unittest import mock
import rlm


class ShutdownErrorTests(unittest.TestCase):
    def test_cleanup_continues_after_each_failure(self):
        calls = []
        def kill(name, fail=False):
            self.assertTrue(rlm._sessions_lock.acquire(blocking=False))
            rlm._sessions_lock.release()
            calls.append(name)
            if fail:
                raise OSError('synthetic kill failure')
        sessions = {name: SimpleNamespace(kill=lambda n=name, f=fail: kill(n, f))
                    for name, fail in [('first', True), ('second', False), ('third', True), ('last', False)]}
        with mock.patch.object(rlm, '_sessions', sessions):
            with self.assertLogs(rlm.__name__, level='ERROR') as logs:
                rlm._shutdown()
            self.assertEqual(calls, ['first', 'second', 'third', 'last'])
            self.assertEqual(rlm._sessions, {})
            self.assertEqual(len(logs.records), 2)
            rlm._shutdown()
            self.assertEqual(len(calls), 4)


if __name__ == '__main__':
    unittest.main()
