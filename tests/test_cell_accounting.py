"""A later serialized cell must not inflate an earlier cell's receipt."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import rlm


class CellAccountingTests(unittest.TestCase):
    def test_each_cell_reports_its_own_completed_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            session = rlm.Session('test', log_dir=Path(root))
            session.proc = mock.Mock()
            session.proc.poll.return_value = None
            first_released, second_released = threading.Event(), threading.Event()
            actual = threading.Lock()
            local = threading.local()
            class InterleavedLock:
                def __enter__(self):
                    actual.acquire()
                def __exit__(self, *args):
                    actual.release()
                    if local.code == 'first':
                        first_released.set()
                        if not second_released.wait(3):
                            raise AssertionError('second cell did not run')
                    else:
                        second_released.set()
            session.lock = InterleavedLock()
            def request(msg, deadline):
                local.code = msg['code']
                session.stats['subcalls'] += 2 if local.code == 'first' else 3
                return {'type': 'done', 'stdout': '', 'error': None}
            with mock.patch.object(session, '_request', side_effect=request), \
                 mock.patch.object(session, 'log') as log, ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(session.exec, 'first')
                self.assertTrue(first_released.wait(3))
                second = pool.submit(session.exec, 'second')
                a, b = first.result(4), second.result(4)
            self.assertEqual(a['subcalls_this_cell'], 2)
            self.assertEqual(b['subcalls_this_cell'], 3)
            self.assertEqual(a['budget']['subcalls_used'], 2)
            self.assertEqual(b['budget']['subcalls_used'], 5)
            events = {call.kwargs['code']: call.kwargs['subcalls'] for call in log.call_args_list}
            self.assertEqual(events, {'first': 2, 'second': 3})
