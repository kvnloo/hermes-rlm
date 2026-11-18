"""Only an llm frame can request host-managed model work."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import rlm


class FrameTypeTests(unittest.TestCase):
    def test_unknown_types_never_dispatch_or_charge(self):
        with tempfile.TemporaryDirectory() as root:
            for kind in ('unknown', 'llm_result', 'load', '', None, 1, True, []):
                with self.subTest(kind=kind):
                    subcall = mock.Mock(return_value=('ok', {}))
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    session.proc = mock.Mock()
                    session.proc.poll.return_value = None
                    frames = [{'type':kind, 'id':1, 'prompts':['unexpected']}, {'type':'done'}]
                    with mock.patch.object(session, '_send'), \
                         mock.patch.object(session, '_read', side_effect=frames):
                        result = session.exec('fixture')
                    self.assertIn('frame type', result['error'] or '')
                    subcall.assert_not_called()
                    self.assertEqual(session.stats['subcalls'], 0)
                    self.assertEqual(session.stats['restarts'], 1)
                    self.assertIsNone(session.proc)

    def test_valid_llm_then_done_still_dispatches(self):
        with tempfile.TemporaryDirectory() as root:
            subcall = mock.Mock(return_value=('answer', {}))
            session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
            frames = [{'type':'llm', 'id':3, 'prompts':['question']}, {'type':'done', 'stdout':'ok'}]
            with mock.patch.object(session, '_send') as send, \
                 mock.patch.object(session, '_read', side_effect=frames):
                self.assertEqual(session._request({'type':'exec'}, time.monotonic()+1)['stdout'], 'ok')
            subcall.assert_called_once()
            self.assertEqual(send.call_args_list[1].args[0],
                             {'type':'llm_result', 'id':3, 'results':['answer']})
            self.assertEqual(session.stats['subcalls'], 1)
