"""A request without a correlation ID must not spend model budget."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm

class FrameIdTests(unittest.TestCase):
    def test_missing_id_refused_before_dispatch_and_charge(self):
        with tempfile.TemporaryDirectory() as root:
            subcall = mock.Mock(return_value=('answer', {}))
            session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
            with self.assertRaisesRegex(RuntimeError, 'id'):
                session._serve_llm({'type': 'llm', 'prompts': ['question']})
            subcall.assert_not_called()
            self.assertEqual(session.stats['subcalls'], 0)

    def test_exec_missing_id_detaches_child_without_provider_work(self):
        with tempfile.TemporaryDirectory() as root:
            subcall = mock.Mock(return_value=('answer', {}))
            session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
            session.proc = mock.Mock()
            session.proc.poll.return_value = None
            with mock.patch.object(session, '_send'), mock.patch.object(
                    session, '_read', return_value={'type': 'llm', 'prompts': ['question']}):
                result = session.exec('trusted fixture')
            self.assertIn('id', result['error'])
            subcall.assert_not_called()
            self.assertEqual(session.stats['subcalls'], 0)
            self.assertEqual(session.stats['restarts'], 1)
            self.assertIsNone(session.proc)

    def test_present_ids_keep_existing_echo_contract(self):
        with tempfile.TemporaryDirectory() as root:
            for request_id in (1, 0, 'request', None, False, ['opaque'], {'opaque': 1}):
                with self.subTest(request_id=request_id):
                    subcall = mock.Mock(return_value=('answer', {}))
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    result = session._serve_llm({'id': request_id, 'prompts': ['question']})
                    self.assertEqual(result, {'type': 'llm_result', 'id': request_id, 'results': ['answer']})
                    subcall.assert_called_once()
                    self.assertEqual(session.stats['subcalls'], 1)

if __name__ == '__main__': unittest.main()
