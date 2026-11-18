"""Bad token settings must neither dispatch nor consume attempted-call budget."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class TokenAdmissionTests(unittest.TestCase):
    def test_bad_requested_tokens_are_refused_before_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            for value in (-1, True, False, 1.5, '7', 'bad', [], {}):
                with self.subTest(value=value):
                    subcall = mock.Mock(return_value=('ok', {}))
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    result = session._serve_llm({'id': 9, 'prompts': ['x'], 'max_tokens': value})
                    self.assertIn('positive integer', result.get('error', ''))
                    self.assertEqual(result['id'], 9)
                    subcall.assert_not_called()
                    self.assertEqual(session.stats['subcalls'], 0)
                    # A refused request must not poison a later valid request.
                    valid = session._serve_llm({'id': 10, 'prompts': ['x'], 'max_tokens': 2})
                    self.assertEqual(valid['results'], ['ok'])
                    self.assertEqual(session.stats['subcalls'], 1)

    def test_bad_configured_cap_is_refused_before_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            for value in (0, -1):
                with self.subTest(value=value):
                    subcall = mock.Mock(return_value=('ok', {}))
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    session.lim['subcall_max_tokens'] = value
                    result = session._serve_llm({'id': 1, 'prompts': ['x'], 'max_tokens': 2})
                    self.assertIn('positive integer', result.get('error', ''))
                    subcall.assert_not_called()
                    self.assertEqual(session.stats['subcalls'], 0)
