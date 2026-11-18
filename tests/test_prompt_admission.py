"""Only a JSON array of text prompts may cross the provider boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class PromptAdmissionTests(unittest.TestCase):
    def test_malformed_batch_refused_without_dispatch_or_charge(self):
        with tempfile.TemporaryDirectory() as root:
            cases = [{}, {'prompts': None}, {'prompts': 'abc'},
                     {'prompts': {'secret': 'value'}}, {'prompts': 7},
                     {'prompts': ['ok', 1]}, {'prompts': [False]},
                     {'prompts': [['nested']]}]
            for fields in cases:
                with self.subTest(fields=fields):
                    subcall = mock.Mock(return_value=('ok', {}))
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    result = session._serve_llm({'id': 9, **fields})
                    self.assertIn('list of strings', result.get('error', ''))
                    self.assertEqual(result['id'], 9)
                    subcall.assert_not_called()
                    self.assertEqual(session.stats['subcalls'], 0)

    def test_valid_empty_batch_and_text_order_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            subcall = mock.Mock(side_effect=lambda text, tokens: (text, {}))
            session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
            self.assertEqual(session._serve_llm({'id': 1, 'prompts': []})['results'], [])
            subcall.assert_not_called()
            self.assertEqual(session.stats['subcalls'], 0)
            prompts = ['first', '', 'last']
            self.assertEqual(session._serve_llm({'id': 2, 'prompts': prompts})['results'], prompts)
            self.assertEqual(session.stats['subcalls'], 3)
