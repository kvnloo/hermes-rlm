"""Budget limits are inclusive: exactly-at-limit requests run, one over is refused."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class AdmissionBoundaryTests(unittest.TestCase):
    def _session(self, root, **lim):
        subcall = mock.Mock(return_value=('ok', {}))
        session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
        session.lim.update(lim)
        return session, subcall

    def test_subcall_budget_is_inclusive(self):
        with tempfile.TemporaryDirectory() as root:
            session, subcall = self._session(root, max_subcalls=2)
            self.assertEqual(session._serve_llm({'id': 1, 'prompts': ['a', 'b']})['results'], ['ok', 'ok'])
            over = session._serve_llm({'id': 2, 'prompts': ['c']})
            self.assertIn('budget exhausted', over['error'])
            self.assertEqual(subcall.call_count, 2)

    def test_prompt_size_limit_is_inclusive(self):
        with tempfile.TemporaryDirectory() as root:
            session, subcall = self._session(root, max_prompt_chars=5)
            self.assertEqual(session._serve_llm({'id': 1, 'prompts': ['x' * 5]})['results'], ['ok'])
            over = session._serve_llm({'id': 2, 'prompts': ['x' * 6]})
            self.assertIn('exceeds RLM_MAX_PROMPT_CHARS', over['error'])
            self.assertEqual(subcall.call_count, 1)

    def test_one_token_is_a_valid_request_and_cap(self):
        with tempfile.TemporaryDirectory() as root:
            session, subcall = self._session(root, subcall_max_tokens=1)
            self.assertEqual(session._serve_llm({'id': 1, 'prompts': ['x'], 'max_tokens': 1})['results'], ['ok'])
            subcall.assert_called_once_with('x', 1)


if __name__ == '__main__':
    unittest.main()
