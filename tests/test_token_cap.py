"""Per-call overrides cannot exceed the configured subcall token budget."""
from pathlib import Path
import tempfile
import unittest
import rlm


class TokenCapTests(unittest.TestCase):
    def test_override_respects_configured_and_absolute_caps(self):
        with tempfile.TemporaryDirectory() as root:
            for configured, requested, expected in (
                    (7, 99, 7), (7, 3, 3), (7, None, 7),
                    (7, 0, 7), (40000, 50000, 32000)):
                with self.subTest(configured=configured, requested=requested):
                    seen = []
                    def subcall(prompt, tokens):
                        seen.append(tokens)
                        return 'ok', {}
                    session = rlm.Session('test', subcall=subcall, log_dir=Path(root))
                    session.lim['subcall_max_tokens'] = configured
                    result = session._serve_llm({'id': 1, 'prompts': ['first', 'second'],
                                                 'max_tokens': requested})
                    self.assertEqual(result['results'], ['ok', 'ok'])
                    self.assertEqual(seen, [expected, expected])
                    self.assertEqual(session.stats['subcalls'], 2)
