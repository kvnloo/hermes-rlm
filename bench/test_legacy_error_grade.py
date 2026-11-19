"""Execution errors are unscored, not wrong benchmark answers."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_bench as bench


class LegacyErrorGradeTests(unittest.TestCase):
    def test_errors_are_unscored_and_successes_remain_graded(self):
        cases = [
            ('timeout', subprocess.TimeoutExpired('fixture', 3), None),
            ('parse', {'answer': '', 'error': 'bad json'}, None),
            ('empty-diagnostic', {'answer': '', 'error': ''}, None),
            ('partial-answer-error', {'answer': 'ANSWER: 123', 'error': 'failed'}, None),
            ('provider', {'answer': 'HTTP 429 rate-limited'}, None),
            ('correct', {'answer': 'ANSWER: 123'}, True),
            ('wrong', {'answer': 'ANSWER: 999'}, False),
        ]
        for label, response, expected in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = root / 'corpus'
                corpus.mkdir()
                (root / 'corpus.gold.json').write_text(json.dumps({
                    'meta': {'approx_tokens': 10, 'records': 1},
                    'questions': [{'id': 'niah', 'q': 'Find code', 'gold': '123'}]}))
                out = root / 'rows.jsonl'
                kwargs = {'side_effect': response} if isinstance(response, Exception) else {'return_value': dict(response)}
                with patch.object(sys, 'argv', ['run_bench.py', str(corpus), str(out), '--arms', 'nano-rlm', '--timeout', '3']), \
                        patch.object(bench, 'run_nano', **kwargs) as run, \
                        patch.object(bench, 'run_hermes') as hermes, \
                        patch.object(bench, 'score', wraps=bench.score) as score, \
                        contextlib.redirect_stdout(io.StringIO()):
                    if label == 'provider':
                        with self.assertRaisesRegex(SystemExit, 'provider unavailable'):
                            bench.main()
                    else:
                        bench.main()
                    run.assert_called_once()
                    hermes.assert_not_called()
                row = json.loads(out.read_text())
                self.assertIs(row['correct'], expected)
                if label == 'timeout':
                    self.assertEqual(row['error'], 'timeout')
                    self.assertEqual(row['wall_s'], 3)
                elif label == 'provider':
                    self.assertEqual(row['error'], 'provider')
                elif expected is None:
                    self.assertEqual(row['error'], response['error'])
                if expected is None:
                    score.assert_not_called()
                else:
                    score.assert_called_once()
