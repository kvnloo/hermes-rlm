"""Unavailable date evaluation is not a wrong model answer."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import paper_matrix as matrix
import paper_tasks


class DateDependencyTests(unittest.TestCase):
    task = {'gold': '2023-01-06T00:00:00',
            'scoring': {'type': 'oolong', 'answer_type': 'ANSWER_TYPE.DATE'}}

    def test_missing_parser_refuses_scoring_but_keeps_direct_match(self):
        with patch.dict(sys.modules, {'dateutil.parser': None}):
            with self.assertRaisesRegex(ValueError, 'python-dateutil'):
                paper_tasks.score_task(self.task, 'January 6, 2023')
            self.assertEqual(paper_tasks.score_task(
                {**self.task, 'gold': '2023-01-06'}, '2023-01-06'), 1)
            self.assertEqual(paper_tasks.score_task({'scoring': 'choice', 'gold': 'B'}, 'B'), 1)

    @unittest.skipUnless(importlib.util.find_spec('dateutil'), 'python-dateutil unavailable')
    def test_installed_parser_keeps_correct_wrong_and_malformed_grades(self):
        for answer, expected in [('January 6, 2023', 1), ('January 7, 2023', 0),
                                 ('January 2023', 0), ('not a date', 0)]:
            with self.subTest(answer=answer):
                self.assertEqual(paper_tasks.score_task(self.task, answer), expected)

    def test_matrix_records_missing_parser_as_invalid_not_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.txt'
            source.write_text('The date is January 6, 2023.')
            task = dict(self.task, id='date', family='OOLONG', size_tokens=8,
                        question='What date?', context_path=str(source),
                        context_sha256=matrix.digest(source), task_sha256='fixture')
            args = SimpleNamespace(ops_dir=root, root_model='fixture', sub_model='fixture',
                timeout=1, disable_thinking=False, base_context_chars=100, max_tokens=32,
                scorer=Path(paper_tasks.__file__))
            summary = dict(requests=[{'status': 200}], root_prompt_tokens=1,
                root_completion_tokens=1, sub_prompt_tokens=0, sub_completion_tokens=0,
                request_attempts=1)
            ledger = SimpleNamespace(summary=lambda _: summary)
            class Router:
                errors = []
                observed = []
                blocked = False
                def __init__(self, *a, **kw): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def logical_snapshot(self): return []
                def forward(self, *a):
                    return 200, json.dumps({'choices': [{'message': {
                        'content': 'January 6, 2023'}}]}).encode(), {}
            with patch.object(matrix, 'Router', Router), \
                    patch.dict(sys.modules, {'dateutil.parser': None}):
                row = matrix.run_one(task, 'base-truncated', 0, args, {}, ledger)
            self.assertFalse(row['valid'], row)
            self.assertEqual(row['status'], 'invalid')
            self.assertIsNone(row['score'])
            self.assertIsNone(row['correct'])
            self.assertEqual(row['error'], 'DateParserUnavailable')
            self.assertEqual(row['accounting'], summary)
