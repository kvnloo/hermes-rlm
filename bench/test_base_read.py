"""The baseline keeps a bounded decoded head without changing corpus counts."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class BaseReadTests(unittest.TestCase):
    def test_bounded_reads_keep_unicode_newlines_and_original_count(self):
        for text in ('é\r\n🚀tail' + 'x' * 140000, 'é\r\n'):
            with self.subTest(length=len(text)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source.txt'
                source.write_bytes(text.encode())
                task = dict(id='t', family='fixture', size_tokens=1, question='Q',
                            context_path=str(source), context_sha256=matrix.digest(source), task_sha256='fixture')
                args = SimpleNamespace(ops_dir=root, root_model='fixture', sub_model='fixture',
                                       timeout=1, disable_thinking=False, base_context_chars=5,
                                       max_tokens=1, scorer=root / 'unused.py')
                summary = dict(requests=[{'status': 200}], root_prompt_tokens=1,
                               root_completion_tokens=1, sub_prompt_tokens=0,
                               sub_completion_tokens=0, request_attempts=1)
                ledger = SimpleNamespace(summary=lambda _: summary)
                bodies = []
                class Router:
                    errors = []
                    observed = []
                    blocked = False
                    def __init__(self, *a, **kw): pass
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                    def logical_snapshot(self): return []
                    def forward(self, body, *a):
                        bodies.append(body)
                        return 200, b'{"choices":[{"message":{"content":"A"}}]}', {}
                reads = []
                class Reader(io.StringIO):
                    def read(self, size=-1):
                        reads.append(size)
                        if size < 0:
                            raise AssertionError('unbounded decoded corpus read')
                        return super().read(size)
                original_open = Path.open
                def opened(path, *a, **kw):
                    if path.name == 'context.txt' and kw.get('encoding') == 'utf-8':
                        self.assertEqual(kw.get('newline'), '')
                        return Reader(text, newline='')
                    return original_open(path, *a, **kw)
                with patch.object(matrix, 'Router', Router), patch.object(Path, 'open', opened), \
                        patch.object(matrix, 'score_answer', return_value={'score': 1}):
                    row = matrix.run_one(task, 'base-truncated', 0, args, {}, ledger)
                self.assertTrue(row['valid'], row)
                self.assertEqual(bodies[0]['messages'][0]['content'], text[:5] + '\n\n' + matrix.task_prompt(task, 'base-truncated'))
                self.assertEqual(row['truncation'], dict(kind='head_characters', chars=len(text[:5]),
                                                       original_chars=len(text), truncated=len(text)>5))
                self.assertEqual(reads[0], 5)
                self.assertTrue(all(0 < n <= 65536 for n in reads))
