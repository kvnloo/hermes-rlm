"""Unicode text separators inside JSON strings are not JSONL record breaks."""
import json
from pathlib import Path
import tempfile
import unittest

import paper_matrix as matrix


class JsonlUnicodeTests(unittest.TestCase):
    def test_resume_unicode_answers_and_physical_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.jsonl'
            for separator in ('\u0085', '\u2028', '\u2029'):
                for ending in ('\n', '\r', '\r\n'):
                    with self.subTest(separator=repr(separator), ending=repr(ending)):
                        rows = [dict(row_key='one', answer='first' + separator + 'second'),
                                dict(row_key='two', answer='done')]
                        data = ending.join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
                        path.write_bytes(data)
                        self.assertEqual(matrix.completed(path), {'one', 'two'})
                        self.assertEqual(path.read_bytes(), data)

    def test_manifest_question_unicode_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'context.txt').write_text('context')
            manifest = root / 'tasks.jsonl'
            question = 'First\u0085second\u2028third\u2029last'
            task = dict(id='task', family='S-NIAH', size_tokens=1, question=question,
                        gold='A', scoring='choice', context_path='context.txt')
            manifest.write_text(json.dumps(task, ensure_ascii=False) + '\n')
            self.assertEqual(matrix.load_tasks(manifest)[0]['question'], question)

    def test_actual_corrupt_record_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.jsonl'
            for text in ('{"row_key": "one"}\n{broken', '{"row_key": "one"}\n\n'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaisesRegex(ValueError, 'corrupt output at line 2'):
                        matrix.completed(path)
