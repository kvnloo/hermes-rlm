"""HRS-SESSION-RESUME: REPL/corpus state is scoped to (task_id, session_id); a task_id change starts empty."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rlm


class TaskIdChangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / 'doc.txt').write_text('needle-in-corpus')
        self.doc = str(root / 'doc.txt')
        for p in (mock.patch.object(rlm, '_sessions', {}),
                  mock.patch.object(rlm, '_data_dir', return_value=root / 'profile'),
                  mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': str(root / 'logs')})):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(rlm._shutdown)

    def run_tool(self, handler, args, task):
        return json.loads(handler(args, task_id=task, session_id='stable-session'))

    def test_new_task_id_sees_no_corpus_or_variables_and_old_state_returns_with_old_id(self):
        load = self.run_tool(rlm._load_handler, {'paths': [self.doc]}, 'task-A')
        self.assertTrue(load['loaded'])
        a = self.run_tool(rlm._exec_handler, {'code': 'x = 41\nlen(context)'}, 'task-A')
        self.assertIsNone(a['error'], a)
        b = self.run_tool(rlm._exec_handler, {'code': 'print(x)'}, 'task-B')
        self.assertIn('NameError', b['error'] or '')
        b = self.run_tool(rlm._exec_handler, {'code': 'print(context)'}, 'task-B')
        self.assertNotIn('needle', b['stdout'])
        back = self.run_tool(rlm._exec_handler, {'code': 'print(x, context)'}, 'task-A')
        self.assertIn('41 needle-in-corpus', back['stdout'])


if __name__ == '__main__':
    unittest.main()
