"""Distinct host identities must never share a REPL or trajectory file."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import rlm


class SessionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(rlm, '_sessions', {}),
                      mock.patch.object(rlm, '_data_dir', return_value=self.root / 'profile'),
                      mock.patch.dict(rlm.os.environ, {'RLM_LOG_DIR': str(self.root / 'logs')})):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(rlm._shutdown)

    def test_lossy_names_and_task_session_precedence_do_not_alias(self):
        pairs = [('tenant/a', 's'), ('tenanta', 's'), ('tenant:a', 's'),
                 ('x' * 80 + 'a', 's'), ('x' * 80 + 'b', 's'),
                 ('shared-task', 'one'), ('shared-task', 'two'),
                 ('same', ''), ('', 'same'), ('default', ''),
                 ('surrogate\ud800', 's'), ('surrogate\ud801', 's')]
        sessions = [rlm.session_for(task, session) for task, session in pairs]
        self.assertEqual(len({id(s) for s in sessions}), len(pairs))
        self.assertEqual(len({s.log_path for s in sessions}), len(pairs))
        for (task, session), s in zip(pairs, sessions):
            self.assertIs(s, rlm.session_for(task, session))
            self.assertEqual(s.log_path.parent, self.root / 'logs')
            self.assertRegex(s.log_path.name, r'^[0-9a-f]{64}\.jsonl$')
        sessions[0].context = 'private corpus'
        self.assertIsNone(sessions[1].context)

    def test_profile_scope_is_part_of_identity_even_with_shared_log_dir(self):
        first = rlm.session_for('task', 'session')
        with mock.patch.object(rlm, '_data_dir', return_value=self.root / 'other-profile'):
            second = rlm.session_for('task', 'session')
        self.assertIsNot(first, second)
        self.assertNotEqual(first.log_path, second.log_path)
        first.log('synthetic', marker='first')
        second.log('synthetic', marker='second')
        self.assertNotIn('second', first.log_path.read_text())
        self.assertNotIn('first', second.log_path.read_text())

    def test_legacy_name_logs_are_left_untouched(self):
        # README: existing legacy logs are not merged or migrated.
        logs = self.root / 'logs'
        logs.mkdir()
        legacy = logs / 'task.jsonl'
        legacy.write_bytes(b'{"legacy": true}\n')
        session = rlm.session_for('task', 'session')
        session.log('synthetic', marker='new')
        self.assertEqual(legacy.read_bytes(), b'{"legacy": true}\n')
        self.assertNotIn('legacy', session.log_path.read_text())
        self.assertEqual(sorted(p.name for p in logs.iterdir()),
                         sorted(['task.jsonl', session.log_path.name]))


if __name__ == '__main__':
    unittest.main()
