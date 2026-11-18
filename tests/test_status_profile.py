"""Status must not disclose another Hermes profile's session metadata."""
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import rlm


class StatusProfileTests(unittest.TestCase):
    def setUp(self):
        self.sessions = {
            ('/profiles/alice', 'alice-task', 'alice-session'):
                SimpleNamespace(stats={'subcalls': 7}, log_path='/private/alice/log'),
            ('/profiles/alice', 'other-alice-task', 'alice-session'):
                SimpleNamespace(stats={'subcalls': 3}, log_path='/private/alice/other'),
            ('/profiles/bob', 'bob-task', 'bob-session'):
                SimpleNamespace(stats={'subcalls': 9}, log_path='/private/bob/log'),
        }

    def test_status_includes_only_current_profile(self):
        with patch.object(rlm, '_sessions', self.sessions):
            for profile, own, other in [('alice', 'alice', 'bob'), ('bob', 'bob', 'alice')]:
                with self.subTest(profile=profile), patch.object(
                        rlm, '_data_dir', return_value=Path('/profiles') / profile):
                    output = rlm._rlm_command(session_id=profile + '-session')
                    self.assertIn(own + '-task', output)
                    self.assertIn('/private/' + own, output)
                    self.assertNotIn(other, output)
            with patch.object(rlm, '_data_dir', return_value=Path('/profiles/empty')):
                output = rlm._rlm_command(session_id='alice-session')
                self.assertIn('no RLM sessions yet', output)
                self.assertNotIn('/private/', output)

    def test_status_lists_every_task_of_current_session(self):
        with patch.object(rlm, '_sessions', self.sessions):
            for profile, logs in [('alice', ['/private/alice/log', '/private/alice/other']),
                                  ('bob', ['/private/bob/log'])]:
                with self.subTest(profile=profile), patch.object(
                        rlm, '_data_dir', return_value=Path('/profiles') / profile):
                    rows = rlm._rlm_command(session_id=profile + '-session').splitlines()[1:]
                    self.assertEqual(len(rows), len(logs))
                    for log in logs:
                        self.assertEqual(sum(r.endswith('log=' + log) for r in rows), 1)

    def test_registry_enumeration_holds_lock(self):
        lock = threading.Lock()
        class GuardedDict(dict):
            def items(inner):
                if not lock.locked():
                    raise AssertionError('registry enumerated without lock')
                return super().items()
        with patch.object(rlm, '_sessions', GuardedDict(self.sessions)), \
             patch.object(rlm, '_sessions_lock', lock), \
             patch.object(rlm, '_data_dir', return_value=Path('/profiles/alice')):
            self.assertIn('alice-task', rlm._rlm_command(session_id='alice-session'))


if __name__ == '__main__':
    unittest.main()
