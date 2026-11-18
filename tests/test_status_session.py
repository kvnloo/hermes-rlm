"""The host session, not a shared profile, bounds command diagnostics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import rlm

class StatusSessionTests(unittest.TestCase):
    def setUp(self):
        sessions = {}
        for profile, task, session, marker in [
            ('/p', 'task-a', 'own', 'own-first'),
            ('/p', 'task-b', 'own', 'own-second'),
            ('/p', 'task-a', 'other', 'other-private'),
            ('/foreign', 'task-a', 'own', 'foreign-private'),
            ('/p', 'own', '', 'task-only-private')]:
            sessions[(profile, task, session)] = SimpleNamespace(stats={}, log_path=marker)
        for p in (patch.object(rlm, '_sessions', sessions),
                  patch.object(rlm, '_data_dir', return_value=Path('/p'))):
            p.start(); self.addCleanup(p.stop)

    def test_host_session_filters_metadata_without_truncating_its_tasks(self):
        out = rlm._rlm_command(session_id='own')
        self.assertIn('own-first', out)
        self.assertIn('own-second', out)
        for forbidden in ('other-private', 'foreign-private', 'task-only-private'):
            self.assertNotIn(forbidden, out)
        self.assertNotIn('private', rlm._rlm_command(session_id='unknown'))

    def test_missing_session_never_falls_back_to_task_or_session_key(self):
        for kwargs in ({}, {'session_id': ''}, {'session_id': None},
                       {'task_id': 'own'}, {'session_key': 'own'},
                       {'session_id': False}, {'session_id': 1}):
            with self.subTest(kwargs=kwargs):
                out = rlm._rlm_command(**kwargs)
                self.assertIn('RLM limits:', out)
                self.assertNotIn('log=', out)
                self.assertNotIn('private', out)
                self.assertIn('session_id', out)

if __name__ == '__main__': unittest.main()
