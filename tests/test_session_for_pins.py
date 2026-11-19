"""Pin session_for: key precedence, sanitising, 80-char cut, and one Session per key."""
import os
import tempfile
import unittest
from unittest import mock
import rlm


class SessionForPins(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patch in (mock.patch.dict(rlm._sessions, clear=True),
                      mock.patch.dict(os.environ, {'RLM_LOG_DIR': tmp.name})):
            patch.start()
            self.addCleanup(patch.stop)

    def test_task_id_beats_session_id_beats_default(self):
        self.assertEqual(rlm.session_for(task_id='t', session_id='s').key, 't')
        self.assertEqual(rlm.session_for(session_id='s').key, 's')
        self.assertEqual(rlm.session_for().key, 'default')
        self.assertEqual(rlm.session_for(task_id='', session_id='').key, 'default')

    def test_key_keeps_alnum_dash_underscore_and_drops_the_rest(self):
        self.assertEqual(rlm.session_for(task_id='aZ09-_/../x y.z').key, 'aZ09-_xyz')

    def test_key_is_cut_at_80_after_sanitising(self):
        self.assertEqual(rlm.session_for(task_id='a' * 100).key, 'a' * 80)
        self.assertEqual(rlm.session_for(task_id='/' * 5 + 'b' * 100).key, 'b' * 80)

    def test_same_key_reuses_the_session_and_other_keys_get_their_own(self):
        a = rlm.session_for(task_id='k')
        self.assertIs(rlm.session_for(task_id='k'), a)
        self.assertIs(rlm.session_for(session_id='k'), a)
        self.assertIsNot(rlm.session_for(task_id='j'), a)
        self.assertEqual(set(rlm._sessions), {'k', 'j'})

    def test_extra_kwargs_are_ignored(self):
        self.assertEqual(rlm.session_for(task_id='k', tool_call_id='x').key, 'k')


if __name__ == "__main__":
    unittest.main()
