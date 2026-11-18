"""Missing host identity must not create a shared execution namespace."""
import json
import unittest
from unittest import mock
import rlm


class IdentityAdmissionTests(unittest.TestCase):
    def test_handlers_refuse_missing_identity_before_state_or_filesystem(self):
        with mock.patch.object(rlm, '_sessions', {}), \
             mock.patch.object(rlm, '_data_dir', side_effect=AssertionError('filesystem lookup')) as data, \
             mock.patch.object(rlm, 'Session') as session:
            for ids in ({}, {'task_id': '', 'session_id': ''},
                        {'task_id': None, 'session_id': None}):
                for handler, args in ((rlm._load_handler, {'paths': ['anything']}),
                                      (rlm._exec_handler, {'code': 'anything'})):
                    with self.subTest(ids=ids, handler=handler.__name__):
                        result = json.loads(handler(args, **ids))
                        self.assertIn('task_id or session_id', result['error'])
            data.assert_not_called()
            session.assert_not_called()
            self.assertEqual(rlm._sessions, {})


if __name__ == '__main__':
    unittest.main()
