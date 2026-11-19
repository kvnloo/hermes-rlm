"""Pin _load_handler/_exec_handler/_rlm_command: arg defaults, task_id routing, JSON error envelope, /rlm output."""
import json
import unittest
from unittest import mock
import rlm


class Fake:
    def __init__(self, boom=None):
        self.calls, self.boom = [], boom
        self.stats, self.log_path = {'cells': 1}, '/x.jsonl'

    def load(self, paths, task_id):
        self.calls.append(('load', paths, task_id))
        if self.boom:
            raise self.boom
        return {'loaded': True}

    def exec(self, code):
        self.calls.append(('exec', code))
        if self.boom:
            raise self.boom
        return {'stdout': 'ok'}


class HandlerPins(unittest.TestCase):
    def patch(self, fake):
        p = mock.patch.object(rlm, 'session_for', return_value=fake)
        self.sf = p.start()
        self.addCleanup(p.stop)

    def test_session_routing_kwargs_forwarded_to_session_for(self):
        self.patch(Fake())
        rlm._load_handler({}, task_id='T', session_id='S')
        self.sf.assert_called_with(task_id='T', session_id='S')
        rlm._exec_handler({'code': 'x'}, task_id='T2', session_id='S2')
        self.sf.assert_called_with(task_id='T2', session_id='S2')

    def test_load_defaults_paths_and_task_id(self):
        f = Fake()
        self.patch(f)
        self.assertEqual(json.loads(rlm._load_handler({})), {'loaded': True})
        self.assertEqual(rlm._load_handler({'paths': ['a']}, task_id='T'), '{"loaded": true}')
        self.assertEqual(f.calls, [('load', [], 'default'), ('load', ['a'], 'T')])

    def test_exec_coerces_code_to_str_and_defaults_empty(self):
        f = Fake()
        self.patch(f)
        rlm._exec_handler({})
        rlm._exec_handler({'code': 5})
        self.assertEqual(f.calls, [('exec', ''), ('exec', '5')])

    def test_exceptions_become_json_error_not_raise(self):
        self.patch(Fake(boom=ValueError('bad')))
        self.assertEqual(json.loads(rlm._load_handler({'paths': ['a']})), {'error': 'ValueError: bad'})
        self.assertEqual(json.loads(rlm._exec_handler({'code': 'x'})), {'error': 'ValueError: bad'})

    def test_session_for_failure_is_also_caught(self):
        with mock.patch.object(rlm, 'session_for', side_effect=OSError('nolog')):
            self.assertEqual(json.loads(rlm._exec_handler({'code': 'x'})), {'error': 'OSError: nolog'})

    def test_none_args_fall_back_to_defaults(self):
        f = Fake()
        self.patch(f)
        rlm._load_handler({'paths': None}, task_id=None)
        rlm._exec_handler({'code': None})
        self.assertEqual(f.calls, [('load', [], 'default'), ('exec', '')])

    def test_exec_success_returns_json_of_result(self):
        self.patch(Fake())
        self.assertEqual(rlm._exec_handler({'code': 'x'}), '{"stdout": "ok"}')

    def test_any_exception_type_is_enveloped(self):
        class Boom(Exception):
            pass
        self.patch(Fake(boom=Boom('b')))
        self.assertEqual(json.loads(rlm._load_handler({})), {'error': 'Boom: b'})
        self.assertEqual(json.loads(rlm._exec_handler({})), {'error': 'Boom: b'})

    def test_rlm_command_lists_limits_and_sessions(self):
        with mock.patch.dict(rlm._sessions, {}, clear=True):
            self.assertTrue(rlm._rlm_command().endswith('\nno RLM sessions yet'))
            rlm._sessions['k'] = Fake()
            rlm._sessions['j'] = Fake()
            out = rlm._rlm_command('ignored')
        row = "{'cells': 1} log=/x.jsonl"
        self.assertEqual(out, 'RLM limits: ' + json.dumps(rlm.limits()) + f"\nk: {row}\nj: {row}")


if __name__ == "__main__":
    unittest.main()
