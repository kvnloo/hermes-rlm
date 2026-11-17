"""Bind validated RLM answers to native tool and finalizer turn identities.

reuse-checked: Hermes pre_llm_call payload, tool_execution middleware identity,
and turn_finalizer.apply_llm_output_transform. No profile-wide stream hold.
"""
import copy
import json
import threading

WITHHELD = ('RLM answer withheld: no successful host-validated FINAL or FINAL_VAR '
            'was recorded for this turn. No citation-validated answer is available.')


class FinalGate:
    def __init__(self, get_session):
        self.get_session = get_session
        self.turns = {}  # (session_id, turn_id) -> in-flight turn state
        self.latest = {}  # session_id -> last state, for same-task load continuity
        self.lock = threading.Lock()

    def begin(self, session_id='', task_id='', turn_id='', **_):
        if not session_id:
            return
        key = (session_id, turn_id)
        with self.lock:
            previous = self.latest.get(session_id)
            existing = self.turns.get(key)
            if existing and existing['task'] == task_id:
                return
            # Session IDs outlive task IDs; a different task never inherits a load.
            state = {'task': task_id, 'turn': turn_id,
                     'loaded': bool(previous and previous['task'] == task_id
                                    and previous['loaded']), 'final': None}
            self.turns[key] = state
            self.latest[session_id] = state
        session = self.get_session(task_id, session_id)
        if session is not None:
            session.last_validated_final = None

    def reset(self, session_id='', **_):
        with self.lock:
            self.latest.pop(session_id, None)
            for key in [key for key in self.turns if key[0] == session_id]:
                self.turns.pop(key, None)

    def end(self, session_id='', turn_id='', **_):
        # The host fires on_session_end after the turn's output transform.
        with self.lock:
            self.turns.pop((session_id, turn_id), None)

    def execute(self, next_call, tool_name, args, task_id='', session_id='', turn_id='', **_):
        if tool_name not in ('rlm_load', 'rlm_exec'):
            return next_call(args)
        with self.lock:
            state = self.turns.get((session_id, turn_id))
            if (not session_id or not task_id or not turn_id or state is None
                    or (state['task'], state['turn']) != (task_id, turn_id)):
                return json.dumps({'error': 'RLM tool refused: missing or mismatched host turn identity'})
            state['final'] = None
        result = next_call(args)
        parsed = json.loads(result) if isinstance(result, str) else result
        session = self.get_session(task_id, session_id)
        with self.lock:
            if self.turns.get((session_id, turn_id)) is not state:
                return result
            if isinstance(parsed, dict) and not parsed.get('error'):
                if tool_name == 'rlm_load' and parsed.get('loaded'):
                    state['loaded'] = True
                if (tool_name == 'rlm_exec' and parsed.get('final') is not None
                        and session is not None and session.last_validated_final is not None):
                    state['validated'] = session.last_validated_final
                    state['final'] = copy.deepcopy(state['validated'])
        return result

    def transform(self, response_text, session_id='', turn_id='', **_):
        with self.lock:
            state = self.turns.get((session_id, turn_id))
            if not state:
                latest = self.latest.get(session_id)
                return WITHHELD if latest and latest['loaded'] else None
            session = self.get_session(state['task'], session_id)
            if not state['loaded'] and (session is None or session.context is None):
                return None
            state['loaded'] = True
            if (not turn_id or state['turn'] != turn_id or state['final'] is None
                    or session is None or session.last_validated_final is not state.get('validated')):
                return WITHHELD
            final = state['final']
            citations = '\n'.join(
                f'- {span["source"]}[{span["start"]}:{span["end"]}]: '
                f'{json.dumps(span["quote"], ensure_ascii=False)}' for span in final['citations'])
            return (final['answer'] + ('\n\nSources:\n' + citations if citations else '')
                    or 'RLM answer withheld: the validated FINAL answer was empty.')
