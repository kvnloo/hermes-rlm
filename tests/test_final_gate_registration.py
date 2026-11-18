"""Registered hooks, not just FinalGate, must keep tasks sharing a session apart."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import rlm
from rlm.final_gate import WITHHELD


class Ctx:
    def __init__(self):
        self.hooks, self.middleware = {}, {}

    def get_config(self, key, default=None):
        return default

    def register_tool(self, *a, **k): pass
    def register_auxiliary_task(self, *a, **k): pass
    def register_command(self, *a, **k): pass
    def on_unload(self, fn): pass
    def register_hook(self, name, fn): self.hooks[name] = fn
    def register_middleware(self, name, fn): self.middleware[name] = fn


class RegisteredHookTests(unittest.TestCase):
    def test_session_end_hook_and_task_scoped_lookup(self):
        ctx = Ctx()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rlm, "_data_dir", lambda: Path(tmp)), \
                mock.patch.object(rlm, "_CTX", None), \
                mock.patch.dict(rlm._sessions, clear=True):
            rlm.register(ctx)
            rlm._sessions[(tmp, "task-a", "shared")] = SimpleNamespace(
                context="evidence", last_validated_final=None, kill=lambda: None)
            hooks = ctx.hooks
            hooks["pre_llm_call"](session_id="shared", task_id="task-a", turn_id="turn-a")
            self.assertEqual(hooks["transform_llm_output"]("x", session_id="shared", turn_id="turn-a"), WITHHELD)
            hooks["pre_llm_call"](session_id="shared", task_id="task-b", turn_id="turn-b")
            self.assertIsNone(hooks["transform_llm_output"]("x", session_id="shared", turn_id="turn-b"))
            hooks["on_session_end"](session_id="shared", turn_id="turn-a")
            self.assertNotIn(("shared", "turn-a"), hooks["on_session_end"].__self__.turns)
            hooks["on_session_reset"](session_id="shared")
            self.assertEqual(rlm._sessions, {})
            self.assertIsNone(hooks["transform_llm_output"]("x", session_id="shared", turn_id="turn-a"))

    def test_session_reset_hook_forgets_loaded_gate(self):
        ctx = Ctx()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rlm, "_data_dir", lambda: Path(tmp)), \
                mock.patch.object(rlm, "_CTX", None), \
                mock.patch.dict(rlm._sessions, clear=True):
            rlm.register(ctx)
            rlm._sessions[(tmp, "task-a", "shared")] = SimpleNamespace(
                context="evidence", last_validated_final=None, kill=lambda: None)
            hooks = ctx.hooks
            hooks["pre_llm_call"](session_id="shared", task_id="task-a", turn_id="turn-a")
            self.assertEqual(hooks["transform_llm_output"]("x", session_id="shared", turn_id="turn-a"), WITHHELD)
            hooks["on_session_end"](session_id="shared", turn_id="turn-a")
            hooks["on_session_reset"](session_id="shared")
            self.assertIsNone(hooks["transform_llm_output"]("x", session_id="shared", turn_id="stray"))
            hooks["pre_llm_call"](session_id="shared", task_id="task-a", turn_id="turn-a2")
            self.assertIsNone(hooks["transform_llm_output"]("x", session_id="shared", turn_id="turn-a2"))


if __name__ == "__main__":
    unittest.main()
