"""Regression: an RLM load must not gate a different Hermes task's reply.

Hermes may assign a fresh task ID on each turn even when the session ID is reused.
"""

import json
import unittest
from types import SimpleNamespace

from rlm.final_gate import FinalGate, WITHHELD


class FinalGateTaskIsolationTests(unittest.TestCase):
    def setUp(self):
        self.loaded_session = SimpleNamespace(context="evidence", last_validated_final=None)
        self.gate = FinalGate(
            lambda task_id, session_id: self.loaded_session
            if (task_id, session_id) == ("task-a", "shared") else None
        )
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a")
        result = self.gate.execute(
            lambda _: json.dumps({"loaded": True}),
            "rlm_load",
            {"paths": ["evidence.txt"]},
            task_id="task-a",
            session_id="shared",
            turn_id="turn-a",
        )
        self.assertTrue(json.loads(result)["loaded"])

    def test_different_task_in_same_session_does_not_inherit_loaded_gate(self):
        self.assertEqual(
            self.gate.transform("uncited", session_id="shared", turn_id="turn-a"),
            WITHHELD,
        )
        self.gate.begin(session_id="shared", task_id="task-b", turn_id="turn-b")
        self.assertIsNone(
            self.gate.transform("ordinary reply", session_id="shared", turn_id="turn-b")
        )

    def test_load_survives_an_interleaved_other_task(self):
        self.gate.get_session = lambda task_id, session_id: None  # host session already gone
        self.gate.end(session_id="shared", turn_id="turn-a")
        self.gate.begin(session_id="shared", task_id="task-b", turn_id="turn-b")
        self.gate.end(session_id="shared", turn_id="turn-b")
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a2")
        self.assertEqual(
            self.gate.transform("uncited", session_id="shared", turn_id="turn-a2"), WITHHELD)

    def test_live_session_context_without_gate_load_is_remembered(self):
        gate = FinalGate(lambda task_id, session_id: SimpleNamespace(
            context="evidence", last_validated_final=None))
        gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a")
        self.assertEqual(gate.transform("uncited", session_id="shared", turn_id="turn-a"), WITHHELD)
        gate.end(session_id="shared", turn_id="turn-a")
        gate.get_session = lambda task_id, session_id: None
        gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a2")
        self.assertEqual(gate.transform("uncited", session_id="shared", turn_id="turn-a2"), WITHHELD)

    def test_overlapping_previous_task_remains_fail_closed_until_its_end(self):
        self.gate.begin(session_id="shared", task_id="task-b", turn_id="turn-b")
        self.assertEqual(
            self.gate.transform("old uncited reply", session_id="shared", turn_id="turn-a"),
            WITHHELD,
        )
        self.gate.end(session_id="shared", turn_id="turn-a")
        self.assertNotIn(("shared", "turn-a"), self.gate.turns)
        self.assertIsNone(
            self.gate.transform("new ordinary reply", session_id="shared", turn_id="turn-b")
        )

    def test_same_task_new_turn_still_requires_fresh_validated_final(self):
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-next")
        self.assertEqual(
            self.gate.transform("uncited", session_id="shared", turn_id="turn-next"),
            WITHHELD,
        )


    def test_load_continuity_is_remembered_per_task_without_live_session(self):
        # Pins the `loaded` set: the host session is gone, only gate state remains.
        self.loaded_session.context = None
        self.gate.end(session_id="shared", turn_id="turn-a")
        self.assertEqual(
            self.gate.transform("stale", session_id="shared", turn_id="turn-a"), WITHHELD)
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-next")
        self.assertEqual(
            self.gate.transform("uncited", session_id="shared", turn_id="turn-next"), WITHHELD)
        self.gate.begin(session_id="shared", task_id="task-b", turn_id="turn-b")
        self.assertIsNone(self.gate.transform("plain", session_id="shared", turn_id="turn-b"))

    def test_validated_final_released_for_its_turn_only_under_overlap(self):
        final = {"answer": "42", "citations": [{"source": "evidence.txt", "start": 0, "end": 2, "quote": "ev"}]}

        def run_exec(_):
            self.loaded_session.last_validated_final = final
            return json.dumps({"final": "42"})

        self.gate.execute(run_exec, "rlm_exec", {}, task_id="task-a", session_id="shared", turn_id="turn-a")
        self.gate.begin(session_id="shared", task_id="task-b", turn_id="turn-b")
        self.assertEqual(
            self.gate.transform("ignored", session_id="shared", turn_id="turn-a"),
            '42\n\nSources:\n- evidence.txt[0:2]: "ev"',
        )
        self.assertIsNone(self.gate.transform("plain", session_id="shared", turn_id="turn-b"))

    def test_repeated_begin_for_same_turn_keeps_validated_final(self):
        # begin is idempotent for a repeated (session, turn, task): a duplicate delivery must not discard the FINAL.
        final = {"answer": "42", "citations": []}

        def run_exec(_):
            self.loaded_session.last_validated_final = final
            return json.dumps({"final": "42"})

        self.gate.execute(run_exec, "rlm_exec", {}, task_id="task-a", session_id="shared", turn_id="turn-a")
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a")
        self.assertEqual(self.gate.transform("x", session_id="shared", turn_id="turn-a"), "42")

    def test_new_turn_of_same_task_does_not_reuse_previous_validated_final(self):
        final = {"answer": "42", "citations": []}

        def run_exec(_):
            self.loaded_session.last_validated_final = final
            return json.dumps({"final": "42"})

        self.gate.execute(run_exec, "rlm_exec", {}, task_id="task-a", session_id="shared", turn_id="turn-a")
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-next")
        self.assertIsNone(self.loaded_session.last_validated_final)
        self.gate.execute(lambda _: json.dumps({"final": "42"}), "rlm_exec", {},
                          task_id="task-a", session_id="shared", turn_id="turn-next")
        self.assertEqual(self.gate.transform("x", session_id="shared", turn_id="turn-next"), WITHHELD)

    def test_later_tool_call_without_final_revokes_earlier_validated_final(self):
        final = {"answer": "42", "citations": []}

        def run_exec(_):
            self.loaded_session.last_validated_final = final
            return json.dumps({"final": "42"})

        ids = dict(task_id="task-a", session_id="shared", turn_id="turn-a")
        self.gate.execute(run_exec, "rlm_exec", {}, **ids)
        self.gate.execute(lambda _: json.dumps({"out": "x"}), "rlm_exec", {}, **ids)
        self.assertEqual(self.gate.transform("x", session_id="shared", turn_id="turn-a"), WITHHELD)

    def test_overlapping_same_task_turn_revokes_older_turns_validated_final(self):
        final = {"answer": "42", "citations": []}

        def run_exec(_):
            self.loaded_session.last_validated_final = final
            return json.dumps({"final": "42"})

        self.gate.execute(run_exec, "rlm_exec", {},
                          task_id="task-a", session_id="shared", turn_id="turn-a")
        self.gate.begin(session_id="shared", task_id="task-a", turn_id="turn-a2")
        self.assertEqual(self.gate.transform("x", session_id="shared", turn_id="turn-a"), WITHHELD)

    def test_rlm_tool_with_mismatched_identity_is_refused_without_running(self):
        calls = []
        for ids in (("task-b", "turn-a"), ("task-a", "turn-x")):
            result = self.gate.execute(
                lambda _: calls.append(1), "rlm_exec", {},
                task_id=ids[0], session_id="shared", turn_id=ids[1],
            )
            self.assertIn("refused", json.loads(result)["error"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
