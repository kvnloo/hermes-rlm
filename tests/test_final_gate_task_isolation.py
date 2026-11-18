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
        # Pins the `latest` pointer: the host session is gone, only gate state remains.
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
