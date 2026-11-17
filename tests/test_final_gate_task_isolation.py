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


if __name__ == "__main__":
    unittest.main()
