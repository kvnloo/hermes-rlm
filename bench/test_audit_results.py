"""Focused offline post-audit regression tests.

reuse-checked: test_paper_matrix.py fixtures and paper_tasks.score_task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_results as audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ops = Path(self.temp.name)
        self.run = self.ops / "runs" / "run-1"
        (self.run / "rlm-logs").mkdir(parents=True)
        self.task = {"id": "t", "gold": "B", "scoring": "choice"}
        self.row = {"run_id": "run-1", "task_id": "t", "arm": "released-rlm",
                    "valid": True, "score": 0.0, "correct": False, "status": "valid",
                    "answer": "ANSWER: A", "root_input_tokens": 123,
                    "protocol": {"max_turns": 10},
                    "accounting": {"requests": [{"role": "root", "status": 200}]}}
        self.events([{"event": "cell", "final": "B", "error": None}])
        self.transcript(self.final_messages())

    def events(self, events):
        (self.run / "rlm-logs" / "hashed.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events))

    def final_messages(self, answer="B", call_id="c", name="rlm_exec"):
        return [{"role": "assistant", "tool_calls": [
                    {"id": call_id, "function": {"name": name, "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(
                    {"error": None, "final": answer, "citations": [
                        {"source": "context.txt", "start": 0, "end": 1, "quote": "x"}]})}]

    def transcript(self, messages, **extra):
        data = {"completed": True, "failed": False, "interrupted": False,
                "turn_exit_reason": "text_response(finish_reason=stop)", "messages": messages}
        data.update(extra)
        (self.run / "hermes-conversation.json").write_text(json.dumps(data))

    def audited(self):
        return audit.audit_row(self.row, self.task, self.ops)

    def test_missing_final_despite_collection_valid(self):
        self.transcript([{"role": "assistant", "content": "FINAL('B')"}])
        result = self.audited()
        self.assertFalse(result["valid"])
        self.assertIsNone(result["score"])
        self.assertIn("missing_validated_final", result["audit_reasons"])
        self.assertTrue(result["collection_valid"])

    def test_valid_final_rescores_not_root_prose_and_keeps_snapshot(self):
        result = self.audited()
        self.assertTrue(result["valid"])
        self.assertEqual((result["answer"], result["score"]), ("B", 1.0))
        self.assertEqual((result["collection_answer"], result["collection_score"]), ("ANSWER: A", 0.0))
        self.assertEqual(result["root_input_tokens"], 123)
        self.assertEqual(result["accounting"], self.row["accounting"])
        self.assertNotIn("collection_valid", self.row)

    def test_failed_child_then_root_200(self):
        self.row["accounting"]["requests"].insert(0, {"role": "sub", "status": 500})
        self.events([{"event": "subcall", "status": "error"}])
        result = self.audited()
        self.assertFalse(result["valid"])
        self.assertIn("child_error", result["audit_reasons"])

    def test_recovered_physical_400_and_429_plugin_ok(self):
        self.row["accounting"]["requests"][:0] = [
            {"role": "sub", "status": 400}, {"role": "sub", "status": 429},
            {"role": "sub", "status": 200}]
        self.events([{"event": "subcall", "status": "ok"}])
        self.assertTrue(self.audited()["valid"])

    def test_missing_transcript_and_budget_exit_fail_closed(self):
        (self.run / "hermes-conversation.json").unlink()
        self.assertIn("missing_validated_final", self.audited()["audit_reasons"])
        self.transcript(self.final_messages(), completed=False,
                        turn_exit_reason="max_iterations_reached(10/10)")
        result = self.audited()
        self.assertFalse(result["valid"])
        self.assertIn("budget_exit", result["audit_reasons"])

    def test_last_successful_final_not_failed_or_nested_or_wrong_tool(self):
        messages = self.final_messages("A") + self.final_messages("B", "c2")
        failed = self.final_messages("A", "c3")
        failed[-1]["content"] = json.dumps({"final": "A", "citations": [{}], "error": "bad"})
        nested = self.final_messages("A", "c4")
        nested[-1]["content"] = json.dumps({"stdout": json.dumps({"final": "A", "citations": [{}]})})
        messages += failed + nested + self.final_messages("A", "c5", "other_tool")
        self.transcript(messages)
        self.assertEqual(self.audited()["answer"], "B")

    def test_empty_citations_unmapped_final_and_malformed_transcript(self):
        messages = self.final_messages()
        body = json.loads(messages[-1]["content"])
        body["citations"] = []
        messages[-1]["content"] = json.dumps(body)
        self.transcript(messages)
        self.assertIn("missing_validated_final", self.audited()["audit_reasons"])
        self.transcript(self.final_messages()[1:])
        self.assertIn("missing_validated_final", self.audited()["audit_reasons"])
        (self.run / "hermes-conversation.json").write_text("not JSON")
        self.assertFalse(self.audited()["valid"])

    def test_failed_or_interrupted_even_with_valid_final(self):
        for flag in ("failed", "interrupted"):
            with self.subTest(flag=flag):
                self.transcript(self.final_messages(), **{flag: True})
                self.assertFalse(self.audited()["valid"])

    def test_collection_invalid_never_promoted(self):
        self.row["valid"] = False
        self.assertFalse(self.audited()["valid"])

    def test_cancelled_and_malformed_or_missing_plugin_logs(self):
        self.events([{"event": "subcall", "status": "cancelled"}])
        self.assertIn("child_cancelled", self.audited()["audit_reasons"])
        log = self.run / "rlm-logs" / "hashed.jsonl"
        log.write_text("{bad\n")
        self.assertFalse(self.audited()["valid"])
        log.unlink()
        self.assertFalse(self.audited()["valid"])

    def test_no_extra_protocol_requirements_or_context_reads(self):
        self.task["context_path"] = "not-present-huge-context.txt"
        self.row["protocol"].update(arbitrary_setting="ignored", max_turns=1)
        self.assertTrue(self.audited()["valid"])
        self.row["arm"] = "base-truncated"
        result = self.audited()
        for key in ("valid", "score", "answer", "accounting"):
            self.assertEqual(result[key], self.row[key])

    def test_nano_has_no_host_final_requirement_but_uncertain_child_fails(self):
        self.row["arm"] = "nano-rlm"
        (self.run / "hermes-conversation.json").unlink()
        self.assertTrue(self.audited()["valid"])
        self.row["accounting"]["requests"].insert(0, {"role": "sub", "status": 500})
        result = self.audited()
        self.assertFalse(result["valid"])
        self.assertEqual(result["audit_adherence"], "unknown")

    def test_cli_freezes_same_basename_and_source_hash(self):
        raw = self.ops / "matrix-v2-semantic.jsonl"
        self.task["context_path"] = str(self.ops / "not-read-context.txt")
        self.row["context_sha256"] = "frozen-context-hash"
        self.row["task_sha256"] = audit.json_sha(dict(
            self.task, context_sha256=self.row["context_sha256"]))
        content = json.dumps(self.row) + "\n"
        raw.write_text(content)
        tasks = self.ops / "tasks.jsonl"
        tasks.write_text(json.dumps(self.task) + "\n")
        out = self.ops / "audited"
        self.assertEqual(audit.main(["--raw", str(raw), "--ops-dir", str(self.ops),
                                     "--tasks", str(tasks), "--out-dir", str(out)]), 0)
        result = json.loads((out / raw.name).read_text())
        self.assertEqual(result["source_sha"], hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(raw.read_text(), content)
        self.assertEqual(result["audit_version"], audit.AUDIT_VERSION)
        with self.assertRaises(ValueError):
            audit.main(["--raw", str(raw), "--ops-dir", str(self.ops),
                        "--tasks", str(tasks), "--out-dir", str(self.ops)])


if __name__ == "__main__":
    unittest.main()
