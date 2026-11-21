"""Synthetic audit receipts only: no live tool calls, models, or archived rows."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from audit_results import audit_row
from protocol_adherence import CAUSES, aggregate_report, classify, evidence_report, main


def call(name, args=None):
    return {"role": "assistant", "tool_calls": [
        {"id": name, "function": {"name": name, "arguments": json.dumps(args or {})}}]}


def receipt(name, **body):
    return {"role": "tool", "tool_call_id": name, "content": json.dumps(body)}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ops = Path(tmp.name)
        self.task = {"id": "t", "gold": "B", "scoring": "choice"}

    def run_audit(self, key, messages, *, child=None, omit_evidence=False):
        run = self.ops / "runs" / key
        logs = run / "rlm-logs"
        logs.mkdir(parents=True)
        if not omit_evidence:
            (run / "hermes-conversation.json").write_text(json.dumps({
                "completed": True, "failed": False, "interrupted": False,
                "turn_exit_reason": "completed", "messages": messages,
            }), encoding="utf-8")
            events = [{"event": "cell", "error": None}]
            if child is not None:
                events.append({"event": "subcall", "status": child})
            (logs / "fixture.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return audit_row({"run_id": key, "task_id": "t", "arm": "released-rlm", "valid": True,
                          "score": 1.0, "accounting": {"requests": []}}, self.task, self.ops)

    def test_missing_final_is_not_unknown_or_uncited_prose(self):
        row = self.run_audit("missing", [{"role": "assistant", "content": "Answer: B"}])
        self.assertEqual(classify(row), dict(missing_validated_final=True, child_error=False,
                                            uncited_answer=False, repl_bypass_other_tool=False))
        missing = self.run_audit("unavailable", [], omit_evidence=True)
        self.assertTrue(all(value is None for value in classify(missing).values()))
        self.assertIn("missing_validated_final", missing["audit_reasons"])

    def test_uncited_final_observed_from_linked_receipt(self):
        for index, body in enumerate((
                {"error": "citation required for an answer over loaded context"},
                {"error": None, "final": "B", "citations": []},
        )):
            row = self.run_audit(f"uncited-{index}", [call("rlm_exec"), receipt("rlm_exec", **body)])
            self.assertEqual(classify(row), dict(missing_validated_final=True, child_error=False,
                                                uncited_answer=True, repl_bypass_other_tool=False))
            self.assertFalse(row["valid"])

    def test_bypass_requires_successful_linked_direct_context_receipt(self):
        for key, messages, expected in (
            ("direct", [call("read_file", {"path": "./context.txt"}),
                        receipt("read_file", content="Synthetic document")], True),
            ("unrelated", [call("read_file", {"path": "notes.txt"}),
                           receipt("read_file", content="Auxiliary note")], False),
            ("failed", [call("read_file", {"path": "context.txt"}),
                        receipt("read_file", success=False, error="Not found")], False),
            ("unlinked", [receipt("read_file", content="Unlinked note")], False),
        ):
            with self.subTest(key=key):
                row = self.run_audit(key, messages)
                self.assertEqual(classify(row)["repl_bypass_other_tool"], expected)
                self.assertTrue(classify(row)["missing_validated_final"])
                self.assertFalse(row["valid"])

    def test_child_error_not_masked_by_cited_final_and_overlap_counted(self):
        cited = [call("rlm_exec"), receipt("rlm_exec", error=None, final="B",
                 citations=[dict(source="context.txt", start=0, end=1, quote="B")])]
        child = self.run_audit("child", cited, child="error")
        self.assertEqual(classify(child), dict(missing_validated_final=False, child_error=True,
                                              uncited_answer=False, repl_bypass_other_tool=False))
        valid = self.run_audit("ok", cited, child="ok")
        self.assertTrue(valid["valid"])
        overlap = self.run_audit("overlap", [
            call("rlm_exec"), receipt("rlm_exec", error="citation required for an answer over loaded context"),
        ], child="error")
        data = evidence_report([child, valid, overlap, dict(child, arm="nano-rlm", run_id="nano")])
        self.assertEqual((data["invalid"], data["invalid_hermes_denominator"], data["invalid_other_arms"]),
                         (3, 2, 1))
        self.assertEqual(data["causes"]["child_error"]["observed"], 2)
        self.assertEqual(data["rows_with_multiple_observations"], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evidence_report([child, child])


class AggregateTests(unittest.TestCase):
    def test_frozen_policy_does_not_invent_historical_causes(self):
        bins = [
            dict(arm="released-rlm", observed=4, scheduled=4, valid=1, invalid=3, collection_valid=2,
                 collection_invalid=2, protocol_adherent=1, protocol_nonadherent=3, protocol_not_applicable=0,
                 protocol_unknown=0),
            dict(arm="base-truncated", observed=2, scheduled=2, valid=2, invalid=0, collection_valid=2,
                 collection_invalid=0, protocol_adherent=0, protocol_nonadherent=0, protocol_not_applicable=2,
                 protocol_unknown=0),
        ]
        data = {"schema_version": "rlm-paper-aggregate.v1", "audit_policy": "paper_strict_v1",
                "primary": bins, "smoke": bins}
        report = aggregate_report(data)
        self.assertEqual((report["observed"], report["strict_invalid"], report["hermes_protocol_adherent"]),
                         (6, 3, 1))
        self.assertEqual((report["hermes_observed"], report["other_arms_observed"]), (4, 2))
        self.assertEqual(report["additional_strict_rejections"], 1)
        self.assertEqual(report["historical_cause_counts"], dict.fromkeys(CAUSES))
        self.assertIsNone(report["historical_top_cause"])
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(json.dumps(data), encoding="utf-8")
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(main(["--summary", str(summary)]), 0)
            self.assertEqual(json.loads(printed.getvalue())["strict_invalid"], 3)
