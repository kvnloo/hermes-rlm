"""Read-only, evaluator-side adjudication of frozen paper matrix rows.

reuse-checked: paper_matrix.py extraction, paper_tasks.score_task, Session.exec,
hermes_client.py transcript format, paper_proxy.Ledger and reviewer P1 evidence.
No model calls, context loading, token backfills or collection-file writes.
Outputs contain private evidence and must not be published as public aggregates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from paper_tasks import score_task

AUDIT_VERSION = "paper-strict-v1"
HERMES_ARMS = {"rlm-no-subcalls", "released-rlm", "fork-rlm"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def read_evidence(path, hashes, label):
    data = path.read_bytes()
    hashes[label] = sha(data)
    return data


def validated_final(transcript):
    """Trust only top-level Session results linked to preceding rlm_exec calls."""
    messages = transcript.get("messages")
    if not isinstance(messages, list):
        raise ValueError("invalid messages")
    calls, consumed = {}, set()
    selected, count = None, 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid message")
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if not isinstance(call_id, str) or call_id in calls:
                    raise ValueError("ambiguous tool call")
                calls[call_id] = call.get("function", {}).get("name")
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if calls.get(call_id) != "rlm_exec":
            continue
        if call_id in consumed:
            raise ValueError("duplicate tool result")
        consumed.add(call_id)
        if any(message.get(k) not in (None, "rlm_exec") for k in ("name", "tool_name")):
            raise ValueError("tool name mismatch")
        result = json.loads(message["content"])
        if not isinstance(result, dict):
            raise ValueError("invalid tool result")
        answer, citations = result.get("final"), result.get("citations")
        if ("error" not in result or result["error"] or not isinstance(answer, str)
                or not answer.strip() or not isinstance(citations, list) or not citations):
            continue
        # Session already checked source membership, bounds and quote equality.
        if not all(isinstance(c, dict) and isinstance(c.get("source"), str)
                   and type(c.get("start")) is int and type(c.get("end")) is int
                   and 0 <= c["start"] < c["end"] and isinstance(c.get("quote"), str)
                   for c in citations):
            continue
        selected = {"answer": answer, "citations": citations, "tool_call_id": call_id}
        count += 1
    return selected, count


def audit_row(row, task, ops_dir):
    """Return a new private row; never promote an invalid collection row."""
    result = deepcopy(row)
    for key in ("valid", "score", "answer", "correct", "status", "grading", "error"):
        if key in row:
            result["collection_" + key] = deepcopy(row[key])
    result.update(audit_version=AUDIT_VERSION, audit_reasons=[], audit_evidence_sha256={},
                  audit_accounting_policy="collection_snapshot_unchanged")
    reasons = result["audit_reasons"]
    hashes = result["audit_evidence_sha256"]
    arm = row.get("arm")
    if arm == "base-truncated":
        result["audit_adherence"] = "not_applicable_base"
        return result
    if arm not in HERMES_ARMS | {"nano-rlm"}:
        reasons.append("unknown_arm")
        result["audit_adherence"] = "unknown"
    if row.get("valid") is not True:
        reasons.append("collection_invalid")

    if arm == "nano-rlm":
        result["audit_adherence"] = "reference_deviation_no_host_final"
        requests = row.get("accounting", {}).get("requests")
        if not isinstance(requests, list):
            reasons.append("missing_child_accounting")
        elif any(r.get("role") == "sub" and r.get("status") != 200 for r in requests):
            # Physical attempts have no logical request IDs or native final outcomes.
            reasons.append("child_recovery_unknown")
        if reasons:
            result["audit_adherence"] = "unknown"
    elif arm in HERMES_ARMS:
        selected = None
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in ("", ".", ".."):
            reasons.append("invalid_run_id")
            run_dir = None
        else:
            runs = Path(ops_dir).resolve() / "runs"
            run_dir = (runs / run_id).resolve()
            if not run_dir.is_relative_to(runs):
                reasons.append("invalid_run_id")
                run_dir = None
        if run_dir is not None:
            try:
                transcript = json.loads(read_evidence(run_dir / "hermes-conversation.json", hashes, "transcript"))
                if not isinstance(transcript, dict):
                    raise ValueError("invalid transcript")
                result["audit_completion"] = {k: transcript.get(k) for k in
                    ("completed", "failed", "interrupted", "turn_exit_reason")}
                exit_reason = str(transcript.get("turn_exit_reason") or "")
                if "max_iterations" in exit_reason:
                    reasons.append("budget_exit")
                if transcript.get("completed") is not True:
                    reasons.append("conversation_incomplete")
                if transcript.get("failed") or transcript.get("error"):
                    reasons.append("conversation_failed")
                if transcript.get("interrupted"):
                    reasons.append("conversation_interrupted")
                selected, count = validated_final(transcript)
                result["audit_validated_final_count"] = count
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                reasons.append("missing_or_malformed_transcript")
            try:
                logs = sorted((run_dir / "rlm-logs").glob("*.jsonl"))
                if not logs:
                    raise ValueError("missing logs")
                statuses = Counter()
                event_count = 0
                for log in logs:
                    data = read_evidence(log, hashes, "rlm-logs/" + log.name)
                    for line in data.splitlines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                            raise ValueError("invalid plugin event")
                        event_count += 1
                        if event["event"] == "subcall":
                            status = event.get("status")
                            if status not in ("ok", "error", "cancelled", "canceled", "cancel"):
                                raise ValueError("unknown subcall status")
                            statuses[status] += 1
                if not event_count:
                    raise ValueError("empty logs")
                result["audit_subcall_statuses"] = dict(statuses)
                if statuses["error"]:
                    reasons.append("child_error")
                if any(statuses[s] for s in ("cancelled", "canceled", "cancel")):
                    reasons.append("child_cancelled")
                requests = row.get("accounting", {}).get("requests", [])
                if (any(r.get("role") == "sub" and r.get("status") != 200 for r in requests)
                        and not statuses):
                    reasons.append("child_recovery_unknown")
            except (OSError, ValueError, TypeError, AttributeError):
                reasons.append("missing_or_malformed_plugin_logs")
        if selected is None:
            reasons.append("missing_validated_final")
        else:
            result["audit_final"] = selected
            if task is None or task.get("id") != row.get("task_id"):
                reasons.append("missing_or_mismatched_task")
            elif not reasons:
                try:
                    result.update(answer=selected["answer"], score=score_task(
                        {"gold": task["gold"], "scoring": task["scoring"]}, selected["answer"]))
                    result.update(correct=result["score"] == 1, grading={"score": result["score"]})
                except (ValueError, KeyError, TypeError, ZeroDivisionError):
                    reasons.append("scoring_failed")
        result["audit_adherence"] = "adherent" if not reasons else "nonadherent"
    if reasons:
        result.update(valid=False, score=None, correct=None, status="invalid", grading=None,
                      error=reasons[0])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", type=Path, required=True)
    parser.add_argument("--ops-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    ops, out = args.ops_dir.resolve(), args.out_dir.resolve()
    if not out.is_relative_to(ops) or out == ops or out.parts[len(ops.parts)] in {"runs", "homes", "tasks"}:
        raise ValueError("output must be a separate evaluator directory inside ops")
    names = [p.name for p in args.raw]
    if len(names) != len(set(names)) or any((out / n).exists() for n in names):
        raise ValueError("duplicate basenames or existing audited output; use a new output directory")
    raw_paths = [p.resolve() for p in args.raw]
    if any(out / p.name == p or p.is_relative_to(out) for p in raw_paths):
        raise ValueError("cannot overwrite or re-audit collection sources")
    manifest = args.tasks.read_bytes()
    tasks = {}
    for line in manifest.splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if task["id"] in tasks:
            raise ValueError("duplicate manifest task ID")
        tasks[task["id"]] = task
    source_hashes = {"auditor": sha(Path(__file__).read_bytes()),
                     "scorer": sha(Path(__file__).with_name("paper_tasks.py").read_bytes()),
                     "manifest": sha(manifest)}
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    stats = {}
    for path in raw_paths:
        data = path.read_bytes()
        source_hash = sha(data)
        audited = []
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "audit_version" in row:
                raise ValueError("input is already audited")
            task = tasks.get(row.get("task_id"))
            # Older rows may not carry a task fingerprint.
            task_matches = task is not None or not row.get("task_sha256")
            if task is not None and row.get("task_sha256"):
                frozen = dict(task)
                context = Path(frozen["context_path"])
                if not context.is_absolute():
                    context = args.tasks.parent / context
                frozen["context_path"] = str(context.resolve())
                frozen["context_sha256"] = row.get("context_sha256")
                task_matches = json_sha(frozen) == row["task_sha256"]
            checked = audit_row(row, task if task_matches else None, ops)
            if not task_matches:
                reason = "missing_or_mismatched_task"
                if reason not in checked["audit_reasons"]:
                    checked["audit_reasons"].append(reason)
                checked.update(valid=False, score=None, correct=None, status="invalid",
                               grading=None, error=checked["audit_reasons"][0])
            checked.update(source_sha=source_hash, source_row_sha256=sha(line),
                           source_line=line_number, audit_source_sha256=source_hashes)
            audited.append(checked)
        target = out / path.name
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out,
                                             prefix=".audit-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                for row in audited:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        stats[path.name] = {"rows": len(audited), "collection_valid": sum(r.get("collection_valid") is True for r in audited),
                            "audited_valid": sum(r.get("valid") is True for r in audited),
                            "adherence": dict(Counter(r["audit_adherence"] for r in audited)),
                            "reasons": dict(Counter(reason for r in audited for reason in r["audit_reasons"])),
                            "source_sha": source_hash}
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
