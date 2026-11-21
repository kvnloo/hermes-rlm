"""Report frozen study validity and classify available private audit evidence.

No model calls. The output contains counts, not answers, paths, or transcripts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_results import AUDIT_VERSION, HERMES_ARMS
from paper_tasks import read_jsonl


CAUSES = ("missing_validated_final", "uncited_answer", "repl_bypass_other_tool", "child_error")
ADHERENCE = ("adherent", "nonadherent", "unknown", "not_applicable")


def aggregate_report(data):
    if (data.get("schema_version"), data.get("audit_policy")) != (
            "rlm-paper-aggregate.v1", "paper_strict_v1"):
        raise ValueError("expected the strict paper aggregate schema")
    totals = Counter()
    by_arm = {}
    for row in data["primary"]:
        keys = ("observed", "scheduled", "valid", "invalid", "collection_valid", "collection_invalid",
                *(f"protocol_{name}" for name in ADHERENCE))
        if any(type(row.get(key)) is not int or row[key] < 0 for key in keys):
            raise ValueError("invalid aggregate counts")
        if (row["valid"] + row["invalid"] != row["observed"]
                or row["collection_valid"] + row["collection_invalid"] != row["observed"]
                or sum(row[f"protocol_{name}"] for name in ADHERENCE) != row["observed"]
                or row["valid"] > row["collection_valid"] or row["observed"] > row["scheduled"]):
            raise ValueError("inconsistent aggregate denominators")
        totals.update({key: row[key] for key in keys})
        arm = by_arm.setdefault(row["arm"], Counter())
        arm.update({k: row[k] for k in ("observed", "invalid", "protocol_adherent")})
    hermes_observed = sum(v["observed"] for k, v in by_arm.items() if k in HERMES_ARMS)
    return {
        "cohort": "primary",
        "observed": totals["observed"],
        "scheduled": totals["scheduled"],
        "collection_invalid": totals["collection_invalid"],
        "additional_strict_rejections": totals["collection_valid"] - totals["valid"],
        "strict_valid": totals["valid"],
        "strict_invalid": totals["invalid"],
        "adherence": {key: totals[f"protocol_{key}"] for key in ADHERENCE},
        "hermes_observed": hermes_observed,
        "other_arms_observed": totals["observed"] - hermes_observed,
        "hermes_protocol_adherent": sum(v["protocol_adherent"] for k, v in by_arm.items() if k in HERMES_ARMS),
        "by_arm": by_arm,
        "historical_cause_counts": dict.fromkeys(CAUSES),
        "historical_top_cause": None,
        "cause_limitation": "Published aggregate excludes row-level reasons and transcripts.",
    }


def classify(row):
    """Return supported observations; None means the evidence was unavailable."""
    n_finals = row.get("audit_validated_final_count")
    linked = row.get("audit_failure_signals")
    statuses = row.get("audit_subcall_statuses")
    reasons = row.get("audit_reasons") or []
    result = {"missing_validated_final": n_finals == 0 if type(n_finals) is int and n_finals >= 0 else None,
              "child_error": True if "child_error" in reasons else
              (bool(statuses.get("error")) if isinstance(statuses, dict) else None)}
    for key in ("uncited_answer", "repl_bypass_other_tool"):
        value = linked.get(key) if isinstance(linked, dict) else None
        result[key] = value if type(value) is bool else None
    return result


def evidence_report(rows):
    counts = {cause: {"observed": 0, "not_observed": 0, "unknown": 0} for cause in CAUSES}
    seen = set()
    observed = invalid = eligible = multi = no_positive = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("audit_version") != AUDIT_VERSION:
            raise ValueError(f"expected audited {AUDIT_VERSION} rows")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise ValueError("missing or duplicate audited run identity")
        seen.add(run_id)
        if type(row.get("valid")) is not bool:
            raise ValueError("invalid audited validity flag")
        observed += 1
        if row["valid"]:
            continue
        invalid += 1
        if row.get("arm") not in HERMES_ARMS:
            continue
        eligible += 1
        classes = classify(row)
        positives = 0
        for cause, value in classes.items():
            if value is True:
                counts[cause]["observed"] += 1
                positives += 1
            elif value is False:
                counts[cause]["not_observed"] += 1
            else:
                counts[cause]["unknown"] += 1
        multi += positives > 1
        no_positive += not positives
    return {
        "scope": "caller-supplied audited rows, not assumed to match the published cohort",
        "observed": observed,
        "invalid": invalid,
        "invalid_hermes_denominator": eligible,
        "invalid_other_arms": invalid - eligible,
        "causes": counts,
        "rows_with_multiple_observations": multi,
        "rows_without_positive_observation": no_positive,
        "limitation": "Observations may overlap; not observed is not proof absent or a causal ranking.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audited-rows", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    raw = args.summary.read_bytes()
    report = aggregate_report(json.loads(raw))
    report["summary_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.audited_rows:
        report["local_evidence"] = evidence_report(row for path in args.audited_rows for row in read_jsonl(path))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
