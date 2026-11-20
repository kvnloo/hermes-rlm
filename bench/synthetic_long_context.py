"""Offline, deterministic long-context tasks for the paper matrix harness.

No external datasets, downloads, model calls, or random-state dependencies.
Gold and questions live in evaluator manifests, never in context files.

    python -m bench.synthetic_long_context --out /tmp/rlm-synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bench.paper_tasks import TaskBuilder

REVISION = "synthetic-long-context-v1"
DEFAULT_RECORDS = 128
RECORD_SIZES = (64, 128, 256, 512)
SIZE_BASIS = "ceil(context characters / 4) heuristic; NOT a tokenizer count"


def _code(kind: str, index: int) -> str:
    payload = f"{REVISION}|{kind}|{index}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:12].upper()


def _needle(records: int) -> tuple[str, str, str]:
    """One ticket's token near the end of a distractor-heavy dispatch ledger."""
    total = records * 3
    target = total - 13
    lines = [
        "DISPATCH LEDGER - independently assigned ticket tokens.",
        "Select by the complete ticket identifier, not by owner, state or position.",
    ]
    states = ("queued", "ready", "held", "closed")
    for index in range(total):
        lines.append(
            f"entry={index:04d} | ticket=T-{index:04d} | "
            f"token=N-{_code('needle', index)} | "
            f"owner=U-{(index * 17 + 11) % 257:03d} | state={states[index % 4]}"
        )
    question = (
        f"In the dispatch ledger, what token belongs to ticket T-{target:04d}? "
        "Return only the token."
    )
    return "\n".join(lines) + "\n", question, f"N-{_code('needle', target)}"


def _multihop(records: int) -> tuple[str, str, str]:
    """Follow agent -> project -> locker -> seal across three separate tables."""
    def project_for(agent: int) -> int:
        return (agent * 5 + 17) % records

    def locker_for(project: int) -> int:
        return (project * 7 + 29) % records

    lines = [
        "ROUTING TABLES - joins require exact, complete identifiers.",
        "The three tables are separate; no agent row contains the final seal.",
        "-- AGENT TO PROJECT --",
    ]
    for index in range(records):
        lines.append(
            f"ALIAS | agent=AG-{index:04d} | "
            f"project=PR-{project_for(index):04d} | active=yes"
        )
    lines.append("-- PROJECT TO LOCKER --")
    for index in range(records):
        lines.append(
            f"PROJECT | project=PR-{index:04d} | "
            f"locker=LK-{locker_for(index):04d} | active=yes"
        )
    lines.append("-- LOCKER TO SEAL --")
    for index in range(records):
        lines.append(
            f"LOCKER | locker=LK-{index:04d} | "
            f"seal=S-{_code('seal', index)} | active=yes"
        )
    target = records - 11
    final_locker = locker_for(project_for(target))
    question = (
        f"For agent AG-{target:04d}, follow agent to project, project to locker, "
        "and locker to seal. Return only the final seal."
    )
    return "\n".join(lines) + "\n", question, f"S-{_code('seal', final_locker)}"


def _aggregation(records: int) -> tuple[str, str, str]:
    """Sum a selective conjunction over many small, independently coded rows."""
    regions = ("north", "east", "south", "west")
    statuses = ("pending", "cleared", "void")
    lines = [
        "TRANSACTION LEDGER - units are integers, without currency conversion.",
        "Only rows matching BOTH requested fields count toward the sum.",
    ]
    total = 0
    for index in range(records * 4):
        region = regions[(index * 3 + 1) % len(regions)]
        status = statuses[(index * 5 + 2) % len(statuses)]
        units = 11 + (index * 37 + 19) % 89
        lines.append(
            f"TX | txn=TX-{index:05d} | region={region} | "
            f"status={status} | units={units:03d} | audit=synthetic"
        )
        if region == "west" and status == "cleared":
            total += units
    question = (
        "What is the sum of units for transactions with region=west AND "
        "status=cleared? Return only the integer."
    )
    return "\n".join(lines) + "\n", question, str(total)


def generate(out: Path, records: int = DEFAULT_RECORDS) -> dict:
    """Materialize three compatible paper-matrix tasks and their offline gold."""
    if type(records) is not int or records not in RECORD_SIZES:
        raise ValueError(f"records must be one of {RECORD_SIZES}")
    builder = TaskBuilder(out, out, out)
    for name, make in (
        ("needle", _needle),
        ("multihop", _multihop),
        ("aggregation", _aggregation),
    ):
        context, question, gold = make(records)
        builder.add(
            f"synthetic-{name}-v1-{records}",
            f"synthetic-{name}",
            (len(context) + 3) // 4,
            context,
            question,
            gold,
            "browse_exact",
            "local deterministic generator",
            REVISION,
            f"{name}/{records}",
            size_basis=SIZE_BASIS,
            records=records,
        )
    return builder.save()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--records", type=int, choices=RECORD_SIZES, default=DEFAULT_RECORDS)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.out, args.records), sort_keys=True))


if __name__ == "__main__":
    main()
