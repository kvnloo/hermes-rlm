"""Offline regression checks for the generated paper-matrix task set.

Run from the repository root:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import unittest

from bench.paper_tasks import read_jsonl, score_task
from bench.synthetic_long_context import generate


class SyntheticLongContextTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name) / "generated"

    def tasks(self, records: int = 64) -> dict[str, dict]:
        summary = generate(self.out, records)
        self.assertEqual(summary["full"], 3)
        self.assertEqual(summary["smoke"], 3)
        return {task["family"]: task for task in read_jsonl(self.out / "full.jsonl")}

    def test_manifest_contract_separation_and_exact_scoring(self) -> None:
        tasks = self.tasks()
        self.assertEqual(
            set(tasks), {"synthetic-needle", "synthetic-multihop", "synthetic-aggregation"}
        )
        self.assertEqual(
            list(read_jsonl(self.out / "full.jsonl")),
            list(read_jsonl(self.out / "smoke.jsonl")),
        )
        for family, task in tasks.items():
            with self.subTest(family=family):
                context_file = Path(task["context_path"])
                context = context_file.read_text(encoding="utf-8")
                self.assertEqual(context_file.name, "context.txt")
                self.assertEqual(task["size_tokens"], (len(context) + 3) // 4)
                self.assertEqual(
                    task["metadata"]["context_sha256"],
                    hashlib.sha256(context.encode("utf-8")).hexdigest(),
                )
                self.assertNotIn("ANSWER:", context)
                self.assertNotIn(str(task["gold"]), task["question"])
                self.assertEqual(task["scoring"], "browse_exact")
                self.assertEqual(score_task(task, f"Reasoning\nANSWER: {task['gold']}"), 1.0)
                self.assertEqual(score_task(task, f"ANSWER: {task['gold']} extra"), 0.0)
                self.assertEqual(score_task(task, "ANSWER: incorrect"), 0.0)
                self.assertEqual(score_task(task, ""), 0.0)

    def test_same_manifest_and_context_bytes_on_repeat_generation(self) -> None:
        self.tasks()
        before = {
            str(path.relative_to(self.out)): path.read_bytes()
            for path in self.out.rglob("*") if path.is_file()
        }
        self.tasks()
        after = {
            str(path.relative_to(self.out)): path.read_bytes()
            for path in self.out.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_needle_gold_is_the_only_matching_ticket(self) -> None:
        task = self.tasks()["synthetic-needle"]
        ticket = re.search(r"ticket (T-\d{4})", task["question"]).group(1)
        lines = Path(task["context_path"]).read_text().splitlines()
        matches = [
            re.search(r"\btoken=(N-[A-F0-9]{12})\b", line).group(1)
            for line in lines if f"ticket={ticket} |" in line
        ]
        self.assertEqual(matches, [task["gold"]])
        self.assertEqual(sum("ticket=T-" in line for line in lines), 64 * 3)

    def test_multihop_gold_requires_three_distinct_joins(self) -> None:
        task = self.tasks()["synthetic-multihop"]
        context = Path(task["context_path"]).read_text()
        agents = dict(re.findall(r"^ALIAS \| agent=(AG-\d{4}) \| project=(PR-\d{4})", context, re.M))
        projects = dict(re.findall(r"^PROJECT \| project=(PR-\d{4}) \| locker=(LK-\d{4})", context, re.M))
        lockers = dict(re.findall(r"^LOCKER \| locker=(LK-\d{4}) \| seal=(S-[A-F0-9]{12})", context, re.M))
        self.assertEqual((len(agents), len(projects), len(lockers)), (64, 64, 64))
        target = re.search(r"agent (AG-\d{4})", task["question"]).group(1)
        self.assertEqual(lockers[projects[agents[target]]], task["gold"])

    def test_aggregation_gold_matches_an_independent_row_parser(self) -> None:
        task = self.tasks()["synthetic-aggregation"]
        context = Path(task["context_path"]).read_text()
        rows = re.findall(
            r"^TX \| txn=(TX-\d{5}) \| region=(\w+) \| "
            r"status=(\w+) \| units=(\d{3}) \| audit=synthetic$",
            context,
            re.M,
        )
        self.assertEqual(len(rows), 64 * 4)
        self.assertEqual(len({row[0] for row in rows}), len(rows))
        expected = sum(
            int(units) for _, region, status, units in rows
            if region == "west" and status == "cleared"
        )
        self.assertGreater(expected, 0)
        self.assertEqual(int(task["gold"]), expected)

    def test_rejects_record_sizes_that_break_bijective_joins(self) -> None:
        for invalid in (0, 63, 65, 513, True):
            with self.subTest(records=invalid), self.assertRaises(ValueError):
                generate(self.out, invalid)


if __name__ == "__main__":
    unittest.main()
