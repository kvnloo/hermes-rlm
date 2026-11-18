"""Offline regression checks for the generated paper-matrix task set.

Run from the repository root:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

from bench.paper_tasks import read_jsonl, score_task
from bench.synthetic_long_context import DEFAULT_RECORDS, RECORD_SIZES, SIZE_BASIS, generate, main


SIZES = (64, 128, 256, 512)  # pinned: the documented --records scales


class SyntheticLongContextTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name) / "generated"

    def tasks(self, records: int = 64) -> dict[str, dict]:
        self.out = self.out.parent / f"generated-{records}"
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
                self.assertEqual(task["metadata"]["size_basis"], SIZE_BASIS)
                self.assertIn("NOT a tokenizer", task["metadata"]["size_basis"])
                self.assertNotIn("ANSWER:", context)
                self.assertNotIn(task["question"], context)
                self.assertNotIn(str(task["gold"]), task["question"])
                self.assertEqual(task["scoring"], "browse_exact")
                self.assertEqual(score_task(task, f"Reasoning\nANSWER: {task['gold']}"), 1.0)
                self.assertEqual(score_task(task, f"ANSWER: {task['gold']} extra"), 0.0)
                self.assertEqual(score_task(task, "ANSWER: incorrect"), 0.0)
                self.assertEqual(score_task(task, ""), 0.0)

    def test_matrix_runner_loads_generated_manifest(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
        self.addCleanup(sys.path.pop, 0)
        from paper_matrix import load_tasks  # script-style sibling imports

        self.tasks(128)
        loaded = load_tasks(self.out / "full.jsonl")
        self.assertEqual(len(loaded), 3)
        sniah = {"sniah", "niah"}  # the filter --smallest-sniah applies
        self.assertFalse([t for t in loaded if re.sub(r"[^a-z]", "", t["family"].lower()) in sniah])

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
        for n in SIZES:
            with self.subTest(records=n):
                task = self.tasks(n)["synthetic-needle"]
                ticket = re.search(r"ticket (T-\d{4})", task["question"]).group(1)
                lines = Path(task["context_path"]).read_text().splitlines()
                matches = [
                    re.search(r"\btoken=(N-[A-F0-9]{12})\b", line).group(1)
                    for line in lines if f"ticket={ticket} |" in line
                ]
                self.assertEqual(matches, [task["gold"]])
                self.assertEqual(sum("ticket=T-" in line for line in lines), n * 3)

    def test_multihop_gold_requires_three_distinct_joins(self) -> None:
        for n in SIZES:
            with self.subTest(records=n):
                task = self.tasks(n)["synthetic-multihop"]
                context = Path(task["context_path"]).read_text()
                agents = dict(re.findall(r"^ALIAS \| agent=(AG-\d{4}) \| project=(PR-\d{4})", context, re.M))
                projects = dict(re.findall(r"^PROJECT \| project=(PR-\d{4}) \| locker=(LK-\d{4})", context, re.M))
                lockers = dict(re.findall(r"^LOCKER \| locker=(LK-\d{4}) \| seal=(S-[A-F0-9]{12})", context, re.M))
                self.assertEqual((len(agents), len(projects), len(lockers)), (n, n, n))
                self.assertEqual(len(set(agents.values())), n)
                self.assertEqual(len(set(projects.values())), n)
                self.assertEqual(len(set(lockers.values())), n)
                target = re.search(r"agent (AG-\d{4})", task["question"]).group(1)
                self.assertEqual(lockers[projects[agents[target]]], task["gold"])

    def test_aggregation_gold_matches_an_independent_row_parser(self) -> None:
        for n in SIZES:
            with self.subTest(records=n):
                task = self.tasks(n)["synthetic-aggregation"]
                context = Path(task["context_path"]).read_text()
                rows = re.findall(
                    r"^TX \| txn=(TX-\d{5}) \| region=(\w+) \| "
                    r"status=(\w+) \| units=(\d{3}) \| audit=synthetic$",
                    context,
                    re.M,
                )
                self.assertEqual(len(rows), n * 4)
                self.assertEqual(len({row[0] for row in rows}), len(rows))
                expected = sum(
                    int(units) for _, region, status, units in rows
                    if region == "west" and status == "cleared"
                )
                self.assertGreater(expected, 0)
                self.assertEqual(int(task["gold"]), expected)

    def test_rejects_record_sizes_that_break_bijective_joins(self) -> None:
        self.assertEqual(RECORD_SIZES, SIZES)  # the accepted set is exactly the documented scales
        for invalid in (0, 63, 65, 513, 1024, True, 64.0):
            with self.subTest(records=invalid), self.assertRaises(ValueError):
                generate(self.out, invalid)

    def test_task_ids_are_unique_per_family_and_scale(self) -> None:
        # Scales must not share ids: result rows and --task-ids select by id.
        for n in SIZES:
            got = sorted(task["id"] for task in self.tasks(n).values())
            self.assertEqual(got, [f"synthetic-{f}-v1-{n}" for f in ("aggregation", "multihop", "needle")])

    def test_second_scale_in_same_out_dir_keeps_first_scale_contexts(self) -> None:
        generate(self.out, 64)
        first = {t["id"]: t for t in read_jsonl(self.out / "full.jsonl")}
        generate(self.out, 128)
        for task in first.values():
            with self.subTest(task=task["id"]):
                text = Path(task["context_path"]).read_text(encoding="utf-8")
                self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 task["metadata"]["context_sha256"])

    def test_documented_matrix_command_parses(self) -> None:
        # The harness doc's model-backed command must stay valid for the real CLI.
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "bench"))
        self.addCleanup(sys.path.pop, 0)
        from paper_matrix import ARMS, parser

        doc = (root / "bench" / "paper-harness.md").read_text(encoding="utf-8")
        block = re.search(r"(python3 bench/paper_matrix\.py \"\$SYNTH/full\.jsonl\".*?)\n```", doc, re.S)
        self.assertIsNotNone(block, "synthetic matrix command missing from paper-harness.md")
        argv = shlex.split(block.group(1).replace("\\\n", " "))[2:]
        argv = [re.sub(r"\$\w+", "/x", a) for a in argv]
        args = parser().parse_args(argv)
        self.assertEqual(args.arms.split(","), ["base-truncated", "fork-rlm"])
        self.assertTrue(set(args.arms.split(",")) <= set(ARMS))
        self.assertTrue(hasattr(args, "base_context_chars"))
        self.assertIn("--base-context-chars", doc)

    def test_cli_prints_summary_and_documented_default_scale(self) -> None:
        def run(*args: str) -> dict:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(["--out", str(self.out), *args])
            return json.loads(buf.getvalue())

        def scales() -> set:
            return {t["metadata"]["records"] for t in read_jsonl(self.out / "full.jsonl")}

        self.assertEqual(run()["full"], 3)
        self.assertEqual(DEFAULT_RECORDS, 128)
        self.assertEqual(scales(), {128})
        run("--records", "64")
        self.assertEqual(scales(), {64})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--out", str(self.out), "--records", "100"])


if __name__ == "__main__":
    unittest.main()
