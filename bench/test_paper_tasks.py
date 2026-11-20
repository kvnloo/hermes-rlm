"""Focused deterministic task-builder and scoring checks, runnable as one file.

reuse-checked: bench/make_corpus.py and run_bench.score; pinned upstream OOLONG
helper and RULER metric; appendix E.1 predicates checked with independent cases.
Run: PYTHONDONTWRITEBYTECODE=1 python3 bench/test_paper_tasks.py
"""
from datetime import date
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from paper_tasks import (A, D, E, H, L, N, TaskBuilder, labeled_users,
                         oolong_gold, oolong_parse, pair_predicates, pairs_gold,
                         parse_pairs, read_jsonl, score_task)


def counts(entries):
    return {label: [date(2023, 6, 1)] * count for label, count in entries}


class PairGoldTests(unittest.TestCase):
    def test_union_predicates(self):
        unions = [(N, L), (E, H), (D, A), (H, L), (E, N),
                  (L, A), (D, N), (H, D), (E, L), (N, A)]
        for task_id, labels in enumerate(unions, 1):
            for label in labels:
                with self.subTest(task=task_id, label=label):
                    when = date(2022, 1, 1) if task_id in (5, 10) else date(2024, 1, 1)
                    first, second = pair_predicates(task_id)
                    self.assertTrue(first({label: [when]}))
                    self.assertTrue(second({label: [when]}))
                    self.assertFalse(first({}))
                    other = next(item for item in (D, E, H, N, L, A) if item not in labels)
                    self.assertFalse(first({other: [when]}))

    def test_strict_dates_and_vacuity(self):
        cases = [(4, H, L, date(2023, 1, 6), 1),
                 (5, E, N, date(2023, 3, 15), -1),
                 (7, N, D, date(2023, 2, 1), 1),
                 (9, L, E, date(2023, 4, 10), 1),
                 (10, A, N, date(2023, 5, 20), -1)]
        from datetime import timedelta
        for task_id, label, alternative, boundary, step in cases:
            predicate, _ = pair_predicates(task_id)
            with self.subTest(task=task_id):
                self.assertFalse(predicate({label: [boundary]}))
                self.assertTrue(predicate({label: [boundary + timedelta(days=step)]}))
                self.assertFalse(predicate({label: [boundary - timedelta(days=step)]}))
                self.assertFalse(predicate({label: [boundary, boundary + timedelta(days=step)]}))
                self.assertTrue(predicate({alternative: [boundary]}))

    def test_all_asymmetric_queries_both_orientations_and_exact_counts(self):
        cases = {
            11: ([(E, 1), (A, 1)], [(E, 1)], E),
            12: ([(N, 2)], [(L, 1), (H, 1)], None),
            13: ([(D, 1)], [(A, 1), (E, 1)], D),
            14: ([(H, 1), (N, 1)], [(L, 2)], L),
            15: ([(E, 1), (L, 1), (A, 1)], [(N, 1)], N),
            16: ([(D, 1), (H, 1)], [(E, 2), (A, 1)], A),
            17: ([(N, 1)], [(L, 1), (D, 1)], N),
            18: ([(A, 1), (H, 1)], [(E, 1), (N, 1)], H),
            19: ([(L, 2), (E, 1)], [(D, 1), (A, 1)], D),
            20: ([(N, 1), (H, 1)], [(L, 1), (E, 1), (A, 1)], A),
        }
        for task_id, (left, right, exact_label) in cases.items():
            with self.subTest(task=task_id):
                self.assertEqual(pairs_gold({9: counts(left), 2: counts(right)}, task_id), [[2, 9]])
                self.assertEqual(pairs_gold({2: counts(left), 9: counts(right)}, task_id), [[2, 9]])
                self.assertEqual(pairs_gold({2: {}}, task_id), [])
                if exact_label:
                    side = left if exact_label in dict(left) and task_id in (13, 17, 18) else right
                    modified = [(label, count + (label == exact_label)) for label, count in side]
                    bad_left = modified if side is left else left
                    bad_right = modified if side is right else right
                    self.assertEqual(pairs_gold({9: counts(bad_left), 2: counts(bad_right)}, task_id), [])
                for label, _ in left:
                    missing = [(key, number) for key, number in left if key != label]
                    self.assertEqual(pairs_gold({9: counts(missing), 2: counts(right)}, task_id), [])

    def test_normalize_and_dedup_gold(self):
        users = {3: counts([(E, 1), (A, 1)]), 1: counts([(E, 1), (A, 1)])}
        self.assertEqual(pairs_gold(users, 11), [[1, 3]])
        with self.assertRaises(ValueError):
            pair_predicates(21)

    def test_labeled_parser(self):
        text = "Header\nDate: Jan 06, 2023 || User: 009 || Instance: Who? || Label: human being\n"
        self.assertEqual(labeled_users(text)[9][H], [date(2023, 1, 6)])
        for bad in ("Date: Jan 99, 2023 || User: 9 || Instance: x || Label: entity", "Date: bad", "header only"):
            with self.assertRaises(ValueError):
                labeled_users(bad)


class ScoringTests(unittest.TestCase):
    def score(self, kind, gold, answer):
        return score_task(dict(scoring=kind, gold=gold), answer)

    def test_pairs_f1(self):
        gold = [[1, 2], [2, 3]]
        self.assertEqual(parse_pairs("(2, 1)\n[1, 2]"), {(1, 2)})
        self.assertAlmostEqual(self.score("pairs_f1", gold, "(2,1) (1,2)"), 2 / 3)
        self.assertEqual(self.score("pairs_f1", gold, "(2,1) (3,2)"), 1)
        self.assertAlmostEqual(self.score("pairs_f1", gold, "(1,2) (2,3) (3,4)"), .8)
        for invalid in ("(1,1)", "(1,2) (3,3)", "garbage", ""):
            self.assertEqual(self.score("pairs_f1", [], invalid), 0)
        self.assertEqual(self.score("pairs_f1", [], "[]"), 1)
        self.assertEqual(self.score("pairs_f1", gold, "[]"), 0)
        self.assertEqual(self.score("pairs_f1", [], "(1,2)"), 0)
        self.assertEqual(self.score("pairs_f1", gold, "ANSWER:\n(1,2)\n(2,3)"), 1)

    def test_oolong_parser_and_numeric(self):
        scorer = dict(type="oolong", answer_type="ANSWER_TYPE.NUMERIC")
        self.assertEqual(oolong_parse("Answer: **[12]**"), "12")
        self.assertEqual(oolong_parse("I think the final answer is 12"), "12")
        self.assertEqual(self.score(scorer, 12, "Answer: 14"), .75 ** 2)
        self.assertEqual(self.score(scorer, 12, "Answer: twelve"), 0)
        self.assertEqual(oolong_gold("['entity']"), "entity")
        self.assertEqual(oolong_gold("[datetime.date(2023, 1, 6)]"), "2023-01-06T00:00:00")

    def test_oolong_numeric_underflows_for_large_integer_distance(self):
        scorer = dict(type="oolong", answer_type="ANSWER_TYPE.NUMERIC")
        huge = 10 ** 400
        for gold, answer in ((12, huge), (huge, 12)):
            with self.subTest(gold_is_huge=(gold == huge)):
                self.assertEqual(self.score(scorer, gold, f"Answer: {answer}"), 0.0)
        self.assertEqual(self.score(scorer, huge, f"Answer: {huge - 2}"), .75 ** 2)

    def test_oolong_other_types(self):
        scorer = dict(type="oolong", answer_type="ANSWER_TYPE.LABEL")
        self.assertEqual(self.score(scorer, "entity", "Label: [entity]"), 1)
        self.assertEqual(self.score(scorer, "entity", "Label: Entity"), 0)
        self.assertEqual(self.score(scorer, "a is more common than b", "Answer: more common"), 1)
        scorer = dict(type="oolong", answer_type="ANSWER_TYPE.DATE")
        self.assertEqual(self.score(scorer, "2023-01-06T00:00:00", "Date: January 6, 2023"), 1)
        self.assertEqual(self.score(scorer, "2023-01-06T00:00:00", "Date: January 2023"), 0)

    def test_ruler_choices_browse(self):
        self.assertEqual(self.score("ruler", ["ABC", "123"], "abc only"), .5)
        self.assertEqual(self.score("ruler", ["123"], "Number 123."), 1)
        self.assertEqual(self.score("ruler", ["123"], "1-2-3"), 0)
        for answer in ("B", "ANSWER: B", "(B)", "The answer is B."):
            self.assertEqual(self.score("choice", "B", answer), 1)
        self.assertEqual(self.score("choice", "B", "A or B"), 0)
        self.assertEqual(self.score("browse_exact", "Some School", "ANSWER: some  SCHOOL"), 1)
        self.assertEqual(self.score("browse_exact", "Some School", "It is Some School."), 0)

    def test_legacy_parity(self):
        spec = importlib.util.spec_from_file_location("legacy_run_bench", Path(__file__).with_name("run_bench.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = [dict(id="niah", gold="1-2-3"), dict(id="count_billing", gold=12),
                 dict(id="top_bug_user", gold="u192 5", ties=["u090", "u192"])]
        for q in cases:
            task = dict(gold=q["gold"], scoring=dict(type="legacy", question_id=q["id"], ties=q.get("ties", [])))
            for answer in ("ANSWER: 123", "ANSWER: 12", "ANSWER: u090 5", "nothing", "ANSWER: wrong\nANSWER: u192 5"):
                self.assertEqual(score_task(task, answer), float(module.score(q, answer)))


class ManifestTests(unittest.TestCase):
    def test_context_separation_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = TaskBuilder(directory, directory, directory)
            builder.add("one", "test", 1024, "pure input\n", "Question?", "SECRET", "browse_exact", "source", "revision", "source-id")
            builder.save()
            path = Path(directory)
            first = (path / "full.jsonl").read_bytes()
            builder.save()
            self.assertEqual(first, (path / "full.jsonl").read_bytes())
            row = next(read_jsonl(path / "smoke.jsonl"))
            context = Path(row["context_path"])
            self.assertEqual(context.read_text(), "pure input\n")
            self.assertNotIn("SECRET", context.read_text())
            self.assertEqual([p.name for p in context.parent.iterdir()], ["context.txt"])
            self.assertEqual(len(row["metadata"]["source_context_sha256"]), 64)

    def test_context_bytes_match_manifest_even_after_newline_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = TaskBuilder(directory, directory, directory)
            fields = ("one", "test", 1024, "caf\u00e9\n", "Question?", "gold",
                      "browse_exact", "source", "revision", "source-id")
            row = builder.add(*fields)
            context = Path(row["context_path"])
            context.write_bytes(b"caf\xc3\xa9\r\n")
            row = builder.add(*fields)
            expected = b"caf\xc3\xa9\n"
            self.assertEqual(context.read_bytes(), expected)
            self.assertEqual(row["metadata"]["context_bytes"], len(expected))
            self.assertEqual(row["metadata"]["context_sha256"],
                             hashlib.sha256(expected).hexdigest())


if __name__ == "__main__":
    unittest.main()
