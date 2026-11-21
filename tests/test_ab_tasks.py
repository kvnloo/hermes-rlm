"""The A/B eval tasks are deterministic, keep gold out of the working directory, and score both answer shapes."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import tasks as T  # noqa: E402


class AbTasks(unittest.TestCase):
    def test_gold_is_never_in_the_working_directory_or_prompt(self):
        for task, (_, _, gold) in zip(T.TASKS, [T._needle(T.RECORDS), T._multihop(T.RECORDS),
                                                T._aggregation(T.RECORDS), T._reviews()]):
            with self.subTest(task.id), tempfile.TemporaryDirectory() as d:
                task.build(Path(d))
                self.assertEqual([p.name for p in Path(d).iterdir()], ["context.txt"])
                self.assertNotIn(gold, task.prompt)
                if task.id in {"needle", "multihop"}:
                    continue  # retrieval tasks: the gold token is in the context by design
                # aggregation answers are computed, not quoted
                self.assertTrue(task.score(Path(d), f"ANSWER: {gold}")["pass"])

    def test_both_answer_shapes_score(self):
        task = T.TASKS[3]
        gold = T._reviews()[2]
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(task.score(Path(d), f"I counted them.\nANSWER: **{gold}**")["pass"])
            self.assertTrue(task.score(Path(d), f"{gold}\n\nSources:\ncontext.txt[0:10] \"CUSTOMER R\"")["pass"])
            self.assertFalse(task.score(Path(d), f"ANSWER: {int(gold) + 1}")["pass"])
            self.assertFalse(task.score(Path(d), "")["pass"])

    def test_reviews_needs_semantics_not_keywords(self):
        context, _, gold = T._reviews()
        keyword_hits = sum(("deliver" in l or "ship" in l) for l in context.splitlines()[1:])
        self.assertNotEqual(keyword_hits, int(gold))
        self.assertGreater(int(gold), 50)

    def test_generation_is_byte_stable(self):
        self.assertEqual(T._reviews(), T._reviews())


if __name__ == "__main__":
    unittest.main()
