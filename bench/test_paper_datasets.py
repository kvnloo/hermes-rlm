"""Offline R1 manifest acceptance and source-scoring examples.

All task content and 1,000 BrowseComp documents in this test are synthetic.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import paper_datasets as datasets
import paper_matrix as matrix
import paper_tasks as tasks


def jsonlines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "out"
        self.oolong_row = dict(id=23, dataset="trec_coarse", context_len=1024, context_window_id=9,
            context_window_text="Two general-knowledge questions. É",
            context_window_text_with_labels=(
                "Date: Jan 06, 2023 || User: 1 || Instance: One || Label: numeric value\n"
                "Date: Jan 06, 2023 || User: 2 || Instance: Two || Label: location\n"),
            question="How many records?", answer="[2]", answer_type="ANSWER_TYPE.NUMERIC")
        self.oolong = jsonlines(self.data / "OOLONG-trec-coarse-small.jsonl", [
            {**self.oolong_row, "id": 999, "dataset": "not_trec"}, self.oolong_row,
            {**self.oolong_row, "id": 24, "question": "What is the count?"}])
        self.questions = self.data / "OOLONG-Pairs-20-queries.json"
        self.questions.write_text(json.dumps([dict(task_id=n, question=f"Synthetic appendix question {n}?")
                                               for n in range(1, 21)]), encoding="utf-8")
        code_rows = [dict(_id=str(i), domain=datasets.CODEQA_DOMAIN, sub_domain=datasets.CODEQA_SUBDOMAIN,
            difficulty="easy", length="short", question="Which option?", choice_A="left", choice_B="right",
            choice_C="up", choice_D="down", answer="B", context="def run(): return 1") for i in range(5)]
        jsonlines(self.data / "LongBench-v2-CodeQA.jsonl", [{**code_rows[0], "domain": "other"}, *code_rows])
        self.browse_query = dict(query_id="q", query="Which fact?", answer="Gold fact",
            gold_docs=[{"docid": "gold", "text": "Evidence A"}],
            evidence_docs=[{"docid": "evidence", "text": "Evidence B"}])
        jsonlines(self.data / "BrowseComp-Plus-20Q.jsonl", [self.browse_query])
        self.ids = ["gold", *(f"d{i}" for i in range(997)), "evidence", "d997"]
        self.selection = jsonlines(self.data / "BrowseComp-Plus-20Q-1000docs-manifest.jsonl",
            [dict(query_id="q", docids=self.ids, required_docids=["gold", "evidence"])])
        self.corpus = jsonlines(self.data / "corpus.jsonl",
            [dict(docid=f"d{i}", text=f"Synthetic background {i}") for i in reversed(range(998))])

    def run_cli(self, *flags):
        with redirect_stdout(io.StringIO()) as console:
            self.assertEqual(datasets.main(["--reference", str(self.root), "--out", str(self.out), *flags]), 0)
        return json.loads(console.getvalue())

    def test_all_four_families_generate_accepted_taskbuilder_manifests(self):
        coverage = self.run_cli("--browse-corpus", str(self.corpus))
        expected = {"full.jsonl": 28, "smoke.jsonl": 4, "pairs-all20.jsonl": 20, "codeqa-optional.jsonl": 5}
        for name, count in expected.items():
            with self.subTest(name=name):
                self.assertEqual(len(matrix.load_tasks(self.out / name)), count)
        self.assertEqual(coverage["family_counts"], {"OOLONG": 2, "OOLONG-Pairs": 20,
                                                "BrowseComp-Plus": 1, "CodeQA": 5})
        rows = matrix.load_tasks(self.out / "full.jsonl")
        pair = next(r for r in rows if r["metadata"].get("appendix_task_id") == 1)
        self.assertEqual(pair["gold"], [[1, 2]])
        self.assertNotIn("Label:", Path(pair["context_path"]).read_text(encoding="utf-8"))
        self.assertEqual(tasks.score_task(pair, "(2, 1)"), 1.0)
        self.assertEqual(tasks.score_task(pair, "(1, 2), (2, 3)"), 2 / 3)
        browse = next(r for r in rows if r["family"] == "BrowseComp-Plus")
        self.assertEqual(browse["metadata"]["document_ids"], self.ids)
        body = Path(browse["context_path"]).read_text(encoding="utf-8")
        self.assertEqual(body.count("[document "), 1000)
        self.assertTrue(body.endswith("[document d997]\nSynthetic background 997"))
        self.assertNotIn(browse["gold"], body)
        self.assertIn("official semantic judge not included", browse["metadata"]["deviation"])
        with self.assertRaises(tasks.OfficialScoreUnavailable):
            tasks.score_task(browse, "ANSWER: gold  fact")
        self.run_cli("--datasets", "browsecomp-plus", "--browse-corpus", str(self.corpus),
                     "--browse-exact-diagnostic")
        diagnostic = matrix.load_tasks(self.out / "full.jsonl")[0]
        self.assertIn("not the official LLM judge", diagnostic["metadata"]["deviation"])
        self.assertEqual(tasks.score_task(diagnostic, "ANSWER: gold  fact"), 1.0)
        code = next(r for r in rows if r["family"] == "CodeQA")
        self.assertIn("B. right", code["question"])
        self.assertEqual(tasks.score_task(code, "ANSWER: B"), 1.0)
        self.assertEqual(tasks.score_task(code, "A"), 0.0)

    def test_published_pairs_reuses_gold_only_with_single_context_per_length(self):
        published = self.data / "pairs"
        published.mkdir()
        (published / "oolong-pairs-1024.json").write_text(json.dumps([
            dict(id=n, question=f"Published-shaped synthetic question {n}?", answer=[[1, 3]], type="Pairs")
            for n in range(1, 21)]), encoding="utf-8")
        self.run_cli("--datasets", "oolong-pairs", "--pairs-reference-dir", str(published))
        rows = matrix.load_tasks(self.out / "full.jsonl")
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["gold"], [[1, 3]])
        self.assertEqual(rows[0]["metadata"]["source_revision"], datasets.UNVERIFIED)
        self.assertEqual(rows[0]["metadata"]["expected_source_revision"], datasets.PAIRS_REV)
        rows = list(tasks.read_jsonl(self.oolong))
        rows.append({**self.oolong_row, "id": 25, "context_window_id": 10,
                     "context_window_text": "A different window."})
        jsonlines(self.oolong, rows)
        with self.assertRaisesRegex(ValueError, "ambiguous published OOLONG-Pairs"):
            self.run_cli("--datasets", "oolong-pairs", "--pairs-reference-dir", str(published))

    def test_default_browse_manifest_dry_run_works_but_live_refuses_before_dispatch(self):
        self.run_cli("--datasets", "browsecomp-plus", "--browse-corpus", str(self.corpus))
        arguments = [str(self.out / "full.jsonl"), str(self.root / "rows.jsonl"),
                     "--ops-dir", str(self.root / "ops"), "--home-template", str(self.root),
                     "--released-source", str(self.root), "--released-python", sys.executable,
                     "--fork-source", str(self.root), "--fork-python", sys.executable,
                     "--nano-source", str(self.root), "--arms", "base-truncated", "--repeats", "1"]
        with (patch.object(matrix, "source_sha", return_value="fixture"),
              patch.object(matrix, "run_one") as run,
              redirect_stdout(io.StringIO()) as console):
            self.assertEqual(matrix.main([*arguments, "--dry-run"]), 0)
            self.assertEqual(json.loads(console.getvalue())["task_id"], "browsecomp-q")
            with self.assertRaisesRegex(ValueError, "official BrowseComp-Plus grading requires --scorer"):
                matrix.main(arguments)
            run.assert_not_called()
        self.assertFalse((self.root / "rows.jsonl").exists())

    def test_author_oolong_missing_dependency_refuses_before_dispatch(self):
        self.run_cli("--datasets", "oolong")
        arguments = [str(self.out / "full.jsonl"), str(self.root / "rows.jsonl"),
                     "--ops-dir", str(self.root / "ops"), "--home-template", str(self.root),
                     "--released-source", str(self.root), "--released-python", sys.executable,
                     "--fork-source", str(self.root), "--fork-python", sys.executable,
                     "--nano-source", str(self.root), "--arms", "base-truncated", "--repeats", "1"]
        with (patch.object(matrix, "source_sha", return_value="fixture"),
              patch.object(matrix, "run_one") as run,
              patch.dict(sys.modules, {"dateutil.parser": None}),
              redirect_stdout(io.StringIO()) as console):
            self.assertEqual(matrix.main([*arguments, "--dry-run"]), 0)
            self.assertEqual(len(console.getvalue().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, "author OOLONG scoring requires python-dateutil"):
                matrix.main(arguments)
            run.assert_not_called()
        self.assertFalse((self.root / "rows.jsonl").exists())

    def test_browse_rejects_incomplete_or_duplicated_selection(self):
        for ids, message in ((self.ids[:-1], "exactly 1000 distinct"),
                             (self.ids[:-1] + [self.ids[0]], "exactly 1000 distinct"),
                             (["not-gold"] + self.ids[1:], "missing a required")):
            with self.subTest(message=message):
                jsonlines(self.selection, [dict(query_id="q", docids=ids)])
                with self.assertRaisesRegex(ValueError, message):
                    self.run_cli("--datasets", "browsecomp-plus", "--browse-corpus", str(self.corpus))
        jsonlines(self.selection, [dict(query_id="q", docids=self.ids)])
        jsonlines(self.corpus, [dict(docid="d0", text="Only one background document")])
        with self.assertRaisesRegex(ValueError, "missing 997 selected BrowseComp corpus documents"):
            self.run_cli("--datasets", "browsecomp-plus", "--browse-corpus", str(self.corpus))


class AuthorScoringParityTests(unittest.TestCase):
    def test_fixed_examples_match_unmodified_author_scorer(self):
        from paper_oolong_author import _synth_score
        examples = [
            ("[10]", "NUMERIC", "ANSWER: 12", 0.5625),
            ("[10]", "NUMERIC", "unknown", 0.0),
            ("['entity']", "LABEL", "ANSWER: ENTITY", 1.0),
            ("['entity']", "LABEL", "My classification is an entity.", 1.0),
            ("['more common than']", "COMPARISON", "It is MORE COMMON THAN y", 1.0),
            ("['more common than']", "COMPARISON", "It is less common than y", 0.0),
            ("[datetime.date(2023, 1, 6)]", "DATE", "Date: January 6, 2023", 1.0),
            ("[datetime.date(2023, 1, 6)]", "DATE", "Date: January 7, 2023", 0.0),
        ]
        for raw, kind, output, expected in examples:
            with self.subTest(kind=kind, output=output):
                answer_type = "ANSWER_TYPE." + kind
                scoring = dict(type="oolong_author", raw_answer=raw, answer_type=answer_type)
                observed = _synth_score(dict(answer=raw, answer_type=answer_type), output)
                self.assertEqual(observed, expected)
                self.assertEqual(tasks.score_task(dict(gold=None, scoring=scoring), output), observed)
