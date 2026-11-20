"""Materialize the paper's four benchmark families from local reference data.

Reuse-checked: TaskBuilder.add/save, labeled_users/pairs_gold, and the existing
reference bundle layout in paper_tasks.py. This command never downloads data.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import itertools
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from paper_tasks import (BROWSE_REV, CODEQA_REV, CORPUS_REV, OOLONG_REV, TaskBuilder,
                         digest, labeled_users, oolong_gold, pairs_gold, read_jsonl)


DATASETS = ("oolong", "oolong-pairs", "browsecomp-plus", "codeqa")
CODEQA_DOMAIN = "Code Repository Understanding"
CODEQA_SUBDOMAIN = "Code repo QA"
UNVERIFIED = "unverified-local-input"
PAIRS_REV = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"


def _id(value):
    # These identifiers become path components in the existing TaskBuilder.
    label = str(value)
    if type(value) not in (int, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(f"invalid dataset row identifier: {label!r}")
    return label


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _rows(path, *, fields, predicate=None):
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return
    if path.suffix != ".parquet":
        raise ValueError(f"expected JSONL or Parquet: {path}")
    try:
        import duckdb
    except ImportError as exc:
        raise ValueError("Parquet input requires duckdb; use JSONL otherwise") from exc
    with duckdb.connect() as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='768MB'")
        db.execute("SET max_temp_directory_size='0B'")
        cursor = db.execute(f"SELECT {fields} FROM read_parquet(?)" +
                            (f" WHERE {predicate}" if predicate else ""), [str(path)])
        keys = [column[0] for column in cursor.description]
        while batch := cursor.fetchmany(1):
            for values in batch:
                yield dict(zip(keys, values))


def _appendix_rows(path, *, published=False):
    queries = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        task_id = row["id" if published else "task_id"]
        if type(task_id) is not int or task_id in queries:
            raise ValueError("duplicate or invalid OOLONG-Pairs question ID")
        question = _text(row["question"], "question")
        gold = row["answer"] if published else None
        if published and (not isinstance(gold, list) or any(
                not isinstance(pair, list) or len(pair) != 2 or
                not all(type(uid) is int for uid in pair) or pair[0] >= pair[1]
                for pair in gold) or len({tuple(pair) for pair in gold}) != len(gold)):
            raise ValueError(f"invalid published OOLONG-Pairs gold for question {task_id}")
        queries[task_id] = question, gold
    if set(queries) != set(range(1, 21)):
        raise ValueError("expected the 20 Appendix D.1 OOLONG-Pairs questions")
    return queries


def materialize_oolong(builder, inputs, *, trec, pairs, questions_path=None, pairs_reference=None):
    queries = _appendix_rows(questions_path) if pairs and pairs_reference is None else {}
    published_windows = {}
    seen_ids, windows = set(), {}
    counts = dict(oolong=0, pairs=0)
    fields = ("id, dataset, context_len, context_window_id, context_window_text, "
              "context_window_text_with_labels, question, answer, answer_type")
    for path in inputs:
        for row in _rows(path, fields=fields, predicate="dataset = 'trec_coarse'"):
            if row.get("dataset", "trec_coarse") != "trec_coarse":
                continue
            size = row["context_len"]
            if type(size) is not int or size < 1:
                raise ValueError("invalid OOLONG context_len")
            row_id = _id(row["id"])
            window = _id(row["context_window_id"])
            text = _text(row["context_window_text"], "context_window_text")
            if row_id in seen_ids:
                raise ValueError(f"duplicate OOLONG row ID: {row_id}")
            seen_ids.add(row_id)
            context_key = f"window-{size}-{window}"
            if trec:
                builder.add(f"oolong-{size}-{row_id}", "OOLONG", size, text,
                            _text(row["question"], "question"), oolong_gold(row["answer"]),
                            {"type": "oolong_author", "answer_type": row["answer_type"],
                             "raw_answer": row["answer"]},
                            "oolongbench/oolong-synth:validation", UNVERIFIED, row_id,
                            expected_source_revision=OOLONG_REV,
                            context_key=context_key, context_window_id=window,
                            raw_answer=row["answer"])
                counts["oolong"] += 1
            if pairs:
                if pairs_reference is not None:
                    fingerprint = digest(text)
                    if published_windows.setdefault(size, fingerprint) != fingerprint:
                        raise ValueError(f"ambiguous published OOLONG-Pairs context length {size}; "
                                         "supply only the matching trec_coarse window")
                    if size in windows:
                        continue
                    windows[size] = fingerprint
                    published = pairs_reference / f"oolong-pairs-{size}.json"
                    for task_id, (question, gold) in sorted(_appendix_rows(published, published=True).items()):
                        task = builder.add(f"pairs-{size}-{task_id}", "OOLONG-Pairs", size, text,
                                           question, gold, "pairs_f1", "mit-oasys/oolong-pairs",
                                           UNVERIFIED, task_id, context_key=context_key,
                                           expected_source_revision=PAIRS_REV,
                                           context_window_id=window, appendix_task_id=task_id,
                                           expected_context_source_revision=OOLONG_REV,
                                           deviation="published gold with caller-supplied matching context; "
                                                     "local set-F1 adapter")
                        builder.all_pairs.append(task)
                        counts["pairs"] += 1
                    continue
                labeled = _text(row["context_window_text_with_labels"], "labeled context")
                fingerprints = digest(text), digest(labeled)
                if context_key in windows:
                    if windows[context_key] != fingerprints:
                        raise ValueError(f"inconsistent OOLONG context window: {context_key}")
                    continue
                windows[context_key] = fingerprints
                users = labeled_users(labeled)
                for task_id, (question, _) in sorted(queries.items()):
                    task = builder.add(f"pairs-{size}-{window}-{task_id}", "OOLONG-Pairs", size,
                                       text, question, pairs_gold(users, task_id), "pairs_f1",
                                       "oolongbench/oolong-synth:validation + arXiv:2512.24601v2 Appendix D.1",
                                       UNVERIFIED, row_id, context_key=context_key,
                                       expected_source_revision=OOLONG_REV,
                                       context_window_id=window, appendix_task_id=task_id,
                                       labeled_context_sha256=fingerprints[1],
                                       deviation="reconstructed gold and local set-F1 adapter")
                    builder.all_pairs.append(task)
                    counts["pairs"] += 1
    if (trec and not counts["oolong"]) or (pairs and not counts["pairs"]):
        raise ValueError("no OOLONG trec_coarse rows in selected input")


def materialize_codeqa(builder, path):
    seen = set()
    fields = ('"_id", domain, sub_domain, difficulty, length, question, '
              'choice_A, choice_B, choice_C, choice_D, answer, context')
    for row in _rows(path, fields=fields):
        # The normalized CodeQA-only export may omit domain metadata.
        if ("domain" in row or "sub_domain" in row) and (
                row.get("domain"), row.get("sub_domain")) != (CODEQA_DOMAIN, CODEQA_SUBDOMAIN):
            continue
        row_id = _id(row["_id"])
        if row_id in seen:
            raise ValueError(f"duplicate CodeQA row ID: {row_id}")
        seen.add(row_id)
        text = _text(row["context"], "context")
        gold = row["answer"]
        if gold not in ("A", "B", "C", "D"):
            raise ValueError(f"invalid CodeQA answer key for {row_id}")
        prompt = _text(row["question"], "question") + "\n" + "\n".join(
            f"{letter}. {_text(row['choice_' + letter], 'choice')}" for letter in "ABCD"
        ) + "\nAnswer with one letter A, B, C, or D."
        task = builder.add(f"codeqa-{row_id}", "CodeQA", (len(text) + 3) // 4,
                           text, prompt, gold, "choice", "zai-org/LongBench-v2",
                           UNVERIFIED, row_id, expected_source_revision=CODEQA_REV,
                           difficulty=row.get("difficulty"),
                           source_length=row.get("length"),
                           size_basis="ceil(context characters / 4) heuristic; not tokenizer count")
        builder.optional.append(task)
    if not seen:
        raise ValueError("no LongBench-v2 CodeQA rows in selected input")


def materialize_browsecomp(builder, queries_path, selection_path, corpus_paths, *, exact_diagnostic=False):
    queries = {}
    for row in read_jsonl(queries_path):
        key = _id(row["query_id"])
        if key in queries:
            raise ValueError(f"duplicate BrowseComp query ID: {key}")
        queries[key] = row
    selections = {}
    required_background = set()
    for row in read_jsonl(selection_path):
        key = _id(row["query_id"])
        if key in selections or key not in queries:
            raise ValueError(f"duplicate or missing BrowseComp query ID: {key}")
        query = queries[key]
        ids = [_id(docid) for docid in row["docids"]]
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise ValueError(f"BrowseComp query {key} must select exactly 1000 distinct documents")
        evidence = {}
        for collection in ("gold_docs", "evidence_docs"):
            for document in query[collection]:
                docid = _id(document["docid"])
                text = _text(document["text"], "evidence text")
                if evidence.setdefault(docid, text) != text:
                    raise ValueError(f"inconsistent BrowseComp evidence document: {docid}")
        required = set(evidence) | {_id(docid) for docid in row.get("required_docids", [])}
        if not required <= set(ids):
            raise ValueError(f"BrowseComp query {key} is missing a required document")
        selections[key] = ids, evidence, row
        required_background.update(set(ids) - evidence.keys())
    if not selections:
        raise ValueError("no BrowseComp document selections in input")
    # Selected background texts can span much of the public corpus. Spool to
    # private temporary storage rather than retaining every document in RAM.
    with tempfile.TemporaryDirectory(prefix="rlm-browsecomp-") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "docs.sqlite3")) as db:
            db.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
            seen = set()
            for row in itertools.chain.from_iterable(
                    _rows(p, fields="docid, text") for p in corpus_paths):
                docid = _id(row["docid"])
                if docid not in required_background:
                    continue
                text = _text(row["text"], "corpus text")
                if docid in seen:
                    if db.execute("SELECT text FROM docs WHERE id=?", (docid,)).fetchone()[0] != text:
                        raise ValueError(f"inconsistent BrowseComp corpus document: {docid}")
                else:
                    db.execute("INSERT INTO docs VALUES (?, ?)", (docid, text))
                    seen.add(docid)
            if missing := required_background - seen:
                raise ValueError(f"missing {len(missing)} selected BrowseComp corpus documents")
            for key, (ids, evidence, row) in selections.items():
                query = queries[key]
                text = "\n\n".join(f"[document {docid}]\n" +
                    (evidence[docid] if docid in evidence else
                     db.execute("SELECT text FROM docs WHERE id=?", (docid,)).fetchone()[0])
                    for docid in ids)
                builder.add(f"browsecomp-{key}", "BrowseComp-Plus", (len(text) + 3) // 4,
                            text, _text(query["query"], "query"), _text(query["answer"], "answer"),
                            "browse_exact" if exact_diagnostic else "browse_official_unavailable",
                            "Tevatron/browsecomp-plus", UNVERIFIED, key,
                            expected_source_revision=BROWSE_REV,
                            expected_corpus_revision=CORPUS_REV, document_ids=ids,
                            size_basis="ceil(context characters / 4) heuristic; not tokenizer count",
                            deviation=row.get("deviation", "caller-supplied 1K selection, not original paper sample") +
                            ("; deterministic exact match, not the official LLM judge" if exact_diagnostic
                             else "; official semantic judge not included"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="local normalized reference tree")
    parser.add_argument("--out", type=Path, required=True, help="private evaluator output")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--oolong-input", type=Path, action="append", help="local JSONL or Parquet, repeatable")
    pairs_inputs = parser.add_mutually_exclusive_group()
    pairs_inputs.add_argument("--pairs-questions", type=Path, help="Appendix D.1 question JSON for reconstructed gold")
    pairs_inputs.add_argument("--pairs-reference-dir", type=Path,
                              help="author oolong-pairs-{size}.json files and caller-matched context")
    parser.add_argument("--codeqa-input", type=Path, help="CodeQA JSONL or full LongBench-v2 Parquet")
    parser.add_argument("--browse-queries", type=Path, help="local decoded query JSONL")
    parser.add_argument("--browse-selection", type=Path, help="local 1000-doc selection JSONL")
    parser.add_argument("--browse-corpus", type=Path, action="append", help="local decoded JSONL/Parquet, repeatable")
    parser.add_argument("--browse-exact-diagnostic", action="store_true",
                        help="opt in to non-official deterministic exact match for BrowseComp")
    args = parser.parse_args(argv)
    data = args.reference / "data"
    builder = TaskBuilder(args.reference, args.reference, args.out)
    if "oolong" in args.datasets or "oolong-pairs" in args.datasets:
        materialize_oolong(builder, args.oolong_input or [data / "OOLONG-trec-coarse-small.jsonl"],
                           trec="oolong" in args.datasets, pairs="oolong-pairs" in args.datasets,
                           questions_path=args.pairs_questions or data / "OOLONG-Pairs-20-queries.json",
                           pairs_reference=args.pairs_reference_dir)
    if "codeqa" in args.datasets:
        materialize_codeqa(builder, args.codeqa_input or data / "LongBench-v2-CodeQA.jsonl")
    if "browsecomp-plus" in args.datasets:
        selection = args.browse_selection or data / "BrowseComp-Plus-20Q-1000docs-manifest.jsonl"
        corpus = args.browse_corpus or sorted({
            args.reference / row["background_parquet"] for row in read_jsonl(selection)
            if row.get("background_parquet")})
        materialize_browsecomp(builder, args.browse_queries or data / "BrowseComp-Plus-20Q.jsonl",
                               selection, corpus, exact_diagnostic=args.browse_exact_diagnostic)
    print(json.dumps(builder.save(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
