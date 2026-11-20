"""Deterministic, evaluator-only paper task preparation and scoring.

reuse-checked: bench/make_corpus.py and run_bench.score; pinned RULER metrics,
OOLONG eval_helpers.py, appendix E.1 queries, and reference/build_reference.py.

Run with --reference DIR --legacy DIR --out DIR. Defaults materialize only small
convenience exports. Use --large on fleet for Parquet extraction; --browsecomp
also requires fleet memory for a roughly 484 MiB Parquet page. No network calls.
Manifests and labeled sources must stay outside the agent cwd. Give an agent only
its context_path's parent and question, never this module's task dictionary.
size_tokens uses source nominal labels, or an explicit characters/4 estimate
where the source supplies no numeric size. It is NOT a root tokenizer count.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import itertools
import json
from pathlib import Path
import re

OOLONG_REV = "49898a421f4b14f2c9cae084d2d270f930ff4c90"
RULER_REV = "5da49e4000224d3289ffb14fc32cbd20211765ed"
CODEQA_REV = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
BROWSE_REV = "144cff8e35b5eaef7e526346aa60774a9deb941f"
CORPUS_REV = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
SIZES = (1024, 32768, 131072, 1048576)
LABELS = ("description and abstract concept", "entity", "human being",
          "numeric value", "location", "abbreviation")
D, E, H, N, L, A = LABELS


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_text(answer):
    """Last ANSWER line, otherwise the whole response; no arbitrary tail limit."""
    text = str(answer or "").strip()
    matches = list(re.finditer(r"(?im)^\s*ANSWER:\s*", text))
    return text[matches[-1].end():].strip() if matches else text


def oolong_parse(answer):
    """Port of synth_attempt_answer_parse, including its case-sensitive quirks."""
    if ":" not in answer:
        return answer if len(answer) < 20 else (answer.split()[-1] if answer.split() else "")
    candidate = answer.split(":")[-1].strip()
    for char in "*[]":
        candidate = candidate.replace(char, "")
    if len(candidate) >= 20:
        for value in ("more common", "less common", "same frequency"):
            if value in candidate:
                return value
    return candidate


def oolong_gold(raw):
    if "datetime" in raw:
        return datetime.strptime(raw, "[datetime.date(%Y, %m, %d)]").isoformat()
    return ast.literal_eval(raw)[0]


def parse_pairs(answer):
    """Accept integer tuple/list pairs; normalize, reject self-pairs, dedup.

    A malformed response with no pairs is not an explicit empty prediction.
    Empty predictions must be [], {}, set(), none, or no pairs.
    """
    text = answer_text(answer)
    pairs = set()
    matches = re.findall(r"[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]", text)
    for left, right in matches:
        a, b = int(left), int(right)
        if a == b:
            return None
        pairs.add(tuple(sorted((a, b))))
    if not matches and text.strip().lower() not in ("[]", "{}", "set()", "none", "no pairs", "no pairs."):
        return None
    return pairs


class DateParserUnavailable(ValueError):
    """Date normalization cannot run without its optional evaluator dependency."""


def score_task(task: dict, answer: str) -> float:
    """Return a deterministic score in [0, 1]; no judge or model access."""
    scoring = task["scoring"]
    kind = scoring["type"] if isinstance(scoring, dict) else scoring
    gold = task["gold"]
    if kind == "oolong":
        prediction = oolong_parse(str(answer or ""))
        if str(prediction) == str(gold):
            return 1.0
        if prediction in ("more common", "less common", "same frequency") and prediction in str(gold):
            return 1.0
        answer_type = scoring["answer_type"]
        if answer_type.endswith("NUMERIC"):
            try:
                return 0.75 ** abs(int(gold) - int(prediction))
            except (ValueError, TypeError, OverflowError):
                # The mathematically tiny score rounds to zero at this distance.
                return 0.0
        if answer_type.endswith("DATE"):
            # The upstream helper uses dateutil. Pin default to prevent today's
            # date from affecting incomplete dates, and reject incomplete dates.
            # A missing dateutil must raise, not silently score every date 0.
            try:
                from dateutil.parser import parse
            except ImportError as exc:
                raise DateParserUnavailable("OOLONG date scoring requires python-dateutil") from exc
            try:
                first = parse(prediction, default=datetime(2000, 1, 1))
                second = parse(prediction, default=datetime(2001, 2, 2))
                return float(first == second == datetime.fromisoformat(gold))
            except (ValueError, OverflowError):
                return 0.0
        return 0.0
    text = answer_text(answer)
    if kind == "ruler":
        return sum(str(token).lower() in text.lower() for token in gold) / len(gold)
    if kind == "pairs_f1":
        predicted = parse_pairs(answer)
        if predicted is None:
            return 0.0
        expected = {tuple(sorted(map(int, pair))) for pair in gold}
        if not expected and not predicted:
            return 1.0
        return 2 * len(expected & predicted) / (len(expected) + len(predicted))
    if kind == "choice":
        cleaned = text.strip().strip("*` ")
        match = re.fullmatch(r"(?:\(?([A-D])\)?[.)]?|(?:The )?(?:correct )?(?:answer|option|choice)(?: is|:)\s*\(?([A-D])\)?[.)]?)", cleaned, re.I)
        return float(bool(match) and (match[1] or match[2]).upper() == gold)
    if kind == "browse_exact":
        normalize = lambda value: " ".join(str(value).casefold().split())
        return float(normalize(text) == normalize(gold))
    if kind == "legacy":
        # Same scoring as run_bench.score, with the original gold schema.
        matches = re.findall(r"ANSWER:\s*(.+)", answer or "")
        candidate = (matches[-1] if matches else (answer or "")[-300:]).strip().lower()
        qid = scoring["question_id"]
        if qid == "niah":
            return float(re.sub(r"\D", "", str(gold)) in re.sub(r"\D", "", candidate))
        if qid == "count_billing":
            nums = [int(n) for n in re.findall(r"\d[\d,]*", candidate.replace(",", ""))]
            return float(gold in nums)
        if qid == "top_bug_user":
            users = re.findall(r"u\d{3}", candidate)
            count = str(gold).split()[1]
            return float(bool(users) and users[0] in scoring["ties"] and re.search(rf"\b{count}\b", candidate) is not None)
    raise ValueError(f"Unknown scorer: {kind}")


def labeled_users(text):
    """Parse only labeled dated rows; fail closed rather than dropping bad rows."""
    users = defaultdict(lambda: defaultdict(list))
    for line in text.splitlines():
        if not line.startswith("Date:"):
            continue
        match = re.fullmatch(r"Date: (.*?) \|\| User: (\d+) \|\| Instance: .* \|\| Label: (.*)", line)
        if not match or match[3] not in LABELS:
            raise ValueError("Invalid labeled TREC record")
        when = datetime.strptime(match[1], "%b %d, %Y").date()
        users[int(match[2])][match[3]].append(when)
    if not users:
        raise ValueError("No labeled TREC records")
    return users


def pair_predicates(task_id):
    """Appendix E.1 literal predicates; universal dates are vacuously true."""
    if not 1 <= task_id <= 20:
        raise ValueError("Pairs task ID must be 1..20")
    unions = [(N, L), (E, H), (D, A), (H, L), (E, N),
              (L, A), (D, N), (H, D), (E, L), (N, A)]
    dates = {4: (H, date(2023, 1, 6), "after"),
             5: (E, date(2023, 3, 15), "before"),
             7: (N, date(2023, 2, 1), "after"),
             9: (L, date(2023, 4, 10), "after"),
             10: (A, date(2023, 5, 20), "before")}
    if task_id <= 10:
        def eligible(user):
            if not any(user.get(label) for label in unions[task_id - 1]):
                return False
            if task_id not in dates:
                return True
            label, boundary, direction = dates[task_id]
            return all(day > boundary if direction == "after" else day < boundary
                       for day in user.get(label, []))
        return eligible, eligible
    # Each requirement is category, minimum, optional exact count.
    requirements = {
        11: (((E, 1, None), (A, 1, None)), ((E, 1, 1),)),
        12: (((N, 2, None),), ((L, 1, None), (H, 1, None))),
        13: (((D, 1, 1),), ((A, 1, None), (E, 1, None))),
        14: (((H, 1, None), (N, 1, None)), ((L, 2, 2),)),
        15: (((E, 1, None), (L, 1, None), (A, 1, None)), ((N, 1, 1),)),
        16: (((D, 1, None), (H, 1, None)), ((E, 2, None), (A, 1, 1))),
        17: (((N, 1, 1),), ((L, 1, None), (D, 1, None))),
        18: (((A, 1, None), (H, 1, 1)), ((E, 1, None), (N, 1, None))),
        19: (((L, 2, None), (E, 1, None)), ((D, 1, 1), (A, 1, 1))),
        20: (((N, 1, None), (H, 1, None)), ((L, 1, None), (E, 1, None), (A, 1, 1))),
    }
    def predicate(requirement):
        return lambda user: all(len(user.get(label, [])) >= minimum and
                                (exact is None or len(user.get(label, [])) == exact)
                                for label, minimum, exact in requirement)
    return tuple(predicate(requirement) for requirement in requirements[task_id])


def pairs_gold(users, task_id):
    first, second = pair_predicates(task_id)
    left = [uid for uid in sorted(users) if first(users[uid])]
    right = [uid for uid in sorted(users) if second(users[uid])]
    return [list(pair) for pair in sorted({tuple(sorted((a, b)))
                                          for a in left for b in right if a != b})]


def parquet_rows(reference, files, columns="*", where="TRUE", limit=3):
    import duckdb
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit='768MB'")
        # No spill files outside the explicit output tree.
        connection.execute("SET max_temp_directory_size='0B'")
        cursor = connection.execute(
            f"SELECT {columns} FROM read_parquet(?) WHERE {where} LIMIT {int(limit)}",
            [[str(reference / path) for path in files]])
        names = [item[0] for item in cursor.description]
        for values in cursor.fetchall():
            yield dict(zip(names, values))
    finally:
        connection.close()


class TaskBuilder:
    def __init__(self, reference, legacy, out):
        self.reference, self.legacy, self.out = map(lambda p: Path(p).resolve(), (reference, legacy, out))
        self.tasks = []
        self.optional = []
        self.all_pairs = []
        self.blocked = []

    def add(self, task_id, family, size, text, question, gold, scoring, source, revision,
            source_id, *, source_text=None, context_key=None,
            size_basis="source nominal; root tokenizer count not measured", **metadata):
        path = self.out / "contexts" / family / (context_key or task_id) / "context.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = text.encode("utf-8")
        if not path.exists() or path.read_bytes() != payload:
            path.write_bytes(payload)
        task = dict(id=task_id, family=family, size_tokens=size, context_path=str(path),
                    question=question, gold=gold, scoring=scoring,
                    metadata=dict(source=source, source_revision=revision, source_id=source_id,
                                  source_context_sha256=digest(text if source_text is None else source_text),
                                  context_sha256=hashlib.sha256(payload).hexdigest(), context_bytes=len(payload),
                                  size_basis=size_basis, **metadata))
        self.tasks.append(task)
        return task

    def niah(self, large=False):
        sources = [(4096, itertools.islice(read_jsonl(self.reference / "data/S-NIAH-ruler-4k-50.jsonl"), 3))]
        if large:
            sources.append((1048576, parquet_rows(self.reference,
                ["data/ruler-1Mrepo-niah_single_1_1M--validation-00000-of-00001.parquet"],
                columns='"index", input, answers, length', limit=3)))
        else:
            self.blocked.append(dict(family="S-NIAH", size_tokens=1048576,
                                     reason="1M Parquet extraction requires fleet --large"))
        for size, rows in sources:
            for row in rows:
                original = row["input"]
                # Preserve source hash, but remove the mirror's Llama chat wrapper
                # and answer prefill from the model-independent text context.
                body = original.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[-1]
                body = body.split("<|eot_id|>", 1)[0]
                question = re.search(r"What is the special magic number[^?]*\?", body)
                if not question:
                    raise ValueError("RULER question not found")
                text = body[:question.start()].rstrip() + "\n"
                self.add(f"sniah-{size}-{row['index']}", "S-NIAH", size, text, question[0],
                         row["answers"], "ruler", "self-long/RULER-llama3-1M", RULER_REV,
                         row["index"], source_text=original, mirror_length=row.get("length"),
                         deviation="third-party RULER instances; nominal Llama size")

    def oolong(self, large=False):
        queries = json.loads((self.reference / "data/OOLONG-Pairs-20-queries.json").read_text())
        if {q["task_id"] for q in queries} != set(range(1, 21)):
            raise ValueError("Expected all twenty literal appendix queries")
        small = list(itertools.islice((r for r in read_jsonl(self.reference / "data/OOLONG-trec-coarse-small.jsonl")
                                      if r["context_len"] == 1024), 3))
        sources = [(1024, small)]
        coverage = json.loads((self.reference / "metadata/oolong-validation-coverage.json").read_text())
        for size in SIZES[1:]:
            files = sorted({r["file"] for r in coverage if r["dataset"] == "trec_coarse" and r["context_len"] == size})
            if large:
                sources.append((size, parquet_rows(self.reference, files,
                    columns="id,context_len,context_window_id,context_window_text,context_window_text_with_labels,question,answer,answer_type,dataset",
                    where=f"dataset='trec_coarse' AND context_len={size}", limit=3)))
            else:
                self.blocked.append(dict(family="OOLONG/OOLONG-Pairs", size_tokens=size,
                                         reason="Parquet extraction requires fleet --large", source_files=files))
        for size, rows in sources:
            rows = sorted(rows, key=lambda row: row["id"])
            for row in rows:
                self.add(f"oolong-{size}-{row['id']}", "OOLONG", size, row["context_window_text"],
                         row["question"], oolong_gold(row["answer"]),
                         dict(type="oolong", answer_type=row["answer_type"]),
                         "oolongbench/oolong-synth:validation", OOLONG_REV, row["id"],
                         context_key=f"window-{size}-{row['context_window_id']}",
                         context_window_id=row["context_window_id"], raw_answer=row["answer"])
            if not rows:
                continue
            row = rows[0]
            users = labeled_users(row["context_window_text_with_labels"])
            for query in queries:
                task = self.add(f"pairs-{size}-{query['task_id']}", "OOLONG-Pairs", size,
                                row["context_window_text"], query["question"], pairs_gold(users, query["task_id"]),
                                "pairs_f1", "oolongbench/oolong-synth:validation + arXiv:2512.24601v1 appendix E.1",
                                OOLONG_REV, row["id"], context_key=f"window-{size}-{row['context_window_id']}",
                                context_window_id=row["context_window_id"], appendix_task_id=query["task_id"],
                                labeled_context_sha256=digest(row["context_window_text_with_labels"]),
                                deviation="reconstructed gold; missing-category universal dates vacuously true; empty/empty F1=1; malformed/self-pairs=0")
                self.all_pairs.append(task)
                if query["task_id"] not in (1, 11):
                    self.tasks.pop()

    def codeqa(self, all_rows=False):
        rows = read_jsonl(self.reference / "data/LongBench-v2-CodeQA.jsonl")
        for i, row in enumerate(rows if all_rows else itertools.islice(rows, 3)):
            question = row["question"] + "\n" + "\n".join(f"{c}. {row['choice_' + c]}" for c in "ABCD") + "\nAnswer with one letter A, B, C, or D."
            task = self.add(f"codeqa-{row['_id']}", "CodeQA", (len(row["context"]) + 3) // 4, row["context"], question,
                            row["answer"], "choice", "zai-org/LongBench-v2", CODEQA_REV, row["_id"],
                            difficulty=row["difficulty"], source_length=row["length"],
                            size_basis="ceil(context characters / 4) heuristic; NOT source or root tokenizer count")
            if i >= 3:
                self.tasks.pop()
            self.optional.append(task)
        if not all_rows:
            self.blocked.append(dict(family="CodeQA", selected_count=3, available_count=50,
                                     remaining_optional_count=47,
                                     reason="optional full50 extraction deferred to fleet --all-codeqa; selected first three source rows"))

    def local(self):
        for name in ("c32k", "c300k", "c2000k"):
            raw = json.loads((self.legacy / f"{name}.gold.json").read_text())
            text = (self.legacy / name / "corpus.txt").read_text(encoding="utf-8")
            for question in raw["questions"]:
                self.add(f"local-{name}-{question['id']}", "local-aggregation", raw["meta"]["approx_tokens"],
                         text, question["q"], question["gold"],
                         dict(type="legacy", question_id=question["id"], ties=question.get("ties", [])),
                         "make_corpus.py", "sha256:" + digest(Path(__file__).with_name("make_corpus.py").read_text()), question["id"],
                         context_key=name, source_corpus=name, source_meta=raw["meta"])

    def browsecomp(self, materialize=False):
        manifests = list(read_jsonl(self.reference / "data/BrowseComp-Plus-20Q-1000docs-manifest.jsonl"))
        if not materialize:
            self.blocked.extend(dict(family="BrowseComp-Plus", source_id=row["query_id"],
                                     document_count=len(row["docids"]), source_revision=BROWSE_REV,
                                     corpus_revision=CORPUS_REV,
                                     reason="1000-document extraction blocked on fleet availability; not a runnable task")
                                for row in manifests)
            return
        queries = {row["query_id"]: row for row in read_jsonl(self.reference / "data/BrowseComp-Plus-20Q.jsonl")}
        files = sorted({row["background_parquet"] for row in manifests})
        docs = {str(row["docid"]): row["text"] for row in parquet_rows(self.reference, files,
                 columns="docid,text", limit=1100)}
        for row in manifests:
            query = queries[row["query_id"]]
            available = dict(docs)
            available.update({str(doc["docid"]): doc["text"] for field in ("gold_docs", "evidence_docs") for doc in query[field]})
            if len(set(row["docids"])) != 1000 or not set(row["required_docids"]) <= set(row["docids"]):
                raise ValueError("Invalid BrowseComp document selection")
            text = "\n\n".join(f"[document {docid}]\n{available[str(docid)]}" for docid in row["docids"])
            self.add(f"browsecomp-{row['query_id']}", "BrowseComp-Plus", (len(text) + 3) // 4, text, query["query"],
                     query["answer"], "browse_exact", "Tevatron/browsecomp-plus", BROWSE_REV,
                     row["query_id"], corpus_revision=CORPUS_REV, document_ids=row["docids"],
                     size_basis="ceil(context characters / 4) heuristic; NOT source or root tokenizer count",
                     deviation=row["deviation"] + "; deterministic casefold whitespace exact match, NOT official LLM judge")

    def save(self):
        tasks = sorted(self.tasks, key=lambda row: (row["family"], row["size_tokens"] or 0, row["id"]))
        smoke = []
        for _, group in itertools.groupby(tasks, key=lambda row: row["family"]):
            smoke.append(next(group))
        write_jsonl(self.out / "smoke.jsonl", smoke)
        write_jsonl(self.out / "full.jsonl", tasks)
        write_jsonl(self.out / "pairs-all20.jsonl", self.all_pairs)
        write_jsonl(self.out / "codeqa-optional.jsonl", self.optional)
        write_jsonl(self.out / "blocked.jsonl", self.blocked)
        summary = dict(smoke=len(smoke), full=len(tasks), family_counts=dict(Counter(t["family"] for t in tasks)),
                       pairs_all20=len(self.all_pairs), codeqa_optional=len(self.optional), blocked=len(self.blocked),
                       repeats="runner owns >=3 repeats; manifests contain unique tasks only")
        (self.out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "legacy", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--large", action="store_true", help="fleet-only Parquet materialization")
    parser.add_argument("--browsecomp", action="store_true", help="fleet-only 1000-document extraction")
    parser.add_argument("--all-codeqa", action="store_true", help="prepare optional complete 50-task manifest")
    args = parser.parse_args()
    builder = TaskBuilder(args.reference, args.legacy, args.out)
    builder.niah(args.large)
    builder.oolong(args.large)
    builder.codeqa(args.all_codeqa)
    builder.local()
    builder.browsecomp(args.browsecomp)
    print(json.dumps(builder.save(), sort_keys=True))


if __name__ == "__main__":
    main()
