"""A/B tasks for hermes-rlm: plain Hermes (file + terminal tools) vs Hermes + RLM (rlm toolset).

Both arms get the same question about the same unlabeled context.txt in their working
directory. Plain Hermes may read, grep or script over the file however it likes. The RLM arm
uses the plugin as documented: rlm_load, then rlm_exec with optional llm_query/llm_batch
sub-calls, then a cited FINAL. Three tasks reuse the repo's offline generator
(bench/synthetic_long_context.py, 512 records); `reviews` adds a semantic count that a regex
cannot solve, which is the case RLM's sub-calls exist for. Gold never enters a prompt or the
working directory. Scoring is exact match after trimming formatting.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench.synthetic_long_context import _aggregation, _multihop, _needle

REVISION = "rlm-ab-v1"
RECORDS = 512


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    build: Callable[[Path], None]
    score: Callable[[Path, str], dict]


SHIP_BAD = ["Took nearly a month to reach my door.", "The courier tossed it over the fence and the box was crushed.",
            "Tracking said delivered, but nothing showed up for another week.", "It arrived way past the promised date.",
            "The parcel sat in some depot for 19 days.", "It showed up late and the box was soaked through.",
            "I waited three weeks for something listed as two-day.", "The package got lost twice before it finally came.",
            "Getting it here was a disaster from start to finish.", "Painfully slow to arrive.",
            "It was meant to come Tuesday; it came the following Friday.", "The box arrived dented and half open."]
SHIP_GOOD = ["Delivery was fast.", "It arrived a day early and was well packed.",
             "Shipping was quick and the courier was polite.", "Showed up the next morning.",
             "Tracking was accurate and it came right on time.", "The box was intact and arrived when promised."]
OTHER = ["The battery dies by lunchtime.", "The screen scratches if you look at it wrong.",
         "Way overpriced for what it is.", "Support never answered my emails.", "Works exactly as described.",
         "My kids love it.", "The manual is useless.", "Feels solid and well made.", "It gets warm after an hour.",
         "Setup took two minutes.", "The color is not what the photos showed.", "Great value for the money.",
         "Stopped working after a week, but the replacement is fine.", "The app keeps logging me out."]
PRODUCTS = ["Kettle K2", "Lamp L9", "Router R5", "Blender B3", "Speaker S7", "Charger C1", "Fan F4", "Scale S2"]


def _reviews():
    r = random.Random(int(hashlib.sha256(f"{REVISION}|reviews".encode()).hexdigest()[:16], 16))
    lines = ["CUSTOMER REVIEWS - one review per line."]
    gold = 0
    for i in range(600):
        parts = r.sample(OTHER, r.randint(1, 2))
        bad = r.random() < 0.22
        if bad:
            parts.append(r.choice(SHIP_BAD))
            gold += 1
        elif r.random() < 0.3:
            parts.append(r.choice(SHIP_GOOD))
        r.shuffle(parts)
        lines.append(f"REVIEW {i:04d} | product={r.choice(PRODUCTS)} | rating={r.randint(1, 5)} | text={' '.join(parts)}")
    question = ("How many reviews complain about shipping or delivery (late, slow, lost, or damaged in transit)? "
                "Count reviews, not sentences. Return only the integer.")
    return "\n".join(lines) + "\n", question, str(gold)


def _normalize(s: str) -> str:
    s = s.strip().strip("`*\"' ").rstrip(".").strip("`*\"' ")
    return re.sub(r"\s+", " ", s).casefold()


def extract(answer: str) -> str:
    """Plain arm: text after the last ANSWER:. RLM arm: the validated FINAL, before its Sources block."""
    if "ANSWER:" in answer:
        answer = answer.rsplit("ANSWER:", 1)[1]
    answer = answer.split("\n\nSources:", 1)[0]
    return _normalize(answer.strip().splitlines()[0] if answer.strip() else "")


def _task(tid: str, make) -> Task:
    context, question, gold = make()

    def build(wd: Path) -> None:
        (wd / "context.txt").write_text(context)

    def score(wd: Path, answer: str) -> dict:
        got = extract(answer)
        return {"pass": got == _normalize(gold), "detail": f"got {got[:60]!r}"}

    return Task(tid, question, build, score)


TASKS = [
    _task("needle", lambda: _needle(RECORDS)),
    _task("multihop", lambda: _multihop(RECORDS)),
    _task("aggregation", lambda: _aggregation(RECORDS)),
    _task("reviews", _reviews),
]

RLM_SETTINGS = {"max_depth": 1, "max_subcalls": 60, "max_tokens": 2000, "timeout_s": 300, "max_cost": None,
                "subcall_task": "rlm_subcall", "concurrency": 4}
ARMS = {
    "plain": {"toolsets": ["file", "terminal"],
              "prompt_template": "The local file context.txt in your working directory holds the document.\n"
                                 "Question: {prompt}\nEnd your reply with a final line: ANSWER: <answer>"},
    "rlm": {"toolsets": ["rlm"], "plugin_dir": "rlm", "plugin_name": "rlm", "aux_tasks": ["rlm_subcall"],
            "plugin_entries": {"rlm": {"settings": RLM_SETTINGS}},
            # Same wording as bench/paper_matrix.py task_prompt for its depth-one RLM arms.
            "prompt_template": "Task input is the local file context.txt. The root prompt contains only this metadata, "
                               "not the raw context. Call rlm_load with paths=['context.txt'], then use rlm_exec to "
                               "inspect context. For semantic work use llm_query/llm_batch over relevant chunks and "
                               "aggregate in Python. \nQuestion: {prompt}\nCompute the answer in a REPL variable named "
                               "answer. Your last rlm_exec cell must call FINAL_VAR('answer', citations=[cite('context.txt', "
                               "start, end)]). Use real supporting zero-based, end-exclusive character offsets. Ordinary "
                               "final prose is not an answer and will be withheld."},
}
