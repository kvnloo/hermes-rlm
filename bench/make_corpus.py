"""Deterministic long-context benchmark corpus with exact gold answers (S-NIAH + OOLONG-style).

Records look like ``[r000123] 2026-03-04 u417: <message>``. Messages are paraphrased support
tickets in five categories, so aggregation needs reading/classifying, not one grep. One record holds
a needle. ``python make_corpus.py <out_dir> <approx_tokens> [seed]`` writes <out_dir>/corpus.txt and <out_dir>.gold.json (outside the agents' cwd).

reuse-checked: OOLONG / S-NIAH upstream generators -- need network/datasets; this is offline,
seeded and sized to any token count. Wired as rlm/bench/run_bench.py's input.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CATS = {
    "billing": ["I was charged twice for {thing} this month", "the invoice for {thing} shows the wrong amount",
                "why did my card get billed again for {thing}?", "refund still missing for {thing}",
                "pricing on {thing} changed without notice and I paid more"],
    "shipping": ["where is my {thing}? tracking hasn't moved", "{thing} arrived at the wrong address",
                 "can you ship {thing} faster?", "the courier lost my {thing}", "delivery of {thing} is a week late"],
    "praise": ["love the new {thing}, great job", "{thing} works perfectly, thank you team",
               "support fixed {thing} in minutes, amazing", "honestly the best {thing} I've used",
               "kudos on {thing}, it made my week"],
    "bug": ["{thing} crashes when I open settings", "{thing} freezes after the update",
            "error 500 every time I save in {thing}", "{thing} logs me out randomly",
            "the export button in {thing} does nothing"],
    "feature": ["could {thing} support dark mode?", "please add CSV export to {thing}",
                "it would be great if {thing} had keyboard shortcuts", "any plans for {thing} on Android?",
                "wish {thing} could sync across devices"],
}
THINGS = ["the app", "my subscription", "the dashboard", "the desktop client", "the API", "the planner",
          "the photo editor", "the music player", "the calendar", "the notes app"]


def build(tokens: int, seed: int = 7):
    rng = random.Random(seed)
    users = [f"u{n:03d}" for n in range(1, 401)]
    lines, labels = [], []
    target_chars = tokens * 4
    size = 0
    while size < target_chars:
        i = len(lines)
        cat = rng.choice(list(CATS))
        user = rng.choice(users)
        msg = rng.choice(CATS[cat]).format(thing=rng.choice(THINGS))
        line = f"[r{i:06d}] 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {user}: {msg}"
        lines.append(line)
        labels.append((user, cat))
        size += len(line) + 1
    code = "-".join(str(rng.randint(0, 9)) for _ in range(5))
    pos = rng.randint(len(lines) // 3, len(lines) - 1)
    lines[pos] = (f"[r{pos:06d}] 2026-07-14 u999: unrelated, but the vault access code is {code}; "
                  "keep it secret")
    labels[pos] = ("u999", "needle")
    per_user = defaultdict(Counter)
    for user, cat in labels:
        per_user[user][cat] += 1
    bug_user, bug_n = max(((u, c["bug"]) for u, c in per_user.items()), key=lambda x: (x[1], x[0]))
    tie = [u for u, c in per_user.items() if c["bug"] == bug_n]
    both = sorted(u for u, c in per_user.items() if c["billing"] and c["praise"])
    cats = Counter(cat for _, cat in labels if cat != "needle")
    questions = [
        {"id": "niah", "q": "What is the vault access code mentioned in the log?", "gold": code},
        {"id": "count_billing", "q": "How many tickets in the log are billing complaints "
         "(charges, invoices, refunds, pricing)? Answer with one integer.", "gold": cats["billing"]},
        {"id": "top_bug_user", "q": "Which user id filed the most bug reports (crashes, freezes, errors, "
         "logouts, broken buttons), and how many? Answer as '<user> <count>'.",
         "gold": f"{bug_user} {bug_n}", "ties": sorted(tie)},
    ]
    return "\n".join(lines) + "\n", questions, {"records": len(lines), "chars": len("\n".join(lines)),
                                                 "approx_tokens": tokens, "seed": seed, "categories": cats}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    text, qs, meta = build(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    (out / "corpus.txt").write_text(text)
    (out.parent / f"{out.name}.gold.json").write_text(json.dumps({"meta": meta, "questions": qs}, indent=2))
    print(json.dumps(meta))
