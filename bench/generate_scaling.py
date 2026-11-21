"""Generate three deterministic synthetic task families at 2^13–2^20 tokens.

Usage: python -m bench.generate_scaling --tokenizer-json /path/to/tokenizer.json --out /tmp/sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.paper_tasks import TaskBuilder
from bench.scaling_tokenizer import LocalTokenizer
from bench.synthetic_long_context import _aggregation, _multihop, _needle


EXPONENTS = tuple(range(13, 21))
REVISION = "synthetic-token-scaling-v1"
GENERATORS = (("needle", _needle), ("multihop", _multihop), ("aggregation", _aggregation))
FILLERS = (" padding", " filler", " a", " x", " 0", " .")


def _fit_context(tokenizer, make, target, variant):
    # Increase substantive records with length. The initial ratio only
    # proposes a record count; every admitted context is actually encoded.
    initial = 32 + 4 * variant
    sample, _, _ = make(initial)
    sample_count = tokenizer.count(sample)
    if not sample_count:
        raise ValueError("tokenizer encoded the substantive context to zero tokens")
    records = max(16, initial * (target * 85 // 100) // sample_count + 4 * variant)
    for _ in range(8):
        core, question, gold = make(records)
        body = f"SYNTHETIC TRIAL {variant}\n" + core + "\nDISTRACTOR PADDING; no task records follow.\n"
        count = tokenizer.count(body)
        if count <= target:
            break
        records = min(records - 1, records * (target * 85 // 100) // count)
        if records < 16:
            raise ValueError("token target is too small for the selected tokenizer")
    else:
        raise ValueError("could not fit the generated records into the token target")
    if count == target:
        return body, question, gold, records, 0
    for filler in FILLERS:
        first = tokenizer.count(body + filler) - count
        stride = (tokenizer.count(body + filler * 17) - count - first)
        if first < 1 or stride <= 0 or stride % 16:
            continue
        unit = stride // 16
        remaining = target - count - first
        if remaining < 0 or remaining % unit:
            continue
        copies = 1 + remaining // unit
        context = body + filler * copies
        if tokenizer.count(context) == target:
            return context, question, gold, records, target - count
    raise ValueError("selected tokenizer cannot make exact token targets with the supported padding pieces")


def generate(out: Path, tokenizer: LocalTokenizer, exponents=EXPONENTS, variants=2):
    if not exponents or set(exponents) - set(EXPONENTS) or len(set(exponents)) != len(exponents):
        raise ValueError("exponents must be distinct values between 13 and 20")
    if type(variants) is not int or not 2 <= variants <= 8:
        raise ValueError("variants must be between 2 and 8")
    builder = TaskBuilder(out, out, out)
    for exponent in sorted(exponents):
        target = 2 ** exponent
        for name, make in GENERATORS:
            for variant in range(variants):
                context, question, gold, records, padding_tokens = _fit_context(tokenizer, make, target, variant)
                builder.add(f"scale-{name}-{exponent}-{variant}", f"synthetic-{name}", target,
                            context, question, gold, "browse_exact", "local deterministic generator",
                            REVISION, f"{name}/{exponent}/{variant}",
                            size_basis="measured tokenizer JSON; special tokens excluded",
                            tokenizer_sha256=tokenizer.sha256, tokenizer_engine=tokenizer.engine,
                            target_exponent=exponent, variant=variant, records=records,
                            padding_tokens=padding_tokens)
    return builder.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--exponents", nargs="+", type=int, choices=EXPONENTS, default=EXPONENTS)
    parser.add_argument("--variants", type=int, choices=range(2, 9), default=2)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.out, LocalTokenizer(args.tokenizer_json), args.exponents, args.variants),
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
