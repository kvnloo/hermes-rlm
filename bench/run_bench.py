"""A/B harness: Hermes+rlm vs plain Hermes vs PrimeIntellect nano-rlm on one corpus, same model.

  python run_bench.py <corpus_dir> <out.jsonl> --arms hermes-rlm,hermes-tools,nano-rlm \
      --provider kilo-free --model nvidia/nemotron-3.5-lightning:free [--repeats N] [--questions id,..]

Each row: arm, question, correct, answer, wall_s, root/sub tokens, api calls, subcalls. Arms are
interleaved per repeat so provider drift hits all of them. Gold comes from make_corpus.py.
Execution errors are unscored (`correct` is null), not incorrect answers.

reuse-checked: z0evals harness-speed-v1 -- times harnesses but has no corpus scoring or RLM arm.
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERMES_HOME = Path(os.environ.get("BENCH_HERMES_HOME", Path(__file__).resolve().parent / "home"))
NANO = Path(os.environ["NANO_RLM_DIR"]) if os.environ.get("NANO_RLM_DIR") else None
HERE = Path(__file__).resolve().parent
TOOLSETS = {"hermes-rlm": "rlm", "hermes-tools": "file,terminal,code_execution"}


def prompt_for(q: dict, meta: dict, corpus: Path) -> str:
    return (f"The file {corpus} is a support log of about "
            f"{meta['approx_tokens']:,} tokens ({meta['records']:,} records), too large to read whole. "
            f"Question: {q['q']} Work it out from the file, then end your reply with one line "
            "'ANSWER: <answer>'.")


def score(q: dict, answer: str) -> bool:
    m = re.findall(r"ANSWER:\s*(.+)", answer or "")
    a = (m[-1] if m else (answer or "")[-300:]).strip().lower()
    if q["id"] == "niah":
        return re.sub(r"\D", "", q["gold"]) in re.sub(r"\D", "", a)
    if q["id"] == "count_billing":
        nums = [int(n) for n in re.findall(r"\d[\d,]*", a.replace(",", ""))]
        return q["gold"] in nums
    if q["id"] == "top_bug_user":
        users = re.findall(r"u\d{3}", a)
        count = q["gold"].split()[1]
        return bool(users) and users[0] in q["ties"] and re.search(rf"\b{count}\b", a) is not None
    return False


def hermes_usage(session_id: str) -> dict:
    db = sqlite3.connect(f"file:{HERMES_HOME / 'state.db'}?mode=ro", uri=True)
    row = db.execute("select input_tokens, output_tokens, cache_read_tokens, api_call_count, "
                     "tool_call_count from sessions where id=?", (session_id,)).fetchone()
    keys = ("root_input_tokens", "root_output_tokens", "root_cache_read_tokens", "api_calls", "tool_calls")
    return dict(zip(keys, row)) if row else {}


def rlm_usage(log: Path) -> dict:
    out = {"subcalls": 0, "sub_prompt_tokens": 0, "sub_completion_tokens": 0, "cells": 0,
           "subcall_statuses": {}, "subcall_latency_ms": 0, "malformed_log_lines": 0,
           "validated_final": None}
    paths = sorted(log.glob("*.jsonl")) if log.is_dir() else ([log] if log.exists() else [])
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                out["malformed_log_lines"] += 1
                continue
            if e.get("event") == "subcall":
                out["subcalls"] += 1
                out["sub_prompt_tokens"] += e.get("prompt_tokens", 0)
                out["sub_completion_tokens"] += e.get("completion_tokens", 0)
                status = e.get("status", "unknown")
                out["subcall_statuses"][status] = out["subcall_statuses"].get(status, 0) + 1
                out["subcall_latency_ms"] += e.get("ms", 0)
            elif e.get("event") == "load":
                out["validated_final"] = None
            elif e.get("event") == "cell":
                out["cells"] += 1
                out["validated_final"] = None
                final, citations = e.get("final"), e.get("citations")
                if (not e.get("error") and isinstance(final, str) and final.strip()
                        and isinstance(citations, list) and citations
                        and all(isinstance(c, dict) and isinstance(c.get("source"), str)
                                and c["source"] and type(c.get("start")) is int
                                and type(c.get("end")) is int and 0 <= c["start"] < c["end"]
                                and isinstance(c.get("quote"), str)
                                and len(c["quote"]) == c["end"] - c["start"] for c in citations)):
                    out["validated_final"] = {"answer": final, "citations": citations}
    # A row must have one session trajectory, not a guessed ordering across sessions.
    if len(paths) != 1 or out["malformed_log_lines"]:
        out["validated_final"] = None
    return out


def run_hermes(arm, prompt, corpus_dir, args, run_dir):
    log_dir = run_dir / uuid.uuid4().hex
    log_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HERMES_HOME=str(HERMES_HOME), RLM_LOG_DIR=str(log_dir))
    cmd = ["hermes", "chat", "--oneshot", "-Q", "-t", TOOLSETS[arm], "--provider", args.provider,
           "-m", args.model, "-q", prompt]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=corpus_dir, env=env, capture_output=True, text=True, timeout=args.timeout)
    wall = time.monotonic() - t0
    text = p.stdout + p.stderr
    sid = (re.findall(r"session_id:\s*(\S+)", text) or [""])[-1]
    answer = text.split("session_id:")[0].strip()
    row = {"answer": answer[-2000:], "wall_s": round(wall, 2), "session_id": sid, **hermes_usage(sid)}
    if arm == "hermes-rlm":
        row.update(rlm_usage(log_dir))
    return row


def kilo_credentials(provider: str):
    import yaml
    cfg = yaml.safe_load((HERMES_HOME / "config.yaml").read_text())
    p = (cfg.get("providers") or {}).get(provider) or {}
    key = p.get("api_key") or os.environ.get(p.get("key_env", ""), "")
    return p["base_url"], key


def run_nano(prompt, corpus_dir, args, run_dir):
    if NANO is None:
        raise ValueError("Set NANO_RLM_DIR to the nano-rlm checkout")
    base, key = kilo_credentials(args.provider)
    env = dict(os.environ, NANO_BASE_URL=base, NANO_API_KEY=key, NANO_MODEL=args.model,
               NANO_RLM_BIN=str(NANO / ".venv/bin/rlm"), RLM_HOME=str(run_dir / "nano-home"))
    t0 = time.monotonic()
    p = subprocess.run([str(NANO / ".venv/bin/python"), str(HERE / "nano_client.py"), prompt],
                       cwd=corpus_dir, env=env, capture_output=True, text=True, timeout=args.timeout)
    wall = time.monotonic() - t0
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"answer": "", "wall_s": round(wall, 2), "error": (p.stderr or p.stdout)[-1500:]}
    snap = res.get("snapshot") or {}
    usage = snap.get("usage") or snap.get("token_usage") or {}
    return {"answer": res["answer"][-2000:], "wall_s": round(wall, 2), "nano_usage": usage,
            "nano_metrics": snap.get("metrics"), "nano_stop": snap.get("last_stop_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir"), ap.add_argument("out")
    ap.add_argument("--arms", default="hermes-rlm,hermes-tools,nano-rlm")
    ap.add_argument("--provider", default="kilo-free")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-lightning:free")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--questions", default="")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    corpus_dir = Path(args.corpus_dir).resolve()
    gold = json.loads((corpus_dir.parent / f"{corpus_dir.name}.gold.json").read_text())
    qs = [q for q in gold["questions"] if not args.questions or q["id"] in args.questions.split(",")]
    run_dir = Path(args.out).resolve().parent / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for rep in range(args.repeats):
        for q in qs:
            for arm in args.arms.split(","):
                prompt = prompt_for(q, gold["meta"], corpus_dir / "corpus.txt")
                try:
                    row = (run_nano(prompt, corpus_dir, args, run_dir) if arm == "nano-rlm"
                           else run_hermes(arm, prompt, corpus_dir, args, run_dir))
                except subprocess.TimeoutExpired:
                    row = {"answer": "", "wall_s": args.timeout, "error": "timeout"}
                row.update(arm=arm, question=q["id"], rep=rep, model=args.model,
                           corpus_tokens=gold["meta"]["approx_tokens"], ts=time.time())
                if not row.get("error") and re.search(r"HTTP 429|rate-limited|didn't produce a reply",
                                                      row.get("answer") or ""):
                    row["error"] = "provider"  # an outage is not a wrong answer; report rows as invalid
                row["correct"] = None if "error" in row else score(q, row.get("answer"))
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if row.get("error") == "provider":
                    sys.exit("provider unavailable (429/no reply); stopping instead of burning quota")
                print(json.dumps({k: row.get(k) for k in ("arm", "question", "correct", "wall_s",
                                  "root_input_tokens", "subcalls", "nano_stop", "error")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
