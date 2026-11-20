"""Rerunnable, interleaved paper matrix over evaluator-only task manifests.

reuse-checked: run_bench.py, nano_client.py, plugin hashed logs, Hermes CLI and nano ACP.
See paper-harness.md for protocol, CLI and limitations. No dataset creation lives here.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from paper_proxy import Ledger, MODELS, ROOT, SUB, Router
from run_bench import rlm_usage

ARMS = ("base-truncated", "rlm-no-subcalls", "released-rlm", "fork-rlm", "nano-rlm")
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_tasks(manifest):
    tasks, seen = [], set()
    with manifest.open(encoding="utf-8") as stream:
        lines = stream.readlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        task = json.loads(line)
        required = {"id", "family", "size_tokens", "context_path", "question", "gold", "scoring"}
        if not required <= task.keys():
            raise ValueError(f"manifest line {n}: missing {sorted(required - task.keys())}")
        if not isinstance(task["id"], str) or task["id"] in seen:
            raise ValueError(f"manifest line {n}: non-string or duplicate id")
        seen.add(task["id"])
        path = Path(task["context_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        # Only normalized plain context files are accepted, never an evaluator directory.
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"manifest line {n}: context_path must be a regular non-symlink file")
        if re.search(r"gold|label|manifest|answer", path.name, re.I):
            raise ValueError(f"manifest line {n}: evaluator-looking context filename")
        size = task["size_tokens"]
        if (not isinstance(task["question"], str) or type(size) not in (int, float)
                or size < 0 or (isinstance(size, float) and not math.isfinite(size))):
            raise ValueError(f"manifest line {n}: invalid question or size_tokens")
        task = dict(task, context_path=str(path.resolve()))
        actual_sha = digest(path)
        metadata = task.get("metadata")
        if isinstance(metadata, dict) and "context_sha256" in metadata:
            if metadata["context_sha256"] != actual_sha:
                raise ValueError(f"manifest line {n}: context_sha256 does not match context bytes")
        task["context_sha256"] = actual_sha
        task["task_sha256"] = json_hash(task)
        tasks.append(task)
    return tasks


def schedule(tasks, arms, repeats):
    # Rotate the first arm per task/repeat, rather than running an arm-sized block.
    for rep in range(repeats):
        for index, task in enumerate(tasks):
            offset = (rep + index) % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                yield task, arm, rep


def row_key(task, arm, rep, protocol):
    return json_hash([task["task_sha256"], arm, rep, protocol])


def completed(out):
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as stream:
            lines = stream.readlines()
        for i, line in enumerate(lines):
            try:
                done.add(json.loads(line)["row_key"])
            except (ValueError, KeyError):
                raise ValueError(f"corrupt output at line {i + 1}; preserve and repair before resume")
    return done


class InvalidScorerResult(ValueError):
    """A scorer result cannot be emitted in the benchmark row."""


def score_answer(task, answer, module_path):
    # Dataset worker owns this implementation. Do not silently substitute an LLM judge.
    spec = importlib.util.spec_from_file_location("paper_tasks_scoring", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.score_task(task, answer)
    scalar = isinstance(result, (float, int))
    if scalar:
        result = {"score": result}
    if not isinstance(result, dict) or "score" not in result:
        raise ValueError("score_task must return a numeric score or dict containing score")
    score = result["score"]
    if not isinstance(score, (float, int)):
        raise ValueError("score_task score must be a finite number")
    try:
        score = float(score)
    except OverflowError as exc:
        raise ValueError("score_task score must be a finite number") from exc
    if not math.isfinite(score):
        raise ValueError("score_task score must be a finite number")
    grade = {"score": score} if scalar else result
    try:
        json.dumps(grade, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidScorerResult("score_task result must serialize as UTF-8 JSON") from exc
    return grade


def task_prompt(task, arm):
    # Explicit allowlist. Neither gold, scoring nor arbitrary metadata enters this prompt.
    choices = task.get("choices")
    question = task["question"]
    if choices:
        question += "\nChoices: " + json.dumps(choices, ensure_ascii=False)
    if arm == "base-truncated":
        return question + "\nEnd with ANSWER: <answer>."
    intro = (f"Task input is the local file context.txt, approximately {task['size_tokens']} dataset-nominal tokens. "
             "The root prompt contains only this metadata, not the raw context. ")
    if arm == "nano-rlm":
        intro += "Load context.txt into your persistent Python REPL as context. Inspect it using code. "
    else:
        intro += "Call rlm_load with paths=['context.txt'], then use rlm_exec to inspect context. "
    if arm == "rlm-no-subcalls":
        intro += "This is the no-subcalls ablation: max_depth=0. Do not use llm_query, llm_batch or any LLM subcall. Solve with Python only. "
    elif arm != "nano-rlm":
        intro += "For semantic work use llm_query/llm_batch over relevant chunks and aggregate in Python. "
    if arm == "nano-rlm":
        finish = "Use source character spans where supported. End with ANSWER: <answer>."
    else:
        finish = ("Compute the answer in a REPL variable named answer. Your last rlm_exec cell must call "
                  "FINAL_VAR('answer', citations=[cite('context.txt', start, end)]). Use real supporting "
                  "zero-based, end-exclusive character offsets. Ordinary final prose is not an answer "
                  "and will be withheld.")
    return intro + "\nQuestion: " + question + "\n" + finish


def fresh_home(home, template, plugin, arm, router, args):
    import yaml
    home.mkdir(parents=True, exist_ok=False)
    # Never copy auth, sessions, databases, memories, .env, hooks or unrelated provider config.
    if (template / "SOUL.md").is_file():
        shutil.copyfile(template / "SOUL.md", home / "SOUL.md")
    shutil.copytree(plugin, home / "plugins" / "rlm",
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "tests", "bench", "*.pyc"))
    settings = {"max_depth": 0 if arm == "rlm-no-subcalls" else 1,
                "max_subcalls": args.max_subcalls, "max_tokens": args.sub_max_tokens,
                "timeout_s": args.cell_timeout, "max_cost": None, "subcall_task": "rlm_subcall",
                "max_corpus_chars": args.max_context_chars}
    config = {
        "_config_version": 12,
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "model": {"provider": "paper-root", "default": args.root_model, "max_tokens": args.max_tokens},
        "providers": {"paper-root": {"base_url": router.url + "/root/v1", "key_env": "PAPER_PROXY_KEY",
                                      "api_key_env": "PAPER_PROXY_KEY", "api_mode": "chat_completions"}},
        "auxiliary": {"transient_retries": 0,
            "rlm_subcall": {"provider": "paper-sub", "base_url": router.url + "/sub/v1",
                            "api_key_env": "PAPER_PROXY_KEY", "model": args.sub_model,
                            "fallback_models": []},
            "background_review": {"enabled": False}},
        "plugins": {"enabled": ["rlm"], "entries": {"rlm": {"settings": settings}}},
        "tools": {"tool_search": {"enabled": "on" if arm == "fork-rlm" else "off",
                                   "eager": ["rlm"] if arm == "fork-rlm" else []}},
        "model_catalog": {"enabled": False},
        "agent": {"max_turns": args.max_turns, "api_max_retries": 1, "auto_recovery_cycles": 0},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    return config


def descendants(pid):
    parents = {}
    proc = Path("/proc")
    if proc.exists():
        for p in proc.iterdir():
            if p.name.isdigit():
                try:
                    # comm may contain spaces and parentheses.
                    fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
                    parents[int(p.name)] = int(fields[1])
                except (OSError, ValueError, IndexError):
                    pass
    found = {pid}
    while True:
        more = {p for p, parent in parents.items() if parent in found}
        if more <= found:
            return found - {pid}
        found |= more


def run_process(command, cwd, env, timeout):
    start = time.monotonic()
    # Daemons can inherit descriptors after leaving the process tree. Files do not
    # require their EOF before collecting partial output, unlike communicate pipes.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                             start_new_session=True)
        known, timed_out = set(), False
        try:
            while True:
                known |= descendants(p.pid)
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    p.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            known |= descendants(p.pid)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for pid in known:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                p.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                # A kernel-stuck child must not turn a deadline into another wait.
                timed_out = True
        # pread leaves the shared writer offset alone and snapshots the current size.
        stdout = os.pread(out.fileno(), os.fstat(out.fileno()).st_size, 0).decode("utf-8", "replace")
        stderr = os.pread(err.fileno(), os.fstat(err.fileno()).st_size, 0).decode("utf-8", "replace")
    return {"stdout": stdout, "stderr": stderr, "exit_code": p.returncode,
            "timed_out": timed_out, "wall_s": round(time.monotonic() - start, 3)}


def hermes_stats(home):
    path = home / "state.db"
    if not path.exists():
        return {"session_ids": [], "discovery_traces": []}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        sessions = db.execute("SELECT id,input_tokens,output_tokens,api_call_count,tool_call_count FROM sessions").fetchall()
        traces = []
        for (calls,) in db.execute("SELECT tool_calls FROM messages WHERE tool_calls IS NOT NULL"):
            try:
                for call in json.loads(calls):
                    name = call.get("function", {}).get("name", call.get("name"))
                    if name in {"tool_search", "tool_describe", "tool_call"}:
                        traces.append(name)
            except (ValueError, TypeError, AttributeError):
                pass
    return {"session_ids": [r[0] for r in sessions], "discovery_traces": traces,
            "host_root_input_tokens": sum(r[1] or 0 for r in sessions),
            "host_root_output_tokens": sum(r[2] or 0 for r in sessions),
            "host_api_calls": sum(r[3] or 0 for r in sessions),
            "tool_calls": sum(r[4] or 0 for r in sessions)}


def source_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def child_outcomes_ok(arm, row):
    physical_children = [r for r in row.get("accounting", {}).get("requests", []) if r.get("role") == "sub"]
    if arm == "nano-rlm":
        logical = row.get("logical_call_outcomes")
        if not isinstance(logical, list):
            return False
        children = [call for call in logical if call["role"] == "sub"]
        return (not physical_children or bool(children)) and all(
            not call["pending"] and call["status"] == 200 for call in children)
    statuses = row.get("subcall_statuses", {})
    return (not physical_children or bool(statuses)) and all(
        status == "ok" for status, count in statuses.items() if count)


def run_one(task, arm, rep, args, protocol, ledger):
    key = row_key(task, arm, rep, protocol)
    run_id = key[:16] + "-" + uuid.uuid4().hex[:8]
    run_dir = args.ops_dir / "runs" / run_id
    cwd = run_dir / "input"
    cwd.mkdir(parents=True)
    # The only original task material in cwd is the unlabeled context.
    shutil.copyfile(task["context_path"], cwd / "context.txt")
    if digest(cwd / "context.txt") != task["context_sha256"]:
        raise ValueError("context changed after manifest validation")
    row = {"row_key": key, "run_id": run_id, "task_id": task["id"], "family": task["family"],
           "size_tokens": task["size_tokens"], "task_sha256": task["task_sha256"],
           "context_sha256": task["context_sha256"], "arm": arm, "rep": rep,
           "root_model": args.root_model, "sub_model": args.sub_model,
           "protocol": protocol, "ts": time.time(), "status": "invalid", "valid": False,
           "correct": None, "score": None}
    prompt = task_prompt(task, arm)
    (run_dir / "prompt.txt").write_text(prompt)
    router = None
    try:
        with Router(ledger, run_id, args.root_model, args.sub_model,
                    upstream_timeout=min(args.timeout, 120),
                    disable_thinking=args.disable_thinking) as router:
            if arm == "base-truncated":
                with (cwd / "context.txt").open(encoding="utf-8", newline="") as context:
                    truncated = context.read(args.base_context_chars)
                    original_chars = len(truncated) + sum(
                        len(chunk) for chunk in iter(lambda: context.read(64 * 1024), ""))
                # Declared UTF-8 character truncation, not invented tokenizer counts.
                body = {"model": args.root_model, "messages": [{"role": "user", "content":
                        truncated + "\n\n" + prompt}], "max_tokens": args.max_tokens, "stream": False}
                start = time.monotonic()
                status, data, _ = router.forward(body, args.root_model, "root")
                response = json.loads(data)
                row.update(answer=response.get("choices", [{}])[0].get("message", {}).get("content") or "",
                           wall_s=round(time.monotonic() - start, 3), exit_code=0 if status == 200 else 1,
                           timed_out=False, truncation={"kind": "head_characters", "chars": len(truncated),
                           "original_chars": original_chars, "truncated": original_chars > len(truncated)})
                (run_dir / "base-response.json").write_bytes(data)
            else:
                env = {k: v for k, v in os.environ.items()
                       if not re.search(r"API_KEY|TOKEN$|SECRET|PASSWORD", k, re.I)}
                env.update(PAPER_PROXY_KEY="paper-local-not-a-provider-key", PYTHONDONTWRITEBYTECODE="1",
                           HERMES_DISABLE_LAZY_INSTALLS="1")
                if arm == "nano-rlm":
                    env.update(NANO_BASE_URL=router.url + "/nano/v1", NANO_API_KEY=env["PAPER_PROXY_KEY"],
                        NANO_MODEL=args.root_model, NANO_RLM_BIN=str(args.nano_source / ".venv/bin/rlm"),
                        RLM_HOME=str(run_dir / "nano-home"),
                        NANO_POLICY=json.dumps({"max_depth": 1, "max_subagent_calls": args.max_subcalls,
                            "max_total_turns": args.max_turns, "max_tokens": args.max_tokens,
                            "max_total_tokens": args.nano_total_tokens, "network_guidance": False}))
                    command = [str(args.nano_python), str(HERE / "nano_client.py"), prompt]
                    row["nano_routing"] = "ACP session header: first session=root; distinct sessions=sub"
                else:
                    home = args.ops_dir / "homes" / run_id
                    config = fresh_home(home, args.home_template, args.plugin_source, arm, router, args)
                    row["config_sha256"] = json_hash(config)
                    source = args.fork_source if arm == "fork-rlm" else args.released_source
                    python = args.fork_python if arm == "fork-rlm" else args.released_python
                    env.update(HERMES_HOME=str(home), PYTHONPATH=str(source), RLM_LOG_DIR=str(run_dir / "rlm-logs"))
                    row["hermes_ingress"] = args.hermes_ingress
                    if args.hermes_ingress == "library":
                        command = [str(python), str(HERE / "hermes_client.py"), prompt,
                                   "--base-url", router.url + "/root/v1", "--provider", "paper-root",
                                   "--model", args.root_model, "--max-iterations", str(args.max_turns),
                                   "--max-tokens", str(args.max_tokens),
                                   "--transcript", str(run_dir / "hermes-conversation.json")]
                    else:
                        command = [str(python), "-c", "from hermes_cli.main import main; main()", "chat",
                                   "--oneshot", "-Q", "-t", "rlm", "--provider", "paper-root",
                                   "-m", args.root_model, "-q", prompt]
                result = run_process(command, cwd, env, args.timeout)
                stdout, stderr = result.pop("stdout"), result.pop("stderr")
                (run_dir / "stdout.txt").write_text(stdout)
                (run_dir / "stderr.txt").write_text(stderr)
                row.update(result)
                if arm == "nano-rlm":
                    response = json.loads(stdout.strip().splitlines()[-1])
                    snap = response.get("snapshot") or {}
                    metrics = snap.get("metrics") or {}
                    row.update(answer=response["answer"], nano_snapshot=snap,
                               subcalls=metrics.get("sub_rlm_num_calls"))
                else:
                    row.update(hermes_stats(home))
                    row.update(rlm_usage(run_dir / "rlm-logs"))
                    if args.hermes_ingress == "library":
                        if stdout.strip():
                            summary = json.loads(stdout.strip().splitlines()[-1])
                            for field in ("answer", "session_ids", "discovery_traces", "tool_calls",
                                          "library_tools", "library_error", "library_partial",
                                          "completed", "failed", "interrupted", "turn_exit_reason",
                                          "host_root_input_tokens", "host_root_output_tokens", "host_api_calls"):
                                if field in summary:
                                    row[field] = summary[field]
                        else:
                            row["answer"] = ""
                    else:
                        row["answer"] = stdout.split("session_id:")[0].strip()
            row["accounting"] = ledger.summary(run_id)
            row["logical_call_outcomes"] = router.logical_snapshot()
            row["proxy_errors"] = router.errors
            row["tool_trace"] = router.observed
            row["root_input_tokens"] = row["accounting"]["root_prompt_tokens"]
            row["root_output_tokens"] = row["accounting"]["root_completion_tokens"]
            row["sub_prompt_tokens"] = row["accounting"]["sub_prompt_tokens"]
            row["sub_completion_tokens"] = row["accounting"]["sub_completion_tokens"]
            row["api_calls"] = row["accounting"]["request_attempts"]
            # Successful retries are retained; failed final attempts invalidate the row.
            requests = row["accounting"]["requests"]
            provider_ok = bool(requests) and requests[-1]["status"] == 200 and not router.blocked
            row["valid"] = bool(provider_ok and not router.errors and not row["timed_out"]
                                and row["exit_code"] == 0 and row.get("answer", "").strip()
                                and not row.get("library_error") and not row.get("library_partial"))
            if arm == "rlm-no-subcalls" and row.get("subcalls", 0):
                row["valid"] = False
                row["error"] = "no-subcalls arm dispatched a child"
            if arm not in ("base-truncated", "nano-rlm"):
                # Prove actual root tool exposure rather than trusting only YAML.
                exposed = [t for x in router.observed if x["role"] == "root" for t in x["tools"]]
                row["rlm_eager_observed"] = "rlm_load" in exposed and "rlm_exec" in exposed
                if not row["rlm_eager_observed"] or not row.get("cells"):
                    row["valid"] = False
                    row["error"] = "RLM tools not exposed or no REPL cell executed"
            if arm not in ("base-truncated", "nano-rlm"):
                row["root_answer"] = row.get("answer", "")
                final = row.get("validated_final")
                if final is None:
                    row["valid"] = False
                    row.setdefault("error", "missing_validated_final")
                else:
                    row["answer"] = final["answer"]
                if args.hermes_ingress == "library" and (
                        row.get("completed") is not True or row.get("failed") or row.get("interrupted")
                        or any(s in str(row.get("turn_exit_reason") or "").lower()
                               for s in ("max_iterations", "budget", "interrupt"))):
                    row["valid"] = False
                    row.setdefault("error", "conversation_incomplete")
            if arm != "base-truncated" and not child_outcomes_ok(arm, row):
                row["valid"] = False
                row.setdefault("error", "unrecovered_or_unknown_child_outcome")
            if row["valid"]:
                grade = score_answer(task, row["answer"], args.scorer)
                row.update(score=grade["score"], grading=grade,
                           correct=grade.get("correct", grade["score"] == 1), status="valid")
            else:
                row.setdefault("error", "provider, timeout, empty answer or process failure")
    except Exception as exc:
        # Do not serialize exception strings from provider/client libraries.
        row.update(status="invalid", valid=False, score=None, correct=None,
                   error=type(exc).__name__, accounting=ledger.summary(run_id),
                   logical_call_outcomes=router.logical_snapshot() if router is not None else [])
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--ops-dir", type=Path, required=True)
    p.add_argument("--home-template", type=Path, required=True)
    p.add_argument("--released-source", type=Path, required=True)
    p.add_argument("--released-python", type=Path, required=True)
    p.add_argument("--fork-source", type=Path, required=True)
    p.add_argument("--fork-python", type=Path, required=True)
    p.add_argument("--nano-source", type=Path, required=True)
    p.add_argument("--nano-python", type=Path)
    p.add_argument("--plugin-source", type=Path, default=HERE.parent / "rlm")
    p.add_argument("--scorer", type=Path, default=HERE / "paper_tasks.py")
    p.add_argument("--root-model", choices=list(MODELS), default=ROOT)
    p.add_argument("--sub-model", choices=list(MODELS), default=SUB)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--hermes-ingress", choices=("cli", "library"), default="library",
                   help="Hermes ingress; library uses actual AIAgent without a gateway daemon")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--task-ids", default="")
    p.add_argument("--smallest-sniah", action="store_true")
    p.add_argument("--limit-tasks", type=int)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--cell-timeout", type=int, default=120)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--sub-max-tokens", type=int, default=2048)
    p.add_argument("--max-subcalls", type=int, default=16)
    p.add_argument("--max-context-chars", type=int, default=100_000_000)
    p.add_argument("--base-context-chars", type=int, default=48_000)
    p.add_argument("--nano-total-tokens", type=int, default=200_000)
    p.add_argument("--provider-cap", type=int, default=2999)
    p.add_argument("--disable-thinking", action="store_true",
                   help="Disable NIM thinking uniformly across all arms; recorded in protocol")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving a venv's python symlink changes sys.prefix and loses its packages.
            setattr(args, name, Path(os.path.abspath(value)))
    args.nano_python = args.nano_python or args.nano_source / ".venv/bin/python"
    arms = tuple(args.arms.split(","))
    if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError("unknown or duplicate arms")
    if min(args.repeats, args.timeout, args.max_subcalls, args.max_turns, args.base_context_chars) < 1:
        raise ValueError("budgets and repeats must be positive")
    if args.limit_tasks is not None and args.limit_tasks < 1:
        raise ValueError("--limit-tasks must be positive")
    if not args.scorer.is_file():
        raise ValueError("scorer must be a regular file")
    scorer_sha256 = digest(args.scorer)
    soul = args.home_template / "SOUL.md"
    soul_sha256 = digest(soul) if soul.is_file() else None
    args.ops_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One controller per output; the request ledger additionally serializes provider reservations.
    import fcntl
    with (args.out.parent / (args.out.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = load_tasks(args.manifest)
        if args.task_ids:
            requested = set(args.task_ids.split(","))
            unknown = requested - {t["id"] for t in tasks}
            if unknown:
                raise ValueError(f"unknown --task-ids: {', '.join(sorted(unknown))}")
            tasks = [t for t in tasks if t["id"] in requested]
        if args.smallest_sniah:
            tasks = [t for t in tasks if re.sub(r"[^a-z]", "", t["family"].lower()) in {"sniah", "niah"}]
            tasks = sorted(tasks, key=lambda t: (t["size_tokens"], t["id"]))[:1]
        if args.limit_tasks:
            tasks = tasks[:args.limit_tasks]
        if not tasks:
            raise ValueError("no tasks selected")
        shas = {k: source_sha(getattr(args, k + "_source")) for k in ("released", "fork", "nano")}
        plugin_hash = json_hash({str(p.relative_to(args.plugin_source)): digest(p)
                               for p in sorted([*args.plugin_source.glob("*.py"),
                                                *args.plugin_source.glob("plugin.yaml")])})
        protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    if k not in {"out", "ops_dir", "manifest", "repeats", "dry_run", "task_ids", "limit_tasks", "smallest_sniah"}}
        protocol.update(source_shas=shas, plugin_sha256=plugin_hash,
                        harness_sha256=digest(Path(__file__)), proxy_sha256=digest(HERE / "paper_proxy.py"),
                        hermes_client_sha256=digest(HERE / "hermes_client.py"),
                        scorer_sha256=scorer_sha256)
        if soul_sha256 is not None:
            protocol["home_soul_sha256"] = soul_sha256
        done = completed(args.out)
        ledger = Ledger(args.ops_dir / "request-ledger.sqlite3", args.provider_cap)
        for task, arm, rep in schedule(tasks, arms, args.repeats):
            key = row_key(task, arm, rep, protocol)
            if key in done:
                continue
            if args.dry_run:
                print(json.dumps({"row_key": key, "task_id": task["id"], "arm": arm, "rep": rep}))
                continue
            row = run_one(task, arm, rep, args, protocol, ledger)
            with args.out.open("a+b") as f:
                separator = b""
                if f.tell():
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) not in (b"\n", b"\r"):
                        separator = b"\n"
                f.write(separator + (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            print(json.dumps({k: row.get(k) for k in ("task_id", "arm", "rep", "valid", "score", "wall_s", "error")}), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
