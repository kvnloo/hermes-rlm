"""rlm: Recursive Language Model mode for Hermes (arXiv:2512.24601, NousResearch/hermes-agent#93303).

The corpus lives as ``context`` inside a persistent, sandboxed Python kernel instead of the prompt.
The root model (the normal Hermes agent) inspects it with ``rlm_exec`` code cells and fans slices out
to a sub-model with ``llm_query``/``llm_batch``. Sub-calls go through the plugin LLM lane ``ctx.llm``
(auxiliary task ``rlm_subcall``), so provider, credentials and usage accounting stay Hermes'; the kernel never
sees a key. Budgets (sub-calls, prompt size, cell time) are enforced here, host-side. Every load,
cell and sub-call is appended to ``<hermes home>/plugin-data/rlm/runs/<session>.jsonl``.

reuse-checked: execute_code / delegate_task -- host-privileged and unbudgeted; RLM needs a sandboxed
corpus REPL whose sub-calls reuse ctx.llm(task=rlm_subcall). Wired via plugins.enabled (config.yaml).
"""
from __future__ import annotations

import atexit
import contextvars
import glob as _glob
import json
import math
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KERNEL = Path(__file__).with_name("kernel.py")
TOOLSET = "rlm"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    return {
        "max_subcalls": _env_int("RLM_MAX_SUBCALLS", 200),       # 0 = depth-0 (REPL only)
        "max_prompt_chars": _env_int("RLM_MAX_PROMPT_CHARS", 400_000),
        "max_output_chars": _env_int("RLM_MAX_OUTPUT_CHARS", 6_000),
        "cell_timeout_s": _env_int("RLM_CELL_TIMEOUT_S", 600),
        "concurrency": _env_int("RLM_CONCURRENCY", 8),
        "subcall_max_tokens": _env_int("RLM_SUBCALL_MAX_TOKENS", 4_000),
        "max_corpus_chars": _env_int("RLM_MAX_CORPUS_CHARS", 400_000_000),
    }


def sandbox_argv() -> list[str]:
    """bubblewrap: read-only /usr, private /tmp, no network, empty env, dies with the host."""
    bwrap = shutil.which("bwrap")
    py = "/usr/bin/python3"
    if bwrap and Path(py).exists():
        return [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib", "/lib64", "--symlink", "usr/bin", "/bin",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--ro-bind", str(KERNEL), "/rlm/kernel.py", "--chdir", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", py, "-I", "/rlm/kernel.py"]
    if os.environ.get("RLM_ALLOW_UNSANDBOXED") == "1":
        return [sys.executable, "-I", str(KERNEL)]
    raise RuntimeError("rlm needs bubblewrap (bwrap) and /usr/bin/python3 for its sandbox; "
                       "set RLM_ALLOW_UNSANDBOXED=1 only for local testing")


_CTX = None  # PluginContext from register(); ctx.llm is the documented host-owned LLM lane


def default_subcall(prompt: str, max_tokens: int) -> tuple[str, dict]:
    if _CTX is None:
        raise RuntimeError("rlm plugin not registered; no host LLM lane")
    r = _CTX.llm.complete(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                          task="rlm_subcall", purpose="rlm.subcall")
    u = r.usage
    if u is None:  # host lane gave text but no usage: keep the text
        return r.text or "", {}
    return r.text or "", {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                          "cost_usd": u.cost_usd}


def _task_path(raw: str, task_id: str) -> str:
    """Anchor a relative path at the task's workspace (what read_file/terminal see), not the process cwd."""
    raw = os.path.expanduser(raw)
    if os.path.isabs(raw):
        return raw
    try:
        from tools.file_tools_paths import _resolve_base_dir
        return str(Path(_resolve_base_dir(task_id or "default")) / raw)
    except Exception:
        return os.path.abspath(raw)


def load_corpus(paths, max_chars: int, task_id: str = "default"):
    """Text files under ``paths`` (files, dirs, globs) -> {relative path: text}; skips binaries/.git."""
    if isinstance(paths, str):
        paths = [paths]
    files: list[Path] = []
    for raw in paths:
        raw = _task_path(raw, task_id)
        for hit in sorted(_glob.glob(raw, recursive=True)) or [raw]:
            p = Path(hit)
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.is_file() and ".git" not in f.parts)
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(raw)
    common = os.path.commonpath([str(f.parent) for f in files]) if files else ""
    corpus, total, skipped = {}, 0, 0
    for f in files:
        data = f.read_bytes()
        if b"\0" in data[:8192]:
            skipped += 1
            continue
        text = data.decode("utf-8", "replace")
        total += len(text)
        if total > max_chars:
            raise ValueError(f"corpus exceeds RLM_MAX_CORPUS_CHARS={max_chars}")
        corpus[os.path.relpath(f, common) if common else str(f)] = text
    return corpus, skipped


class Session:
    """One sandboxed kernel plus budgets and a trajectory log for one Hermes session."""

    def __init__(self, key: str, subcall=default_subcall, log_dir: Path | None = None):
        self.key, self.subcall, self.lim = key, subcall, limits()
        if log_dir is None:
            log_dir = Path(os.environ.get("RLM_LOG_DIR") or _data_dir() / "runs")
        self.log_path = log_dir / f"{key or 'default'}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.context = None
        self.stats = {"subcalls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "subcall_prompt_chars": 0, "cells": 0, "restarts": 0}
        self.lock = threading.Lock()

    def log(self, event: str, **data) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **data}) + "\n")

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(sandbox_argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        if self.context is not None:
            self._request({"type": "load", "context": self.context}, time.monotonic() + 120)

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def _read(self, deadline: float) -> dict:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            ready, _, _ = select.select([self.proc.stdout], [], [], min(left, 1.0))
            if ready:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("sandbox kernel exited")
                return json.loads(line)

    def _send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _request(self, msg: dict, deadline: float) -> dict:
        self._send(msg)
        while True:
            reply = self._read(deadline)
            if reply["type"] == "done":
                return reply
            self._send(self._serve_llm(reply))

    def _serve_llm(self, req: dict) -> dict:
        prompts = req["prompts"]
        lim = self.lim
        if self.stats["subcalls"] + len(prompts) > lim["max_subcalls"]:
            self.log("budget_refused", kind="subcalls", requested=len(prompts), **self.stats)
            return {"type": "llm_result", "id": req["id"], "error":
                    f"sub-call budget exhausted ({self.stats['subcalls']}/{lim['max_subcalls']} used); "
                    "answer from what you have"}
        big = max((len(p) for p in prompts), default=0)
        if big > lim["max_prompt_chars"]:
            return {"type": "llm_result", "id": req["id"], "error":
                    f"prompt of {big} chars exceeds RLM_MAX_PROMPT_CHARS={lim['max_prompt_chars']}; "
                    "split the slice into smaller chunks"}
        try:
            max_tokens = min(int(req.get("max_tokens") or lim["subcall_max_tokens"]), 32_000)
            if max_tokens < 1:
                raise ValueError("below 1")
        except (TypeError, ValueError, OverflowError) as exc:  # model code controls this value
            return {"type": "llm_result", "id": req["id"], "error":
                    f"max_tokens must be a positive integer ({type(exc).__name__}); omit it or pass a number"}
        self.stats["subcalls"] += len(prompts)

        def one(prompt: str):
            t0 = time.monotonic()
            try:
                text, usage = self.subcall(prompt, max_tokens)
                text = text or ""  # no text / no usage must not sink the batch
                if not isinstance(text, str):
                    raise TypeError(f"subcall returned {type(text).__name__}, not text")
                usage = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd")
                         if isinstance(usage, dict) and isinstance(usage.get(k), (int, float))
                         and not isinstance(usage[k], bool)
                         and not (isinstance(usage[k], float) and not math.isfinite(usage[k]))
                         and usage[k] >= 0
                         and (k == "cost_usd" or usage[k] % 1 == 0)}  # finite, non-negative; token counts whole
            except Exception as exc:  # one failed slice must not sink the batch
                text, usage = f"[llm_query error: {type(exc).__name__}: {exc}]", {}
            self.log("subcall", prompt_chars=len(prompt), output_chars=len(text),
                     ms=int((time.monotonic() - t0) * 1000), **usage)
            return text, usage

        with ThreadPoolExecutor(max_workers=max(1, lim["concurrency"])) as pool:
            # copy_context: worker threads keep the call's profile scope, which ctx.llm reads
            out = [f.result() for f in [pool.submit(contextvars.copy_context().run, one, pr) for pr in prompts]]
        for p, (_, usage) in zip(prompts, out):
            self.stats["subcall_prompt_chars"] += len(p)
            self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
            self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
        return {"type": "llm_result", "id": req["id"], "results": [t for t, _ in out]}

    def load(self, paths, task_id: str = "default") -> dict:
        corpus, skipped = load_corpus(paths, self.lim["max_corpus_chars"], task_id)
        with self.lock:
            self.kill()
            self.context = next(iter(corpus.values())) if len(corpus) == 1 else corpus
            self._spawn()
        chars = sum(len(t) for t in corpus.values())
        self.log("load", paths=paths, files=len(corpus), chars=chars, skipped_binary=skipped)
        kind = "str" if len(corpus) == 1 else "dict[path -> text]"
        return {"loaded": True, "context_type": kind, "files": len(corpus), "chars": chars,
                "approx_tokens": chars // 4, "skipped_binary": skipped,
                "sample_keys": list(corpus)[:20] if len(corpus) > 1 else None}

    def exec(self, code: str) -> dict:
        lim = self.lim
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            t0 = time.monotonic()
            before = dict(self.stats)
            try:
                reply = self._request({"type": "exec", "code": code}, t0 + lim["cell_timeout_s"])
            except TimeoutError:
                self.kill()
                self.stats["restarts"] += 1
                reply = {"stdout": "", "final": None, "error":
                         f"cell exceeded {lim['cell_timeout_s']}s and was killed; kernel restarted "
                         "with `context` reloaded, other variables are gone"}
            except RuntimeError as exc:
                self.kill()
                self.stats["restarts"] += 1
                reply = {"stdout": "", "final": None, "error": f"{exc}; kernel restarted, variables lost"}
            self.stats["cells"] += 1
        out = reply.get("stdout") or ""
        shown = out if len(out) <= lim["max_output_chars"] else (
            out[: lim["max_output_chars"]] + f"\n... [{len(out) - lim['max_output_chars']} more chars truncated;"
            " keep large values in variables and print summaries]")
        ms = int((time.monotonic() - t0) * 1000)
        self.log("cell", code=code, stdout_chars=len(out), error=reply.get("error"), ms=ms,
                 final=reply.get("final"), subcalls=self.stats["subcalls"] - before["subcalls"])
        result = {"stdout": shown, "error": reply.get("error"), "ms": ms,
                  "subcalls_this_cell": self.stats["subcalls"] - before["subcalls"],
                  "budget": {"subcalls_used": self.stats["subcalls"], "subcalls_max": lim["max_subcalls"]}}
        if reply.get("final") is not None:
            result["final"] = reply["final"]
        return result


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def session_for(task_id: str = "", session_id: str = "", **_kw) -> Session:
    key = "".join(c for c in (task_id or session_id or "default") if c.isalnum() or c in "-_")[:80]
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = Session(key)
        return _sessions[key]


@atexit.register
def _shutdown() -> None:
    for s in _sessions.values():
        s.kill()


LOAD_SCHEMA = {
    "name": "rlm_load",
    "description": (
        "RLM mode: load files/dirs/globs as the read-only variable `context` inside a persistent "
        "sandboxed Python REPL (no network, no host files) WITHOUT putting them in your context. Use "
        "for corpora too large to read directly. One file -> str; several -> dict[path -> text]. "
        "Then inspect it with rlm_exec."),
    "parameters": {"type": "object", "properties": {
        "paths": {"type": "array", "items": {"type": "string"},
                  "description": "Files, directories or globs (** allowed)"}},
        "required": ["paths"]},
}
EXEC_SCHEMA = {
    "name": "rlm_exec",
    "description": (
        "Run Python in the RLM REPL (state persists across calls). Available: `context`; "
        "`llm_query(prompt) -> str` asks a sub-model about a slice you embed in the prompt (it sees "
        "nothing else); `llm_batch([prompts]) -> [str]` runs many in parallel; `FINAL(answer)` records "
        "the answer. Strategy: probe structure with len()/slicing/regex first, chunk, map with "
        "llm_batch, keep results in variables, print only short summaries (output is truncated)."),
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python source; the last expression is echoed"}},
        "required": ["code"]},
}


def _load_handler(args: dict, **kw) -> str:
    try:
        return json.dumps(session_for(**kw).load(args.get("paths") or [], kw.get("task_id") or "default"))
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _exec_handler(args: dict, **kw) -> str:
    try:
        return json.dumps(session_for(**kw).exec(str(args.get("code") or "")))
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _rlm_command(raw_args: str = "", **_kw) -> str:
    rows = [f"{k}: {v.stats} log={v.log_path}" for k, v in _sessions.items()]
    return "RLM limits: " + json.dumps(limits()) + ("\n" + "\n".join(rows) if rows else "\nno RLM sessions yet")


def _data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir("rlm")
    except Exception:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "plugin-data" / "rlm"


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_tool("rlm_load", TOOLSET, LOAD_SCHEMA, _load_handler, emoji="🧮",
                      description=LOAD_SCHEMA["description"])
    ctx.register_tool("rlm_exec", TOOLSET, EXEC_SCHEMA, _exec_handler, emoji="🧮",
                      description=EXEC_SCHEMA["description"])
    ctx.register_auxiliary_task("rlm_subcall", display_name="RLM sub-call",
                                description="Sub-model behind llm_query/llm_batch in the rlm plugin")
    ctx.register_command("rlm", _rlm_command, description="Show RLM budgets, usage and trajectory logs")
