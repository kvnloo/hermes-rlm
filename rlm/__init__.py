"""rlm: Recursive Language Model mode for Hermes (arXiv:2512.24601, NousResearch/hermes-agent#93303).

The corpus lives as ``context`` inside a persistent, sandboxed Python kernel instead of the prompt.
The root model (the normal Hermes agent) inspects it with ``rlm_exec`` code cells and fans slices out
to a sub-model with ``llm_query``/``llm_batch``. Sub-calls go through the plugin LLM lane ``ctx.llm``
(auxiliary task ``rlm_subcall``), so provider, credentials and usage accounting stay Hermes'; the kernel never
sees a key. Budgets (sub-calls, prompt size, cell time) are enforced here, host-side. Every load,
cell and sub-call is appended to ``<hermes home>/plugin-data/rlm/runs/<session>.jsonl``.

reuse-checked: execute_code / delegate_task -- host-privileged and unbudgeted; RLM needs a sandboxed
corpus REPL whose sub-calls reuse ctx.llm.acomplete(task=rlm_subcall). Wired via plugins.enabled.
"""
from __future__ import annotations

import atexit
import asyncio
import hashlib
import math
import glob as _glob
import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

from .final_gate import FinalGate
from .prompts import RLM_GUIDANCE, SUBCALL_SYSTEM
from .settings import limits as _limits, validate as validate_settings

KERNEL = Path(__file__).with_name("kernel.py")
TOOLSET = "rlm"


def limits(ctx=None) -> dict:
    return _limits(ctx if ctx is not None else _CTX)


def host_interrupted() -> bool:
    try:
        from tools.interrupt import is_interrupted
    except ImportError:
        return False
    return is_interrupted()


class CellCancelled(RuntimeError):
    pass


def sandbox_argv(*, allow_unsandboxed=False) -> list[str]:
    """bubblewrap: read-only /usr, private /tmp, no network, empty env, dies with the host."""
    if allow_unsandboxed:
        return [sys.executable, "-I", str(KERNEL)]
    bwrap = shutil.which("bwrap")
    py = "/usr/bin/python3"
    if bwrap and Path(py).exists():
        return [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib", "/lib64", "--symlink", "usr/bin", "/bin",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--ro-bind", str(KERNEL), "/rlm/kernel.py", "--chdir", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", py, "-I", "/rlm/kernel.py"]
    raise RuntimeError("rlm needs bubblewrap (bwrap) and /usr/bin/python3 for its sandbox; "
                       "tests may explicitly pass allow_unsandboxed=True")


_CTX = None  # PluginContext from register(); ctx.llm is the documented host-owned LLM lane


async def default_subcall(prompt: str, max_tokens: int, *, ctx=None,
                          task="rlm_subcall", timeout=600) -> tuple[str, dict]:
    ctx = ctx if ctx is not None else _CTX
    if ctx is None:
        raise RuntimeError("rlm plugin not registered; no host LLM lane")
    r = await ctx.llm.acomplete(
        messages=[{"role": "system", "content": SUBCALL_SYSTEM},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout, task=task, purpose="rlm.subcall")
    u = r.usage
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
                files += sorted(f for f in p.rglob("*") if f.is_file() and ".git" not in f.relative_to(p).parts)
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(raw)
    # Overlapping files, directories and globs must not consume the budget twice.
    files = list(dict.fromkeys(files))
    common = os.path.commonpath([str(f.parent) for f in files]) if files else ""
    corpus, total, skipped = {}, 0, 0
    for f in files:
        data = f.read_bytes()
        if b"\0" in data[:8192]:
            skipped += 1
            continue
        text = data.decode("utf-8", "replace")
        source = os.path.relpath(f, common) if common else str(f)
        # The corpus keeps the last value for a source name, not every read.
        total += len(text) - len(corpus.get(source, ""))
        if total > max_chars:
            raise ValueError(f"corpus exceeds RLM_MAX_CORPUS_CHARS={max_chars}")
        corpus[source] = text
    return corpus, skipped


class Session:
    """One sandboxed kernel plus budgets and a trajectory log for one Hermes session."""

    def __init__(self, key: str, subcall=None, log_dir: Path | None = None, *,
                 settings=None, allow_unsandboxed=False, cancelled=None, ctx=None):
        self.key, self.subcall = key, subcall
        self.ctx = ctx if ctx is not None else _CTX
        self.lim = validate_settings({**limits(self.ctx), **(settings or {})})
        self.allow_unsandboxed = allow_unsandboxed
        self.cancelled = cancelled if cancelled is not None else host_interrupted
        self._stop = threading.Event()
        if log_dir is None:
            legacy_log_dir = os.environ.get("RLM_LOG_DIR")
            if legacy_log_dir:
                warnings.warn("RLM_LOG_DIR is deprecated; use the profile's plugin-data/rlm/runs",
                              DeprecationWarning, stacklevel=2)
            log_dir = Path(legacy_log_dir) if legacy_log_dir else _data_dir() / "runs"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        self.log_path = log_dir / f"{digest}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.context = None
        self.sources = {}
        self.last_validated_final = None
        self.stats = {"subcalls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cost_usd": 0.0, "cost_unknown": 0,
                      "subcall_prompt_chars": 0, "cells": 0, "restarts": 0}
        self.lock = threading.Lock()

    def log(self, event: str, **data) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **data}) + "\n")

    def _spawn(self, deadline=None) -> None:
        self.proc = subprocess.Popen(sandbox_argv(allow_unsandboxed=self.allow_unsandboxed),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self._read_buffer = b""
        os.set_blocking(self.proc.stdin.fileno(), False)
        if self.context is not None:
            self._request({"type": "load", "context": self.context},
                          deadline if deadline is not None else time.monotonic() + self.lim["timeout_s"])

    def kill(self) -> None:
        self.last_validated_final = None
        self._stop.set()
        proc, self.proc = self.proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdin.close()
            proc.stdout.close()

    def _check(self, deadline):
        if self._stop.is_set() or self.cancelled():
            raise CellCancelled("cell cancelled; in-flight subcalls stopped")
        if time.monotonic() >= deadline:
            raise TimeoutError

    def _read(self, deadline: float) -> dict:
        while True:
            self._check(deadline)
            if b"\n" in self._read_buffer:
                line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                return json.loads(line)
            ready, _, _ = select.select([self.proc.stdout], [], [],
                                        max(0, min(deadline - time.monotonic(), 0.05)))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("sandbox kernel exited")
                self._read_buffer += chunk

    def _send(self, msg: dict, deadline: float) -> None:
        data = memoryview((json.dumps(msg) + "\n").encode())
        while data:
            self._check(deadline)
            _, ready, _ = select.select([], [self.proc.stdin], [], 0.05)
            if ready:
                try:
                    written = os.write(self.proc.stdin.fileno(), data[:65536])
                except BlockingIOError:
                    continue
                data = data[written:]

    def _request(self, msg: dict, deadline: float) -> dict:
        self._send(msg, deadline)
        while True:
            reply = self._read(deadline)
            if reply["type"] == "done":
                return reply
            if reply.get("type") != "llm":
                raise RuntimeError("invalid sandbox protocol message")
            try:
                answer = self._serve_llm(reply, deadline)
            except (Exception, asyncio.CancelledError, SystemExit) as exc:
                # The kernel is blocked on this reply; never leave it unanswered.
                answer = {"type": "llm_result", "id": reply.get("id"),
                          "error": f"llm_batch failed on the host: {type(exc).__name__}"}
            self._send(answer, deadline)

    def _cost_refusal(self):
        cap = self.lim["max_cost"]
        if cap is not None:
            if self.stats["cost_unknown"]:
                return "cost budget unavailable: provider did not report a price; further calls disabled"
            if self.stats["cost_usd"] >= cap:
                return "cost budget exhausted; answer from existing evidence"
        return None

    def _serve_llm(self, req: dict, deadline: float) -> dict:
        response = {"type": "llm_result", "id": req["id"]}
        prompts, lim = req.get("prompts"), self.lim
        error = None
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            error = "prompts must be a list of strings"
        elif lim["max_depth"] == 0:
            error = "max_depth=0 disables subcalls"
        elif self.stats["subcalls"] + len(prompts) > lim["max_subcalls"]:
            error = f"sub-call budget exhausted ({self.stats['subcalls']}/{lim['max_subcalls']} used)"
        elif max(map(len, prompts), default=0) > lim["max_prompt_chars"]:
            error = f"prompt exceeds max_prompt_chars (RLM_MAX_PROMPT_CHARS={lim['max_prompt_chars']})"
        else:
            error = self._cost_refusal()
        tokens = req.get("max_tokens")
        if tokens is None:
            tokens = lim["max_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            error = "max_tokens must be a positive integer"
        if error:
            self.log("budget_refused", reason=error)
            return {**response, "error": error}
        response["results"] = asyncio.run(self._batch(prompts, min(tokens, lim["max_tokens"]), deadline))
        return response

    async def _batch(self, prompts, max_tokens, deadline):
        gate = asyncio.Semaphore(self.lim["concurrency"])

        async def one(prompt):
            async with gate:
                self._check(deadline)
                refusal = self._cost_refusal()
                if refusal:
                    return f"[llm_query error: {refusal}]"
                self.stats["subcalls"] += 1
                attempt = self.stats["subcalls"]
                self.stats["subcall_prompt_chars"] += len(prompt)
                t0, status, text, usage = time.monotonic(), "ok", "", {}
                try:
                    if self.subcall is None:
                        text, usage = await default_subcall(
                            prompt, max_tokens, ctx=self.ctx, task=self.lim["subcall_task"],
                            timeout=max(0.001, deadline - time.monotonic()))
                    else:
                        text, usage = await self.subcall(prompt, max_tokens)
                    self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    cost = usage.get("cost_usd")
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                        self.stats["cost_usd"] += cost
                    else:
                        self.stats["cost_unknown"] += 1
                    return text
                except asyncio.CancelledError:
                    status = "cancelled"
                    self.stats["cost_unknown"] += 1
                    raise
                except Exception as exc:
                    status = "error"
                    self.stats["cost_unknown"] += 1
                    # Provider exception text may contain credentials or request content.
                    text = f"[llm_query error: {type(exc).__name__}]"
                    return text
                except SystemExit:
                    status = "error"
                    self.stats["cost_unknown"] += 1
                    raise
                finally:
                    self.log("subcall", attempt=attempt, status=status, prompt_chars=len(prompt),
                             output_chars=len(text), ms=int((time.monotonic() - t0) * 1000),
                             prompt_tokens=usage.get("prompt_tokens", 0),
                             completion_tokens=usage.get("completion_tokens", 0),
                             cost_usd=usage.get("cost_usd"))

        tasks = [asyncio.create_task(one(p)) for p in prompts]
        try:
            pending = set(tasks)
            while pending:
                self._check(deadline)
                _, pending = await asyncio.wait(pending, timeout=min(0.05, max(0, deadline-time.monotonic())))
            return [task.result() for task in tasks]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def load(self, paths, task_id: str = "default") -> dict:
        self.last_validated_final = None
        corpus, skipped = load_corpus(paths, self.lim["max_corpus_chars"], task_id)
        with self.lock:
            self.kill()
            self._stop.clear()
            self.sources = corpus
            self.context = next(iter(corpus.values())) if len(corpus) == 1 else corpus
            try:
                self._spawn()
            except BaseException:
                self.kill()
                raise
        chars = sum(len(t) for t in corpus.values())
        self.log("load", paths=paths, files=len(corpus), chars=chars, skipped_binary=skipped)
        kind = "str" if len(corpus) == 1 else "dict[path -> text]"
        return {"loaded": True, "context_type": kind, "files": len(corpus), "chars": chars,
                "approx_tokens": chars // 4, "skipped_binary": skipped,
                "sample_keys": list(corpus)[:20] if len(corpus) > 1 else None,
                "sources": [{"source": k, "chars": len(v)} for k, v in list(corpus.items())[:100]],
                "sources_truncated": len(corpus) > 100,
                "limits": self.lim, "guidance": RLM_GUIDANCE}

    def _validate_final(self, answer, citations):
        if not isinstance(answer, str):
            raise ValueError("final answer must be text")
        if not isinstance(citations, list):
            raise ValueError("citations must be a list")
        if any(self.sources.values()) and not citations:
            raise ValueError("citation required for an answer over loaded context")
        validated = []
        for span in citations:
            if not isinstance(span, dict):
                raise ValueError("citation must be an object")
            source, start, end = (span.get(k) for k in ("source", "start", "end"))
            if not isinstance(source, str) or source not in self.sources:
                raise ValueError("citation source is not in the loaded corpus")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(self.sources[source])):
                raise ValueError("citation range must satisfy 0 <= start < end <= source length")
            quote = self.sources[source][start:end]
            if "quote" in span and span["quote"] != quote:
                raise ValueError("citation quote does not match the original source")
            if len(quote) > self.lim["max_output_chars"]:
                raise ValueError("citation span exceeds max_output_chars; cite a shorter span")
            validated.append({"source": source, "start": start, "end": end, "quote": quote})
        if len(answer) + sum(len(s["quote"]) for s in validated) > self.lim["max_output_chars"]:
            raise ValueError("final answer and citations exceed max_output_chars")
        return validated

    def exec(self, code: str) -> dict:
        lim = self.lim
        with self.lock:
            self.last_validated_final = None
            self._stop.clear()
            t0 = time.monotonic()
            before = dict(self.stats)
            try:
                self._check(t0 + lim["timeout_s"])
                if self.proc is None or self.proc.poll() is not None:
                    self._spawn(t0 + lim["timeout_s"])
                reply = self._request({"type": "exec", "code": code}, t0 + lim["timeout_s"])
                if reply.get("error"):
                    reply["final"] = None
                if reply.get("final") is not None:
                    try:
                        reply["citations"] = self._validate_final(reply["final"], reply.get("citations", []))
                        self.last_validated_final = {"answer": reply["final"], "citations": reply["citations"]}
                    except ValueError as exc:
                        reply["final"], reply["error"] = None, str(exc)
            except TimeoutError:
                self.kill()
                self.stats["restarts"] += 1
                reply = {"stdout": "", "final": None, "error":
                         f"cell exceeded {lim['timeout_s']}s and was killed; next cell reloads "
                         "`context`, other variables are gone"}
            except RuntimeError as exc:
                self.kill()
                self.stats["restarts"] += 1
                reply = {"stdout": "", "final": None, "error": f"{exc}; next cell reloads context, variables lost"}
            self.stats["cells"] += 1
        out = reply.get("stdout") or ""
        shown = out if len(out) <= lim["max_output_chars"] else (
            out[: lim["max_output_chars"]] + f"\n... [{len(out) - lim['max_output_chars']} more chars truncated;"
            " keep large values in variables and print summaries]")
        ms = int((time.monotonic() - t0) * 1000)
        self.log("cell", code=code, stdout_chars=len(out), error=reply.get("error"), ms=ms,
                 final=reply.get("final"), citations=reply.get("citations") if reply.get("final") is not None else None,
                 subcalls=self.stats["subcalls"] - before["subcalls"])
        result = {"stdout": shown, "error": reply.get("error"), "ms": ms,
                  "subcalls_this_cell": self.stats["subcalls"] - before["subcalls"],
                  "budget": {"subcalls_used": self.stats["subcalls"], "subcalls_max": lim["max_subcalls"],
                             "max_depth": lim["max_depth"], "max_tokens": lim["max_tokens"],
                             "cost_usd": self.stats["cost_usd"], "cost_unknown": self.stats["cost_unknown"],
                             "max_cost": lim["max_cost"]}}
        if reply.get("final") is not None:
            result["final"] = reply["final"]
            result["citations"] = reply["citations"]
        return result


_sessions: dict[tuple, Session] = {}
_sessions_lock = threading.Lock()


def session_for(task_id: str = "", session_id: str = "", **_kw) -> Session:
    key = (str(_data_dir()), str(task_id), str(session_id))
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = Session(json.dumps(key))
        return _sessions[key]


@atexit.register
def _shutdown() -> None:
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for s in sessions:
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
        "the answer. FINAL_VAR('name', citations=[cite(source,start,end)]) returns an existing "
        "variable; all finals over loaded sources need verified character spans. Depth is 1: "
        "sub-LLMs see no tools or implicit context. Probe structure, then use llm_batch for semantic "
        "extraction over ALL relevant chunks, store buffers and aggregate them. Do not solve a "
        "long semantic task by printing context into your own prompt. Outputs are truncated.\n\n"
        + RLM_GUIDANCE),
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
    for task in {"rlm_subcall", limits(ctx)["subcall_task"]}:
        ctx.register_auxiliary_task(task, display_name="RLM sub-call",
                                    description="Depth-1 sub-model behind llm_query/llm_batch")
    ctx.register_command("rlm", _rlm_command, description="Show RLM budgets, usage and trajectory logs")
    def existing_session(task_id, session_id):
        with _sessions_lock:
            return _sessions.get((str(_data_dir()), task_id, session_id))

    gate = FinalGate(existing_session)

    def reset_session(session_id='', **_):
        gate.reset(session_id=session_id)
        profile = str(_data_dir())
        with _sessions_lock:
            keys = [key for key in _sessions if key[0] == profile and key[2] == session_id]
            sessions = [_sessions.pop(key) for key in keys]
        for session in sessions:
            session.kill()

    ctx.register_hook('pre_llm_call', gate.begin)
    ctx.register_hook('transform_llm_output', gate.transform)
    ctx.register_hook('on_session_start', reset_session)
    ctx.register_hook('on_session_reset', reset_session)
    ctx.register_hook('on_session_end', gate.end)
    ctx.register_middleware('tool_execution', gate.execute)
    ctx.on_unload(_shutdown)
