"""rlm: Recursive Language Model mode for Hermes (arXiv:2512.24601, NousResearch/hermes-agent#93303).

The corpus lives as ``context`` inside a persistent, sandboxed Python kernel instead of the prompt.
The root model (the normal Hermes agent) inspects it with ``rlm_exec`` code cells and fans slices out
to a sub-model with ``llm_query``/``llm_batch``. Sub-calls go through the plugin LLM lane ``ctx.llm``
(auxiliary task ``rlm_subcall``), so provider, credentials and usage accounting stay Hermes'; the kernel never
sees a key. Budgets (sub-calls, prompt size, cell time) are enforced here, host-side. Every load,
cell and sub-call is appended to ``<hermes home>/plugin-data/rlm/runs/<identity-sha256>.jsonl``.

reuse-checked: execute_code / delegate_task -- host-privileged and unbudgeted; RLM needs a sandboxed
corpus REPL whose sub-calls reuse ctx.llm(task=rlm_subcall). Wired via plugins.enabled (config.yaml).
"""
from __future__ import annotations

import atexit
import contextvars
import hashlib
import glob as _glob
import json
import logging
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
    return r.text or "", {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                          "cost_usd": u.cost_usd}


def _task_path(raw: str, task_id: str) -> str:
    """Anchor a relative path at the task's workspace (what read_file/terminal see), not the process cwd."""
    raw = os.path.expanduser(raw)
    if os.path.isabs(raw):
        return raw
    try:
        base = _CTX.session_cwd()
    except Exception as exc:
        raise RuntimeError("RLM cannot resolve the session workspace for relative paths") from exc
    if not isinstance(base, str) or not os.path.isabs(base):
        raise RuntimeError("RLM requires an absolute session workspace for relative paths")
    return str(Path(base) / raw)


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
    # Preserve first-seen order while counting each real file once.
    seen: dict[Path, Path] = {}
    for f in files:
        seen.setdefault(f.resolve(), f)
    files = list(seen.values())
    common = os.path.commonpath([str(f.parent) for f in files]) if files else ""
    corpus, total, skipped = {}, 0, 0
    for f in files:
        data = f.read_bytes()
        if b"\0" in data[:8192]:
            skipped += 1
            continue
        text = data.decode("utf-8", "replace")
        source = os.path.relpath(f, common) if common else str(f)
        # Match the retained mapping: later text replaces the same source key.
        total += len(text) - len(corpus.get(source, ""))
        if total > max_chars:
            raise ValueError(f"corpus exceeds RLM_MAX_CORPUS_CHARS={max_chars}")
        corpus[source] = text
    return corpus, skipped


class Session:
    """One sandboxed kernel plus budgets and a trajectory log for one Hermes session."""

    def __init__(self, key: str, subcall=default_subcall, log_dir: Path | None = None):
        self.key, self.subcall, self.lim = key, subcall, limits()
        if log_dir is None:
            log_dir = Path(os.environ.get("RLM_LOG_DIR") or _data_dir() / "runs")
        digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()
        self.log_path = log_dir / f"{digest}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self._read_buffer = b""
        self.context = None
        self._closing = False
        self.stats = {"subcalls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "subcall_prompt_chars": 0, "cells": 0, "restarts": 0}
        self.lock = threading.Lock()

    def log(self, event: str, **data) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **data}) + "\n")

    def _ensure_open(self) -> None:
        if self._closing:
            raise RuntimeError("RLM session is closed")

    def _spawn(self) -> None:
        self._ensure_open()
        self._read_buffer = b""
        self.proc = subprocess.Popen(sandbox_argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            # Unload can race with Popen before the child is assigned to proc.
            self._ensure_open()
            if self.context is not None:
                self._request({"type": "load", "context": self.context}, time.monotonic() + 120)
        except BaseException:
            # A failed handshake must not leave a child for the next cell to reuse.
            self._teardown()
            raise

    def kill(self) -> None:
        """Stop the kernel from any thread. An exchange in flight owns the pipes, so
        it only gets the child killed and tears itself down on the resulting EOF."""
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.kill()
        if self.lock.acquire(blocking=False):
            try:
                self._teardown()
            finally:
                self.lock.release()

    def _teardown(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        finally:
            if self.proc:
                for pipe in (self.proc.stdin, self.proc.stdout):
                    try:
                        pipe.close()
                    except OSError:
                        pass
            self.proc = None
            self._read_buffer = b""

    def _read(self, deadline: float) -> dict:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            if b"\n" in self._read_buffer:
                line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                return json.loads(line)
            ready, _, _ = select.select([self.proc.stdout], [], [], min(left, 1.0))
            if ready:
                # Only this reader consumes stdout; do not enter TextIOWrapper
                # readline(), which can wait past the deadline for a newline.
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("sandbox kernel exited")
                self._read_buffer += chunk

    def _send(self, msg: dict, deadline: float) -> None:
        payload = memoryview((json.dumps(msg) + "\n").encode("utf-8"))
        fd = self.proc.stdin.fileno()
        # This is the sole stdin writer. Avoid TextIOWrapper buffering/flush,
        # which can block indefinitely when a live child stops reading.
        os.set_blocking(fd, False)
        while payload:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            _, ready, _ = select.select([], [fd], [], min(left, 1.0))
            if not ready:
                continue
            try:
                sent = os.write(fd, payload[:65536])
            except BlockingIOError:
                continue
            except BrokenPipeError:
                raise RuntimeError("sandbox kernel exited") from None
            if sent <= 0:
                raise RuntimeError("sandbox kernel write made no progress")
            payload = payload[sent:]

    def _request(self, msg: dict, deadline: float) -> dict:
        self._send(msg, deadline)
        while True:
            reply = self._read(deadline)
            if reply["type"] == "done":
                # Validate before leaving the guarded exchange: formatting an
                # invalid completion later must not leave a live child attached.
                for field in ("stdout", "error", "final"):
                    value = reply.get(field)
                    if value is not None and not isinstance(value, str):
                        raise RuntimeError(f"invalid sandbox done field: {field}")
                return reply
            if reply["type"] != "llm":
                raise RuntimeError("invalid sandbox frame type")
            self._send(self._serve_llm(reply), deadline)

    def _serve_llm(self, req: dict) -> dict:
        if "id" not in req:
            raise RuntimeError("sandbox llm request missing id")
        prompts = req.get("prompts")
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            return {"type": "llm_result", "id": req["id"], "error":
                    "prompts must be a list of strings"}
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
        tokens, cap = req.get("max_tokens"), lim["subcall_max_tokens"]
        # Preserve the catalog API's integer-zero default without accepting
        # falsey containers or booleans as token counts.
        if tokens is None or (type(tokens) is int and tokens == 0):
            tokens = cap
        if type(tokens) is not int or tokens < 1 or type(cap) is not int or cap < 1:
            return {"type": "llm_result", "id": req["id"], "error":
                    "max_tokens and RLM_SUBCALL_MAX_TOKENS must be a positive integer"}
        max_tokens = min(tokens, cap, 32_000)
        self.stats["subcalls"] += len(prompts)

        def one(prompt: str):
            t0 = time.monotonic()
            try:
                text, usage = self.subcall(prompt, max_tokens)
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
            self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
        return {"type": "llm_result", "id": req["id"], "results": [t for t, _ in out]}

    def load(self, paths, task_id: str = "default") -> dict:
        self._ensure_open()
        corpus, skipped = load_corpus(paths, self.lim["max_corpus_chars"], task_id)
        with self.lock:
            self._ensure_open()
            self._teardown()
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
            self._ensure_open()
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            t0 = time.monotonic()
            before = dict(self.stats)
            try:
                reply = self._request({"type": "exec", "code": code}, t0 + lim["cell_timeout_s"])
            except TimeoutError:
                self._teardown()
                self.stats["restarts"] += not self._closing
                note = ("session closed" if self._closing else
                        "kernel restarted with `context` reloaded, other variables are gone")
                reply = {"stdout": "", "final": None, "error":
                         f"cell exceeded {lim['cell_timeout_s']}s and was killed; {note}"}
            except RuntimeError as exc:
                self._teardown()
                note = "session closed" if self._closing else "kernel restarted, variables lost"
                self.stats["restarts"] += not self._closing
                reply = {"stdout": "", "final": None, "error": f"{exc}; {note}"}
            except BaseException:
                # Transport/decoding failure or interruption leaves reply framing
                # uncertain. Never let the next cell reuse that child.
                self._teardown()
                self.stats["restarts"] += 1
                raise
            self.stats["cells"] += 1
            subcalls_used = self.stats["subcalls"]
            subcalls_this_cell = subcalls_used - before["subcalls"]
        out = reply.get("stdout") or ""
        shown = out if len(out) <= lim["max_output_chars"] else (
            out[: lim["max_output_chars"]] + f"\n... [{len(out) - lim['max_output_chars']} more chars truncated;"
            " keep large values in variables and print summaries]")
        ms = int((time.monotonic() - t0) * 1000)
        self.log("cell", code=code, stdout_chars=len(out), error=reply.get("error"), ms=ms,
                 final=reply.get("final"), subcalls=subcalls_this_cell)
        result = {"stdout": shown, "error": reply.get("error"), "ms": ms,
                  "subcalls_this_cell": subcalls_this_cell,
                  "budget": {"subcalls_used": subcalls_used, "subcalls_max": lim["max_subcalls"]}}
        if reply.get("final") is not None:
            result["final"] = reply["final"]
        return result


_sessions: dict[tuple[str, str, str], Session] = {}
_sessions_lock = threading.Lock()


def session_for(task_id: str = "", session_id: str = "", **_kw) -> Session:
    if not task_id and not session_id:
        raise ValueError("RLM requires a host task_id or session_id")
    key = (str(_data_dir()), str(task_id), str(session_id))
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = Session(json.dumps(key))
        return _sessions[key]


@atexit.register
def _shutdown() -> None:
    with _sessions_lock:
        sessions = list(_sessions.values())
        for session in sessions:
            session._closing = True
        _sessions.clear()
    for session in sessions:
        try:
            session.kill()
        except Exception:
            logging.getLogger(__name__).exception("RLM session cleanup failed; continuing remaining cleanup")


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


def _rlm_command(raw_args: str = "", session_id: str = "", **_kw) -> str:
    header = "RLM limits: " + json.dumps(limits())
    if not isinstance(session_id, str) or not session_id:
        return header + "\nHost session_id required to show RLM session diagnostics"
    profile = str(_data_dir())
    with _sessions_lock:
        rows = [f"{k}: {v.stats} log={v.log_path}" for k, v in _sessions.items()
                if k[0] == profile and k[2] == session_id]
    return header + ("\n" + "\n".join(rows) if rows else "\nno RLM sessions yet")


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
    ctx.on_unload(_shutdown)
