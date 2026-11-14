"""Local model router and durable upstream-attempt cap for the paper matrix.

reuse-checked: run_bench.py provider guard and nano ACP semantic session headers.
Only accounting metadata is persisted here, never prompts, responses or credentials.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS = {
    "nvidia/nemotron-3-ultra-550b-a55b": ("nim", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "nvidia/nemotron-3.5-lightning-30b-a3b": ("nim", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "qwen/qwen3.8-27b": ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
ROOT = "nvidia/nemotron-3-ultra-550b-a55b"
SUB = "nvidia/nemotron-3.5-lightning-30b-a3b"


class Ledger:
    def __init__(self, path: Path, cap: int = 2999):
        if not 0 < cap < 3000:
            raise ValueError("provider cap must be between 1 and 2999")
        self.path, self.cap = path, cap
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, run TEXT, "
                       "provider TEXT, role TEXT, model TEXT, status INTEGER, latency REAL, "
                       "prompt_tokens INTEGER, completion_tokens INTEGER, usage_known INTEGER, ts REAL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, run, model, role):
        provider = MODELS[model][0]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            n = db.execute("SELECT count(*) FROM attempts WHERE provider=?", (provider,)).fetchone()[0]
            if n >= self.cap:
                raise RuntimeError("provider_request_cap")
            return db.execute("INSERT INTO attempts(run,provider,role,model,ts) VALUES(?,?,?,?,?)",
                              (run, provider, role, model, time.time())).lastrowid

    def finish(self, request_id, status, latency, usage):
        with self.connect() as db:
            db.execute("UPDATE attempts SET status=?,latency=?,prompt_tokens=?,completion_tokens=?,"
                       "usage_known=? WHERE id=?", (status, latency, usage.get("prompt_tokens"),
                       usage.get("completion_tokens"), int(bool(usage)), request_id))

    def summary(self, run):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute("SELECT * FROM attempts WHERE run=? ORDER BY id", (run,))]
        out = {"request_attempts": len(rows), "requests": rows,
               "provider_failure": any(r["status"] != 200 for r in rows),
               "usage_complete": all(r["usage_known"] for r in rows)}
        for role in ("root", "sub"):
            rr = [r for r in rows if r["role"] == role]
            out[f"{role}_api_attempts"] = len(rr)
            for kind in ("prompt_tokens", "completion_tokens"):
                out[f"{role}_{kind}"] = sum(r[kind] or 0 for r in rr)
        return out


class Router:
    def __init__(self, ledger, run, root=ROOT, sub=SUB, backoff=5, upstream_timeout=120,
                 disable_thinking=False):
        self.ledger, self.run, self.root, self.sub = ledger, run, root, sub
        self.backoff, self.upstream_timeout = backoff, upstream_timeout
        self.disable_thinking = disable_thinking
        self.nano_root = None
        self.lock = threading.Lock()
        self.blocked = set()
        self.observed = []
        self.errors = []
        self.logical_calls = {}
        router = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, code, data, content_type="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                self.reply(200, json.dumps({"object": "list", "data": [
                    {"id": m, "object": "model", "owned_by": "paper-bench"} for m in (root, sub)]}).encode())

            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    role = router.role(self.path, self.headers.get("X-ACP-Agent-Session-ID"))
                    model = root if role == "root" else sub
                    body["model"] = model
                    tools = [t.get("function", {}).get("name") for t in body.get("tools", [])]
                    calls = [c.get("function", {}).get("name") for m in body.get("messages", [])
                             for c in m.get("tool_calls", [])]
                    router.observed.append({"role": role, "tools": tools, "tool_calls": calls})
                    logical_id = self.headers.get("X-ACP-Model-Request-ID") if self.path.startswith("/nano/") else None
                    if self.path.startswith("/nano/") and not logical_id:
                        raise ValueError("nano request missing ACP model request identity")
                    status, data, ct = router.forward(body, model, role, logical_id=logical_id)
                    self.reply(status, data, ct)
                except Exception as exc:
                    router.errors.append(type(exc).__name__ + ":" + str(exc) if isinstance(exc, ValueError)
                                         else type(exc).__name__)
                    self.reply(503, b'{"error":{"message":"paper proxy refused request"}}')

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def role(self, path, session):
        if path.startswith("/sub/"):
            return "sub"
        if path.startswith("/root/"):
            return "root"
        if not path.startswith("/nano/") or not session:
            raise ValueError("nano request missing ACP session identity")
        with self.lock:
            if self.nano_root is None:
                self.nano_root = session
            return "root" if session == self.nano_root else "sub"

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def logical_snapshot(self):
        with self.lock:
            return [dict(value) for value in self.logical_calls.values()]

    def forward(self, body, model, role, logical_id=None):
        logical_id = logical_id or uuid.uuid4().hex
        with self.lock:
            previous = self.logical_calls.get(logical_id)
            if previous and (previous["role"] != role or previous["model"] != model or previous["pending"]):
                raise ValueError("conflicting logical request identity")
            self.logical_calls[logical_id] = {"id": logical_id, "role": role, "model": model,
                                              "status": None, "pending": True}
        try:
            result = self._forward(body, model, role)
        except Exception:
            with self.lock:
                self.logical_calls[logical_id].update(status=599, pending=False)
            raise
        with self.lock:
            self.logical_calls[logical_id].update(status=result[0], pending=False)
        return result

    def _forward(self, body, model, role):
        provider, base, key_env = MODELS[model]
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"missing credential environment {key_env}")
        if provider in self.blocked:
            return 503, b'{"error":{"message":"provider circuit open"}}', "application/json"
        if self.disable_thinking and provider == "nim":
            body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}),
                                            "enable_thinking": False}
        # Buffer SSE unchanged. Request usage in its terminal chunk for actual accounting.
        if body.get("stream"):
            body["stream_options"] = {"include_usage": True}
        for attempt in range(3):
            request_id = self.ledger.reserve(self.run, model, role)
            start = time.monotonic()
            req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                    {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.upstream_timeout) as response:
                    data, status = response.read(), response.status
                    ct = response.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as exc:
                # Never persist provider exception bodies, which can echo request content.
                status, data, ct = exc.code, json.dumps({"error": {"message": f"upstream HTTP {exc.code}"}}).encode(), "application/json"
                exc.close()
            except (OSError, TimeoutError):
                status, data, ct = 599, b'{"error":{"message":"upstream transport error"}}', "application/json"
            usage = response_usage(data)
            self.ledger.finish(request_id, status, time.monotonic() - start, usage)
            if status not in (429, 401):
                return status, data, ct
            if attempt < 2:
                time.sleep(self.backoff * 2 ** attempt)
        self.blocked.add(provider)
        return status, data, ct


def response_usage(data):
    usage = {}
    try:
        if data.lstrip().startswith(b"{"):
            return json.loads(data).get("usage") or {}
        for line in data.splitlines():
            if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                usage = json.loads(line[6:]).get("usage") or usage
    except (ValueError, TypeError):
        pass
    return usage
