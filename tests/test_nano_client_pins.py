"""Pin bench/nano_client.py (Client chunk collection, ACP handshake, runtime payload, snapshot pick-up).

`acp` is not installed here, so a fake module is injected via sys.modules for the import only.
"""
import asyncio
import contextlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).resolve().parent.parent / "bench" / "nano_client.py"
META = "ai.prime.rlm/session-v1"


class FakeAgent:
    def __init__(self, client, closed, resp):
        self.client, self.closed, self.resp, self.calls = client, closed, resp, []

    async def initialize(self, version):
        self.calls.append(("initialize", version))

    async def new_session(self, **kw):
        self.calls.append(("new_session", kw))
        return types.SimpleNamespace(session_id="S1")

    async def prompt(self, session_id, prompt):
        self.calls.append(("prompt", session_id, prompt))
        for t in ("he", "llo"):
            await self.client.session_update(session_id, types.SimpleNamespace(
                session_update="agent_message_chunk", content=types.SimpleNamespace(text=t)))
        await self.client.session_update(session_id, types.SimpleNamespace())  # no session_update attr
        await self.client.session_update(session_id, types.SimpleNamespace(
            session_update="tool_call", content=types.SimpleNamespace(text="IGNORED")))
        return self.resp

    async def close_session(self, session_id):
        self.calls.append(("close_session", session_id))
        return self.closed


def load(closed, resp):
    spawned = {}
    holder = {}

    @contextlib.asynccontextmanager
    async def spawn(client, *argv, **kw):
        spawned.update(argv=argv, kw=kw)
        holder["agent"] = FakeAgent(client, closed, resp)
        yield holder["agent"], None

    acp = types.ModuleType("acp")
    acp.PROTOCOL_VERSION = 7
    acp.spawn_agent_process = spawn
    acp.text_block = lambda t: {"text": t}
    with mock.patch.dict(sys.modules, {"acp": acp}):
        spec = importlib.util.spec_from_file_location("nano_client_under_test", PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod, spawned, holder


ENV = {"NANO_MODEL": "m1", "NANO_BASE_URL": "http://u", "NANO_API_KEY": "k"}


def run(closed=None, resp=None, env=None):
    mod, spawned, holder = load(closed, resp)
    out = io.StringIO()
    with mock.patch.dict(os.environ, {**ENV, **(env or {})}, clear=True), contextlib.redirect_stdout(out):
        asyncio.run(mod.main("PROMPT"))
    lines = out.getvalue().splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0]), spawned, holder["agent"]


class NanoClientPins(unittest.TestCase):
    def test_answer_joins_only_message_chunks_and_prints_one_json_line(self):
        res, _, _ = run()
        self.assertEqual(res["answer"], "hello")
        self.assertIsNone(res["snapshot"])
        self.assertIsInstance(res["wall_s"], float)

    def test_unknown_client_callbacks_are_noops(self):
        mod, _, _ = load(None, None)
        c = mod.Client()
        self.assertIsNone(asyncio.run(c.request_permission(1, x=2)))
        self.assertEqual(c.text, [])

    def test_handshake_order_spawn_args_and_prompt(self):
        _, spawned, agent = run()
        self.assertEqual(spawned["argv"], ("rlm", "--acp"))
        self.assertEqual(spawned["kw"], {"transport_kwargs": {"limit": 64 * 1024 * 1024}})
        names = [c[0] for c in agent.calls]
        self.assertEqual(names, ["initialize", "new_session", "prompt", "close_session"])
        self.assertEqual(agent.calls[0][1], 7)
        self.assertEqual(agent.calls[2], ("prompt", "S1", [{"text": "PROMPT"}]))
        self.assertEqual(agent.calls[3], ("close_session", "S1"))

    def test_rlm_bin_env_overrides_binary(self):
        _, spawned, _ = run(env={"NANO_RLM_BIN": "/x/rlm"})
        self.assertEqual(spawned["argv"], ("/x/rlm", "--acp"))

    def test_runtime_payload_from_env(self):
        _, _, agent = run(env={"NANO_POLICY": '{"max_depth": 2}'})
        kw = agent.calls[1][1]
        self.assertEqual(kw["cwd"], os.getcwd())
        self.assertEqual(kw["mcp_servers"], [])
        rt = kw["ai.prime.rlm/runtime-v1"]
        self.assertRegex(rt.pop("session_id"), r"^bench-\d+$")
        self.assertEqual(rt, {
            "model": "m1", "provider": {"base_url": "http://u", "api_key": "k", "headers": {}},
            "policy": {"max_depth": 2}, "system_prompt_path": None, "append_to_system_prompt": None,
            "skills": ["edit"], "kernel_env": {}, "search_api_key": None})

    def test_default_policy_is_empty(self):
        _, _, agent = run()
        self.assertEqual(agent.calls[1][1]["ai.prime.rlm/runtime-v1"]["policy"], {})

    def test_snapshot_prefers_close_response_over_prompt_response(self):
        closed = types.SimpleNamespace(field_meta={META: {"from": "closed"}})
        resp = types.SimpleNamespace(field_meta={META: {"from": "resp"}})
        self.assertEqual(run(closed, resp)[0]["snapshot"], {"from": "closed"})

    def test_snapshot_falls_back_to_prompt_response_and_underscore_meta(self):
        closed = types.SimpleNamespace(field_meta={"other": 1})
        resp = types.SimpleNamespace(_meta={META: {"from": "resp"}})
        self.assertEqual(run(closed, resp)[0]["snapshot"], {"from": "resp"})

    def test_non_dict_meta_is_ignored(self):
        closed = types.SimpleNamespace(field_meta="junk")
        self.assertIsNone(run(closed, None)[0]["snapshot"])


if __name__ == "__main__":
    unittest.main()
