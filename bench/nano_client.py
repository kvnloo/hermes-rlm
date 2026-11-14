"""Drive PrimeIntellect-ai/nano-rlm over ACP for one prompt; print one JSON result line.

Run with nano-rlm's own venv python, cwd = the task workspace:
  NANO_BASE_URL=... NANO_API_KEY=... NANO_MODEL=... <nano-rlm>/.venv/bin/python nano_client.py "<prompt>"
Mirrors nano-rlm's README quickstart; policy defaults are nano-rlm's own (max_depth 1, 1M tokens).

reuse-checked: nano-rlm README quickstart client -- same code, plus timing and snapshot capture.
"""
import asyncio
import json
import os
import sys
import time

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block


class Client:
    def __init__(self):
        self.text = []

    async def session_update(self, session_id, update, **kwargs):
        if getattr(update, "session_update", None) == "agent_message_chunk":
            self.text.append(update.content.text)

    def __getattr__(self, name):  # accept any other client callback
        async def _noop(*a, **k):
            return None
        return _noop


async def main(prompt: str) -> None:
    runtime = {
        "session_id": f"bench-{int(time.time())}", "model": os.environ["NANO_MODEL"],
        "provider": {"base_url": os.environ["NANO_BASE_URL"], "api_key": os.environ["NANO_API_KEY"],
                     "headers": {}},
        "policy": json.loads(os.environ.get("NANO_POLICY", "{}")), "system_prompt_path": None,
        "append_to_system_prompt": None, "skills": ["edit"], "kernel_env": {}, "search_api_key": None,
    }
    client = Client()
    t0 = time.monotonic()
    snap = None
    async with spawn_agent_process(client, os.environ.get("NANO_RLM_BIN", "rlm"), "--acp",
                                   transport_kwargs={"limit": 64 * 1024 * 1024}) as (agent, _p):
        await agent.initialize(PROTOCOL_VERSION)
        session = await agent.new_session(cwd=os.getcwd(), mcp_servers=[],
                                          **{"ai.prime.rlm/runtime-v1": runtime})
        resp = await agent.prompt(session_id=session.session_id, prompt=[text_block(prompt)])
        closed = await agent.close_session(session.session_id)
        for r in (closed, resp):
            meta = getattr(r, "field_meta", None) or getattr(r, "_meta", None) or {}
            if isinstance(meta, dict) and meta.get("ai.prime.rlm/session-v1"):
                snap = meta["ai.prime.rlm/session-v1"]
                break
    print(json.dumps({"answer": "".join(client.text), "wall_s": round(time.monotonic() - t0, 2),
                      "snapshot": snap}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
