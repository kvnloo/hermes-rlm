"""One-turn Hermes library ingress without the CLI's per-home gateway startup.

reuse-checked: AIAgent init, _load_tools/get_tool_definitions, TurnFacadeMixin and CLI caller.
Plugin discovery, tool-search assembly, dispatch and auxiliary LLM calls remain Hermes-owned.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import uuid


def result_summary(result, agent):
    calls = [c.get("function", {}).get("name", c.get("name"))
             for m in result.get("messages", []) for c in (m.get("tool_calls") or [])]
    return {"answer": result.get("final_response") or "", "session_ids": [agent.session_id],
            "hermes_ingress": "library", "library_error": bool(result.get("error")),
            "library_partial": bool(result.get("partial")),
            "completed": result.get("completed"), "failed": result.get("failed"),
            "interrupted": result.get("interrupted"), "turn_exit_reason": result.get("turn_exit_reason"),
            "discovery_traces": [n for n in calls if n in {"tool_search", "tool_describe", "tool_call"}],
            "tool_calls": len(calls), "library_tools": sorted(agent.valid_tool_names),
            "host_root_input_tokens": agent.session_input_tokens,
            "host_root_output_tokens": agent.session_output_tokens,
            "host_api_calls": agent.session_api_calls}


def run(args):
    # Import from the selected source via PYTHONPATH, after HERMES_HOME is set by
    # the controller. AIAgent performs real plugin discovery before tool assembly.
    from run_agent import AIAgent

    agent = AIAgent(base_url=args.base_url, api_key=os.environ[args.key_env],
                    provider=args.provider, requested_provider=args.provider,
                    api_mode="chat_completions", model=args.model,
                    max_iterations=args.max_iterations, max_tokens=args.max_tokens,
                    enabled_toolsets=["rlm"], quiet_mode=True, save_trajectories=False,
                    skip_context_files=True, load_soul_identity=True,
                    skip_memory=True, skip_background_review=True,
                    session_db=None, fallback_model={},
                    session_id="paper-" + uuid.uuid4().hex, platform="cli", cwd=os.getcwd())
    try:
        result = agent.run_conversation(user_message=args.prompt)
        if args.transcript:
            Path(args.transcript).write_text(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        return result_summary(result, agent)
    finally:
        agent.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt")
    p.add_argument("--base-url", default=os.environ.get("PAPER_BASE_URL"), required=not os.environ.get("PAPER_BASE_URL"))
    p.add_argument("--provider", default=os.environ.get("PAPER_PROVIDER", "paper-root"))
    p.add_argument("--key-env", default="PAPER_PROXY_KEY")
    p.add_argument("--model", default=os.environ.get("PAPER_MODEL"), required=not os.environ.get("PAPER_MODEL"))
    p.add_argument("--max-iterations", type=int, default=int(os.environ.get("PAPER_MAX_ITERATIONS", "10")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("PAPER_MAX_TOKENS", "3072")))
    p.add_argument("--transcript")
    args = p.parse_args(argv)
    # Keep stdout a single JSON result even if a dependency prints during import.
    with redirect_stdout(sys.stderr):
        summary = run(args)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    incomplete = (summary["completed"] is not True or summary["failed"] or summary["interrupted"]
                  or any(s in str(summary["turn_exit_reason"] or "").lower()
                         for s in ("max_iterations", "budget", "interrupt")))
    return 1 if summary["library_error"] or summary["library_partial"] or incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
