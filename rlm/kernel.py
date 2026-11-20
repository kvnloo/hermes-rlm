"""RLM sandbox kernel: one persistent Python namespace driven by JSON lines on stdin/stdout.

Runs inside bubblewrap (no network, no host filesystem, empty env). Stdlib only. The model's
code sees ``context`` (the corpus), ``llm_query``/``llm_batch`` (sub-model calls proxied to the
host, which owns credentials and budgets) and ``FINAL``. ``print`` output is captured per cell.

reuse-checked: execute_code -- host Python with filesystem/tool access and no corpus/subcall
budget; RLM needs a no-network, no-host-fs kernel. Wired as the rlm plugin's sandbox (__init__.py).
"""
import ast
import io
import json
import os
import sys
import traceback

_proto_out = os.fdopen(os.dup(1), "w", buffering=1)
_proto_in = sys.stdin
_next_id = 0
_final = None
_citations = []


def _send(msg):
    _proto_out.write(json.dumps(msg) + "\n")
    _proto_out.flush()


def _recv():
    line = _proto_in.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def llm_batch(prompts, max_tokens=None):
    """Run sub-model calls concurrently on the host; returns one string per prompt."""
    global _next_id
    if isinstance(prompts, (str, bytes)):
        raise TypeError("llm_batch takes a list of prompts; use llm_query for a single prompt")
    prompts = [str(p) for p in prompts]
    _next_id += 1
    _send({"type": "llm", "id": _next_id, "prompts": prompts, "max_tokens": max_tokens})
    reply = _recv()
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    return reply["results"]


def llm_query(prompt, max_tokens=None):
    """One sub-model call over ``prompt`` (put the slice of ``context`` you need inside it)."""
    return llm_batch([prompt], max_tokens)[0]


def cite(source, start, end):
    """Reference an original source with Unicode character offsets, end exclusive."""
    return {"source": source, "start": start, "end": end}


def FINAL(answer, citations=None):
    """Submit an answer; the host validates every citation against original sources."""
    global _final, _citations
    if isinstance(answer, dict) and 'answer' in answer:
        citations = answer.get('citations', citations)
        answer = answer['answer']
    _final = str(answer)
    _citations = [] if citations is None else citations
    print("FINAL recorded; awaiting host citation validation.")


def FINAL_VAR(variable_name, citations=None):
    """Submit an existing REPL variable by name, without printing its full value."""
    if not isinstance(variable_name, str) or variable_name not in ns:
        raise ValueError('FINAL_VAR needs the name of an existing REPL variable')
    FINAL(ns[variable_name], citations=citations)


ns = {"__name__": "__rlm__", "llm_query": llm_query, "llm_batch": llm_batch,
      "llm_query_batched": llm_batch, "FINAL": FINAL, "FINAL_VAR": FINAL_VAR,
      "cite": cite, "context": None}


def _run(code):
    tree = ast.parse(code, "<cell>", "exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))


def main():
    global _final, _citations
    while True:
        msg = _recv()
        if msg["type"] == "load":
            ns["context"] = msg["context"]
            _send({"type": "done", "stdout": "", "error": None, "final": None})
            continue
        buf, _final, _citations = io.StringIO(), None, []
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        error = None
        try:
            _run(msg["code"])
        except BaseException:  # report every failure, including SystemExit from model code
            error = traceback.format_exc(limit=-3)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        _send({"type": "done", "stdout": buf.getvalue(), "error": error,
               "final": _final, "citations": _citations})


if __name__ == "__main__":
    main()
