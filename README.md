# hermes-rlm

Experimental [Recursive Language Model](https://arxiv.org/abs/2512.24601) plugin for Hermes Agent. The corpus stays in a sandboxed Python REPL, outside the root model's prompt. The root uses `rlm_exec` to inspect it and delegates slices through Hermes's `ctx.llm.acomplete` lane.

## Enable and configure

Install `rlm/` as a Hermes plugin. Enable the plugin and opt in to its `rlm` toolset. It registers only `rlm_load` and `rlm_exec`, with no core-tool changes. Linux, `bwrap`, and system Python are required for normal use. The worker has no network, host filesystem access, or credentials.

Plugin settings belong in `config.yaml`:

```yaml
plugins:
  enabled: [rlm]
  entries:
    rlm:
      settings:
        max_subcalls: 200
        max_depth: 1
        max_tokens: 4000
        timeout_s: 600
        max_cost: null
        subcall_task: rlm_subcall
        concurrency: 8
        max_prompt_chars: 400000
        max_output_chars: 6000
        max_corpus_chars: 400000000
auxiliary:
  rlm_subcall:
    provider: your-configured-provider
    model: your-sub-model
```

Settings use `ctx.get_config`, which reads `plugins.entries.rlm.settings` before Hermes's legacy `config` subtree. Missing settings fall back to deprecated `RLM_*` environment variables, then defaults. `RLM_SUBCALL_MAX_TOKENS` and `RLM_CELL_TIMEOUT_S` are legacy aliases for `max_tokens` and `timeout_s`. Invalid settings fail rather than silently removing limits.

`max_depth` supports only `0` and `1`. Depth zero or `max_subcalls: 0` disables subcalls. At depth one, children receive only a completion prompt, with no tools, REPL, or further recursion. `max_tokens` caps each completion, including per-call overrides. `timeout_s` sets each cell's deadline, including in-flight batches. Deadline polling cannot preempt synchronous host-client setup. Subcall and cost counters last for the session and do not reset on corpus reload.

`max_cost` is an observed USD admission cutoff, not a prepaid spending cap. Concurrent requests can overshoot it. If a provider does not report cost, the next requests fail closed when a finite cutoff is configured. `null` disables that cutoff. Usage reports include `cost_unknown`; unavailable prices never count as known zero cost.

## REPL protocol

`rlm_load` accepts files, directories, or globs and returns metadata, source names, lengths, budgets, and guidance, never source text. A single file becomes a string `context`; multiple files become a dictionary keyed by source name. The root explicitly inspects selected slices with `rlm_exec`.

These functions are available in each persistent kernel:

- `llm_query(prompt, max_tokens=None)` returns a sub-model answer.
- `llm_batch(prompts, max_tokens=None)` returns answers in input order. `llm_query_batched` is an alias.
- `cite(source, start, end)` identifies an original source span.
- `FINAL(answer, citations=[...])` submits an answer.
- `FINAL_VAR('variable_name', citations=[...])` submits an existing answer buffer.

Example `rlm_exec` cell after loading a file named `source.txt`:

```python
start = context.index('needle=')
end = context.index('\n', start) if '\n' in context[start:] else len(context)
evidence = [cite('source.txt', start, end)]
answer = llm_query(f'What value is stated here? {context[start:end]}')
FINAL_VAR('answer', citations=evidence)
```

Citations use zero-based Unicode character offsets with an exclusive end, not byte offsets. The host validates source names, ranges, and optional quotes against the original text, even if REPL code changes `context`. Answers over nonempty loaded sources require a citation. Validated results include the exact quote. This proves span integrity, not that the span supports the answer's meaning.

The guidance adapts the authors' [paper-era prompt](https://github.com/alexzhang13/rlm/blob/0243c641be434b864505b998dcf7038faafa1086/rlm/utils/prompts.py). It retains metadata-only initialization, programmatic chunking, semantic subcalls, buffered aggregation, and `FINAL`/`FINAL_VAR`. Hermes tool calls replace REPL fences, and executable final functions add source validation. This is depth-one RLM, not upstream's later recursive-child implementation.

Cancellation polls Hermes's thread-scoped interrupt predicate on the tool caller thread. It cancels and awaits active async completions, discards queued requests, and destroys the kernel. Only dispatched attempts consume subcall slots, including failed or cancelled attempts. Cancellation cannot retract work already accepted by a remote provider. Production has no blocking sync-LLM fallback.

For a session that has loaded RLM context, the finished user-facing reply is fail-closed. A registered `transform_llm_output` hook returns only the current turn's host-validated `FINAL` answer and formatted `source[start:end]` quotes. Without one, it replaces root prose with an explicit withholding notice. Contradictory root prose never overrides a validated answer. Session start, new turns, reloads, later cells, failed finals, and timeouts clear authorization. Exact task, session, and turn identities come from native `pre_llm_call` and tool-execution middleware; missing or mismatched identities refuse RLM tool execution. Sessions that have not loaded RLM context are unchanged.

This is a finished-response contract, not a streaming redaction guarantee. Hermes's supported `hold_stream` flag applies profile-wide, so this plugin deliberately leaves it off. Interim streamed text may already have been visible before the final response is replaced. No scoped stream-hold API is assumed.

Each session has an independent process, namespace, budgets, and trajectory log under its profile's plugin data directory. Task and session identifiers both participate in identity. `/rlm` shows budgets and logs. Logs can contain code and answers; treat them as sensitive data.

## Tests

Cross-platform tests explicitly use unsandboxed **test** workers, without requiring Hermes or Linux:

```sh
python -m pytest -q rlm/tests --rlm-unsandboxed
```

Actual Linux namespace and isolation checks require bubblewrap:

```sh
python -m pytest -q rlm/tests --rlm-sandbox
```

The unsandboxed constructor option is test-only. It is not a plugin setting or an environment escape hatch. The optional host transport test uses `RLM_HERMES_SOURCE` to locate a Hermes checkout and verifies that `/stop` closes a hanging HTTP connection. It uses only a temporary home and local server.

For replication harnesses, see `bench/` and the z0evals study `rlm-paper-replication-v1`.
