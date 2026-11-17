# Paper matrix reference

`paper_matrix.py` runs normalized tasks through five interleaved arms. Dataset preparation and scoring belong to `paper_tasks.py`.

## Invocation

All installation paths come from CLI arguments. The source directory and Python executable are separate arguments. Venv Python symlinks retain their original paths.

```sh
bws-env-exec --scope hermes-jev-provider-bench -- nice -n 10 python3 bench/paper_matrix.py \
  "$MANIFEST" "$OPS/smoke.jsonl" --ops-dir "$OPS" \
  --home-template "$HOME_TEMPLATE" \
  --released-source "$RELEASED_SOURCE" --released-python "$RELEASED_PYTHON" \
  --fork-source "$FORK_SOURCE" --fork-python "$FORK_PYTHON" \
  --nano-source "$NANO_SOURCE" \
  --smallest-sniah --repeats 1 --timeout 150 --max-turns 8 --max-subcalls 4
```

`--nano-python` defaults to `$NANO_SOURCE/.venv/bin/python`. A broader run omits `--smallest-sniah` and uses the default three repeats. `--dry-run` prints pending row identities without inference. `--task-ids` and `--limit-tasks` bound selection. `--arms` accepts a comma-separated subset of the names below.

## Arms

| Arm | Runtime | Root input | Configuration |
|---|---|---|---|
| `base-truncated` | Direct chat completion | Head of context, then question | No system prompt or tools. Default 48,000-character cap. |
| `rlm-no-subcalls` | Released Hermes | File metadata and question | Toolset `rlm`, discovery off, `max_depth=0`, explicit no-subcall guidance. |
| `released-rlm` | Released Hermes | File metadata and question | Toolset `rlm`, discovery off, depth one. |
| `fork-rlm` | Fork Hermes | File metadata and question | Toolset `rlm`, discovery on, `eager: [rlm]`, depth one. |
| `nano-rlm` | nano ACP | File metadata and question | Depth one, local proxy routes distinct child sessions to the submodel. |

`--hermes-ingress library` is the default. `hermes_client.py` calls the selected source's actual `run_agent.AIAgent.run_conversation` directly, without a per-home gateway. Plugin discovery, tool assembly, dispatch, and auxiliary subcalls remain Hermes-owned. `--hermes-ingress cli` retains the canonical Python CLI entrypoint. The ingress and adapter content hash are part of each row's protocol, so old CLI results remain distinct.

Library ingress skips cwd context files, memory, and background review. It loads the same home `SOUL.md` identity. Session-database accounting is optional; proxy accounting remains authoritative. Full library messages are saved in `runs/<run_id>/hermes-conversation.json`, outside the input cwd. Discovery traces also come from those actual messages.

Fresh homes contain only `SOUL.md`, generated YAML at config version 12, and the current plugin. No template credentials, sessions, histories, memories, hooks, or databases are copied. Automatic dependency installation and telemetry collection/upload are disabled.

The proxy's observed request tool names prove root exposure. A Hermes RLM row without exposed `rlm_load` and `rlm_exec`, or without an executed cell, is invalid. Discovery bridge calls are collected from session messages and model-request histories. Zero discovery calls alone does not prove eager exposure. With only `rlm` enabled, the fork can have no deferred tools and therefore no bridge calls.

## Task and scorer contract

Each manifest line contains `id`, `family`, `size_tokens`, `context_path`, `question`, `gold`, and `scoring`. `choices` and `metadata` are optional. Relative context paths resolve against the manifest directory. `context_path` must name one plain file, not a directory or symlink. Evaluator-looking filenames are rejected.

Each run gets a new `input/` working directory containing only an unlabeled copy named `context.txt`. The manifest and gold remain outside that directory. Prompt construction allows only the question, choices, and size metadata. Arbitrary metadata and scoring fields never enter a model prompt. This is a benchmark separation rule, not a hostile-agent filesystem security boundary. Nano's reference code executes host-side Python and is not given a new filesystem sandbox by this harness.

`paper_tasks.score_task(task, answer)` supplies deterministic grading. Numeric returns become `{"score": value}`. A dictionary return must contain `score`. The dataset worker defines exact parsing, OOLONG numeric grades, CodeQA choices, and Pairs F1. The harness never inserts an LLM judge or substitutes binary correctness for a graded score.

`size_tokens` is the dataset label, not a measured root tokenizer count. Base truncation is explicitly character-based, controlled by `--base-context-chars`. It is not presented as exact token-window replication. The chosen cap must leave room for the question and output within the selected provider's model window.

`--disable-thinking` sends `chat_template_kwargs.enable_thinking=false` to every NIM request across every arm, including root, child and base requests. The flag is part of each row's protocol. Omitting it preserves provider defaults. This is a study choice, not the paper's reasoning configuration.

## Accounting and retries

Only these model IDs are accepted:

- `nvidia/nemotron-3-ultra-550b-a55b`, default root through NIM.
- `nvidia/nemotron-3.5-lightning-30b-a3b`, default submodel through NIM.
- `qwen/qwen3.8-27b`, optional model through Groq.

Provider credentials stay in the controller environment. Child processes receive a loopback placeholder, not provider keys. YAML contains environment-variable references only. The proxy stores no prompts, answers, or credentials in its SQLite ledger.

Each upstream attempt reserves a durable slot before network dispatch. All arms and resumes sharing `--ops-dir` share `request-ledger.sqlite3`. The default cap is 2,999 attempts per provider, never 3,000. Requests made outside this harness are not included. The parent must account for other workers and can lower `--provider-cap` to reserve headroom.

HTTP 429 and 401 trigger at most three upstream attempts, with delays of five and ten seconds. Exhaustion opens that provider's circuit for the row. Host and nano retries cannot bypass the circuit. Nano also has reference-implementation retries beyond its configurable SDK retries. Those can consume wall time, but the circuit prevents extra upstream attempts after exhaustion. Every row has a process deadline.

Proxy accounting includes upstream attempts, statuses, latency, and reported root and submodel tokens. `usage_complete=false` means at least one attempt lacked usage. Token sums include known usage only, not estimated tokens for failed requests. Host session counters and plugin attempt counters remain separate to avoid double counting. Plugin logs are discovered by glob inside a unique row directory, not by session-ID filename. One `subcall` event counts once. Cell summary totals do not add calls again.

Nano's native ACP runtime has one inherited model setting and no child-model override. The proxy uses `X-ACP-Agent-Session-ID`: the first session is the root, and other session IDs are children. Missing identity fails closed. This changes transport routing without changing the reference checkout. Nano children are recursive agents, not the paper's plain LM subcalls. That remains a protocol deviation.

## Rows and resume

Results include source SHAs, plugin content hash, harness and scorer hashes, task and context hashes, full answers, process status, validity, score, call counts, timing, and trace metadata. Full stdout and stderr live in `runs/<run_id>/`. Agent homes live in `homes/<run_id>/`, outside the input working directory.

The row key includes task content, arm, repeat, and protocol. Completed invalid rows are terminal too, so an unavailable provider does not cause an automatic retry storm on resume. Increasing the repeat count adds only new repeats. Changing a model, budget, source SHA, plugin, harness, or scorer creates new row keys. Corrupt output is rejected rather than silently skipped. An output lock prevents concurrent controllers from writing the same JSONL file.

The row order rotates the first arm for each task and repeat. Output uses temporary files, not pipes that a detached daemon can keep open. Timeouts preserve partial stdout and stderr, kill the process group and observed descendants, and bound reaping to half a second. This includes observed detached Jupyter kernels. Already accepted remote inference cannot be retracted. A crash between dispatch and response leaves a reserved ledger slot with unknown completion status.

## Verification

```sh
fleet-build --node mbp --light --src-include bench --wait 0 -- \
  python3 -m unittest discover -s bench -p 'test_paper_matrix.py'
```

The bounded single-file check is `python3 bench/test_paper_matrix.py`. It uses local fake HTTP responses and no model credentials. It covers the persistent cap, retry circuit, nano model routing, fresh homes, scoring hooks, hash logs, invalid-row resume, interleaving, prompt isolation, and detached-child timeout cleanup.

## Offline synthetic long-context quick comparison

Generate three small, deterministic, download-free tasks (needle retrieval,
agent-to-project-to-locker multi-hop, and filtered integer aggregation).
Only the evaluator manifest contains gold. The agent receives the generated
unlabeled `context.txt`; all three tasks use the existing deterministic
casefold/whitespace **exact** scorer (`browse_exact`), not an LLM judge.

```sh
SYNTH=/tmp/rlm-synthetic
python -m bench.synthetic_long_context --out "$SYNTH" --records 128
python -m unittest discover -s tests
```

`full.jsonl` and `smoke.jsonl` both contain these three tasks. `--records`
accepts 64, 128, 256 or 512; default 128 keeps runs inexpensive.
`size_tokens` is an explicitly labeled characters/4 estimate, not a
tokenizer measurement. Repeating generation at the same path is byte-stable.

For a bounded model-backed comparison, with model-provider credentials and
the source paths from the invocation above already configured:

```sh
OPS=/tmp/rlm-synthetic-ops
python3 bench/paper_matrix.py "$SYNTH/full.jsonl" "$OPS/rows.jsonl" \
  --ops-dir "$OPS" --home-template "$HOME_TEMPLATE" \
  --released-source "$RELEASED_SOURCE" --released-python "$RELEASED_PYTHON" \
  --fork-source "$FORK_SOURCE" --fork-python "$FORK_PYTHON" \
  --nano-source "$NANO_SOURCE" \
  --arms base-truncated,fork-rlm --repeats 1 \
  --timeout 150 --max-turns 8 --max-subcalls 4
```

Do **not** add `--smallest-sniah`; that flag filters out every synthetic task.
`base-truncated` passes a head slice of the raw context directly to the root
model; `fork-rlm` reads the file via RLM tools. Use `--base-context-chars`
at least as large as the longest generated context to compare against *full*
plain-context input, provided the selected model's window also fits.
Otherwise label the baseline a truncation comparison.

No model runs or provider calls are needed for generation or the unit tests.
