# A/B eval: plain Hermes vs Hermes + RLM

This directory asks plain Hermes and Hermes with the RLM plugin the same questions about the same long `context.txt`, on the same model. It reports exact-match accuracy and total input tokens. Results live in `results/`.

## What is compared

| arm | toolsets | how it reads the context |
|---|---|---|
| `plain` | `file`, `terminal` | any way it likes: `read_file`, `search_files`, shell, Python |
| `rlm` | `rlm` | `rlm_load`, then `rlm_exec` cells with optional `llm_query`/`llm_batch` sub-calls, then a cited `FINAL_VAR` |

The RLM prompt uses the same wording as `bench/paper_matrix.py` for its depth-one RLM arms. RLM settings are `max_depth 1`, `max_subcalls 60`, `concurrency 4`. The root model is `nvidia/nemotron-3-ultra-550b-a55b:free`, and RLM sub-calls use `nvidia/nemotron-3.5-lightning:free`. Each row gets a fresh `HERMES_HOME` with only a generated `config.yaml` (and the plugin, in the RLM arm). The RLM kernel runs in its real bubblewrap sandbox on Linux. The child process calls Hermes's own `run_agent.AIAgent.run_conversation`.

## Tasks (`tasks.py`)

- `needle`, `multihop`, `aggregation`: the repo's offline generator (`bench/synthetic_long_context.py`) at 512 records: 120 KB, 86 KB and 161 KB. A script or a grep can solve all three.
- `reviews`: count which of 600 short reviews complain about shipping or delivery (gold 131, 69 KB). The complaints are paraphrased, and positive delivery remarks are mixed in. A count of lines containing "ship" or "deliver" gives 7, not 131 (`tests/test_ab_tasks.py` checks that the two differ). This is the semantic aggregation case RLM sub-calls exist for.

Gold is computed by the generator and never enters a prompt or the working directory. Scoring is exact match on the text after the last `ANSWER:` (plain) or on the validated FINAL answer before its `Sources:` block (RLM).

## Run it

```sh
export AB_PROVIDER_KEY=...   # an OpenRouter key; :free models cost $0
python3 evals/abharness.py run --tasks evals.tasks --out evals/results/<name> --work /tmp/ab-work \
  --hermes-source ~/.hermes/hermes-agent --hermes-python ~/.hermes/hermes-agent/venv/bin/python --repeats 3
python3 evals/abharness.py report evals/results/<name>
```

RLM needs Linux with `bwrap` and `/usr/bin/python3`. The runner refuses non-`:free` models unless you pass `--allow-paid`, and reruns rows that ended on HTTP 429/5xx or a timeout.

## What the numbers do and don't mean

- **Token measures** are sums of provider-reported usage. **prompt** = root prompt tokens plus RLM sub-call prompt tokens (from the plugin's `subcall` log events, each counted once). **cache read** = the root tokens served from the provider's prompt cache. **uncached** = prompt − cache read; for root requests that is Hermes's own `input_tokens` counter. Sub-call cache use isn't reported, so sub-call prompts count as uncached. Sub-calls run on a smaller model, so equal token counts aren't equal cost on a paid route.
- **Wall time** depends on the free route's queueing and isn't a result.
- **n is small:** 4 tasks times 3 repeats. Read the per-task rows, not just the totals.
