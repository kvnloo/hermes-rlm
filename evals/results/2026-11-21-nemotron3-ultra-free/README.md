# RLM vs plain Hermes, 2026-11-21

Root model: Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`). RLM sub-calls: Nemotron 3.5 Lightning (`:free`). Both run on OpenRouter for $0. Hermes `791df314d2`, RLM kernel in its bubblewrap sandbox on Linux. Two cohorts:

- this directory: 4 tasks × 2 arms × 3 repeats, sub-call `max_tokens` 2000 (harness commit `dc8094a`).
- [`../2026-11-21-reviews-subcalls-16k/`](../2026-11-21-reviews-subcalls-16k/RESULTS.md): `reviews` again with sub-call `max_tokens` 16000, both arms × 3 repeats (commit `b052a86`).

Full tables are in each `RESULTS.md`.

## Headline

**On the three scriptable long-context tasks, RLM was right 9/9 times against plain Hermes's 7/9, and sent 58% fewer prompt tokens.** On the semantic counting task, neither arm was reliable.

| task (context size) | plain pass | RLM pass | prompt tokens, RLM / plain (paired median) |
|---|---|---|---|
| `needle` (120 KB) | 3/3 | 3/3 | 0.20 |
| `multihop` (86 KB) | 3/3 | 3/3 | 0.52 |
| `aggregation` (161 KB) | 1/3 | 3/3 | 0.41 |
| `reviews` (69 KB), sub-calls 2000 tok | 2/3 | 0/3 | 2.57 (sub-calls broken, see below) |
| `reviews`, sub-calls 16000 tok | 0/3 | 1/3 | 1.36 (totals −18%, because one plain run sent 896K) |

Summed over the three scriptable tasks (18 runs): prompt tokens 1,419,890 plain vs 594,282 RLM (−58.1%), uncached 582,546 vs 239,178 (−58.9%). RLM wins on both measures, so it's cheaper at any cache price.

## Why

- **Plain Hermes pulls the file into the prompt; RLM doesn't.** Every plain run started with `read_file` on the whole context (101,517 chars for `aggregation`), and that result rode along in every later request. RLM's root saw only metadata plus small `rlm_exec` outputs. That's where the token gap comes from (measured from the rows' tool calls and token sums).
- **Plain's two `aggregation` misses are a silent truncation.** Both times, `search_files` returned its default 50 matches out of 170. The model summed the 50 values it could see and answered 2399 (gold 9402). The passing plain run piped `grep | awk` over the whole file instead. RLM's REPL computes over the full `context` string, so this failure mode doesn't exist there (measured from transcripts).
- **RLM costs more round trips.** It averaged 12.7 API calls per run against plain's 3.8 (first cohort). That doesn't raise tokens here, because each RLM request is small, but it adds latency.
- **The first `reviews` cohort is invalid for RLM.** The sub-model spent the whole 2,000-token budget reasoning and returned empty text on 94 of 105 sub-calls (115,828 completion tokens, 1,196 output chars). The plugin logged these as `ok` (filed as kvnloo/hermes-rlm#66). The root fell back to keyword matching and got it wrong. With 16,000 tokens, 0 of 119 sub-calls came back empty.
- **`reviews` is harder than it looks for both arms.** With working sub-calls, RLM answered 124, 93 and 131 (gold 131), and plain answered 116, 120 and 129, after up to 21 `search_files` calls. Across both cohorts plain went 2/6. Only 12 distinct complaint phrasings exist, so a model that reads enough lines can enumerate them, which is why plain sometimes gets it. This task doesn't show an RLM win on semantic aggregation at n = 3.

## What limits these numbers

- n = 3 per task and arm, synthetic tasks, one root model, one free route.
- The scriptable tasks reward any arm that runs code over the whole file. Plain Hermes can do that through `terminal` and sometimes does (`aggregation` rep 0). The RLM advantage measured here is mostly that its protocol always does it.
- Wall time reflects free-route queueing and is not a result.
- `prompt_tokens` and `cache_read_tokens` in this directory's `rows.jsonl` were added after the run from each run's saved Hermes result (`backfilled_from`); `input_tokens` matched on all 24 rows first. The second cohort recorded them directly. The report code changed after the first run.
