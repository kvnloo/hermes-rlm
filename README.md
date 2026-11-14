# hermes-rlm

Recursive Language Model (arXiv:2512.24601) plugin for Hermes Agent: the corpus lives as `context` in a sandboxed Python REPL; the root model inspects it with code and fans slices out to a sub-model via `ctx.llm`.

Status: experimental. See `bench/` and the z0evals study `rlm-paper-replication-v1`.
