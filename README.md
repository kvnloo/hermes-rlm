# hermes-rlm

Recursive Language Model (arXiv:2512.24601) plugin for Hermes Agent: a corpus becomes `context` in a persistent Python REPL; the root model inspects it with code and can pass slices to a sub-model through Hermes' LLM service.

Status: experimental. See `bench/` and the z0evals study `rlm-paper-replication-v1`.

## Install

The repository root is the Hermes plugin directory (`plugin.yaml` and `__init__.py`); the implementation lives in `rlm/`. The normal sandbox requires Linux, `bwrap` (bubblewrap), and `/usr/bin/python3`. Load a host corpus using `rlm_load`, then inspect it with `rlm_exec`.

## What it touches

- **Tools**: `rlm_load`, `rlm_exec`.
- **Auxiliary tasks**: `rlm_subcall`.
- **Commands**: `rlm` (status for the host-supplied session ID within the current Hermes profile, including all tasks in that session; missing, empty, or non-string session IDs show only limits and a diagnostic notice, never session metadata. Task IDs and session keys are not used as fallbacks. The host remains responsible for authenticating and supplying the correct session ID).
- **Hooks**: none.
- **Middleware**: none.
- **Reads**: `rlm_load` opens host files, directories, and glob matches requested by the model, relative to the calling session workspace via `ctx.session_cwd()` for relative paths; missing, failing, or non-absolute workspace resolution is refused rather than falling back to the gateway process directory. Explicit absolute paths remain supported; it loads text on the host before handing it to the sandbox. The plugin also reads optional `RLM_*` environment settings for budgets and runtime behavior.
- **Writes**: per-session trajectory JSONL at `<hermes home>/plugin-data/rlm/runs/<identity-sha256>.jsonl`, or at the location set by `RLM_LOG_DIR`. Runtime identity includes the exact profile data directory, task ID, and session ID; filenames hash their JSON encoding instead of truncating or stripping IDs. Existing legacy logs are left untouched (not merged or migrated). Tool calls with neither a host task ID nor a session ID are refused before state creation; either identity alone remains supported. Logs contain loaded paths, executed Python cell source, finals, and usage metadata; the plugin does not enforce private permissions on log files. Inspect local storage permissions before processing confidential material.
- **Network**: `llm_query` and `llm_batch` call Hermes' `ctx.llm` provider, which may send prompts to a remote model. The default bubblewrap kernel has no direct network access. Setting `RLM_ALLOW_UNSANDBOXED=1` removes that network restriction.
- **Credentials**: the plugin does not directly read provider keys or token stores; Hermes manages model credentials through `ctx.llm`. In unsandboxed mode, executed Python has the host process's access to environment variables and files, potentially including secrets.
- **Subprocesses**: the plugin starts a persistent Python `kernel.py` subprocess per active session, normally via `bwrap` with a private `/tmp`, no network, and no arbitrary host filesystem access (read-only `/usr` is mounted). Kernels are killed on reload, cell timeout, and process exit. Missing sandbox prerequisites fail closed unless `RLM_ALLOW_UNSANDBOXED=1` is explicitly set; that fallback executes model-authored Python with host privileges and is intended **only for trusted local testing**.
- **Telemetry**: no separate telemetry or usage-reporting endpoint. Hermes' provider handles model request accounting, and the plugin writes the local trajectory log described above.

Protocol replies are read in byte chunks with a deadline, including partial lines and buffered multi-line replies. This is not a whole-operation hard deadline: stdin writes, host corpus I/O, provider calls, and process cleanup may still block. Reply-frame size is not independently capped.

Unload retires detached Session instances: retained references cannot start another load or exec. Ordinary kernel kill remains restartable. This does not cancel an already-running provider call or independently block new host calls admitted after unload. A timeout observed after retirement reports a closed session, not a restart.
