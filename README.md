# hermes-rlm

Recursive Language Model (arXiv:2512.24601) plugin for Hermes Agent: the corpus lives as `context` in a sandboxed Python REPL; the root model inspects it with code and fans slices out to a sub-model via `ctx.llm`.

Status: experimental. See `bench/` and the z0evals study `rlm-paper-replication-v1`.

## Install

```bash
hermes plugins install kvnloo/hermes-rlm
hermes plugins enable rlm
```

The repository root is the Hermes directory plugin (`plugin.yaml` + `__init__.py`); the implementation and sandbox kernel are in the installable `rlm` Python package.

The sandbox requires Linux `bwrap` (bubblewrap) and `/usr/bin/python3`. Do not enable `RLM_ALLOW_UNSANDBOXED=1` for production.

## Tests

```bash
python -m pip install .
python -m unittest discover -s tests
```

The host-integrated sandbox tests in `rlm/tests/` use pytest and require a Hermes development environment with bubblewrap.
