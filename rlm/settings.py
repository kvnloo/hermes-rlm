"""Validate plugin settings once per session, with deprecated environment fallbacks.

reuse-checked: Hermes PluginContext.get_config settings precedence and old RLM limits.
"""
import math
import os
import warnings

DEFAULTS = {
    'max_subcalls': 200,
    'max_depth': 1,
    'max_tokens': 4000,
    'timeout_s': 600.0,
    'max_cost': None,
    'subcall_task': 'rlm_subcall',
    'max_prompt_chars': 400_000,
    'max_output_chars': 6000,
    'concurrency': 8,
    'max_corpus_chars': 400_000_000,
}
ENV_NAMES = {
    'max_tokens': ('RLM_MAX_TOKENS', 'RLM_SUBCALL_MAX_TOKENS'),
    'timeout_s': ('RLM_TIMEOUT_S', 'RLM_CELL_TIMEOUT_S'),
}


def validate(settings):
    result = {**DEFAULTS, **settings}
    unknown = result.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'unknown RLM settings: {sorted(unknown)}')
    for key, value in result.items():
        if key == 'subcall_task':
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError('subcall_task must be a nonempty task key')
        elif key == 'max_cost' and value is None:
            continue
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        elif key in ('timeout_s', 'max_cost'):
            if value < 0 or (key == 'timeout_s' and value == 0):
                raise ValueError(f'{key} is out of range')
        elif not isinstance(value, int) or value < (0 if key in ('max_subcalls', 'max_depth') else 1):
            raise ValueError(f'{key} must be an integer in range')
    if result['max_depth'] > 1:
        raise ValueError('max_depth must be 0 or 1; recursive child REPLs are not supported')
    if result['max_tokens'] > 32_000:
        raise ValueError('max_tokens must be <= 32000')
    return result


def limits(ctx=None):
    result = {}
    missing = object()
    for key, default in DEFAULTS.items():
        value = ctx.get_config(key, missing) if ctx is not None else missing
        if value is missing:
            value = default
            for env in ENV_NAMES.get(key, (f'RLM_{key.upper()}',)):
                if env not in os.environ:
                    continue
                warnings.warn(f'{env} is deprecated; use plugins.entries.rlm.settings.{key}',
                              DeprecationWarning, stacklevel=2)
                raw = os.environ[env]
                if key == 'subcall_task':
                    value = raw
                else:
                    try:
                        value = float(raw) if key in ('timeout_s', 'max_cost') else int(raw)
                    except ValueError as exc:
                        raise ValueError(f'invalid deprecated setting {env}') from exc
                break
        result[key] = value
    return validate(result)
