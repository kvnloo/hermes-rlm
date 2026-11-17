"""Hermes directory-plugin entry point; implementation lives in the rlm package."""

from __future__ import annotations

from .rlm import register

__all__ = ["register"]
