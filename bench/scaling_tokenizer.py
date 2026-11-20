"""Offline real-tokenizer loader shared by the scaling generator and matrix."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path


class LocalTokenizer:
    def __init__(self, path: Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ValueError("scaling requires the tokenizers package") from exc
        data = path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.engine = version("tokenizers")
        self._tokenizer = Tokenizer.from_str(data.decode("utf-8"))
        # Saved truncation/padding options would invalidate actual token counts.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
