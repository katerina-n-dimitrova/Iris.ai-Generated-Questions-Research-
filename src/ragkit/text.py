"""Tokenizers and JSONL I/O shared by every experiment.

The two tokenizers are deliberately distinct dialects and must not be merged:
ASCII tokenization is what the MultiHop-RAG BM25 index was built with, while
the Unicode tokenizer is required for Bulgarian text. Swapping one for the
other silently shifts BM25 scores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WORD_ASCII = re.compile(r"[a-z0-9]+")
WORD_UNICODE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize_ascii(text: str) -> list[str]:
    return WORD_ASCII.findall((text or "").casefold())


def tokenize_unicode(text: str) -> list[str]:
    return [word.casefold() for word in WORD_UNICODE.findall(text or "")]


def read_jsonl(
    path: Path, *, encoding: str | None = None, missing_ok: bool = False
) -> list[dict]:
    if missing_ok and not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding=encoding) if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_jsonl_utf8(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
