"""
Chunking for SPIQA papers.

Produces two kinds of *meaningful retrieval units* (not random fragments):

1. Text chunks  -- paragraphs within a section merged up to a token budget
   (default 416 tokens) with token overlap (default 100). Uses tiktoken for
   accurate token counts.

2. Figure/table units -- each figure/table is its own retrieval unit combining:
      * caption
      * figure/table metadata (kind + type)
      * section heading (best-effort)
      * nearby paragraph context (paragraphs that mention the figure/table)

Every chunk carries rich metadata (paper id, split, section heading, paragraph
index range, referenced figures/tables, source_type) so retrieval hits can be
mapped back to gold evidence during evaluation.

Chunk record schema (JSONL, one row per chunk):
    {
      "chunk_id": "<split>::<paper_id>::<kind>::<n>",
      "paper_id", "split",
      "source_type": "text" | "figure_caption" | "table_caption",
      "text": "<the retrieval-unit text (also the evidence shown at eval time)>",
      "metadata": {
          "section_heading", "paragraph_start", "paragraph_end",
          "figure_refs": [...], "fig_id", "fig_kind", "fig_type"
      }
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import spiqa_config as C
from spiqa_loader import Paper, Section, Figure, load_papers

# Regex to spot "Figure 3", "Fig. 3", "Table 1" mentions inside paragraph text.
_FIG_MENTION = re.compile(r"\b(?:fig(?:ure)?|tab(?:le)?)\.?\s*([0-9]+)", re.IGNORECASE)
# Pull the figure/table number out of a SPIQA figure id like ".../3-Figure1-1.png".
_FIG_NUM_IN_ID = re.compile(r"(?:figure|table)\s*([0-9]+)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding(C.TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    split: str
    source_type: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def to_row(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "split": self.split,
            "source_type": self.source_type,
            "text": self.text,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Text chunking (token-based, section-aware, with overlap)
# --------------------------------------------------------------------------- #
def _pack_paragraphs(paragraphs: List[str], size: int, overlap: int):
    """
    Greedily pack whole paragraphs into token-budgeted windows. If a single
    paragraph exceeds the budget it is split on token boundaries. Yields
    (text, para_start_idx, para_end_idx). Overlap is applied by re-including a
    token-suffix of the previous window as leading context.
    """
    enc = _encoder()
    windows = []
    cur_tokens: List[int] = []
    cur_start = 0
    cur_end = 0

    def flush():
        if cur_tokens:
            windows.append((enc.decode(cur_tokens), cur_start, cur_end))

    for idx, para in enumerate(paragraphs):
        p_tokens = enc.encode(para.strip())
        if not p_tokens:
            continue
        # Oversized single paragraph -> hard-split it.
        if len(p_tokens) > size:
            flush()
            cur_tokens, cur_start, cur_end = [], idx, idx
            for i in range(0, len(p_tokens), size - overlap):
                piece = p_tokens[i : i + size]
                windows.append((enc.decode(piece), idx, idx))
            continue
        if len(cur_tokens) + len(p_tokens) <= size:
            if not cur_tokens:
                cur_start = idx
            cur_tokens += p_tokens
            cur_end = idx
        else:
            flush()
            tail = cur_tokens[-overlap:] if overlap else []
            cur_tokens = tail + p_tokens
            cur_start, cur_end = idx, idx
    flush()
    return windows


def _text_chunks(paper: Paper, size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    n = 0
    for section in paper.sections:
        if not section.paragraphs:
            continue
        for text, p0, p1 in _pack_paragraphs(section.paragraphs, size, overlap):
            refs = sorted({m.group(0).title() for m in _FIG_MENTION.finditer(text)})
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.split}::{paper.paper_id}::text::{n}",
                    paper_id=paper.paper_id,
                    split=paper.split,
                    source_type="text",
                    text=text,
                    metadata={
                        "section_heading": section.name,
                        "paragraph_start": p0,
                        "paragraph_end": p1,
                        "figure_refs": refs,
                        "n_tokens": count_tokens(text),
                    },
                )
            )
            n += 1
    return chunks


# --------------------------------------------------------------------------- #
# Figure/table units (caption + metadata + nearby context + heading)
# --------------------------------------------------------------------------- #
def _fig_number(fig_id: str) -> Optional[str]:
    m = _FIG_NUM_IN_ID.search(fig_id)
    return m.group(1) if m else None


def _nearby_context(paper: Paper, fig: Figure, max_chars: int = 500) -> str:
    """Concatenate (a snippet of) paragraphs that mention this figure/table."""
    num = _fig_number(fig.fig_id)
    if not num:
        return ""
    want = fig.kind  # "figure" or "table"
    hits = []
    for section in paper.sections:
        for para in section.paragraphs:
            for m in _FIG_MENTION.finditer(para):
                kind = "table" if m.group(0).lower().startswith(("tab",)) else "figure"
                if m.group(1) == num and kind == want:
                    hits.append(para.strip())
                    break
    ctx = " ".join(hits)
    return ctx[:max_chars]


def _figure_units(paper: Paper) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, fig in enumerate(paper.figures):
        if not fig.caption and not fig.fig_id:
            continue
        context = _nearby_context(paper, fig)
        source_type = "table_caption" if fig.kind == "table" else "figure_caption"
        # Assemble a self-contained retrieval unit.
        parts = []
        if paper.title:
            parts.append(f"Paper: {paper.title}")
        parts.append(f"{fig.kind.title()} ({fig.fig_type or 'n/a'}) — {fig.fig_id}")
        if fig.caption:
            parts.append(f"Caption: {fig.caption}")
        if context:
            parts.append(f"Context: {context}")
        chunks.append(
            Chunk(
                chunk_id=f"{paper.split}::{paper.paper_id}::fig::{i}",
                paper_id=paper.paper_id,
                split=paper.split,
                source_type=source_type,
                text="\n".join(parts),
                metadata={
                    "section_heading": "",
                    "fig_id": fig.fig_id,
                    "fig_kind": fig.kind,
                    "fig_type": fig.fig_type,
                    "caption": fig.caption,
                    "has_context": bool(context),
                    "n_tokens": count_tokens("\n".join(parts)),
                },
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_chunks(
    papers: List[Paper],
    *,
    size: int = C.CHUNK_SIZE_TOKENS,
    overlap: int = C.CHUNK_OVERLAP_TOKENS,
    include_figures: bool = True,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for paper in papers:
        chunks.extend(_text_chunks(paper, size, overlap))
        if include_figures:
            chunks.extend(_figure_units(paper))
    return chunks


def chunks_path(split: str) -> Path:
    return C.PROCESSED_DIR / f"{split}_chunks.jsonl"


def save_chunks(chunks: List[Chunk], split: str) -> Path:
    out = chunks_path(split)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_row(), ensure_ascii=False) + "\n")
    return out


def load_chunks(split: str) -> List[Dict]:
    return [
        json.loads(l) for l in chunks_path(split).open(encoding="utf-8") if l.strip()
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Chunk a SPIQA split.")
    ap.add_argument("--split", default=C.DEFAULT_SPLIT, choices=list(C.SPLIT_FILES))
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=C.CHUNK_SIZE_TOKENS)
    ap.add_argument("--overlap", type=int, default=C.CHUNK_OVERLAP_TOKENS)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    papers = load_papers(args.split, args.max_papers)
    chunks = build_chunks(
        papers,
        size=args.chunk_size,
        overlap=args.overlap,
        include_figures=not args.no_figures,
    )
    out = save_chunks(chunks, args.split)

    by_type: Dict[str, int] = {}
    toks = []
    for c in chunks:
        by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
        toks.append(c.metadata.get("n_tokens", 0))
    import statistics as st

    print(f"Chunks: {len(chunks)}  by_type={by_type}")
    if toks:
        print(
            f"tokens/chunk: min={min(toks)} mean={st.mean(toks):.0f} "
            f"median={st.median(toks):.0f} max={max(toks)}"
        )
    print("Saved ->", out)
