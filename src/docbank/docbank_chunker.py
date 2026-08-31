"""
Layout-aware chunking for DocBank documents.

* FLOW blocks (paragraph / section / list / author / abstract / reference /
  footer / small inline equations) are packed into ~500-token chunks (cap 600,
  100-token overlap), carrying the current section heading.
* SPECIAL blocks (table, caption, and large display equations >= 40 tokens) are
  kept as their OWN retrieval units, combined with the current section heading +
  a short snippet of the preceding flow text (surrounding explanation), so the
  table/equation/caption stays with its context.

Each chunk records: chunk_id, doc_id, arxiv_id, page, section heading, layout
type (paragraph / section / mixed / table / equation / caption), block labels,
n_tokens. Chunks PARTITION the primary content (each block's primary content is
in exactly one chunk) so a synthetic-QA gold chunk id is unambiguous.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import docbank_config as C
from docbank_loader import Document, Block

_PARAGRAPHISH = {"paragraph", "list", "author", "abstract", "footer", "reference"}
_EQUATION_UNIT_MIN_TOKENS = 60  # only large display equations become own units
_MIN_CHUNK_TOKENS = 8  # drop degenerate fragments


@lru_cache(maxsize=1)
def _enc():
    import tiktoken

    return tiktoken.get_encoding(C.TIKTOKEN_ENCODING)


def _ntok(text: str) -> int:
    return len(_enc().encode(text or ""))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    arxiv_id: str
    page: int
    section: str
    layout_type: str
    block_labels: List[str]
    text: str
    n_tokens: int = 0

    def to_row(self) -> Dict:
        d = asdict(self)
        return d


def _is_special(b: Block) -> bool:
    if b.ltype in ("table", "caption"):
        return True
    if b.ltype == "equation" and _ntok(b.text) >= _EQUATION_UNIT_MIN_TOKENS:
        return True
    return False


def _flow_layout_type(blocks: List[Block]) -> str:
    labels = {b.ltype for b in blocks}
    if labels <= {"section", "title"}:
        return "section"
    if labels <= _PARAGRAPHISH | {"equation"} and "section" not in labels:
        return "paragraph"
    return "mixed"


def _pack_flow(
    flow: List[Block],
    doc: Document,
    section_at,
    next_id,
    size: int,
    cap: int,
    overlap: int,
) -> List[Chunk]:
    """Token-budget pack a run of flow blocks (all sharing rough locality)."""
    enc = _enc()
    chunks: List[Chunk] = []
    cur: List[Block] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if not cur:
            return
        text = " ".join(b.text for b in cur).strip()
        if text:
            cid = next_id()
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    arxiv_id=doc.arxiv_id,
                    page=cur[0].page,
                    section=section_at(cur[0]),
                    layout_type=_flow_layout_type(cur),
                    block_labels=[b.label for b in cur],
                    text=text,
                    n_tokens=_ntok(text),
                )
            )
        cur = []
        cur_tok = 0

    for b in flow:
        btok = _enc().encode(b.text)
        if len(btok) > cap:  # oversized block -> hard split
            flush()
            for i in range(0, len(btok), cap):
                piece = enc.decode(btok[i : i + cap]).strip()
                if piece:
                    cid = next_id()
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=doc.doc_id,
                            arxiv_id=doc.arxiv_id,
                            page=b.page,
                            section=section_at(b),
                            layout_type=b.ltype,
                            block_labels=[b.label],
                            text=piece,
                            n_tokens=_ntok(piece),
                        )
                    )
            continue
        if cur and cur_tok + len(btok) > cap:
            flush()
        cur.append(b)
        cur_tok += len(btok)
        if cur_tok >= size:
            # close window; re-seed with ~overlap tokens of trailing blocks
            tail, t = [], 0
            for prev in reversed(cur):
                pt = len(_enc().encode(prev.text))
                if t + pt > overlap:
                    break
                tail.insert(0, prev)
                t += pt
            flush()
            cur = list(tail)
            cur_tok = t
    flush()
    return chunks


def build_chunks(
    docs: List[Document],
    *,
    size: int = C.CHUNK_SIZE_TOKENS,
    cap: int = C.CHUNK_MAX_TOKENS,
    overlap: int = C.CHUNK_OVERLAP_TOKENS,
    tag: str = "",
    section_break: bool = False,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    ns = f"::{tag}" if tag else ""

    for doc in docs:
        counter = {"n": 0}

        def next_id():
            cid = f"{doc.arxiv_id}{ns}::c{counter['n']}"
            counter["n"] += 1
            return cid

        # track section heading as we walk
        heading = ""
        last_flow_snippet = ""

        def section_at(_b):
            return heading

        flow_run: List[Block] = []

        def flush_flow():
            nonlocal flow_run
            if flow_run:
                chunks.extend(
                    _pack_flow(flow_run, doc, section_at, next_id, size, cap, overlap)
                )
                flow_run = []

        for b in doc.blocks:
            if b.ltype in ("section", "title") and _ntok(b.text) < 40:
                if section_break and heading and b.text[:160] != heading:
                    flush_flow()  # section boundary -> variable size
                heading = b.text[:160]
            if _is_special(b):
                flush_flow()
                # surrounding explanation: last ~40 tokens of prior flow text
                ctx = last_flow_snippet
                parts = []
                if heading:
                    parts.append(f"Section: {heading}")
                if ctx:
                    parts.append(f"Context: {ctx}")
                parts.append(f"{b.ltype.title()}: {b.text}")
                text = "\n".join(parts)
                chunks.append(
                    Chunk(
                        chunk_id=next_id(),
                        doc_id=doc.doc_id,
                        arxiv_id=doc.arxiv_id,
                        page=b.page,
                        section=heading,
                        layout_type=b.ltype,
                        block_labels=[b.label],
                        text=text,
                        n_tokens=_ntok(text),
                    )
                )
            else:
                flow_run.append(b)
                if b.ltype in _PARAGRAPHISH:
                    words = b.text.split()
                    last_flow_snippet = " ".join(words[-40:])
        flush_flow()
    return [c for c in chunks if c.n_tokens >= _MIN_CHUNK_TOKENS]


# --------------------------------------------------------------------------- #
def chunks_path() -> Path:
    return C.PROCESSED_DIR / "chunks.jsonl"


def save_chunks(chunks: List[Chunk]) -> Path:
    out = chunks_path()
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_row(), ensure_ascii=False) + "\n")
    return out


def load_chunks() -> List[Dict]:
    return [json.loads(l) for l in chunks_path().open(encoding="utf-8") if l.strip()]


def chunk_stats(chunks: List[Chunk]) -> Dict:
    import statistics as st, collections

    toks = [c.n_tokens for c in chunks]
    by_type = collections.Counter(c.layout_type for c in chunks)
    by_doc = collections.Counter(c.arxiv_id for c in chunks)
    return {
        "num_chunks": len(chunks),
        "num_documents": len(by_doc),
        "avg_chunks_per_doc": round(len(chunks) / max(len(by_doc), 1), 1),
        "chunk_size_target": C.CHUNK_SIZE_TOKENS,
        "chunk_size_cap": C.CHUNK_MAX_TOKENS,
        "overlap": C.CHUNK_OVERLAP_TOKENS,
        "tokens_per_chunk": {
            "min": min(toks) if toks else 0,
            "max": max(toks) if toks else 0,
            "mean": round(st.mean(toks), 1) if toks else 0,
            "median": round(st.median(toks), 1) if toks else 0,
        },
        "chunks_by_type": dict(by_type.most_common()),
    }


if __name__ == "__main__":
    from docbank_loader import load_documents

    docs = load_documents()
    chunks = build_chunks(docs)
    save_chunks(chunks)
    print(json.dumps(chunk_stats(chunks), indent=2))
