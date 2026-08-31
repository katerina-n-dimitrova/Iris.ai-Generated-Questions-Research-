"""
PeerQA loader + chunker + gold-label construction.

Loads the redistributable PeerQA release (qa.jsonl + papers.jsonl), which ships
the paper text ALREADY SEGMENTED INTO SENTENCES — one row per sentence with a
global sentence index ``idx``, paragraph index ``pidx``, sentence-in-paragraph
index ``sidx``, a ``type`` (sentence/heading/table/figure/formula/...), the
``content``, and the current ``last_heading``. Gold evidence is given per
question as sentence ``idx`` sets in ``answer_evidence_mapped`` — i.e. the
dataset provides sentence- (and paragraph-) level relevance labels.

Because individual sentences are far below a normal RAG chunk (~10-30 tokens),
we PACK consecutive sentences (in ``idx`` order, within a paper) into
token-budgeted chunks (~500 tokens, hard cap 600) with ~100-token overlap, using
tiktoken for accurate counts. Every chunk records the SET of source sentence
idxs it covers, so a query's gold sentence idxs map cleanly onto parent chunks.

Paper selection: only papers whose full text is present in papers.jsonl and that
carry answerable questions with mapped evidence are eligible; we take the top-N
papers by usable-query count (deterministic).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.request import urlopen

import peerqa_config as C


# --------------------------------------------------------------------------- #
# Download / extract
# --------------------------------------------------------------------------- #
def ensure_data() -> Path:
    """Download+extract the PeerQA zip into RAW_DIR if not already present.
    Returns the directory holding qa.jsonl / papers.jsonl."""
    qa = C.RAW_DIR / "qa.jsonl"
    papers = C.RAW_DIR / "papers.jsonl"
    if qa.exists() and papers.exists():
        return C.RAW_DIR
    zpath = C.RAW_DIR / "peerqa-data-v1.0.zip"
    if not zpath.exists():
        print(f"[peerqa] downloading data zip -> {zpath}")
        with urlopen(C.DATA_ZIP_URL) as r, zpath.open("wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(C.RAW_DIR)
    return C.RAW_DIR


# --------------------------------------------------------------------------- #
# Tokeniser
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding(C.TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Sentence:
    idx: int  # global sentence index within the paper
    pidx: int  # paragraph index
    sidx: int  # sentence-in-paragraph index
    type: str  # sentence | heading | table | figure | formula | ...
    content: str
    heading: str


@dataclass
class QAItem:
    question_id: str
    question: str
    answer: str
    gold_sent_idxs: Set[int]  # gold evidence sentence idxs (relevance labels)
    answerable: bool


@dataclass
class Paper:
    paper_id: str
    corpus: str
    sentences: List[Sentence]
    questions: List[QAItem] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    sent_idxs: List[int]  # source sentence idxs covered by this chunk
    n_tokens: int
    heading: str
    pidx_start: int
    pidx_end: int


# --------------------------------------------------------------------------- #
# Load + select papers
# --------------------------------------------------------------------------- #
def _corpus(pid: str) -> str:
    return "/".join(pid.split("/")[:2])


def load_dataset(num_papers: int = C.NUM_PAPERS) -> List[Paper]:
    """Load PeerQA and return the top-N text-available papers by usable-query
    count (usable = answerable, mapped evidence, evidence idxs in range)."""
    data_dir = ensure_data()
    qa_rows = [json.loads(l) for l in (data_dir / "qa.jsonl").open(encoding="utf-8")]
    pap_rows = [
        json.loads(l) for l in (data_dir / "papers.jsonl").open(encoding="utf-8")
    ]

    # Group sentences per paper (keep idx order).
    sents_by_paper: Dict[str, Dict[int, Sentence]] = {}
    for r in pap_rows:
        s = Sentence(
            idx=int(r["idx"]),
            pidx=int(r["pidx"]),
            sidx=int(r["sidx"]),
            type=str(r.get("type") or ""),
            content=str(r.get("content") or ""),
            heading=str(r.get("last_heading") or ""),
        )
        sents_by_paper.setdefault(r["paper_id"], {})[s.idx] = s

    # Group usable questions per paper.
    q_by_paper: Dict[str, List[QAItem]] = {}
    for r in qa_rows:
        pid = r["paper_id"]
        if pid not in sents_by_paper:
            continue  # text not redistributed (openreview/arxiv-only)
        mapped = r.get("answer_evidence_mapped") or []
        gold: Set[int] = set()
        for m in mapped:
            for ix in m.get("idx") or []:
                if int(ix) in sents_by_paper[pid]:
                    gold.add(int(ix))
        if not gold:
            continue  # need locatable gold evidence for retrieval eval
        q_by_paper.setdefault(pid, []).append(
            QAItem(
                question_id=r["question_id"],
                question=r["question"],
                answer=str(r.get("answer_free_form") or ""),
                gold_sent_idxs=gold,
                answerable=bool(r.get("answerable_mapped")),
            )
        )

    # Rank papers by usable-query count; deterministic tie-break by paper_id.
    ranked = sorted(q_by_paper.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = ranked[:num_papers]

    papers: List[Paper] = []
    for pid, qs in chosen:
        sents = [sents_by_paper[pid][i] for i in sorted(sents_by_paper[pid])]
        papers.append(
            Paper(paper_id=pid, corpus=_corpus(pid), sentences=sents, questions=qs)
        )
    return papers


# --------------------------------------------------------------------------- #
# Chunking — token-budgeted packing of consecutive sentences, with overlap.
# --------------------------------------------------------------------------- #
def _pack(paper: Paper, size: int, cap: int, overlap: int) -> List[Chunk]:
    enc = _encoder()
    chunks: List[Chunk] = []
    cur: List[Sentence] = []
    cur_tok = 0
    n = 0

    def flush():
        nonlocal cur, cur_tok, n
        if not cur:
            return
        text = " ".join(s.content for s in cur if s.content).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}::chunk::{n}",
                    paper_id=paper.paper_id,
                    text=text,
                    sent_idxs=[s.idx for s in cur],
                    n_tokens=count_tokens(text),
                    heading=next((s.heading for s in cur if s.heading), ""),
                    pidx_start=cur[0].pidx,
                    pidx_end=cur[-1].pidx,
                )
            )
            n += 1

    for s in paper.sentences:
        stok = len(enc.encode(s.content))
        # Oversized single unit (e.g. a big flattened table) -> hard-split on
        # token boundaries into <=cap pieces, each its own chunk (same sent idx).
        if stok > cap:
            flush()
            cur = []
            cur_tok = 0
            toks = enc.encode(s.content)
            for i in range(0, len(toks), cap):
                piece = enc.decode(toks[i : i + cap]).strip()
                if piece:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{paper.paper_id}::chunk::{n}",
                            paper_id=paper.paper_id,
                            text=piece,
                            sent_idxs=[s.idx],
                            n_tokens=count_tokens(piece),
                            heading=s.heading,
                            pidx_start=s.pidx,
                            pidx_end=s.pidx,
                        )
                    )
                    n += 1
            continue
        if cur_tok + stok > cap and cur:
            flush()
            # overlap: re-seed with trailing sentences up to ~overlap tokens
            tail: List[Sentence] = []
            t = 0
            for prev in reversed(cur):
                pt = len(enc.encode(prev.content))
                if t + pt > overlap:
                    break
                tail.insert(0, prev)
                t += pt
            cur = tail + [s]
            cur_tok = t + stok
        else:
            cur.append(s)
            cur_tok += stok
            if cur_tok >= size:  # reached target; close the window
                flush()
                tail = []
                t = 0
                for prev in reversed(cur):
                    pt = len(enc.encode(prev.content))
                    if t + pt > overlap:
                        break
                    tail.insert(0, prev)
                    t += pt
                cur = tail
                cur_tok = t
    flush()
    return chunks


def build_chunks(
    papers: List[Paper],
    *,
    size: int = C.CHUNK_SIZE_TOKENS,
    cap: int = C.CHUNK_MAX_TOKENS,
    overlap: int = C.CHUNK_OVERLAP_TOKENS,
) -> List[Chunk]:
    out: List[Chunk] = []
    for p in papers:
        out.extend(_pack(p, size, cap, overlap))
    return out


# --------------------------------------------------------------------------- #
# Gold: map each query's gold sentence idxs -> the set of chunk_ids covering them
# --------------------------------------------------------------------------- #
def build_gold(papers: List[Paper], chunks: List[Chunk]) -> List[Dict]:
    by_paper: Dict[str, List[Chunk]] = {}
    for c in chunks:
        by_paper.setdefault(c.paper_id, []).append(c)

    queries: List[Dict] = []
    for p in papers:
        pchunks = by_paper.get(p.paper_id, [])
        # sentence idx -> chunk_ids containing it (overlap => can be >1)
        sent2chunks: Dict[int, List[str]] = {}
        for c in pchunks:
            for ix in c.sent_idxs:
                sent2chunks.setdefault(ix, []).append(c.chunk_id)
        for q in p.questions:
            gold: Set[str] = set()
            for ix in q.gold_sent_idxs:
                gold.update(sent2chunks.get(ix, []))
            if not gold:
                continue
            queries.append(
                {
                    "query_id": q.question_id,
                    "paper_id": p.paper_id,
                    "question": q.question,
                    "gold_chunk_ids": gold,
                    "n_gold_sentences": len(q.gold_sent_idxs),
                }
            )
    return queries


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def chunks_path() -> Path:
    return C.PROCESSED_DIR / "chunks.jsonl"


def save_chunks(chunks: List[Chunk]) -> Path:
    out = chunks_path()
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    return out


def load_chunks() -> List[Dict]:
    return [json.loads(l) for l in chunks_path().open(encoding="utf-8") if l.strip()]


def dataset_summary(papers: List[Paper], chunks: List[Chunk]) -> Dict:
    import statistics as st

    toks = [c.n_tokens for c in chunks]
    n_sent = sum(len(p.sentences) for p in papers)
    return {
        "num_papers": len(papers),
        "corpora": sorted({p.corpus for p in papers}),
        "num_sentences_total": n_sent,
        "num_questions_usable": sum(len(p.questions) for p in papers),
        "num_chunks": len(chunks),
        "avg_sentences_per_chunk": round(
            sum(len(c.sent_idxs) for c in chunks) / max(len(chunks), 1), 1
        ),
        "tokens_per_chunk": {
            "min": min(toks) if toks else 0,
            "max": max(toks) if toks else 0,
            "mean": round(st.mean(toks), 1) if toks else 0,
            "median": round(st.median(toks), 1) if toks else 0,
        },
        "chunk_size_target": C.CHUNK_SIZE_TOKENS,
        "chunk_size_cap": C.CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": C.CHUNK_OVERLAP_TOKENS,
    }


if __name__ == "__main__":
    papers = load_dataset()
    chunks = build_chunks(papers)
    save_chunks(chunks)
    queries = build_gold(papers, chunks)
    summ = dataset_summary(papers, chunks)
    summ["num_eval_queries"] = len(queries)
    print(json.dumps(summ, indent=2))
    print("papers:")
    for p in papers:
        print(f"  {len(p.questions):2d}q {len(p.sentences):4d}sent  {p.paper_id}")
