"""
DocBank chunk-size × question-enrichment experiment — engine.

Mirrors the PeerQA chunk-size study on DocBank layout text:
* Table 1 — 4 fixed chunk sizes (200/400/600/800 tok, ~22% overlap) × fixed
  counts q5/q10/q13/q15 (questions-only) + a chunk-text baseline per size.
* Table 2 — one section-aware VARIABLE chunking; fixed q10 vs adaptive
  length-based ("bigger chunk ⇒ more questions") vs adaptive density-based vs
  fused. => 5 chunkings total (4 fixed + 1 variable).
* Table 3 — best / best-enrichment / cheapest-strong / fastest.

The 118 synthetic eval questions are reused as queries; each question's gold is
remapped onto every chunking by locating its verbatim evidence text in the new
chunks (single query set → sizes are directly comparable). Only questions whose
evidence maps in ALL chunkings are kept (the common set).
"""

from __future__ import annotations

import re
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import docbank_config as C
import docbank_chunker as CH
import docbank_experiment as DX  # _client/_reset (docbank dir), embed_all
import peerqa_experiment as E  # generation + path-agnostic helpers
from chunksize_experiment import _fused_parents
from embeddings import get_embedder, embedding_signature

FIXED_SIZES = [(200, 45), (400, 90), (600, 125), (800, 175)]
FIXED_COUNTS = [5, 10, 13, 15]
MAXQ = 15
VAR_CAP, VAR_OVERLAP = 800, 120
QCACHE = C.PROCESSED_DIR / "chunksize_enrichment_questions.jsonl"


# --------------------------------------------------------------------------- #
# Gold remap by evidence text
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").replace("\xa0", " ")).strip().lower()


def _shingle(s: str) -> str:
    s = _norm(s)
    return s[:60] if len(s) >= 20 else s


def gold_for_chunking(qa_rows: List[Dict], chunks: List[Dict]) -> Dict[str, Set[str]]:
    """{question_id: {gold_chunk_ids}} by locating each QA's evidence text."""
    norm_chunks = [(c["chunk_id"], _norm(c["text"])) for c in chunks]
    out: Dict[str, Set[str]] = {}
    for q in qa_rows:
        ev = q.get("gold_evidence_text") or ""
        cand = _shingle(ev) or _shingle(q.get("answer") or "")
        if not cand or len(cand) < 12:
            continue
        gold = {cid for cid, nt in norm_chunks if cand in nt}
        if not gold and len(cand) > 30:  # retry on a middle shingle
            mid = _norm(ev)[15:55]
            if len(mid) >= 20:
                gold = {cid for cid, nt in norm_chunks if mid in nt}
        if gold:
            out[q["question_id"]] = gold
    return out


# --------------------------------------------------------------------------- #
# Adaptive question-count allocation
# --------------------------------------------------------------------------- #
def length_to_nq(ntok: int) -> int:
    pts = [(200, 4), (400, 6), (600, 9), (800, 12)]
    if ntok <= pts[0][0]:
        nq = pts[0][1]
    elif ntok >= pts[-1][0]:
        nq = pts[-1][1] + (ntok - 800) / 200 * 3
    else:
        nq = pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= ntok <= x1:
                nq = y0 + (y1 - y0) * (ntok - x0) / (x1 - x0)
                break
    return int(max(3, min(MAXQ, round(nq))))


_RE_NUM = re.compile(r"\d+\.?\d*%?")
_RE_ACRO = re.compile(r"\b[A-Z]{2,6}\b")
_RE_CAP = re.compile(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+\b")
_RE_STRUCT = re.compile(
    r"\b(?:Table|Figure|Fig\.?|Eq\.?|Equation|Section|Lemma|Theorem)\s*\d+", re.I
)


def _density(text: str, ntok: int) -> float:
    return (
        len(set(_RE_NUM.findall(text)))
        + 1.5 * len(set(_RE_ACRO.findall(text)))
        + len(set(_RE_CAP.findall(text)))
        + 2.0 * len(_RE_STRUCT.findall(text))
        + ntok / 60.0
    )


def density_to_nq_map(
    chunks: List[Dict], all_q: Dict[str, List[str]]
) -> Dict[str, int]:
    scored = [(c["chunk_id"], _density(c["text"], c["n_tokens"])) for c in chunks]
    order = sorted(scored, key=lambda x: x[1])
    rank = {cid: i / max(len(order) - 1, 1) for i, (cid, _) in enumerate(order)}
    out = {}
    for cid, _ in scored:
        nq = int(round(4 + 11 * rank[cid]))
        out[cid] = max(3, min(nq, len(all_q.get(cid, [])) or nq))
    return out


# --------------------------------------------------------------------------- #
# Build collections (isolated docbank_cs_* names, docbank chroma dir)
# --------------------------------------------------------------------------- #
def _name(tag: str) -> str:
    return f"docbank_cs_{tag}"


def build_baseline(tag, chunks, cvecs):
    coll = DX._reset(_name(tag))
    ids, embs, docs, metas = [], [], [], []
    for c, v in zip(chunks, cvecs):
        ids.append(c["chunk_id"])
        embs.append(v)
        docs.append(c["text"])
        metas.append({"record_type": "chunk", "parent_chunk_id": c["chunk_id"]})
    E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "num_questions": 0,
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / _name(tag)),
    }


def build_questions(tag, chunks, qtext_by, qvec_by, nq):
    coll = DX._reset(_name(tag))
    ids, embs, docs, metas = [], [], [], []
    nq_total = 0
    for c in chunks:
        cid = c["chunk_id"]
        k = nq if isinstance(nq, int) else nq.get(cid, 0)
        for j, (qt, qv) in enumerate(
            zip(qtext_by.get(cid, [])[:k], qvec_by.get(cid, [])[:k])
        ):
            ids.append(f"{cid}::q{j}")
            embs.append(qv)
            docs.append(qt)
            metas.append({"record_type": "question", "parent_chunk_id": cid})
            nq_total += 1
    if ids:
        E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "num_questions": nq_total,
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / _name(tag)),
    }


def build_fused(tag, chunks, cvecs, qtext_by, qvec_by, nq):
    coll = DX._reset(_name(tag))
    ids, embs, docs, metas = [], [], [], []
    for c, v in zip(chunks, cvecs):
        cid = c["chunk_id"]
        ids.append(cid)
        embs.append(v)
        docs.append(c["text"])
        metas.append({"record_type": "chunk", "parent_chunk_id": cid})
        k = nq if isinstance(nq, int) else nq.get(cid, 0)
        for j, (qt, qv) in enumerate(
            zip(qtext_by.get(cid, [])[:k], qvec_by.get(cid, [])[:k])
        ):
            ids.append(f"{cid}::q{j}")
            embs.append(qv)
            docs.append(qt)
            metas.append({"record_type": "question", "parent_chunk_id": cid})
    E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "num_questions": len(ids) - len(chunks),
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / _name(tag)),
    }


# --------------------------------------------------------------------------- #
# Eval (gold is a SET of chunk ids)
# --------------------------------------------------------------------------- #
def evaluate(coll, queries, *, fused=False, k_values=None, overfetch_factor=30):
    k_values = k_values or C.K_VALUES
    embedder = get_embedder()
    overfetch = max(max(k_values) * overfetch_factor, 150)
    per_query, emb_ms, search_ms = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        ranked = (
            _fused_parents(coll, qvec, max(k_values), overfetch)
            if fused
            else E._retrieve_parents(coll, qvec, max(k_values), overfetch)
        )
        search_ms.append((time.perf_counter() - t1) * 1000)
        per_query.append(E._query_metrics(ranked, q["gold_chunk_ids"], k_values))

    def avg(k):
        return round(sum(x[k] for x in per_query) / max(len(per_query), 1), 4)

    def pct(v, p):
        if not v:
            return 0.0
        v = sorted(v)
        return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 3)

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {
        "metrics": metrics,
        "num_queries": len(queries),
        "latency_ms": {
            "query_embed_mean": round(st.mean(emb_ms), 3) if emb_ms else 0,
            "search_mean": round(st.mean(search_ms), 3) if search_ms else 0,
            "search_p95": pct(search_ms, 95),
            "total_p95": round(pct(emb_ms, 95) + pct(search_ms, 95), 3),
        },
    }


# --------------------------------------------------------------------------- #
# Embedding (reuse DX.embed_all)
# --------------------------------------------------------------------------- #
def embed_set(chunks, all_q, embedder):
    return DX.embed_all(
        chunks, all_q, embedder
    )  # cvecs, qvec_by, qtext_by, cs, qs, nqv
