"""
PeerQA chunk-size × question-enrichment experiment — engine.

Research question
-----------------
Does the amount of generated-question enrichment needed for retrieval depend on
chunk size and/or information density?

Design
------
* Table 1 — fixed counts by chunk size: 4 chunkings (200/400/600/800 tokens,
  overlap ~22%), each swept at q5/q10/q13/q15 questions-only, plus a chunk-text
  baseline per size. Each (size, count) is a self-contained index+eval.
* Table 2 — adaptive strategies: one VARIABLE-size chunking (section/paragraph
  aware, ~100-800 tokens) is the testbed. On the SAME chunk set + gold we compare
  fixed q10, adaptive length-based (#q from the chunk's token length), and
  adaptive density-based (#q from an information-density score, calibrated to
  average ~10 so it is a matched-budget test vs fixed q10). Plus chunk-text
  baseline and a fused (chunk-vector + best-question) condition.
* Table 3 — best quality / cheapest-strong / fastest trade-off across everything.

Reuses peerqa_data (loader/chunker/gold) and peerqa_experiment (LLM generation,
Chroma helpers, metrics). Generation caches to its own file so the previous
questions-only experiment is untouched.
"""

from __future__ import annotations

import json
import re
import statistics as st
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Callable

import peerqa_config as C
import peerqa_data as D
import peerqa_experiment as E
from embeddings import get_embedder, embedding_signature

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
# (target_size, overlap) — overlap ~22% of size, within the requested bands.
FIXED_SIZES = [(200, 45), (400, 90), (600, 125), (800, 175)]
FIXED_COUNTS = [5, 10, 13, 15]
MAXQ = 15  # generation pool ceiling
VAR_CAP = 800  # variable chunking hard cap
VAR_OVERLAP = 120

RESULTS = C.RESULTS_DIR / "chunksize_results.json"
QCACHE = C.PROCESSED_DIR / "chunksize_questions.jsonl"


# --------------------------------------------------------------------------- #
# Enriched, grounded question-generation prompt (spec §5)
# --------------------------------------------------------------------------- #
def _prompt_grounded(chunk_text: str, n: int):
    system = (
        "You generate retrieval questions for a scientific-paper RAG system. Given "
        "a passage, output natural user questions that are EACH fully answerable "
        "from the passage alone. Requirements: (1) ground every question in the "
        "passage — never introduce facts, numbers, entities, methods, datasets or "
        "metrics not present; (2) each question should name the specific "
        "entity / method / dataset / metric / number it targets; (3) cover DIFFERENT "
        "facts and subtopics — for longer passages spread questions across distinct "
        "subtopics, do not rephrase the same idea; (4) no duplicate or "
        "near-duplicate wording. Output ONLY a numbered list, one question per line."
    )
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Write up to {n} distinct, grounded questions covering different facts in "
        f"this passage. If it genuinely supports fewer than {n}, output only as many "
        f"as it supports."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Chunking helpers (namespaced ids so chunk sets never collide)
# --------------------------------------------------------------------------- #
def _renamespace(chunks: List[D.Chunk], tag: str) -> List[D.Chunk]:
    out = []
    for c in chunks:
        n = c.chunk_id.rsplit("::", 1)[-1]
        out.append(replace(c, chunk_id=f"{c.paper_id}::{tag}::{n}"))
    return out


def build_fixed_chunks(papers, size: int, overlap: int) -> List[D.Chunk]:
    chunks = D.build_chunks(papers, size=size, cap=size, overlap=overlap)
    return _renamespace(chunks, f"s{size}")


def build_variable_chunks(
    papers, *, cap: int = VAR_CAP, overlap: int = VAR_OVERLAP
) -> List[D.Chunk]:
    """Section/paragraph-aware packing: flush at heading changes and at the cap,
    yielding naturally variable chunk sizes (~100-800 tokens)."""
    enc = D._encoder()
    out: List[D.Chunk] = []
    for p in papers:
        cur: List[D.Sentence] = []
        cur_tok = 0
        n = 0

        def flush():
            nonlocal cur, cur_tok, n
            if not cur:
                return
            text = " ".join(s.content for s in cur if s.content).strip()
            if text:
                out.append(
                    D.Chunk(
                        chunk_id=f"{p.paper_id}::svar::{n}",
                        paper_id=p.paper_id,
                        text=text,
                        sent_idxs=[s.idx for s in cur],
                        n_tokens=D.count_tokens(text),
                        heading=next((s.heading for s in cur if s.heading), ""),
                        pidx_start=cur[0].pidx,
                        pidx_end=cur[-1].pidx,
                    )
                )
                n += 1

        prev_heading = None
        for s in p.sentences:
            stok = len(enc.encode(s.content))
            heading_change = (s.heading or "") != (
                prev_heading or ""
            ) and prev_heading is not None
            if stok > cap:  # oversized unit -> hard-split
                flush()
                cur = []
                cur_tok = 0
                toks = enc.encode(s.content)
                for i in range(0, len(toks), cap):
                    piece = enc.decode(toks[i : i + cap]).strip()
                    if piece:
                        out.append(
                            D.Chunk(
                                chunk_id=f"{p.paper_id}::svar::{n}",
                                paper_id=p.paper_id,
                                text=piece,
                                sent_idxs=[s.idx],
                                n_tokens=D.count_tokens(piece),
                                heading=s.heading,
                                pidx_start=s.pidx,
                                pidx_end=s.pidx,
                            )
                        )
                        n += 1
                prev_heading = s.heading
                continue
            if cur and (heading_change or cur_tok + stok > cap):
                flush()
                # small overlap re-seed
                tail, t = [], 0
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
            prev_heading = s.heading
        flush()
    return out


# --------------------------------------------------------------------------- #
# Information-density scoring + adaptive question-count allocation
# --------------------------------------------------------------------------- #
_RE_NUM = re.compile(r"\d+\.?\d*%?")
_RE_ACRO = re.compile(r"\b[A-Z]{2,6}\b")
_RE_CAPPHRASE = re.compile(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+\b")
_RE_STRUCT = re.compile(
    r"\b(?:Table|Figure|Fig\.?|Eq\.?|Equation|Section|Appendix)\s*\d+", re.I
)
_METRIC_TERMS = (
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "rmse",
    "mae",
    "precision",
    "recall",
    "auc",
    "perplexity",
    "ndcg",
    "map",
    "correlation",
    "r2",
    "p-value",
    "significance",
)


def density_signals(text: str, n_sent: int) -> Dict[str, int]:
    low = text.lower()
    return {
        "n_sentences": n_sent,
        "n_numbers": len(set(_RE_NUM.findall(text))),
        "n_acronyms": len(set(_RE_ACRO.findall(text))),
        "n_cap_phrases": len(set(_RE_CAPPHRASE.findall(text))),
        "n_struct_refs": len(_RE_STRUCT.findall(text)),
        "n_metric_terms": sum(low.count(t) for t in _METRIC_TERMS),
        "n_equals": text.count("="),
    }


def density_score(sig: Dict[str, int]) -> float:
    return (
        1.0 * sig["n_numbers"]
        + 1.5 * sig["n_acronyms"]
        + 1.0 * sig["n_cap_phrases"]
        + 2.0 * sig["n_struct_refs"]
        + 1.5 * sig["n_metric_terms"]
        + 0.5 * sig["n_sentences"]
        + 0.5 * sig["n_equals"]
    )


def length_to_nq(n_tokens: int) -> int:
    """Piecewise map of chunk token length -> #questions (spec bands), cap 15."""
    pts = [(200, 4), (400, 6), (600, 9), (800, 12)]
    if n_tokens <= pts[0][0]:
        nq = pts[0][1]
    elif n_tokens >= pts[-1][0]:
        nq = pts[-1][1] + (n_tokens - 800) / 200 * 3  # gentle extrapolation
    else:
        nq = pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= n_tokens <= x1:
                nq = y0 + (y1 - y0) * (n_tokens - x0) / (x1 - x0)
                break
    return int(max(3, min(MAXQ, round(nq))))


def density_to_nq_map(
    chunks: List[D.Chunk], all_q: Dict[str, List[str]]
) -> Dict[str, int]:
    """Percentile-rank density -> #questions in [4,15] (mean ~9.5, matched budget
    vs fixed q10). Capped by the number actually generated for the chunk."""
    scored = []
    for c in chunks:
        sig = density_signals(c.text, len(c.sent_idxs))
        scored.append((c.chunk_id, density_score(sig)))
    order = sorted(scored, key=lambda x: x[1])
    rank = {cid: i / max(len(order) - 1, 1) for i, (cid, _) in enumerate(order)}
    out = {}
    for cid, _ in scored:
        nq = int(round(4 + 11 * rank[cid]))  # [4,15]
        out[cid] = max(3, min(nq, len(all_q.get(cid, [])) or nq))
    return out


# --------------------------------------------------------------------------- #
# Embedding a chunk set once (chunk texts + question pool)
# --------------------------------------------------------------------------- #
def embed_set(chunks: List[Dict], all_q: Dict[str, List[str]], embedder):
    t0 = time.perf_counter()
    chunk_vecs = embedder.embed_documents([c["text"] for c in chunks])
    chunk_s = time.perf_counter() - t0

    flat_q, owner = [], []
    for c in chunks:
        qs = all_q.get(c["chunk_id"], [])[:MAXQ]
        flat_q.extend(qs)
        owner.extend([c["chunk_id"]] * len(qs))
    t1 = time.perf_counter()
    flat_qv = embedder.embed_documents(flat_q) if flat_q else []
    q_s = time.perf_counter() - t1

    qvec_by, qtext_by = {}, {}
    for cid, qt, qv in zip(owner, flat_q, flat_qv):
        qvec_by.setdefault(cid, []).append(qv)
        qtext_by.setdefault(cid, []).append(qt)
    return chunk_vecs, qvec_by, qtext_by, round(chunk_s, 2), round(q_s, 2), len(flat_q)


# --------------------------------------------------------------------------- #
# Build collections
# --------------------------------------------------------------------------- #
def _coll_name(tag: str) -> str:
    return f"peerqa_cs_{tag}"


def build_baseline(tag: str, chunks: List[Dict], chunk_vecs) -> tuple:
    name = _coll_name(tag)
    coll = E._reset(name)
    ids, embs, docs, metas = [], [], [], []
    for c, cv in zip(chunks, chunk_vecs):
        cid = c["chunk_id"]
        ids.append(cid)
        embs.append(cv)
        docs.append(c["text"])
        metas.append(
            {"record_type": "chunk", "parent_chunk_id": cid, "paper_id": c["paper_id"]}
        )
    t0 = time.perf_counter()
    E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "index_add_s": round(time.perf_counter() - t0, 3),
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / name),
    }


def build_questions(
    tag: str, chunks: List[Dict], qtext_by, qvec_by, nq_by_chunk
) -> tuple:
    """Questions-only collection. nq_by_chunk: dict cid->count, or int for fixed."""
    name = _coll_name(tag)
    coll = E._reset(name)
    ids, embs, docs, metas = [], [], [], []
    total_q = 0
    for c in chunks:
        cid = c["chunk_id"]
        nq = nq_by_chunk if isinstance(nq_by_chunk, int) else nq_by_chunk.get(cid, 0)
        qs = qtext_by.get(cid, [])[:nq]
        qvs = qvec_by.get(cid, [])[:nq]
        for j, (qt, qv) in enumerate(zip(qs, qvs)):
            ids.append(f"{cid}::q{j}")
            embs.append(qv)
            docs.append(qt)
            metas.append(
                {
                    "record_type": "question",
                    "parent_chunk_id": cid,
                    "paper_id": c["paper_id"],
                }
            )
            total_q += 1
    t0 = time.perf_counter()
    if ids:
        E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "num_questions": total_q,
        "index_add_s": round(time.perf_counter() - t0, 3),
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / name),
    }


def build_fused(
    tag: str, chunks: List[Dict], chunk_vecs, qtext_by, qvec_by, nq_by_chunk
) -> tuple:
    """Chunk vector + question vectors in one collection (for score fusion)."""
    name = _coll_name(tag)
    coll = E._reset(name)
    ids, embs, docs, metas = [], [], [], []
    for c, cv in zip(chunks, chunk_vecs):
        cid = c["chunk_id"]
        ids.append(cid)
        embs.append(cv)
        docs.append(c["text"])
        metas.append(
            {"record_type": "chunk", "parent_chunk_id": cid, "paper_id": c["paper_id"]}
        )
        nq = nq_by_chunk if isinstance(nq_by_chunk, int) else nq_by_chunk.get(cid, 0)
        for j, (qt, qv) in enumerate(
            zip(qtext_by.get(cid, [])[:nq], qvec_by.get(cid, [])[:nq])
        ):
            ids.append(f"{cid}::q{j}")
            embs.append(qv)
            docs.append(qt)
            metas.append(
                {
                    "record_type": "question",
                    "parent_chunk_id": cid,
                    "paper_id": c["paper_id"],
                }
            )
    t0 = time.perf_counter()
    E._add_batches(coll, ids, embs, docs, metas)
    return coll, {
        "num_embeddings": len(ids),
        "index_add_s": round(time.perf_counter() - t0, 3),
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / name),
    }


# --------------------------------------------------------------------------- #
# Evaluation (generic over a collection; parent-resolve or score-fusion)
# --------------------------------------------------------------------------- #
def _fused_parents(coll, qvec, k, overfetch, w=0.5):
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(overfetch, n),
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []
    chunk_sim, bestq_sim = {}, {}
    for m, d in zip(metas, dists):
        p = m.get("parent_chunk_id")
        sim = 1.0 - d
        if m.get("record_type") == "chunk":
            chunk_sim[p] = max(chunk_sim.get(p, -1), sim)
        else:
            bestq_sim[p] = max(bestq_sim.get(p, -1), sim)
    parents = set(chunk_sim) | set(bestq_sim)
    fused = []
    for p in parents:
        cs = chunk_sim.get(p)
        qs = bestq_sim.get(p)
        if cs is not None and qs is not None:
            s = w * cs + (1 - w) * qs
        else:
            s = cs if cs is not None else qs
        fused.append((p, s))
    fused.sort(key=lambda x: -x[1])
    return [p for p, _ in fused[:k]]


def eval_collection(
    coll, queries, *, fused=False, k_values=None, overfetch_factor=30
) -> Dict:
    k_values = k_values or C.K_VALUES
    embedder = get_embedder()
    overfetch = max(max(k_values) * overfetch_factor, 150)
    per_query, emb_ms, search_ms = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        if fused:
            ranked = _fused_parents(coll, qvec, max(k_values), overfetch)
        else:
            ranked = E._retrieve_parents(coll, qvec, max(k_values), overfetch)
        search_ms.append((time.perf_counter() - t1) * 1000)
        per_query.append(E._query_metrics(ranked, q["gold_chunk_ids"], k_values))

    def _avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def _pct(vals, p):
        if not vals:
            return 0.0
        vals = sorted(vals)
        return round(vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))], 3)

    metrics = {}
    for k_ in k_values:
        metrics[f"hit@{k_}"] = _avg(f"hit@{k_}")
        metrics[f"recall@{k_}"] = _avg(f"recall@{k_}")
    metrics["mrr"] = _avg("mrr")
    metrics["ndcg@10"] = _avg("ndcg@10")
    return {
        "metrics": metrics,
        "num_queries": len(queries),
        "latency_ms": {
            "query_embed_mean": round(st.mean(emb_ms), 3) if emb_ms else 0,
            "search_mean": round(st.mean(search_ms), 3) if search_ms else 0,
            "search_p95": _pct(search_ms, 95),
            "total_p95": round(_pct(emb_ms, 95) + _pct(search_ms, 95), 3),
        },
    }
