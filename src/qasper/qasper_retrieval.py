"""
Retrieval + Reciprocal Rank Fusion for an arm, and saving rankings to disk.

Three rankings are produced per query so we can see where gains come from:
  * dense  : the Chroma dense index. For enrichment arms this is questions-only,
             so a chunk's dense score is the MAX similarity over its questions
             (implemented by deduplicating question hits to their parent chunk,
             keeping the first/highest-scoring occurrence).
  * bm25   : BM25 over chunk text (+ appended questions in enrichment arms).
  * hybrid : Reciprocal Rank Fusion of dense + bm25, k = RRF_K (60).

All rankings are written to results/qasper/rankings/<arm>__<mode>.jsonl so metrics
can be recomputed without re-running retrieval.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

import qasper_config as C
import qasper_index as IDX
from embeddings import get_embedder


# --------------------------------------------------------------------------- #
# Per-mode rankings
# --------------------------------------------------------------------------- #
def dense_ranking(coll, qvec, depth: int, overfetch: int) -> List[str]:
    """Ranked distinct parent chunk ids from the dense index (max-over-questions)."""
    n = coll.count()
    k = min(depth * overfetch, n)
    res = coll.query(query_embeddings=[qvec], n_results=k, include=["metadatas"])
    metas = res["metadatas"][0] if res["metadatas"] else []
    seen, out = set(), []
    for m in metas:  # Chroma returns best-first
        parent = m.get("parent_chunk_id")
        if parent and parent not in seen:
            seen.add(parent)
            out.append(parent)
            if len(out) >= depth:
                break
    return out


def bm25_ranking(bm25, chunk_ids: List[str], query: str, depth: int) -> List[str]:
    scores = bm25.get_scores(IDX.tokenize(query))
    order = np.argsort(-scores)[:depth]
    return [chunk_ids[i] for i in order]


def rrf_fuse(rankings: List[List[str]], k: int, depth: int) -> List[str]:
    """Reciprocal Rank Fusion. score(d) = sum_r 1 / (k + rank_r(d)), 1-indexed."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in fused[:depth]]


# --------------------------------------------------------------------------- #
# Run an arm end-to-end (build indices -> retrieve -> save rankings)
# --------------------------------------------------------------------------- #
def _save_rankings(arm_name: str, mode: str, rows: List[dict]) -> None:
    path = C.ranking_path(arm_name, mode)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_arm(
    arm_name: str,
    chunks: List[dict],
    queries: List[dict],
    questions: Dict[str, List[str]] = None,
) -> dict:
    """Build the arm's dense+BM25 indices, run all queries, save rankings.

    Content is loaded from the generation caches by arm/source inside the index
    builders, so ``questions`` is optional. Returns a stats dict."""
    print(f"[retrieval] building indices for {arm_name} ...", flush=True)
    idx_stats = IDX.build_dense_index(arm_name, chunks, questions)
    bm25, bm25_ids = IDX.build_bm25(arm_name, chunks, questions)
    coll = IDX.get_dense_collection(arm_name)
    embedder = get_embedder()
    depth = C.RANK_DEPTH

    per_mode: Dict[str, List[dict]] = {m: [] for m in C.RETRIEVAL_MODES}
    emb_ms, dense_ms, bm25_ms = [], [], []

    for q in queries:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        d_rank = dense_ranking(coll, qvec, depth, C.DENSE_OVERFETCH)
        dense_ms.append((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        b_rank = bm25_ranking(bm25, bm25_ids, q["question"], depth)
        bm25_ms.append((time.perf_counter() - t2) * 1000)

        h_rank = rrf_fuse([d_rank, b_rank], C.RRF_K, depth)

        for mode, ranking in (("dense", d_rank), ("bm25", b_rank), ("hybrid", h_rank)):
            per_mode[mode].append(
                {
                    "query_id": q["query_id"],
                    "answer_type": q["answer_type"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "ranked": ranking,
                }
            )

    for mode, rows in per_mode.items():
        _save_rankings(arm_name, mode, rows)

    return {
        **idx_stats,
        "num_queries": len(queries),
        "latency_ms": {
            "query_embed_mean": round(float(np.mean(emb_ms)), 3) if emb_ms else 0,
            "dense_search_mean": round(float(np.mean(dense_ms)), 3) if dense_ms else 0,
            "bm25_search_mean": round(float(np.mean(bm25_ms)), 3) if bm25_ms else 0,
        },
        "rankings_saved": {
            m: str(C.ranking_path(arm_name, m)) for m in C.RETRIEVAL_MODES
        },
    }


def load_rankings(arm_name: str, mode: str) -> List[dict]:
    path = C.ranking_path(arm_name, mode)
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out
