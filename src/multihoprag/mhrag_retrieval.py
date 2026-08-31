"""
Retrieval + Reciprocal Rank Fusion for an arm, and saving rankings to disk.

Three rankings are produced per query so we can see where gains come from:
  * dense  : the Chroma dense index. For enrichment arms this is questions-only,
             so a chunk's dense score is the MAX similarity over its questions
             (implemented by deduplicating question hits to their parent chunk,
             keeping the first/highest-scoring occurrence).
  * bm25   : BM25 over chunk text (+ appended questions in enrichment arms).
  * hybrid : Reciprocal Rank Fusion of dense + bm25, k = RRF_K (60).

All rankings are written to results/multihoprag/rankings/<arm>__<mode>.jsonl so
metrics can be recomputed without re-running retrieval.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

import mhrag_config as C
import mhrag_index as IDX
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
    dense_questions: Dict[str, List[str]],
    bm25_questions: Dict[str, List[str]],
) -> dict:
    """Build the arm's dense+BM25 indices, run all queries, save rankings.

    ``dense_questions`` feeds the dense multi-vectors / doc2query concat;
    ``bm25_questions`` feeds the BM25 append terms. They may come from different
    arms' caches (Exp-4 c reuses B1 for dense and E4b keywords for BM25)."""
    print(f"[retrieval] building indices for {arm_name} ...", flush=True)
    arm = C.ARMS[arm_name]
    modes = modes_for_arm(arm)

    idx_stats = {}
    coll = None
    if "dense" in modes:
        idx_stats = IDX.build_dense_index(arm_name, chunks, dense_questions)
        coll = IDX.get_dense_collection(arm_name)
    bm25 = bm25_ids = None
    if "bm25" in modes:
        bm25, bm25_ids = IDX.build_bm25(arm_name, chunks, bm25_questions)

    embedder = get_embedder()
    depth = C.RANK_DEPTH
    per_mode: Dict[str, List[dict]] = {m: [] for m in modes}
    emb_ms, dense_ms, bm25_ms = [], [], []

    for q in queries:
        d_rank = b_rank = None
        if "dense" in modes:
            t0 = time.perf_counter()
            qvec = embedder.embed_query(q["question"])
            emb_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            d_rank = dense_ranking(coll, qvec, depth, C.DENSE_OVERFETCH)
            dense_ms.append((time.perf_counter() - t1) * 1000)
        if "bm25" in modes:
            t2 = time.perf_counter()
            b_rank = bm25_ranking(bm25, bm25_ids, q["question"], depth)
            bm25_ms.append((time.perf_counter() - t2) * 1000)

        rankings = {}
        if d_rank is not None:
            rankings["dense"] = d_rank
        if b_rank is not None:
            rankings["bm25"] = b_rank
        if "hybrid" in modes:
            fuse_inputs = [r for r in (d_rank, b_rank) if r is not None]
            rankings["hybrid"] = rrf_fuse(fuse_inputs, C.RRF_K, depth)

        for mode, ranking in rankings.items():
            per_mode[mode].append(
                {
                    "query_id": q["query_id"],
                    "query_type": q["query_type"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "ranked": ranking,
                }
            )

    for mode, rows in per_mode.items():
        _save_rankings(arm_name, mode, rows)

    return {
        **idx_stats,
        "arm": arm_name,
        "modes": list(modes),
        "num_queries": len(queries),
        "latency_ms": {
            "query_embed_mean": round(float(np.mean(emb_ms)), 3) if emb_ms else 0,
            "dense_search_mean": round(float(np.mean(dense_ms)), 3) if dense_ms else 0,
            "bm25_search_mean": round(float(np.mean(bm25_ms)), 3) if bm25_ms else 0,
        },
        "rankings_saved": {m: str(C.ranking_path(arm_name, m)) for m in modes},
    }


def modes_for_arm(arm) -> tuple:
    """Which retrieval modes an arm produces. Standard arms produce all three;
    Exp-4 (a) is dense-only, (b) is BM25-only (declared via arm attributes)."""
    only = getattr(arm, "only_modes", None)
    if only:
        return tuple(only)
    return C.RETRIEVAL_MODES


def load_rankings(arm_name: str, mode: str) -> List[dict]:
    path = C.ranking_path(arm_name, mode)
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out
