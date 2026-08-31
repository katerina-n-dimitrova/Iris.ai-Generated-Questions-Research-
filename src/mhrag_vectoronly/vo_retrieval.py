"""
Stage: dense-VECTOR-ONLY retrieval for both conditions (§11, §12).

Condition A (baseline): query vector vs original-chunk vectors -> ranked chunks.
Condition B (generated): query vector vs generated-question vectors -> retrieve a
    LARGE candidate list of questions (rank_depth * candidate_multiplier, >=100),
    map each to its parent chunk, dedup keeping the MAX cosine similarity per
    parent (parent_chunk_score_method), return ranked unique parent chunks.

The benchmark query is embedded transiently; the query vector is never stored.
Cosine similarity = 1 - chroma_cosine_distance (vectors are L2-normalized).
No BM25 / sparse / hybrid anywhere.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

import vo_config as C
import vo_data as D
from embeddings import get_embedder


def _sim(distance: float) -> float:
    return round(1.0 - float(distance), 6)


# --------------------------------------------------------------------------- #
# Condition A — original chunk vectors
# --------------------------------------------------------------------------- #
def retrieve_baseline(coll, qvec, depth: int) -> List[dict]:
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(depth, n),
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out = []
    for rank, (m, d) in enumerate(zip(metas, dists), 1):
        out.append(
            {
                "rank": rank,
                "chunk_id": m["parent_chunk_id"],
                "parent_document_id": m["parent_document_id"],
                "score": _sim(d),
                "title": m.get("title", ""),
                "source": m.get("source", ""),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Condition B — generated-question vectors -> parent chunks (max, dedup)
# --------------------------------------------------------------------------- #
def retrieve_generated(coll, qvec, depth: int, multiplier: int) -> List[dict]:
    n = coll.count()
    k = min(max(depth * multiplier, 100), n)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    docs = res["documents"][0]

    best: Dict[str, dict] = {}  # parent chunk -> best matching question hit
    for m, d, doc in zip(metas, dists, docs):
        parent = m["parent_chunk_id"]
        s = _sim(d)
        if parent not in best or s > best[parent]["score"]:
            best[parent] = {
                "chunk_id": parent,
                "parent_document_id": m["parent_document_id"],
                "score": s,
                "best_question": doc,
                "best_question_id": m["generated_question_id"],
                "title": m.get("title", ""),
                "source": m.get("source", ""),
            }
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:depth]
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    return ranked


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_retrieval() -> dict:
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    coll_a = C.get_collection(C.BASELINE_COLLECTION)
    coll_b = C.get_collection(C.GENQ_COLLECTION)
    depth = C.RANK_DEPTH

    base_rows, gen_rows = [], []
    lat = {"query_embed_ms": [], "baseline_search_ms": [], "generated_search_ms": []}

    for q in queries:
        qid, qtext = q["query_id"], q["query"]
        g = gold[qid]

        t0 = time.perf_counter()
        qvec = embedder.embed_query(qtext)
        lat["query_embed_ms"].append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        a = retrieve_baseline(coll_a, qvec, depth)
        lat["baseline_search_ms"].append((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        b = retrieve_generated(coll_b, qvec, depth, C.CANDIDATE_MULTIPLIER)
        lat["generated_search_ms"].append((time.perf_counter() - t2) * 1000)

        common = {
            "query_id": qid,
            "query": qtext,
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        base_rows.append({**common, "condition": C.COND_A, "ranked": a})
        gen_rows.append({**common, "condition": C.COND_B, "ranked": b})

    D._write_jsonl(C.BASELINE_RANKINGS, base_rows)
    D._write_jsonl(C.GENQ_RANKINGS, gen_rows)

    def _stats(xs):
        a = np.array(xs)
        return {
            "mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
        }

    summary = {
        "num_queries": len(queries),
        "latency_ms": {k: _stats(v) for k, v in lat.items()},
    }
    # persist per-metric latency to CSV (§20)
    with C.LATENCY_CSV.open("w") as fh:
        fh.write("stage,mean_ms,p50_ms,p95_ms,p99_ms\n")
        for name, s in summary["latency_ms"].items():
            fh.write(f"{name},{s['mean']},{s['p50']},{s['p95']},{s['p99']}\n")
    with (C.RESULTS_DIR / "retrieval_latency.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"[retrieval] {len(queries)} queries | "
        f"embed {summary['latency_ms']['query_embed_ms']['mean']}ms "
        f"A {summary['latency_ms']['baseline_search_ms']['mean']}ms "
        f"B {summary['latency_ms']['generated_search_ms']['mean']}ms"
    )
    return summary


def load_rankings(condition: str) -> List[dict]:
    path = C.BASELINE_RANKINGS if condition == C.COND_A else C.GENQ_RANKINGS
    return D.read_jsonl(path)


if __name__ == "__main__":
    import pprint

    pprint.pp(run_retrieval())
