"""
Stage: dense-VECTOR-ONLY retrieval, Condition A vs Condition E (§7, §17).

A: query vector vs original-chunk vectors -> ranked unique chunks.
E: query vector vs POOLED atomic+chunk-level question vectors -> retrieve a large
   candidate list, map each to its parent chunk, dedup keeping MAX cosine per
   parent, rank unique parent chunks. The winning question's type/atom/view are
   preserved on the chunk for later diagnostics (they never change the ranking).

Benchmark query embedded transiently; query vector never stored. No BM25/hybrid.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

import am_config as C
import am_data as D
from embeddings import get_embedder


def _sim(d):
    return round(1.0 - float(d), 6)


def retrieve_baseline(coll, qvec, depth):
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(depth, n),
        include=["metadatas", "distances"],
    )
    out = []
    for rank, (m, d) in enumerate(zip(res["metadatas"][0], res["distances"][0]), 1):
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


def retrieve_mixed(coll, qvec, depth, multiplier):
    n = coll.count()
    k = min(max(depth * multiplier, 100), n)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )
    best: Dict[str, dict] = {}
    for m, d, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        parent = m["parent_chunk_id"]
        s = _sim(d)
        if parent not in best or s > best[parent]["score"]:
            best[parent] = {
                "chunk_id": parent,
                "parent_document_id": m["parent_document_id"],
                "score": s,
                "best_question": doc,
                "best_question_type": m.get("question_type", ""),
                "best_question_view": m.get("question_view", ""),
                "best_atom_id": m.get("atom_id", ""),
                "title": m.get("title", ""),
                "source": m.get("source", ""),
            }
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:depth]
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    return ranked


def run_retrieval() -> dict:
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    coll_a = C.get_collection(C.BASELINE_COLLECTION)
    coll_e = C.get_collection(C.MIXED_COLLECTION)
    depth = C.RANK_DEPTH

    base_rows, mix_rows = [], []
    lat = {"query_embed_ms": [], "baseline_search_ms": [], "mixed_search_ms": []}
    for q in queries:
        qid = q["query_id"]
        g = gold[qid]
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["query"])
        lat["query_embed_ms"].append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        a = retrieve_baseline(coll_a, qvec, depth)
        lat["baseline_search_ms"].append((time.perf_counter() - t1) * 1000)
        t2 = time.perf_counter()
        e = retrieve_mixed(coll_e, qvec, depth, C.CANDIDATE_MULTIPLIER)
        lat["mixed_search_ms"].append((time.perf_counter() - t2) * 1000)
        common = {
            "query_id": qid,
            "query": q["query"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        base_rows.append({**common, "condition": C.COND_A, "ranked": a})
        mix_rows.append({**common, "condition": C.COND_E, "ranked": e})

    D._write_jsonl(C.BASELINE_RANKINGS, base_rows)
    D._write_jsonl(C.MIXED_RANKINGS, mix_rows)

    def st(xs):
        a = np.array(xs)
        return {
            "mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
        }

    summary = {
        "num_queries": len(queries),
        "latency_ms": {k: st(v) for k, v in lat.items()},
    }
    with C.LATENCY_CSV.open("w") as fh:
        fh.write("stage,mean_ms,p50_ms,p95_ms,p99_ms\n")
        for name, s in summary["latency_ms"].items():
            fh.write(f"{name},{s['mean']},{s['p50']},{s['p95']},{s['p99']}\n")
    json.dump(summary, open(C.RETRIEVAL_LATENCY, "w"), indent=2)
    print(
        f"[retrieval] {len(queries)} queries | embed {summary['latency_ms']['query_embed_ms']['mean']}ms "
        f"A {summary['latency_ms']['baseline_search_ms']['mean']}ms "
        f"E {summary['latency_ms']['mixed_search_ms']['mean']}ms"
    )
    return summary


def load_rankings(condition: str):
    path = C.BASELINE_RANKINGS if condition == C.COND_A else C.MIXED_RANKINGS
    return D.read_jsonl(path)


if __name__ == "__main__":
    import pprint

    pprint.pp(run_retrieval())
