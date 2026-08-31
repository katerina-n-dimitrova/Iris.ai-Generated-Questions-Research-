"""
Run retrieval for every (dataset, condition): embed each query, search the
matching Chroma collection, assemble context, and log online latency.

Outputs:
  - results/latency_logs/online_query_latency.csv   (one row per query)
  - results/retrieval_metrics/retrieved_<dataset>_<condition>.jsonl
        (per-query retrieved chunk ids + source ids + gold, for evaluation)

Online latency columns:
    query_id, dataset, condition, top_k, query_embedding_latency_ms,
    chroma_search_latency_ms, context_assembly_latency_ms,
    total_retrieval_latency_ms, retrieved_chunk_ids

Usage:
    python src/run_retrieval_experiments.py
    python src/run_retrieval_experiments.py --datasets scifact --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Dict, List

from tqdm import tqdm

import common
import config
import embeddings

ONLINE_LOG = config.LATENCY_LOG_DIR / "online_query_latency.csv"
ONLINE_FIELDS = [
    "query_id",
    "dataset",
    "condition",
    "top_k",
    "query_embedding_latency_ms",
    "chroma_search_latency_ms",
    "context_assembly_latency_ms",
    "total_retrieval_latency_ms",
    "retrieved_chunk_ids",
]


def _load_queries(dataset: str, max_samples: int) -> List[Dict]:
    path = config.PROCESSED_DIR / f"{dataset}_queries.jsonl"
    if not path.exists():
        print(f"  no queries for {dataset} ({path.name}); skipping")
        return []
    return list(common.read_jsonl(path))[:max_samples]


def run_one(
    client, dataset: str, condition: str, top_k: int, retrieve_k: int, max_samples: int
) -> List[Dict]:
    spec = config.DATASETS[dataset]
    coll_name = spec.collection_name(condition)
    try:
        collection = client.get_collection(coll_name)
    except Exception:
        print(f"  collection {coll_name} not found; run build_chroma_indexes.py")
        return []

    queries = _load_queries(dataset, max_samples)
    if not queries:
        return []

    embedder = embeddings.get_embedder()
    latency_rows: List[Dict] = []
    retrieved_rows: List[Dict] = []

    for q in tqdm(queries, desc=f"  {dataset}/{condition}", leave=False):
        qtext = q["text"]

        t_embed = time.perf_counter()
        qvec = embedder.embed_query(qtext)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_search = time.perf_counter()
        res = collection.query(
            query_embeddings=[qvec],
            n_results=retrieve_k,
            include=["documents", "metadatas", "distances"],
        )
        search_ms = (time.perf_counter() - t_search) * 1000

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        t_ctx = time.perf_counter()
        context = "\n\n---\n\n".join(docs[:top_k])  # assemble top_k as context
        ctx_ms = (time.perf_counter() - t_ctx) * 1000

        total_ms = embed_ms + search_ms + ctx_ms

        latency_rows.append(
            {
                "query_id": q["query_id"],
                "dataset": dataset,
                "condition": condition,
                "top_k": top_k,
                "query_embedding_latency_ms": round(embed_ms, 4),
                "chroma_search_latency_ms": round(search_ms, 4),
                "context_assembly_latency_ms": round(ctx_ms, 4),
                "total_retrieval_latency_ms": round(total_ms, 4),
                "retrieved_chunk_ids": "|".join(ids[:top_k]),
            }
        )

        retrieved_rows.append(
            {
                "query_id": q["query_id"],
                "dataset": dataset,
                "condition": condition,
                "query_text": qtext,
                "gold_source_ids": q.get("gold_source_ids", []),
                "gold_answer": q.get("gold_answer", ""),
                "retrieved": [
                    {
                        "chunk_id": ids[i],
                        "source_id": (metas[i] or {}).get("source_id"),
                        "distance": dists[i] if i < len(dists) else None,
                        "document": docs[i] if i < len(docs) else "",
                    }
                    for i in range(len(ids))
                ],
                "context": context,
            }
        )

    # persist per-(dataset,condition) retrieval output for evaluation/answers
    out_path = config.RETRIEVAL_METRICS_DIR / f"retrieved_{dataset}_{condition}.jsonl"
    common.write_jsonl(out_path, retrieved_rows)
    print(f"  {dataset}/{condition}: {len(retrieved_rows)} queries -> {out_path.name}")
    return latency_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=config.ALL_DATASETS,
        choices=config.ALL_DATASETS,
    )
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=list(config.CONDITIONS),
        choices=config.CONDITIONS,
    )
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    # retrieve enough to compute Recall@10 / nDCG@10 regardless of top_k
    retrieve_k = max(max(config.RETRIEVAL_K_VALUES), args.top_k)
    print(f"Embedding signature: {embeddings.embedding_signature()}")
    print(f"Chroma backend:      {config.chroma_signature()}")
    print(f"top_k={args.top_k}, retrieving {retrieve_k} per query")

    client = config.get_chroma_client()

    all_latency: List[Dict] = []
    for dataset in args.datasets:
        for condition in args.conditions:
            all_latency += run_one(
                client, dataset, condition, args.top_k, retrieve_k, args.max_samples
            )

    if all_latency:
        groups = {(r["dataset"], r["condition"]) for r in all_latency}
        common.upsert_csv(
            ONLINE_LOG, ONLINE_FIELDS, all_latency, ("dataset", "condition"), groups
        )
        print(f"\nOnline latency log -> {ONLINE_LOG.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
