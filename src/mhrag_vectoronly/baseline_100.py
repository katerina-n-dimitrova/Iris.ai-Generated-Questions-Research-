"""No-question baseline on the adaptive experiments' frozen 100 articles."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

import adaptive_questions_100 as A


BASELINE_METRICS = A.RESULTS / "baseline_metrics.json"
BASELINE_RANKINGS = A.RESULTS / "baseline_rankings.jsonl"


def run() -> dict:
    chunks = A.read_jsonl(A.CHUNKS_PATH)
    queries = A.read_jsonl(A.QUERIES_PATH)
    summary = json.loads(A.SUMMARY_PATH.read_text())
    chunk_vectors = A.embed_resumable(
        A.CHUNK_VECTORS, [chunk["content"] for chunk in chunks]
    )
    query_vectors = A.embed_resumable(
        A.QUERY_VECTORS, [query["query"] for query in queries]
    )
    normalized_chunks = chunk_vectors / np.maximum(
        np.linalg.norm(chunk_vectors, axis=1, keepdims=True), 1e-12
    )
    normalized_queries = query_vectors / np.maximum(
        np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12
    )
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    bm25 = BM25Okapi([A.tokenize(chunk["content"]) for chunk in chunks])

    scores, rankings = [], []
    for index, query in enumerate(queries):
        dense_order = np.argsort(
            -np.einsum(
                "ij,j->i",
                normalized_chunks,
                normalized_queries[index],
                optimize=False,
            )
        )
        sparse_order = np.argsort(-bm25.get_scores(A.tokenize(query["query"])))
        rrf_scores = defaultdict(float)
        for order in (dense_order, sparse_order):
            for rank, item in enumerate(order, 1):
                rrf_scores[chunk_ids[item]] += 1 / (A.RRF_K + rank)
        ranking = [
            chunk_id
            for chunk_id, _ in sorted(
                rrf_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        values = A.metric_row(query, ranking)
        scores.append(values)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": values,
            }
        )

    keys = list(scores[0])
    condition = {
        "condition": "Baseline",
        "question_rule": "No generated questions",
        "quality": {
            "generated_questions": 0,
            "questions_per_chunk_min": 0,
            "questions_per_chunk_mean": 0.0,
            "questions_per_chunk_max": 0,
        },
        "stored_vectors": len(chunks),
        "metrics": {key: float(np.mean([row[key] for row in scores])) for key in keys},
    }
    payload = {
        "protocol": {
            **summary,
            "generation_model": "None for Baseline; gpt-5.4-mini for adaptive conditions",
            "embedding": "Iris dim-384",
            "retrieval": "dense chunk vectors + BM25 RRF for Baseline; 0.5 chunk/question vector fusion + BM25 RRF for adaptive conditions",
            "rrf_k": A.RRF_K,
        },
        "condition": condition,
    }
    BASELINE_METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    A.write_jsonl(BASELINE_RANKINGS, rankings)

    combined = json.loads(A.METRICS.read_text())
    combined["protocol"] = payload["protocol"]
    combined["conditions"] = [condition] + [
        row for row in combined["conditions"] if row["condition"] != "Baseline"
    ]
    A.METRICS.write_text(json.dumps(combined, indent=2) + "\n")
    combined_rankings = json.loads(A.RANKINGS.read_text())
    combined_rankings["Baseline"] = rankings
    A.RANKINGS.write_text(json.dumps(combined_rankings))
    A.render(combined)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
