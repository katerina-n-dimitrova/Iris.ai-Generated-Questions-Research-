"""Evaluate the 1024/128 no-generated-question dense + BM25 baseline."""

from __future__ import annotations

import json

import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_hierarchical_hybrid as H


METRICS = R.RESULTS / "metrics_no_question_baseline.json"
RANKINGS = R.RESULTS / "rankings_no_question_baseline.json"


def run() -> dict:
    chunks = R.read_jsonl(G.CHUNKS_1024)
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    gold = {row["query_id"]: row for row in R.read_jsonl(G.GOLD_1024)}
    queries = H.load_queries()
    split = {row["query_id"]: row["split"] for row in R.read_jsonl(R.SPLIT_PATH)}
    test = [query for query in queries if split[query["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)
    # Reuse the exact chunk embeddings used by Article-E1-LargeChunk-BM25.
    chunk_vectors = R._embed_cached(
        "article_article_manual1024_chunks",
        [chunk["content"] for chunk in chunks],
    )

    rows = []
    for query in test:
        dense, _ = R._dense_chunks(
            chunks, chunk_vectors, query_vectors[query["query_id"]]
        )
        bm25 = R._bm25(chunks, query["query"])
        ranking = R._rrf([dense, bm25])
        rows.append(R._metric_row(query, gold[query["query_id"]], ranking, chunk_map))

    payload = {
        "protocol": {
            "dataset": "MultiHop-RAG closed 15-article pilot",
            "test_queries": len(test),
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "generated_questions": 0,
            "stored_vectors": len(chunks),
            "retrieval": "dense chunk vectors + BM25 chunk RRF",
            "rrf_k": R.RRF_K,
        },
        "condition": "Baseline",
        "metrics": R._evaluate(rows),
        "index": "mhrag_ctl_article_article_manual1024_chunks",
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    RANKINGS.write_text(json.dumps({"Baseline": rows}, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
