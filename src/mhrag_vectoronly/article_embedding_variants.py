"""Whole-article embedding variants of the locked and adaptive baselines."""

from __future__ import annotations

import json

import numpy as np

import adaptive_article_fact_pipeline as F
import controlled_article_first as A
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

METRICS = R.RESULTS / "metrics_article_embedding_variants.json"
RANKINGS = R.RESULTS / "rankings_article_embedding_variants.json"


def _article_vectors(articles: list[dict]) -> np.ndarray:
    vectors = R._embed_cached(
        "whole_articles", [article["cleaned_body"] for article in articles]
    )
    collection = C.reset_collection("mhrag_ctl_article_whole_articles")
    collection.add(
        ids=[article["article_id"] for article in articles],
        embeddings=vectors.tolist(),
        documents=[article["cleaned_body"] for article in articles],
        metadatas=[{"document_title": article["title"]} for article in articles],
    )
    return vectors


def _three_way_dense(
    chunks: list[dict],
    chunk_vectors: np.ndarray,
    questions: list[dict],
    question_vectors: np.ndarray,
    articles: list[dict],
    article_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> list[str]:
    _, chunk_scores = R._dense_chunks(chunks, chunk_vectors, query_vector)
    question_scores = A._question_scores(questions, question_vectors, query_vector)
    raw_article_scores = H._cosine_matrix(article_vectors, query_vector)
    article_scores = {
        article["article_id"]: float(score)
        for article, score in zip(articles, raw_article_scores)
    }
    scores = {
        chunk["chunk_id"]: (
            chunk_scores[chunk["chunk_id"]]
            + question_scores.get(chunk["chunk_id"], -1.0)
            + article_scores[chunk["document_id"]]
        )
        / 3.0
        for chunk in chunks
    }
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _fixed_questions(chunks: list[dict]) -> list[dict]:
    return A._question_rows(A.MANUAL_Q, chunks, remap=True)


def run() -> dict:
    articles = H.read_jsonl(H.ARTICLES_PATH)
    chunks = G.read_jsonl(G.CHUNKS_1024)
    gold = {row["query_id"]: row for row in G.read_jsonl(G.GOLD_1024)}
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    test = [query for query in queries if split[query["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)
    article_vectors = _article_vectors(articles)

    adaptive_cache = G.read_jsonl(F.CACHE)
    if len(adaptive_cache) != len(articles):
        raise RuntimeError("Run adaptive_article_fact_pipeline.py first")
    configs = (
        (
            "Baseline B",
            "article_manual1024",
            _fixed_questions(chunks),
            "Fixed 10 questions per complete article",
        ),
        (
            "Adaptive facts B",
            "adaptive_fact1024",
            F._question_rows(adaptive_cache),
            "Adaptive 5–20 questions from distinct-fact analysis",
        ),
    )
    conditions, rankings = [], {}
    for condition, config_name, questions, generation in configs:
        chunk_vectors, question_vectors = A._index(config_name, chunks, questions)
        rows = []
        for query in test:
            dense = _three_way_dense(
                chunks,
                chunk_vectors,
                questions,
                question_vectors,
                articles,
                article_vectors,
                query_vectors[query["query_id"]],
            )
            ranking = R._rrf([dense, R._bm25(chunks, query["query"])])
            rows.append(
                R._metric_row(query, gold[query["query_id"]], ranking, chunk_map)
            )
        rankings[condition] = rows
        conditions.append(
            {
                "condition": condition,
                "question_generation": generation,
                "generated_questions": len(questions),
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "articles_embedded": len(articles),
                "chunks_embedded": len(chunks),
                "stored_vectors": len(articles) + len(chunks) + len(questions),
                "generation_model": C.gen_model(),
                "embedding": "Iris dim384",
                "dense_fusion": (
                    "1/3 whole-article + 1/3 chunk + 1/3 mapped-question cosine"
                ),
                "retrieval": "three-way dense score fusion + BM25 chunk RRF",
                "metrics": R._evaluate(rows),
                "indexes": [
                    "mhrag_ctl_article_whole_articles",
                    f"mhrag_ctl_article_{config_name}_chunks",
                    f"mhrag_ctl_article_{config_name}_questions",
                ],
            }
        )
    payload = {
        "protocol": {
            "test_queries": len(test),
            "article_vectors": len(articles),
            "chunking": "1024 / 128",
            "dense_fusion": "equal 1/3 article, chunk, and question cosine",
            "sparse_fusion": "BM25 chunk ranking via RRF k=60",
        },
        "conditions": conditions,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(rankings))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
