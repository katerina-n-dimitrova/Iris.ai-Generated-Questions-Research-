"""Controlled chunk/question fusion-weight selection on adaptive questions."""

from __future__ import annotations

import json

import numpy as np

import adaptive_article_fact_pipeline as F
import controlled_article_first as A
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_hierarchical_hybrid as H

METRICS = R.RESULTS / "metrics_chunk_question_fusion_weights.json"
RANKINGS = R.RESULTS / "rankings_chunk_question_fusion_weight_winner.json"
CONDITIONS = (
    ("W0", 0.00),
    ("W25", 0.25),
    ("W50", 0.50),
    ("W75", 0.75),
    ("W100", 1.00),
)


def _minmax(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _scores(
    chunks: list[dict],
    questions: list[dict],
    chunk_vectors: np.ndarray,
    question_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_chunk = H._cosine_matrix(chunk_vectors, query_vector)
    mapped_question = A._question_scores(questions, question_vectors, query_vector)
    observed = np.asarray(list(mapped_question.values()), dtype=float)
    missing_value = float(observed.min()) if len(observed) else -1.0
    raw_question = np.asarray(
        [mapped_question.get(chunk["chunk_id"], missing_value) for chunk in chunks]
    )
    return _minmax(raw_chunk), _minmax(raw_question)


def _ranking(
    chunks: list[dict],
    chunk_scores: np.ndarray,
    question_scores: np.ndarray,
    alpha: float,
) -> list[str]:
    fusion = alpha * chunk_scores + (1.0 - alpha) * question_scores
    order = H._rank_indices(fusion)
    return [chunks[index]["chunk_id"] for index in order]


def _evaluate_condition(
    queries: list[dict],
    alpha: float,
    chunks: list[dict],
    questions: list[dict],
    chunk_vectors: np.ndarray,
    question_vectors: np.ndarray,
    query_vectors: dict[str, np.ndarray],
    gold: dict,
    chunk_map: dict,
) -> tuple[dict, list[dict]]:
    rows = []
    for query in queries:
        chunk_scores, question_scores = _scores(
            chunks,
            questions,
            chunk_vectors,
            question_vectors,
            query_vectors[query["query_id"]],
        )
        ranking = _ranking(chunks, chunk_scores, question_scores, alpha)
        rows.append(R._metric_row(query, gold[query["query_id"]], ranking, chunk_map))
    return R._evaluate(rows), rows


def run() -> dict:
    chunks = G.read_jsonl(G.CHUNKS_1024)
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    gold = {row["query_id"]: row for row in G.read_jsonl(G.GOLD_1024)}
    adaptive_cache = G.read_jsonl(F.CACHE)
    questions = F._question_rows(adaptive_cache)

    # Reuse the exact cached vectors. No embedding or generation calls occur.
    chunk_vectors = R._embed_cached(
        "article_adaptive_fact1024_chunks",
        [chunk["content"] for chunk in chunks],
    )
    question_vectors = R._embed_cached(
        "article_adaptive_fact1024_questions",
        [question["text"] for question in questions],
    )
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    development = [
        query for query in queries if split[query["query_id"]] == "development"
    ]
    test = [query for query in queries if split[query["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)

    development_rows = []
    for condition, alpha in CONDITIONS:
        metrics, _ = _evaluate_condition(
            development,
            alpha,
            chunks,
            questions,
            chunk_vectors,
            question_vectors,
            query_vectors,
            gold,
            chunk_map,
        )
        development_rows.append(
            {
                "condition": condition,
                "alpha": alpha,
                "chunk_weight": alpha,
                "question_weight": 1.0 - alpha,
                "metrics": metrics,
            }
        )

    # Primary metric and tie-breakers are fixed before looking at test results.
    # A final exact tie prefers the weight closest to the established W50.
    winner = max(
        development_rows,
        key=lambda row: (
            row["metrics"]["all_evidence_hit@5"],
            row["metrics"]["evidence_recall@5"],
            row["metrics"]["mrr@10"],
            -abs(row["alpha"] - 0.5),
        ),
    )
    test_metrics, test_rows = _evaluate_condition(
        test,
        winner["alpha"],
        chunks,
        questions,
        chunk_vectors,
        question_vectors,
        query_vectors,
        gold,
        chunk_map,
    )
    w50 = next(row for row in development_rows if row["condition"] == "W50")
    for row in development_rows:
        row["delta_vs_w50"] = {
            key: row["metrics"][key] - w50["metrics"][key] for key in row["metrics"]
        }

    payload = {
        "experiment": "Chunk–Question Fusion Weight Analysis",
        "representation": (
            "existing 44 1024-token chunk vectors and 239 adaptive "
            "article-question vectors"
        ),
        "embedding": "existing Iris dim384 caches (not Qwen)",
        "score_normalization": (
            "per-query min-max, independently for chunk and question scores; "
            "chunks without mapped questions receive normalized question score 0"
        ),
        "disabled": ["BM25", "RRF", "geometry reranking", "thresholds"],
        "candidate_pool": 44,
        "final_rank_depth": 10,
        "selection": {
            "split": "development only",
            "primary": "all_evidence_hit@5",
            "tie_breakers": ["evidence_recall@5", "mrr@10"],
            "final_exact_tie": "closest to W50",
            "winner": winner["condition"],
            "alpha": winner["alpha"],
        },
        "development_conditions": development_rows,
        "untouched_test_winner": {
            "condition": winner["condition"],
            "alpha": winner["alpha"],
            "metrics": test_metrics,
        },
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps({winner["condition"]: test_rows}))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
