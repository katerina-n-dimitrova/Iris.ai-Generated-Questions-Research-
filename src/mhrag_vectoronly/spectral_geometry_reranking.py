"""Spectral-geometry-inspired reranking for adaptive article questions.

This is an inspired controlled experiment, not a reproduction of any paper.
All hyperparameters are selected on development queries; test queries are used
only once after selection.
"""

from __future__ import annotations

import json

import numpy as np

import adaptive_article_fact_pipeline as F
import controlled_article_first as A
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_hierarchical_hybrid as H
import vo_metrics as VM

METRICS = R.RESULTS / "metrics_spectral_geometry_reranking.json"
RANKINGS = R.RESULTS / "rankings_spectral_geometry_reranking.json"
POOL_OPTIONS = (20, 30, 44)
WEIGHT_OPTIONS = (0.6, 0.7, 0.8)
PCA_THRESHOLDS = (0.8, 0.9, 0.95)


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _pca(chunk_vectors: np.ndarray) -> dict[float, np.ndarray]:
    centered = chunk_vectors - chunk_vectors.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / variance.sum()
    output = {}
    for threshold in PCA_THRESHOLDS:
        n_components = int(np.searchsorted(cumulative, threshold) + 1)
        projected = np.einsum("ij,kj->ik", centered, vt[:n_components], optimize=False)
        output[threshold] = _l2(projected)
    return output


def _relevance(pool_size: int) -> np.ndarray:
    if pool_size == 1:
        return np.ones(1)
    return 1.0 - np.arange(pool_size, dtype=float) / (pool_size - 1)


def _greedy(
    base_ranking: list[str],
    vectors_by_id: dict[str, np.ndarray],
    pool_size: int,
    relevance_weight: float,
    method: str,
) -> list[str]:
    pool = base_ranking[:pool_size]
    matrix = np.vstack([vectors_by_id[chunk_id] for chunk_id in pool])
    relevance = _relevance(len(pool))
    selected: list[int] = []
    remaining = set(range(len(pool)))
    while remaining:
        best = None
        for index in remaining:
            if not selected:
                gain = 1.0
            elif method == "mmr":
                similarity = float(
                    np.max(
                        np.einsum(
                            "ij,j->i", matrix[selected], matrix[index], optimize=False
                        )
                    )
                )
                gain = float(np.clip(1.0 - similarity, 0.0, 1.0))
            else:
                basis_source = matrix[selected].T
                basis, _ = np.linalg.qr(basis_source)
                coefficients = np.einsum(
                    "ij,i->j", basis, matrix[index], optimize=False
                )
                projection = np.einsum("ij,j->i", basis, coefficients, optimize=False)
                residual = matrix[index] - projection
                gain = float(np.clip(np.linalg.norm(residual), 0.0, 1.0))
            score = (
                relevance_weight * relevance[index] + (1.0 - relevance_weight) * gain
            )
            candidate = (score, relevance[index], -index, index)
            if best is None or candidate > best:
                best = candidate
        selected.append(best[-1])
        remaining.remove(best[-1])
    reranked_pool = [pool[index] for index in selected]
    return reranked_pool + base_ranking[pool_size:]


def _base_rankings(
    queries: list[dict],
    chunks: list[dict],
    questions: list[dict],
    chunk_vectors: np.ndarray,
    question_vectors: np.ndarray,
    query_vectors: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    output = {}
    for query in queries:
        vector = query_vectors[query["query_id"]]
        chunk_rank, chunk_scores = R._dense_chunks(chunks, chunk_vectors, vector)
        question_scores = A._question_scores(questions, question_vectors, vector)
        dense = R._dual(chunk_rank, chunk_scores, question_scores)
        output[query["query_id"]] = R._rrf([dense, R._bm25(chunks, query["query"])])
    return output


def _rows(
    queries: list[dict],
    rankings: dict[str, list[str]],
    gold: dict,
    chunk_map: dict,
) -> list[dict]:
    return [
        R._metric_row(
            query,
            gold[query["query_id"]],
            rankings[query["query_id"]],
            chunk_map,
        )
        for query in queries
    ]


def _redundancy(
    queries: list[dict],
    rankings: dict[str, list[str]],
    normalized_chunks: dict[str, np.ndarray],
    k: int = 5,
) -> dict:
    similarities = []
    duplicate_pairs = 0
    for query in queries:
        ids = rankings[query["query_id"]][:k]
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                similarity = float(
                    np.dot(
                        normalized_chunks[ids[left]],
                        normalized_chunks[ids[right]],
                    )
                )
                similarities.append(similarity)
                duplicate_pairs += int(similarity >= 0.90)
    return {
        "mean_pairwise_cosine@5": float(np.mean(similarities)),
        "high_similarity_pairs@5": duplicate_pairs,
        "high_similarity_pair_rate@5": duplicate_pairs / len(similarities),
    }


def _score(rows: list[dict], redundancy: dict) -> tuple:
    metrics = R._evaluate(rows)
    return (
        metrics["evidence_recall@5"],
        metrics["all_evidence_hit@5"],
        -redundancy["mean_pairwise_cosine@5"],
        metrics["mrr@10"],
    )


def run() -> dict:
    chunks = G.read_jsonl(G.CHUNKS_1024)
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    gold = {row["query_id"]: row for row in G.read_jsonl(G.GOLD_1024)}
    adaptive_cache = G.read_jsonl(F.CACHE)
    questions = F._question_rows(adaptive_cache)
    chunk_vectors, question_vectors = A._index("adaptive_fact1024", chunks, questions)
    normalized_matrix = _l2(chunk_vectors)
    normalized_chunks = {
        chunk["chunk_id"]: normalized_matrix[index]
        for index, chunk in enumerate(chunks)
    }
    spectral_matrices = _pca(chunk_vectors)
    spectral_by_threshold = {
        threshold: {
            chunk["chunk_id"]: matrix[index] for index, chunk in enumerate(chunks)
        }
        for threshold, matrix in spectral_matrices.items()
    }

    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    development = [
        query for query in queries if split[query["query_id"]] == "development"
    ]
    test = [query for query in queries if split[query["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)
    all_base = _base_rankings(
        development + test,
        chunks,
        questions,
        chunk_vectors,
        question_vectors,
        query_vectors,
    )

    best_mmr = None
    for pool_size in POOL_OPTIONS:
        for weight in WEIGHT_OPTIONS:
            rankings = {
                query["query_id"]: _greedy(
                    all_base[query["query_id"]],
                    normalized_chunks,
                    pool_size,
                    weight,
                    "mmr",
                )
                for query in development
            }
            rows = _rows(development, rankings, gold, chunk_map)
            redundancy = _redundancy(development, rankings, normalized_chunks)
            candidate = (
                _score(rows, redundancy),
                -pool_size,
                -weight,
                {"candidate_pool": pool_size, "relevance_weight": weight},
            )
            if best_mmr is None or candidate[:3] > best_mmr[:3]:
                best_mmr = candidate

    best_spectral = None
    for pool_size in POOL_OPTIONS:
        for threshold in PCA_THRESHOLDS:
            for weight in WEIGHT_OPTIONS:
                vectors = spectral_by_threshold[threshold]
                rankings = {
                    query["query_id"]: _greedy(
                        all_base[query["query_id"]],
                        vectors,
                        pool_size,
                        weight,
                        "spectral",
                    )
                    for query in development
                }
                rows = _rows(development, rankings, gold, chunk_map)
                redundancy = _redundancy(development, rankings, normalized_chunks)
                candidate = (
                    _score(rows, redundancy),
                    -pool_size,
                    -threshold,
                    -weight,
                    {
                        "candidate_pool": pool_size,
                        "pca_variance_threshold": threshold,
                        "pca_components": spectral_matrices[threshold].shape[1],
                        "relevance_weight": weight,
                    },
                )
                if best_spectral is None or candidate[:4] > best_spectral[:4]:
                    best_spectral = candidate

    mmr_params = best_mmr[-1]
    spectral_params = best_spectral[-1]
    test_rankings = {"Base": {q["query_id"]: all_base[q["query_id"]] for q in test}}
    test_rankings["Cosine-MMR"] = {
        query["query_id"]: _greedy(
            all_base[query["query_id"]],
            normalized_chunks,
            mmr_params["candidate_pool"],
            mmr_params["relevance_weight"],
            "mmr",
        )
        for query in test
    }
    spectral_vectors = spectral_by_threshold[spectral_params["pca_variance_threshold"]]
    test_rankings["Spectral-geometry"] = {
        query["query_id"]: _greedy(
            all_base[query["query_id"]],
            spectral_vectors,
            spectral_params["candidate_pool"],
            spectral_params["relevance_weight"],
            "spectral",
        )
        for query in test
    }

    conditions = []
    ranking_rows = {}
    for condition, rankings in test_rankings.items():
        rows = _rows(test, rankings, gold, chunk_map)
        ranking_rows[condition] = rows
        conditions.append(
            {
                "condition": condition,
                "metrics": R._evaluate(rows),
                "redundancy": _redundancy(test, rankings, normalized_chunks),
                "parameters": (
                    None
                    if condition == "Base"
                    else mmr_params
                    if condition == "Cosine-MMR"
                    else spectral_params
                ),
            }
        )
    payload = {
        "experiment": "Spectral-geometry-inspired retrieval reranking",
        "not_a_direct_paper_reproduction": True,
        "question_filtering": "disabled to isolate retrieval reranking",
        "representation": (
            "44 original 1024-token chunk vectors + 239 adaptive "
            "article-question vectors; no other embeddings"
        ),
        "selection_protocol": {
            "selected_on": "development split only",
            "primary_metric": "evidence_recall@5",
            "tie_breakers": [
                "all_evidence_hit@5",
                "lower mean pairwise cosine@5",
                "mrr@10",
            ],
            "candidate_pool_options": POOL_OPTIONS,
            "relevance_weight_options": WEIGHT_OPTIONS,
            "pca_variance_threshold_options": PCA_THRESHOLDS,
            "mmr_selected": mmr_params,
            "spectral_selected": spectral_params,
        },
        "conditions": conditions,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(ranking_rows))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
