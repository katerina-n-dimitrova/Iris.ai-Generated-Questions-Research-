"""Controlled retrieval/evaluation core for the three requested experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import controlled_suite_generate as G
import controlled_suite_optimize as O
import vo_config as C
import vo_hierarchical_hybrid as H
import vo_metrics as VM
from embeddings import get_embedder

RESULTS = C.RESULTS_DIR / "controlled_three_experiments"
RESULTS.mkdir(parents=True, exist_ok=True)
RANKINGS = RESULTS / "rankings_manual_conditions.json"
METRICS = RESULTS / "metrics_manual_conditions.json"
MANIFEST = RESULTS / "retrieval_manifest.json"
SPLIT_PATH = H.DATA / "query_split_seed42.jsonl"
QUERY_VECTORS = RESULTS / "query_vectors_iris.json"
RRF_K = 60
DEPTH = C.RANK_DEPTH
METRIC_KEYS = (
    "evidence_recall@1",
    "evidence_recall@5",
    "evidence_recall@10",
    "all_evidence_hit@5",
    "mrr@10",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _questions(path: Path) -> list[dict]:
    output = []
    for row in read_jsonl(path):
        if "questions" in row:
            questions = row["questions"]
        else:
            questions = [
                question
                for fact in row.get("facts", [])
                for question in fact.get("questions", [])
            ]
        output.extend(
            {
                "id": f"{row['chunk_id']}::q{index}",
                "chunk_id": row["chunk_id"],
                "text": question,
            }
            for index, question in enumerate(questions)
        )
    return output


def _embed_cached(name: str, texts: list[str]) -> np.ndarray:
    path = RESULTS / f"vectors_{name}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("texts") == texts:
            return np.asarray(payload["vectors"], dtype=float)
    vectors = np.asarray(get_embedder().embed_documents(texts), dtype=float)
    path.write_text(json.dumps({"texts": texts, "vectors": vectors.tolist()}))
    return vectors


def _query_vectors(queries: list[dict]) -> dict[str, np.ndarray]:
    if QUERY_VECTORS.exists():
        payload = json.loads(QUERY_VECTORS.read_text())
        if set(payload) == {query["query_id"] for query in queries}:
            return {
                key: np.asarray(value, dtype=float) for key, value in payload.items()
            }
    embedder = get_embedder()
    payload = {}
    for query in queries:
        payload[query["query_id"]] = embedder.embed_query(query["query"])
    QUERY_VECTORS.write_text(json.dumps(payload))
    return {key: np.asarray(value, dtype=float) for key, value in payload.items()}


def _index(
    name: str, chunks: list[dict], question_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    chunk_vectors = _embed_cached(
        f"{name}_chunks", [chunk["content"] for chunk in chunks]
    )
    question_vectors = _embed_cached(
        f"{name}_questions", [row["text"] for row in question_rows]
    )
    chunk_collection = C.reset_collection(f"mhrag_ctl_{name}_chunks")
    chunk_collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=chunk_vectors.tolist(),
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[
            {
                "parent_chunk_id": chunk["chunk_id"],
                "parent_document_id": chunk["document_id"],
            }
            for chunk in chunks
        ],
    )
    question_collection = C.reset_collection(f"mhrag_ctl_{name}_questions")
    for start in range(0, len(question_rows), 5000):
        batch = question_rows[start : start + 5000]
        question_collection.add(
            ids=[row["id"] for row in batch],
            embeddings=question_vectors[start : start + len(batch)].tolist(),
            documents=[row["text"] for row in batch],
            metadatas=[{"parent_chunk_id": row["chunk_id"]} for row in batch],
        )
    return chunk_vectors, question_vectors


def _rrf(rankings: list[list[str]]) -> list[str]:
    score = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] += 1 / (RRF_K + rank)
    return [
        item for item, _ in sorted(score.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _dense_chunks(
    chunks: list[dict], chunk_vectors: np.ndarray, qvec: np.ndarray
) -> tuple[list[str], dict[str, float]]:
    scores = H._cosine_matrix(chunk_vectors, qvec)
    order = H._rank_indices(scores)
    ids = [chunks[index]["chunk_id"] for index in order]
    return ids, {chunks[index]["chunk_id"]: float(scores[index]) for index in order}


def _dense_questions(
    question_rows: list[dict], question_vectors: np.ndarray, qvec: np.ndarray
) -> tuple[list[str], dict[str, float], float]:
    scores = H._cosine_matrix(question_vectors, qvec)
    best = {}
    for index in H._rank_indices(scores):
        chunk_id = question_rows[index]["chunk_id"]
        best.setdefault(chunk_id, float(scores[index]))
    ranking = [
        chunk_id
        for chunk_id, _ in sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return ranking, best, float(scores.max())


def _dual(
    chunk_rank: list[str],
    chunk_scores: dict[str, float],
    question_scores: dict[str, float],
) -> list[str]:
    # Equal-weight score fusion is fixed for every dual-index condition.
    score = {
        chunk_id: 0.5 * chunk_scores[chunk_id]
        + 0.5 * question_scores.get(chunk_id, -1.0)
        for chunk_id in chunk_rank
    }
    return [
        chunk_id
        for chunk_id, _ in sorted(score.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _bm25(chunks: list[dict], query: str) -> list[str]:
    model = BM25Okapi([H.tokens(chunk["content"]) for chunk in chunks])
    scores = model.get_scores(H.tokens(query))
    return [chunks[index]["chunk_id"] for index in H._rank_indices(scores)]


def _metric_row(query: dict, gold: dict, ranking: list[str], chunk_map: dict) -> dict:
    return {
        "query_id": query["query_id"],
        "question_type": query["question_type"],
        "n_required_documents": query["n_required_documents"],
        "n_required_evidence_facts": query["n_required_evidence_facts"],
        "gold_chunk_ids": gold["gold_chunk_ids"],
        "evidence_units": gold["evidence_units"],
        "required_article_ids": query["required_article_ids"],
        "ranked": [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "parent_document_id": chunk_map[chunk_id]["document_id"],
                "score": float(len(ranking) - rank + 1),
            }
            for rank, chunk_id in enumerate(ranking[:DEPTH], 1)
        ],
    }


def _dataset(chunk_size: int, question_path: Path) -> dict:
    if chunk_size == 512:
        chunks = H.load_chunks()
        gold = H.load_gold()
    else:
        chunks = read_jsonl(G.CHUNKS_1024)
        gold = {row["query_id"]: row for row in read_jsonl(G.GOLD_1024)}
    questions = _questions(question_path)
    return {
        "chunks": chunks,
        "chunk_map": {chunk["chunk_id"]: chunk for chunk in chunks},
        "gold": gold,
        "questions": questions,
    }


def _route_rankings(
    dataset: dict,
    query: dict,
    qvec: np.ndarray,
    chunk_vectors: np.ndarray,
    question_vectors: np.ndarray,
) -> dict:
    chunk_rank, chunk_scores = _dense_chunks(dataset["chunks"], chunk_vectors, qvec)
    question_rank, question_scores, confidence = _dense_questions(
        dataset["questions"], question_vectors, qvec
    )
    dual = _dual(chunk_rank, chunk_scores, question_scores)
    sparse = _bm25(dataset["chunks"], query["query"])
    return {
        "chunk": chunk_rank,
        "question": question_rank,
        "dual": dual,
        "dual_bm25": _rrf([dual, sparse]),
        "question_bm25": _rrf([question_rank, sparse]),
        "confidence": confidence,
    }


def _select_threshold(
    dev_queries: list[dict], routes: dict[str, dict], dataset: dict
) -> float:
    candidates = sorted(
        {round(routes[query["query_id"]]["confidence"], 6) for query in dev_queries}
    )
    best = None
    for threshold in candidates:
        rows = []
        for query in dev_queries:
            route = routes[query["query_id"]]
            ranking = (
                route["question"]
                if route["confidence"] >= threshold
                else route["chunk"]
            )
            rows.append(
                _metric_row(
                    query,
                    dataset["gold"][query["query_id"]],
                    ranking,
                    dataset["chunk_map"],
                )
            )
        score = np.mean([VM.per_query(row)["evidence_recall@5"] for row in rows])
        candidate = (score, -threshold, threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[2])


def _evaluate(rows: list[dict]) -> dict:
    per = [VM.per_query(row) for row in rows]
    return {key: float(np.mean([item[key] for item in per])) for key in METRIC_KEYS}


def run() -> dict:
    queries = H.load_queries()
    split = {row["query_id"]: row["split"] for row in read_jsonl(SPLIT_PATH)}
    development = [q for q in queries if split[q["query_id"]] == "development"]
    test = [q for q in queries if split[q["query_id"]] == "test"]
    qvectors = _query_vectors(queries)

    configs = {
        "general512": _dataset(512, G.MANUAL_GENERAL_512),
        "general1024": _dataset(1024, G.MANUAL_GENERAL_1024),
        "adaptive512": _dataset(512, G.ADAPTIVE_GENERAL_512),
        "atomic512": _dataset(512, G.MANUAL_ATOMIC_512),
        "atomic1024": _dataset(1024, G.MANUAL_ATOMIC_1024),
        "atomic_adaptive512": _dataset(512, G.ADAPTIVE_ATOMIC_512),
        "gepa_general512": _dataset(512, O.GEPA_GENERAL_Q),
        "mipro_general512": _dataset(512, O.MIPRO_GENERAL_Q),
        "gepa_atomic512": _dataset(512, O.GEPA_ATOMIC_Q),
        "mipro_atomic512": _dataset(512, O.MIPRO_ATOMIC_Q),
    }
    indexed = {}
    all_routes = {}
    for name, dataset in configs.items():
        indexed[name] = _index(name, dataset["chunks"], dataset["questions"])
        chunk_vectors, question_vectors = indexed[name]
        all_routes[name] = {
            query["query_id"]: _route_rankings(
                dataset,
                query,
                qvectors[query["query_id"]],
                chunk_vectors,
                question_vectors,
            )
            for query in queries
        }

    thresholds = {
        name: _select_threshold(development, all_routes[name], configs[name])
        for name in ("general512", "gepa_general512", "mipro_general512")
    }
    threshold = thresholds["general512"]
    threshold_bm25 = _select_threshold_bm25(
        development, all_routes["general512"], configs["general512"]
    )
    specs = [
        ("Experiment 1", "E1-ManualPrompt", "general512", "dual"),
        ("Experiment 1", "E1-BM25", "general512", "dual_bm25"),
        ("Experiment 1", "E1-LargeChunk", "general1024", "dual"),
        ("Experiment 1", "E1-GEPA", "gepa_general512", "dual"),
        ("Experiment 1", "E1-MIPROv2", "mipro_general512", "dual"),
        ("Experiment 1", "E1-Adaptive", "adaptive512", "dual"),
        ("Experiment 2", "E2-ManualPrompt", "general512", "fallback"),
        ("Experiment 2", "E2-BM25", "general512", "fallback_bm25"),
        ("Experiment 2", "E2-LargeChunk", "general1024", "fallback"),
        ("Experiment 2", "E2-GEPA", "gepa_general512", "fallback"),
        ("Experiment 2", "E2-MIPROv2", "mipro_general512", "fallback"),
        ("Experiment 3", "E3-ManualPrompt", "atomic512", "question"),
        ("Experiment 3", "E3-BM25", "atomic512", "question_bm25"),
        ("Experiment 3", "E3-LargeChunk", "atomic1024", "question"),
        ("Experiment 3", "E3-GEPA", "gepa_atomic512", "question"),
        ("Experiment 3", "E3-MIPROv2", "mipro_atomic512", "question"),
        ("Experiment 3", "E3-Adaptive", "atomic_adaptive512", "question"),
    ]
    output_rows, rankings = [], {}
    for experiment, condition, config_name, method in specs:
        dataset = configs[config_name]
        rows, fallback_count = [], 0
        question_route_rows, fallback_route_rows = [], []
        weak_question_rows, weak_fallback_rows = [], []
        for query in test:
            route = all_routes[config_name][query["query_id"]]
            if method == "fallback":
                active_threshold = thresholds.get(config_name, threshold)
                fallback = route["confidence"] < active_threshold
                ranking = route["chunk"] if fallback else route["question"]
                fallback_count += int(fallback)
            elif method == "fallback_bm25":
                fallback = route["confidence"] < threshold_bm25
                ranking = (
                    route["question"]
                    if not fallback
                    else _rrf(
                        [route["chunk"], _bm25(dataset["chunks"], query["query"])]
                    )
                )
                fallback_count += int(fallback)
            else:
                ranking = route[method]
            row = _metric_row(
                query,
                dataset["gold"][query["query_id"]],
                ranking,
                dataset["chunk_map"],
            )
            rows.append(row)
            if experiment == "Experiment 2":
                (fallback_route_rows if fallback else question_route_rows).append(row)
                if fallback:
                    weak_fallback_rows.append(row)
                    weak_question_rows.append(
                        _metric_row(
                            query,
                            dataset["gold"][query["query_id"]],
                            route["question"],
                            dataset["chunk_map"],
                        )
                    )
        rankings[condition] = rows
        route_analysis = None
        if experiment == "Experiment 2":
            weak_question = _evaluate(weak_question_rows)
            weak_fallback = _evaluate(weak_fallback_rows)
            route_analysis = {
                "question_route_queries": len(question_route_rows),
                "fallback_route_queries": len(fallback_route_rows),
                "question_route_metrics": (
                    _evaluate(question_route_rows) if question_route_rows else {}
                ),
                "fallback_route_metrics": (
                    _evaluate(fallback_route_rows) if fallback_route_rows else {}
                ),
                "weak_question_without_fallback": weak_question,
                "weak_question_with_fallback": weak_fallback,
                "fallback_delta_evidence_recall@5": (
                    weak_fallback["evidence_recall@5"]
                    - weak_question["evidence_recall@5"]
                ),
            }
        output_rows.append(
            {
                "experiment": experiment,
                "condition": condition,
                "config": config_name,
                "retrieval_method": method,
                "test_queries": len(test),
                "stored_vectors": (
                    len(dataset["chunks"]) + len(dataset["questions"])
                    if experiment == "Experiment 1"
                    else len(dataset["questions"])
                    if experiment == "Experiment 3"
                    else len(dataset["chunks"]) + len(dataset["questions"])
                ),
                "fallback_count": fallback_count
                if experiment == "Experiment 2"
                else None,
                "fallback_rate": (
                    fallback_count / len(test) if experiment == "Experiment 2" else None
                ),
                "route_analysis": route_analysis,
                "metrics": _evaluate(rows),
            }
        )
    payload = {
        "protocol": {
            "embedding": (
                "iris:https://llm-api-dev.iris.ai/embeddings/generate/:dim384"
            ),
            "generator": C.gen_model(),
            "temperature": C.GEN_TEMPERATURE,
            "seed": C.SEED,
            "test_queries": 17,
            "development_queries": 17,
            "threshold_selected_on": "development only",
            "e2_threshold": threshold,
            "e2_thresholds_by_prompt": thresholds,
            "e2_bm25_threshold": threshold_bm25,
            "score_fusion": "0.5 chunk cosine + 0.5 best question cosine",
            "rrf_k": RRF_K,
            "rank_depth": DEPTH,
        },
        "conditions": output_rows,
    }
    RANKINGS.write_text(json.dumps(rankings))
    METRICS.write_text(json.dumps(payload, indent=2))
    MANIFEST.write_text(
        json.dumps(
            {
                name: {
                    "chunks": len(dataset["chunks"]),
                    "questions": len(dataset["questions"]),
                    "chunk_collection": f"mhrag_ctl_{name}_chunks",
                    "question_collection": f"mhrag_ctl_{name}_questions",
                }
                for name, dataset in configs.items()
            },
            indent=2,
        )
    )
    return payload


def _select_threshold_bm25(
    dev_queries: list[dict], routes: dict[str, dict], dataset: dict
) -> float:
    candidates = sorted(
        {round(routes[query["query_id"]]["confidence"], 6) for query in dev_queries}
    )
    best = None
    for threshold in candidates:
        rows = []
        for query in dev_queries:
            route = routes[query["query_id"]]
            ranking = route["question"]
            if route["confidence"] < threshold:
                ranking = _rrf(
                    [route["chunk"], _bm25(dataset["chunks"], query["query"])]
                )
            rows.append(
                _metric_row(
                    query,
                    dataset["gold"][query["query_id"]],
                    ranking,
                    dataset["chunk_map"],
                )
            )
        score = np.mean([VM.per_query(row)["evidence_recall@5"] for row in rows])
        candidate = (score, -threshold, threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[2])


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
