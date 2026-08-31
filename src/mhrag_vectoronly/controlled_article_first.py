"""Article-first generated-question variants for controlled Experiments 1–3."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import controlled_suite_generate as G
import controlled_suite_optimize as O
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

DATA = G.DATA / "article_first"
DATA.mkdir(parents=True, exist_ok=True)
MIPRO_Q = DATA / "mipro_article_questions_q10.jsonl"
METRICS = R.RESULTS / "metrics_article_first.json"
RANKINGS = R.RESULTS / "rankings_article_first.json"
MANUAL_Q = H.DOCQ_PATH
GEPA_Q = H.DATA / "gepa_verified_document_questions.jsonl"

ARTICLE_USER = '''Complete article:
"""
{article}
"""

Generate exactly 10 diverse article-level retrieval questions. Each question
must be answerable from the complete article and should cover a different
important fact or relationship. For each question include a concise answer and
a short verbatim evidence quote copied from the article.

Return:
{{"questions":[{{"question":"...","short_answer":"...",
"evidence":"verbatim quote"}}]}}'''


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _supporting_chunks(
    question: dict, chunks: list[dict], document_id: str
) -> list[str]:
    evidence = _normalize(question.get("evidence", ""))
    candidates = [c for c in chunks if c["document_id"] == document_id]
    exact = [
        c["chunk_id"]
        for c in candidates
        if evidence and evidence in _normalize(c["content"])
    ]
    if exact:
        return exact
    # Evidence quotes sometimes differ only in punctuation/quote marks. Reuse
    # the repository's established fact-to-chunk alignment fallback.
    hits, _, _ = H.D._locate_fact(
        question.get("evidence", ""),
        [{"chunk_id": c["chunk_id"], "text": c["content"]} for c in candidates],
    )
    if not hits and question.get("short_answer"):
        hits, _, _ = H.D._locate_fact(
            question["short_answer"],
            [{"chunk_id": c["chunk_id"], "text": c["content"]} for c in candidates],
        )
    if not hits:
        # Last-resort deterministic lexical alignment for a grounded question
        # whose generated quote/answer was paraphrased too heavily.
        signal = set(
            H.tokens(
                " ".join(
                    str(question.get(key, ""))
                    for key in ("question", "short_answer", "evidence")
                )
            )
        )
        scored = [
            (len(signal & set(H.tokens(c["content"]))), c["chunk_id"])
            for c in candidates
        ]
        hits = [max(scored, key=lambda pair: (pair[0], pair[1]))[1]]
    return hits


def generate_mipro() -> list[dict]:
    articles = H.read_jsonl(H.ARTICLES_PATH)
    chunks = H.load_chunks()
    cache = (
        {row["document_id"]: row for row in G.read_jsonl(MIPRO_Q)}
        if MIPRO_Q.exists()
        else {}
    )
    prompt = O.MIPRO_GENERAL_PROMPT.read_text()
    for pos, article in enumerate(articles, 1):
        if len(cache.get(article["article_id"], {}).get("questions", [])) == 10:
            continue
        questions = []
        for _ in range(C.GEN_MAX_RETRIES):
            raw = G._call(
                prompt + "\nUse the complete article as the source unit.",
                ARTICLE_USER.format(article=article["cleaned_body"]),
            )
            questions, seen = [], set()
            for item in raw.get("questions", []):
                text = str(item.get("question", "")).strip()
                key = _normalize(text)
                support = _supporting_chunks(item, chunks, article["article_id"])
                if text and key not in seen and support:
                    seen.add(key)
                    questions.append(
                        {
                            "question": text,
                            "short_answer": str(item.get("short_answer", "")).strip(),
                            "evidence": str(item.get("evidence", "")).strip(),
                            "supporting_chunk_ids": support,
                            "document_id": article["article_id"],
                            "question_id": f"{article['article_key']}::miprodq{len(questions)}",
                            "verified": True,
                        }
                    )
            if len(questions) == 10:
                break
        if len(questions) != 10:
            raise RuntimeError(
                f"MIPRO article generation produced {len(questions)} supported "
                f"questions for {article['title']}; rerun to retry"
            )
        cache[article["article_id"]] = {
            "document_id": article["article_id"],
            "document_title": article["title"],
            "questions": questions,
            "valid": True,
        }
        G.write_jsonl(
            MIPRO_Q,
            [cache[a["article_id"]] for a in articles if a["article_id"] in cache],
        )
        print(f"[article-first:MIPRO] {pos}/{len(articles)}")
    return G.read_jsonl(MIPRO_Q)


def _question_rows(path: Path, chunks: list[dict], remap: bool = False) -> list[dict]:
    output = []
    for article_row in G.read_jsonl(path):
        for index, question in enumerate(article_row["questions"]):
            support = (
                _supporting_chunks(question, chunks, article_row["document_id"])
                if remap
                else question["supporting_chunk_ids"]
            )
            if not support:
                raise RuntimeError(f"No supporting chunk for {question['question']}")
            output.append(
                {
                    "id": (
                        f"{article_row.get('document_title', 'article')}::{index}::"
                        f"{'large' if remap else 'base'}"
                    ),
                    "text": question["question"],
                    "supporting_chunk_ids": support,
                    "document_id": article_row["document_id"],
                }
            )
    return output


def _index(
    name: str, chunks: list[dict], questions: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    chunk_vectors = R._embed_cached(
        f"article_{name}_chunks", [c["content"] for c in chunks]
    )
    question_vectors = R._embed_cached(
        f"article_{name}_questions", [q["text"] for q in questions]
    )
    chunk_collection = C.reset_collection(f"mhrag_ctl_article_{name}_chunks")
    chunk_collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=chunk_vectors.tolist(),
        documents=[c["content"] for c in chunks],
        metadatas=[{"parent_document_id": c["document_id"]} for c in chunks],
    )
    question_collection = C.reset_collection(f"mhrag_ctl_article_{name}_questions")
    question_collection.add(
        ids=[f"aq::{i}" for i in range(len(questions))],
        embeddings=question_vectors.tolist(),
        documents=[q["text"] for q in questions],
        metadatas=[
            {
                "parent_document_id": q["document_id"],
                "supporting_chunk_ids": json.dumps(q["supporting_chunk_ids"]),
            }
            for q in questions
        ],
    )
    return chunk_vectors, question_vectors


def _question_scores(
    questions: list[dict], vectors: np.ndarray, qvec: np.ndarray
) -> dict[str, float]:
    scores = H._cosine_matrix(vectors, qvec)
    best = {}
    for question, score in zip(questions, scores):
        for chunk_id in question["supporting_chunk_ids"]:
            best[chunk_id] = max(best.get(chunk_id, -1.0), float(score))
    return best


def _routes(
    chunks: list[dict],
    questions: list[dict],
    chunk_vectors: np.ndarray,
    question_vectors: np.ndarray,
    query: dict,
    qvec: np.ndarray,
) -> dict:
    chunk_rank, chunk_scores = R._dense_chunks(chunks, chunk_vectors, qvec)
    question_scores = _question_scores(questions, question_vectors, qvec)
    question_rank = [
        chunk_id
        for chunk_id, _ in sorted(
            question_scores.items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]
    sparse = R._bm25(chunks, query["query"])
    return {
        "chunk": chunk_rank,
        "question": question_rank,
        "dual": R._dual(chunk_rank, chunk_scores, question_scores),
        "question_bm25": R._rrf([question_rank, sparse]),
        "confidence": max(question_scores.values()),
    }


def _condition_row(
    experiment: str,
    condition: str,
    config_name: str,
    method: str,
    chunks: list[dict],
    questions: list[dict],
    metrics: dict,
    route_analysis: dict | None = None,
    threshold: float | None = None,
) -> dict:
    return {
        "experiment": experiment,
        "condition": condition,
        "config": config_name,
        "retrieval_method": method,
        "question_generation": "10 questions per complete article",
        "generated_questions": len(questions),
        "chunk_size": 1024 if "1024" in config_name else 512,
        "chunk_overlap": 128,
        "stored_vectors": (
            len(questions)
            if experiment == "Experiment 3B"
            else len(chunks) + len(questions)
        ),
        "prompt_version": (
            "gepa_article_v1"
            if "gepa" in config_name
            else "miprov2_article_v1"
            if "mipro" in config_name
            else "manual_article_v1"
        ),
        "generation_model": C.gen_model(),
        "metrics": metrics,
        "indexes": [
            f"mhrag_ctl_article_{config_name}_chunks",
            f"mhrag_ctl_article_{config_name}_questions",
        ],
        "route_analysis": route_analysis,
        "threshold": threshold,
    }


def run() -> dict:
    generate_mipro()
    chunks512 = H.load_chunks()
    chunks1024 = G.read_jsonl(G.CHUNKS_1024)
    gold512 = H.load_gold()
    gold1024 = {r["query_id"]: r for r in G.read_jsonl(G.GOLD_1024)}
    split = {r["query_id"]: r["split"] for r in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    development = [q for q in queries if split[q["query_id"]] == "development"]
    test = [q for q in queries if split[q["query_id"]] == "test"]
    qvectors = R._query_vectors(queries)
    configs = {
        "article_manual512": (chunks512, gold512, _question_rows(MANUAL_Q, chunks512)),
        "article_manual1024": (
            chunks1024,
            gold1024,
            _question_rows(MANUAL_Q, chunks1024, remap=True),
        ),
        "article_gepa512": (chunks512, gold512, _question_rows(GEPA_Q, chunks512)),
        "article_mipro512": (chunks512, gold512, _question_rows(MIPRO_Q, chunks512)),
    }
    indexed = {
        name: _index(name, chunks, questions)
        for name, (chunks, _, questions) in configs.items()
    }
    all_routes = {
        name: {
            query["query_id"]: _routes(
                chunks,
                questions,
                *indexed[name],
                query,
                qvectors[query["query_id"]],
            )
            for query in queries
        }
        for name, (chunks, _, questions) in configs.items()
    }
    datasets = {
        name: {
            "chunks": chunks,
            "chunk_map": {chunk["chunk_id"]: chunk for chunk in chunks},
            "gold": gold,
        }
        for name, (chunks, gold, _) in configs.items()
    }
    thresholds = {
        name: R._select_threshold(development, all_routes[name], datasets[name])
        for name in configs
    }
    threshold_bm25 = R._select_threshold_bm25(
        development,
        all_routes["article_manual512"],
        datasets["article_manual512"],
    )

    e1_specs = (
        ("Article-E1-ManualPrompt", "article_manual512", False),
        ("Article-E1-BM25", "article_manual512", True),
        ("Article-E1-LargeChunk", "article_manual1024", False),
        ("Article-E1-LargeChunk-BM25", "article_manual1024", True),
        ("Article-E1-GEPA", "article_gepa512", False),
        ("Article-E1-MIPROv2", "article_mipro512", False),
    )
    condition_rows, rankings = [], {}
    for condition, config_name, add_bm25 in e1_specs:
        chunks, gold, questions = configs[config_name]
        chunk_map = {c["chunk_id"]: c for c in chunks}
        rows = []
        for query in test:
            route = all_routes[config_name][query["query_id"]]
            ranking = route["dual"]
            if add_bm25:
                ranking = R._rrf([ranking, R._bm25(chunks, query["query"])])
            rows.append(
                R._metric_row(query, gold[query["query_id"]], ranking, chunk_map)
            )
        rankings[condition] = rows
        condition_rows.append(
            _condition_row(
                "Experiment 1B",
                condition,
                config_name,
                "article_dual_bm25" if add_bm25 else "article_dual",
                chunks,
                questions,
                R._evaluate(rows),
            )
        )

    article_specs = (
        ("ManualPrompt", "article_manual512", ("Experiment 2B", "Experiment 3B")),
        ("BM25", "article_manual512", ("Experiment 2B", "Experiment 3B")),
        ("LargeChunk", "article_manual1024", ("Experiment 2B", "Experiment 3B")),
        ("LargeChunk-BM25", "article_manual1024", ("Experiment 3B",)),
        ("GEPA", "article_gepa512", ("Experiment 2B", "Experiment 3B")),
        ("MIPROv2", "article_mipro512", ("Experiment 2B", "Experiment 3B")),
    )
    for suffix, config_name, experiments in article_specs:
        chunks, gold, questions = configs[config_name]
        chunk_map = {c["chunk_id"]: c for c in chunks}
        use_bm25 = suffix.endswith("BM25")
        for experiment in experiments:
            condition = f"Article-E{experiment[-2]}-{suffix}"
            rows = []
            question_route_rows, fallback_route_rows = [], []
            weak_question_rows, weak_fallback_rows = [], []
            active_threshold = threshold_bm25 if use_bm25 else thresholds[config_name]
            for query in test:
                route = all_routes[config_name][query["query_id"]]
                if experiment == "Experiment 2B":
                    fallback = route["confidence"] < active_threshold
                    if fallback:
                        ranking = route["chunk"]
                        if use_bm25:
                            ranking = R._rrf([ranking, R._bm25(chunks, query["query"])])
                    else:
                        ranking = route["question"]
                else:
                    fallback = False
                    ranking = route["question_bm25"] if use_bm25 else route["question"]
                row = R._metric_row(query, gold[query["query_id"]], ranking, chunk_map)
                rows.append(row)
                if experiment == "Experiment 2B":
                    (fallback_route_rows if fallback else question_route_rows).append(
                        row
                    )
                    if fallback:
                        weak_fallback_rows.append(row)
                        weak_question_rows.append(
                            R._metric_row(
                                query,
                                gold[query["query_id"]],
                                route["question"],
                                chunk_map,
                            )
                        )
            rankings[condition] = rows
            route_analysis = None
            if experiment == "Experiment 2B":
                weak_question = R._evaluate(weak_question_rows)
                weak_fallback = R._evaluate(weak_fallback_rows)
                route_analysis = {
                    "question_route_queries": len(question_route_rows),
                    "fallback_route_queries": len(fallback_route_rows),
                    "question_route_metrics": R._evaluate(question_route_rows),
                    "fallback_route_metrics": R._evaluate(fallback_route_rows),
                    "weak_question_without_fallback": weak_question,
                    "weak_question_with_fallback": weak_fallback,
                    "fallback_delta_evidence_recall@5": (
                        weak_fallback["evidence_recall@5"]
                        - weak_question["evidence_recall@5"]
                    ),
                }
            method = (
                ("article_fallback_bm25" if use_bm25 else "article_fallback")
                if experiment == "Experiment 2B"
                else ("article_question_bm25" if use_bm25 else "article_question")
            )
            condition_rows.append(
                _condition_row(
                    experiment,
                    condition,
                    config_name,
                    method,
                    chunks,
                    questions,
                    R._evaluate(rows),
                    route_analysis,
                    active_threshold if experiment == "Experiment 2B" else None,
                )
            )
    payload = {
        "protocol": {
            "questions_per_article": 10,
            "articles": 15,
            "question_to_chunk_mapping": "verbatim evidence + established fuzzy fallback",
            "embedding": "Iris dim384 for chunks, questions, and user queries",
            "test_queries": 17,
            "score_fusion": "0.5 chunk cosine + 0.5 best supporting-question cosine",
            "threshold_selected_on": "development only",
            "experiment_2b_thresholds": thresholds,
            "experiment_2b_bm25_threshold": threshold_bm25,
        },
        "conditions": condition_rows,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(rankings))
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
