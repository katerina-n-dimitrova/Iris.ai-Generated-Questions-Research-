#!/usr/bin/env python3
"""Add the unbounded chunk + unbounded whole-article fifth retrieval arm."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rank_bm25 import BM25Okapi

import adaptive_questions_100 as A
import article_chunk_questions_full as C

DATA = C.DATA
RESULTS = C.RESULTS
ARTICLE_GENERATIONS = DATA / "article_question_generations_unbounded.jsonl"
ARTICLE_VECTORS = RESULTS / "article_question_vectors_unbounded_iris.json"
ARTICLE_RANKINGS = RESULTS / "article_chunk_question_unbounded_rankings.jsonl"
CONDITION = C.UNBOUNDED_CONDITION
MAX_WORKERS = 8


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def generate_one(article: dict) -> dict:
    facts = article["facts"]
    if not facts:
        user = f'''Source article:\n"""\n{article["content"]}\n"""'''
        for _ in range(A.MAX_RETRIES):
            facts = A.dedup_facts(A.call_json(A.FACT_PROMPT, user).get("facts", []))
            if facts:
                break
        if not facts:
            raise RuntimeError(f"No article facts for {article['article_id']}")
    budget = round(len(facts) * 0.5)
    proxy = {"chunk_id": article["article_id"], "content": article["content"]}
    questions = A.generate_questions(proxy, facts, budget)
    for question in questions:
        question["supporting_article_id"] = article["article_id"]
        question["supporting_chunk_ids"] = article["chunk_ids"]
    return {
        "article_id": article["article_id"],
        "title": article["title"],
        "chunk_ids": article["chunk_ids"],
        "n_distinct_article_facts": len(facts),
        "question_budget": budget,
        "questions": questions,
    }


def generate_articles(articles: list[dict]) -> list[dict]:
    cached = (
        {
            row["article_id"]: row
            for row in read_jsonl(ARTICLE_GENERATIONS)
            if row.get("questions") and row.get("question_budget", 0) > 0
        }
        if ARTICLE_GENERATIONS.exists()
        else {}
    )
    todo = [article for article in articles if article["article_id"] not in cached]
    print(
        f"[unbounded-article-generation] cached={len(cached)} todo={len(todo)}",
        flush=True,
    )
    failures = []
    with ARTICLE_GENERATIONS.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_one, article): article for article in todo
            }
            for position, future in enumerate(as_completed(futures), 1):
                article = futures[future]
                try:
                    row = future.result()
                    cached[article["article_id"]] = row
                    checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    print(
                        f"[unbounded-article-generation] {position}/{len(todo)} "
                        f"{article['article_id']} questions={len(row['questions'])}",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(
                        {"article_id": article["article_id"], "error": str(error)}
                    )
                    print(
                        f"[unbounded-article-generation:error] {article['article_id']}: {error}",
                        flush=True,
                    )
    if failures:
        (DATA / "article_question_unbounded_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} unbounded articles need a cached retry")
    ordered = [cached[article["article_id"]] for article in articles]
    A.write_jsonl(ARTICLE_GENERATIONS, ordered)
    return ordered


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def evaluate(chunks, queries, generated, articles):
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vectors = A.embed_resumable(
        RESULTS / "chunk_vectors_iris.json", [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        RESULTS / "query_vectors_iris.json", [row["query"] for row in queries]
    )
    chunk_questions = [
        (q["question"], row["chunk_id"])
        for row in generated
        for q in row["unbounded_questions"]
    ]
    article_questions = [
        (q["question"], row["article_id"]) for row in articles for q in row["questions"]
    ]
    chunk_q_vectors = A.embed_resumable(
        RESULTS / "unbounded_question_vectors_iris.json",
        [x[0] for x in chunk_questions],
    )
    article_q_vectors = A.embed_resumable(
        ARTICLE_VECTORS, [x[0] for x in article_questions]
    )
    qv, cv, cqv, aqv = map(
        unit, (query_vectors, chunk_vectors, chunk_q_vectors, article_q_vectors)
    )
    chunk_score = np.einsum("ik,jk->ij", qv, cv, optimize=True)
    chunk_question_score = np.einsum("ik,jk->ij", qv, cqv, optimize=True)
    article_question_score = np.einsum("ik,jk->ij", qv, aqv, optimize=True)
    by_chunk, by_article = defaultdict(list), defaultdict(list)
    for i, (_, owner) in enumerate(chunk_questions):
        by_chunk[owner].append(i)
    for i, (_, owner) in enumerate(article_questions):
        by_article[owner].append(i)
    chunk_q_max = np.column_stack(
        [
            chunk_question_score[:, by_chunk[cid]].max(axis=1)
            if by_chunk[cid]
            else chunk_score[:, i]
            for i, cid in enumerate(chunk_ids)
        ]
    )
    article_q_max = np.column_stack(
        [
            article_question_score[:, by_article[row["document_id"]]].max(axis=1)
            for row in chunks
        ]
    )
    dense_score = (chunk_score + chunk_q_max + article_q_max) / 3
    bm25 = BM25Okapi([A.tokenize(row["content"]) for row in chunks])
    metric_rows, rankings = [], []
    for qi, query in enumerate(queries):
        dense = np.argsort(-dense_score[qi])
        sparse = np.argsort(-bm25.get_scores(A.tokenize(query["query"])))
        rrf = defaultdict(float)
        for order in (dense, sparse):
            for rank, index in enumerate(order, 1):
                rrf[chunk_ids[index]] += 1 / (A.RRF_K + rank)
        ranking = [item for item, _ in sorted(rrf.items(), key=lambda x: (-x[1], x[0]))]
        metrics = A.metric_row(query, ranking)
        metric_rows.append(metrics)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": metrics,
            }
        )
        if (qi + 1) % 250 == 0:
            print(f"[retrieve:{CONDITION}] {qi + 1}/{len(queries)}", flush=True)
    A.write_jsonl(ARTICLE_RANKINGS, rankings)
    chunk_counts = [len(row["unbounded_questions"]) for row in generated]
    article_counts = [len(row["questions"]) for row in articles]
    keys = list(metric_rows[0])
    return {
        "condition": CONDITION,
        "question_rule": "chunk and article: round(facts * 0.5), no bounds",
        "quality": {
            "generated_questions": sum(chunk_counts) + sum(article_counts),
            "chunk_questions": sum(chunk_counts),
            "article_questions": sum(article_counts),
            "questions_per_chunk_min": min(chunk_counts),
            "questions_per_chunk_max": max(chunk_counts),
            "questions_per_article_min": min(article_counts),
            "questions_per_article_max": max(article_counts),
        },
        "stored_vectors": len(chunks) + sum(chunk_counts) + sum(article_counts),
        "metrics": {
            key: float(np.mean([row[key] for row in metric_rows])) for key in keys
        },
    }


def run():
    chunks = read_jsonl(DATA / "chunks.jsonl")
    queries = read_jsonl(DATA / "queries.jsonl")
    generated_raw = read_jsonl(DATA / "adaptive_generations.jsonl")
    by_id = {row["chunk_id"]: row for row in generated_raw}
    generated = [
        by_id.get(
            chunk["chunk_id"],
            {
                "chunk_id": chunk["chunk_id"],
                "facts": [],
                "bounded_questions": [],
                "unbounded_questions": [],
            },
        )
        for chunk in chunks
    ]
    article_inputs = C.article_inputs(chunks, generated)
    article_generated = generate_articles(article_inputs)
    condition = evaluate(chunks, queries, generated, article_generated)
    payload = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    payload["conditions"] = [
        row for row in payload["conditions"] if row["condition"] != CONDITION
    ] + [condition]
    payload["protocol"]["unbounded_article_question_generation"] = {
        "articles": len(article_inputs),
        "rule": "round(article facts * 0.5), no bounds",
        "dense_fusion": "equal 1/3 chunk + chunk-question + article-question",
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_payload = dict(payload)
    report_payload["conditions"] = [
        row for row in payload["conditions"] if row["condition"] != C.SUMMARY_CONDITION
    ]
    C.render(report_payload)
    print(json.dumps(condition, ensure_ascii=False, indent=2))
    return condition


if __name__ == "__main__":
    run()
