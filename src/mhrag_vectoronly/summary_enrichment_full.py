"""Add adaptive chunk-question + whole-article summary enrichment at full scale."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rank_bm25 import BM25Okapi

import adaptive_questions_100 as A
import article_chunk_questions_full as X


DATA = X.DATA
RESULTS = X.RESULTS
SUMMARY_GENERATIONS = DATA / "article_summaries.jsonl"
SUMMARY_VECTORS = RESULTS / "article_summary_vectors_iris.json"
SUMMARY_RANKINGS = RESULTS / "summary_enrichment_rankings.jsonl"
CONDITION = X.SUMMARY_CONDITION
MAX_WORKERS = 24

SUMMARY_PROMPT = """Summarize the supplied article in 120-180 words for semantic
retrieval. Preserve the central topic, named entities, dates, numbers, causal
relationships, comparisons, and distinctive facts that could help match a
specific information-seeking question. Do not add unsupported information.
Return valid JSON only: {"summary":"..."}"""


def article_sources(chunks: list[dict]) -> list[dict]:
    raw = json.loads(
        (X.ROOT / "data" / "raw" / "multihoprag" / "corpus.json").read_text()
    )
    raw_by_url = {row["url"]: row for row in raw}
    chunks_by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk)
    return [
        {
            "article_id": document_id,
            "title": raw_by_url[document_id].get("title", rows[0]["document_title"]),
            "content": raw_by_url[document_id].get("body", ""),
            "chunk_ids": [
                row["chunk_id"]
                for row in sorted(rows, key=lambda item: item["chunk_position"])
            ],
        }
        for document_id, rows in sorted(chunks_by_document.items())
    ]


def generate_one(article: dict) -> dict:
    user = f'''Article title: {article["title"]}\n\nSource article:\n"""\n{article["content"]}\n"""'''
    summary = ""
    for attempt in range(6):
        summary = str(
            X.call_json_resilient(SUMMARY_PROMPT, user, A.SEED + attempt).get(
                "summary", ""
            )
        ).strip()
        if 120 <= len(summary.split()) <= 180:
            break
    if len(summary.split()) > 180:
        summary = " ".join(summary.split()[:180]).rstrip(" ,;:") + "."
    if not 120 <= len(summary.split()) <= 180:
        raise RuntimeError(f"Summary outside 120-180 words for {article['article_id']}")
    return {
        "article_id": article["article_id"],
        "title": article["title"],
        "chunk_ids": article["chunk_ids"],
        "summary": summary,
    }


def generate_summaries(articles: list[dict]) -> list[dict]:
    cached = (
        {
            row["article_id"]: row
            for row in X.read_jsonl(SUMMARY_GENERATIONS)
            if 120 <= len(row.get("summary", "").split()) <= 180
        }
        if SUMMARY_GENERATIONS.exists()
        else {}
    )
    todo = [article for article in articles if article["article_id"] not in cached]
    print(f"[summary-generation] cached={len(cached)} todo={len(todo)}", flush=True)
    failures = []
    with SUMMARY_GENERATIONS.open("a") as checkpoint:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_one, article): article for article in todo
            }
            for position, future in enumerate(as_completed(futures), 1):
                article = futures[future]
                try:
                    row = future.result()
                    cached[article["article_id"]] = row
                    checkpoint.write(json.dumps(row) + "\n")
                    checkpoint.flush()
                    print(
                        f"[summary-generation] {position}/{len(todo)} "
                        f"{article['title']}",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "article_id": article["article_id"],
                            "title": article["title"],
                            "error": str(error),
                        }
                    )
                    print(
                        f"[summary-generation:error] {article['title']}: {error}",
                        flush=True,
                    )
    if failures:
        (DATA / "article_summary_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError(f"{len(failures)} articles need a cached retry")
    failure_path = DATA / "article_summary_failures.json"
    if failure_path.exists():
        failure_path.unlink()
    ordered = [cached[article["article_id"]] for article in articles]
    A.write_jsonl(SUMMARY_GENERATIONS, ordered)
    return ordered


def evaluate(
    chunks: list[dict],
    queries: list[dict],
    chunk_generated: list[dict],
    summaries: list[dict],
) -> tuple[dict, list[dict]]:
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vectors = A.embed_resumable(
        RESULTS / "chunk_vectors_iris.json", [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        RESULTS / "query_vectors_iris.json", [row["query"] for row in queries]
    )
    chunk_questions = [
        (question["question"], row["chunk_id"])
        for row in chunk_generated
        for question in row["bounded_questions"]
    ]
    chunk_question_vectors = A.embed_resumable(
        RESULTS / "bounded_question_vectors_iris.json",
        [question for question, _ in chunk_questions],
    )
    chunk_question_indices: dict[str, list[int]] = defaultdict(list)
    for index, (_, chunk_id) in enumerate(chunk_questions):
        chunk_question_indices[chunk_id].append(index)

    summary_vectors = A.embed_resumable(
        SUMMARY_VECTORS, [row["summary"] for row in summaries]
    )
    summary_index = {row["article_id"]: index for index, row in enumerate(summaries)}
    bm25 = BM25Okapi([A.tokenize(row["content"]) for row in chunks])
    metric_rows, rankings = [], []
    for qi, query in enumerate(queries):
        chunk_scores = X.cosine(chunk_vectors, query_vectors[qi])
        question_scores = X.cosine(chunk_question_vectors, query_vectors[qi])
        summary_scores = X.cosine(summary_vectors, query_vectors[qi])
        fused = []
        for index, chunk in enumerate(chunks):
            question_score = (
                max(question_scores[chunk_question_indices[chunk["chunk_id"]]])
                if chunk_question_indices[chunk["chunk_id"]]
                else chunk_scores[index]
            )
            summary_score = summary_scores[summary_index[chunk["document_id"]]]
            fused.append((chunk_scores[index] + question_score + summary_score) / 3)
        dense_order = np.argsort(-np.asarray(fused))
        sparse_order = np.argsort(-bm25.get_scores(A.tokenize(query["query"])))
        rrf = defaultdict(float)
        for order in (dense_order, sparse_order):
            for rank, item in enumerate(order, 1):
                rrf[chunk_ids[item]] += 1 / (A.RRF_K + rank)
        ranking = [
            chunk_id
            for chunk_id, _ in sorted(rrf.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        metrics = A.metric_row(query, ranking)
        metric_rows.append(metrics)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": metrics,
            }
        )
        if (qi + 1) % 50 == 0:
            print(f"[retrieve] {qi + 1}/{len(queries)}", flush=True)
    keys = list(metric_rows[0])
    chunk_question_count = len(chunk_questions)
    condition = {
        "condition": CONDITION,
        "question_rule": "adaptive bounded chunk questions + one 120-180-word article summary",
        "quality": {
            "generated_questions": chunk_question_count,
            "chunk_questions": chunk_question_count,
            "article_summaries": len(summaries),
            "questions_per_chunk_min": min(
                len(row["bounded_questions"]) for row in chunk_generated
            ),
            "questions_per_chunk_max": max(
                len(row["bounded_questions"]) for row in chunk_generated
            ),
        },
        "stored_vectors": len(chunks) + chunk_question_count + len(summaries),
        "metrics": {
            key: float(np.mean([row[key] for row in metric_rows])) for key in keys
        },
    }
    A.write_jsonl(SUMMARY_RANKINGS, rankings)
    return condition, rankings


def run() -> dict:
    chunks = X.read_jsonl(DATA / "chunks.jsonl")
    queries = X.read_jsonl(DATA / "queries.jsonl")
    generated_by_chunk = {
        row["chunk_id"]: row
        for row in X.read_jsonl(DATA / "adaptive_generations.jsonl")
    }
    chunk_generated = [
        generated_by_chunk.get(
            chunk["chunk_id"],
            {
                "chunk_id": chunk["chunk_id"],
                "bounded_questions": [],
                "unbounded_questions": [],
                "facts": [],
                "generation_skipped": True,
            },
        )
        for chunk in chunks
    ]
    articles = article_sources(chunks)
    summaries = generate_summaries(articles)
    condition, rankings = evaluate(chunks, queries, chunk_generated, summaries)

    payload = json.loads((RESULTS / "metrics.json").read_text())
    payload["conditions"] = [
        row for row in payload["conditions"] if row["condition"] != CONDITION
    ] + [condition]
    payload["protocol"]["article_summary_generation"] = {
        "articles": len(articles),
        "summaries": len(summaries),
        "summary_length": "120-180 words",
        "dense_fusion": "equal 1/3 chunk + chunk-question + article-summary",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    ranking_payload = json.loads((RESULTS / "rankings.json").read_text())
    ranking_payload[CONDITION] = rankings
    (RESULTS / "rankings.json").write_text(json.dumps(ranking_payload))
    X.render(payload)
    print(json.dumps(condition, indent=2))
    return payload


if __name__ == "__main__":
    run()
