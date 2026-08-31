"""Baseline and adaptive-question retrieval study on 300 MuSiQue documents."""

from __future__ import annotations

import hashlib
import json
import random

import adaptive_questions_100 as A
import baseline_100 as B


A.ARTICLE_COUNT = 300
A.MAX_WORKERS = 12
A.MAX_RETRIES = 8
A.DATA = A.ROOT / "data" / "processed" / "musique_adaptive_questions_300"
A.RESULTS = A.ROOT / "results" / "musique_adaptive_questions_300"
A.REPORT = A.ROOT / "report" / "musique_adaptive_questions_300.html"
for directory in (A.DATA, A.RESULTS, A.REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

A.CHUNKS_PATH = A.DATA / "chunks.jsonl"
A.QUERIES_PATH = A.DATA / "queries.jsonl"
A.SUMMARY_PATH = A.DATA / "summary.json"
A.GEN_PATH = A.DATA / "adaptive_generations.jsonl"
A.CHUNK_VECTORS = A.RESULTS / "chunk_vectors_iris.json"
A.QUERY_VECTORS = A.RESULTS / "query_vectors_iris.json"
A.BOUNDED_VECTORS = A.RESULTS / "bounded_question_vectors_iris.json"
A.UNBOUNDED_VECTORS = A.RESULTS / "unbounded_question_vectors_iris.json"
A.METRICS = A.RESULTS / "metrics.json"
A.RANKINGS = A.RESULTS / "rankings.json"
B.BASELINE_METRICS = A.RESULTS / "baseline_metrics.json"
B.BASELINE_RANKINGS = A.RESULTS / "baseline_rankings.jsonl"

RAW = A.ROOT / "data" / "raw" / "musique" / "musique_ans_v1.0_dev.jsonl"


def document_id(paragraph: dict) -> str:
    payload = f"{paragraph.get('title', '')}\n{paragraph['paragraph_text']}"
    return "musique://" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def prepare_musique() -> tuple[list[dict], list[dict], dict]:
    if A.CHUNKS_PATH.exists() and A.QUERIES_PATH.exists() and A.SUMMARY_PATH.exists():
        return (
            A.read_jsonl(A.CHUNKS_PATH),
            A.read_jsonl(A.QUERIES_PATH),
            json.loads(A.SUMMARY_PATH.read_text()),
        )
    rows = A.read_jsonl(RAW)
    answerable = [row for row in rows if row.get("answerable")]
    prepared = []
    document_map = {}
    for row in answerable:
        by_index = {paragraph["idx"]: paragraph for paragraph in row["paragraphs"]}
        support_indices = [
            item["paragraph_support_idx"] for item in row["question_decomposition"]
        ]
        support_documents = []
        for index in support_indices:
            paragraph = by_index[index]
            doc_id = document_id(paragraph)
            document_map[doc_id] = paragraph
            support_documents.append(doc_id)
        prepared.append((row, support_documents))

    shuffled = prepared[:]
    random.Random(A.SEED).shuffle(shuffled)
    selected = set()
    for _, required_list in shuffled:
        required = set(required_list)
        if len(selected | required) <= A.ARTICLE_COUNT:
            selected |= required
        if len(selected) >= A.ARTICLE_COUNT:
            break
    ordered_ids = sorted(selected)
    chunks = []
    chunks_by_document = {}
    for article_index, doc_id in enumerate(ordered_ids):
        paragraph = document_map[doc_id]
        article = {
            "url": doc_id,
            "title": paragraph.get("title", ""),
            "body": paragraph["paragraph_text"],
        }
        article_chunks = A.chunk_article(article, article_index)
        chunks.extend(article_chunks)
        chunks_by_document[doc_id] = [chunk["chunk_id"] for chunk in article_chunks]

    queries = []
    for row, support_documents in prepared:
        if not set(support_documents) <= selected:
            continue
        units = [chunks_by_document[doc_id] for doc_id in support_documents]
        queries.append(
            {
                "query_id": row["id"],
                "query": row["question"].strip(),
                "question_type": f"{len(row['question_decomposition'])}hop",
                "required_article_ids": sorted(set(support_documents)),
                "n_required_documents": len(set(support_documents)),
                "n_required_evidence_facts": len(units),
                "evidence_units": units,
                "gold_chunk_ids": sorted(
                    {chunk_id for unit in units for chunk_id in unit}
                ),
            }
        )
    summary = {
        "dataset": "dgslibisey/MuSiQue",
        "source_split": "development (answerable)",
        "text_only": True,
        "articles": len(selected),
        "chunks": len(chunks),
        "eligible_queries": len(queries),
        "unresolved_queries": 0,
        "alignment_methods": {
            "paragraph_support_idx": sum(
                q["n_required_evidence_facts"] for q in queries
            )
        },
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "selection_seed": A.SEED,
    }
    A.write_jsonl(A.CHUNKS_PATH, chunks)
    A.write_jsonl(A.QUERIES_PATH, queries)
    A.SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print("[musique:data]", summary, flush=True)
    return chunks, queries, summary


A.prepare = prepare_musique


if __name__ == "__main__":
    A.run()
    B.run()
