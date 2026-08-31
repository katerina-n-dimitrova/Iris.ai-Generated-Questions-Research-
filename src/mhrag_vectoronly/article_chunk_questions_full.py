"""Add a whole-article + chunk-question arm to the 609-article study.

Dense retrieval embeds three item types separately and assigns each chunk the
equal-weight mean of its chunk-vector similarity, its best bounded chunk-level
question similarity, and its article's best whole-article question similarity.
The resulting dense ranking is fused with the unchanged chunk BM25 ranking by
RRF (k=60). Existing three-arm metrics and caches are preserved.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import adaptive_questions_100 as A
import adaptive_questions_full as F


ROOT = A.ROOT
DATA = ROOT / "data" / "processed" / "mhrag_adaptive_questions_full"
RESULTS = ROOT / "results" / "mhrag_adaptive_questions_full"
REPORT = ROOT / "report" / "mhrag_adaptive_questions_full.html"
ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
ARTICLE_VECTORS = RESULTS / "article_question_vectors_iris.json"
ARTICLE_RANKINGS = RESULTS / "article_chunk_question_rankings.jsonl"
CONDITION = "Adaptive chunk + whole-article questions 5–20"
UNBOUNDED_CONDITION = "Adaptive chunk + whole-article questions unbounded"
SUMMARY_CONDITION = "Adaptive chunk questions + whole-article summary"
MAX_WORKERS = 24
ALLOW_PARTIAL_ARTICLES = os.getenv("MHRAG_ALLOW_PARTIAL_ARTICLES", "0") == "1"
SKIP_MISSING_ARTICLES = os.getenv("MHRAG_SKIP_MISSING_ARTICLES", "0") == "1"

_call_json_once = A.call_json


def call_json_resilient(system: str, user: str, seed: int = A.SEED) -> dict:
    last_error = None
    for attempt in range(6):
        try:
            return _call_json_once(system, user, seed)
        except Exception as error:
            last_error = error
            time.sleep(min(16, 2**attempt))
    raise last_error


A.call_json = call_json_resilient


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def article_inputs(chunks: list[dict], generated: list[dict]) -> list[dict]:
    raw = json.loads(
        (ROOT / "data" / "raw" / "multihoprag" / "corpus.json").read_text()
    )
    raw_by_url = {row["url"]: row for row in raw}
    generation_by_chunk = {row["chunk_id"]: row for row in generated}
    chunks_by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk)
    articles = []
    for document_id in sorted(chunks_by_document):
        document_chunks = sorted(
            chunks_by_document[document_id], key=lambda row: row["chunk_position"]
        )
        facts, seen = [], set()
        for chunk in document_chunks:
            for fact in generation_by_chunk.get(chunk["chunk_id"], {}).get("facts", []):
                key = re.sub(r"\W+", " ", fact.get("fact", "").casefold()).strip()
                if key and key not in seen:
                    seen.add(key)
                    facts.append(fact)
        source = raw_by_url[document_id]
        budget = min(20, max(5, round(len(facts) * 0.5)))
        articles.append(
            {
                "article_id": document_id,
                "title": source.get("title", document_chunks[0]["document_title"]),
                "content": source.get("body", ""),
                "chunk_ids": [chunk["chunk_id"] for chunk in document_chunks],
                "facts": facts,
                "question_budget": budget,
            }
        )
    return articles


def generate_one(article: dict) -> dict:
    facts = article["facts"]
    if not facts:
        user = f'''Source article:\n"""\n{article["content"]}\n"""'''
        facts = A.dedup_facts(A.call_json(A.FACT_PROMPT, user).get("facts", []))
        if not facts:
            raise RuntimeError(f"No article facts for {article['article_id']}")
    proxy = {"chunk_id": article["article_id"], "content": article["content"]}
    budget = min(20, max(5, round(len(facts) * 0.5)))
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
        {row["article_id"]: row for row in read_jsonl(ARTICLE_GENERATIONS)}
        if ARTICLE_GENERATIONS.exists()
        else {}
    )
    missing = [article for article in articles if article["article_id"] not in cached]
    todo = [] if SKIP_MISSING_ARTICLES else missing
    failures = []
    with tqdm(
        total=len(articles),
        initial=len(cached),
        desc="Article generation",
        unit="article",
        dynamic_ncols=True,
    ) as progress:
        with ARTICLE_GENERATIONS.open("a") as checkpoint:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(generate_one, article): article for article in todo
                }
                for future in as_completed(futures):
                    article = futures[future]
                    try:
                        row = future.result()
                        cached[article["article_id"]] = row
                        checkpoint.write(json.dumps(row) + "\n")
                        checkpoint.flush()
                        progress.set_postfix(
                            questions=len(row["questions"]),
                            errors=len(failures),
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "article_id": article["article_id"],
                                "title": article["title"],
                                "error": str(error),
                            }
                        )
                        progress.write(
                            f"[article-generation:error] "
                            f"{article['title']}: {error}"
                        )
                    finally:
                        progress.update()
    if failures:
        (DATA / "article_question_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        if not ALLOW_PARTIAL_ARTICLES:
            raise RuntimeError(f"{len(failures)} articles need a cached retry")
    elif not missing:
        failure_path = DATA / "article_question_failures.json"
        if failure_path.exists():
            failure_path.unlink()
    ordered = [
        cached.get(
            article["article_id"],
            {
                "article_id": article["article_id"],
                "title": article["title"],
                "chunk_ids": article["chunk_ids"],
                "n_distinct_article_facts": len(article["facts"]),
                "question_budget": article["question_budget"],
                "questions": [],
                "generation_skipped": True,
            },
        )
        for article in articles
    ]
    # Keep the checkpoint restricted to successes so future runs can retry
    # skipped articles rather than treating placeholders as completed work.
    A.write_jsonl(
        ARTICLE_GENERATIONS,
        [cached[article["article_id"]] for article in articles if article["article_id"] in cached],
    )
    return ordered


def cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    dots = np.einsum("ij,j->i", matrix, vector, optimize=False)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return dots / np.maximum(norms, 1e-12)


def evaluate(
    chunks: list[dict],
    queries: list[dict],
    chunk_generated: list[dict],
    article_generated: list[dict],
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

    article_questions = [
        (question["question"], row["article_id"])
        for row in article_generated
        for question in row["questions"]
    ]
    article_question_vectors = A.embed_resumable(
        ARTICLE_VECTORS, [question for question, _ in article_questions]
    )
    article_question_indices: dict[str, list[int]] = defaultdict(list)
    for index, (_, article_id) in enumerate(article_questions):
        article_question_indices[article_id].append(index)

    bm25 = BM25Okapi([A.tokenize(row["content"]) for row in chunks])
    metric_rows, rankings = [], []
    for qi, query in enumerate(queries):
        chunk_scores = cosine(chunk_vectors, query_vectors[qi])
        chunk_q_scores = cosine(chunk_question_vectors, query_vectors[qi])
        article_q_scores = cosine(article_question_vectors, query_vectors[qi])
        fused = []
        for index, chunk in enumerate(chunks):
            chunk_q = (
                max(chunk_q_scores[chunk_question_indices[chunk["chunk_id"]]])
                if chunk_question_indices[chunk["chunk_id"]]
                else chunk_scores[index]
            )
            article_indices = article_question_indices[chunk["document_id"]]
            if article_indices:
                article_q = max(article_q_scores[article_indices])
                fused.append((chunk_scores[index] + chunk_q + article_q) / 3)
            else:
                # If Iris could not enrich this article, evaluate it using the
                # two available signals instead of dropping it from retrieval.
                fused.append((chunk_scores[index] + chunk_q) / 2)
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
    chunk_counts = [len(row["bounded_questions"]) for row in chunk_generated]
    article_counts = [len(row["questions"]) for row in article_generated]
    condition = {
        "condition": CONDITION,
        "question_rule": (
            "chunk: clamp(5, 20, round(facts * 0.5)); "
            "article: clamp(5, 20, round(article facts * 0.5))"
        ),
        "quality": {
            "generated_questions": sum(chunk_counts) + sum(article_counts),
            "chunk_questions": sum(chunk_counts),
            "article_questions": sum(article_counts),
            "questions_per_chunk_min": min(chunk_counts),
            "questions_per_chunk_max": max(chunk_counts),
            "questions_per_article_min": min(article_counts),
            "questions_per_article_max": max(article_counts),
            "articles_with_questions": sum(count > 0 for count in article_counts),
            "articles_without_questions": sum(count == 0 for count in article_counts),
        },
        "stored_vectors": len(chunks) + sum(chunk_counts) + sum(article_counts),
        "metrics": {
            key: float(np.mean([row[key] for row in metric_rows])) for key in keys
        },
    }
    A.write_jsonl(ARTICLE_RANKINGS, rankings)
    return condition, rankings


def render(payload: dict) -> None:
    keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    )
    rows = ""
    for condition in payload["conditions"]:
        q, metrics = condition["quality"], condition["metrics"]
        cells = "".join(f"<td><b>{metrics[key]:.3f}</b></td>" for key in keys)
        baseline = condition["condition"] == "Baseline"
        combined = condition["condition"] in (CONDITION, UNBOUNDED_CONDITION)
        summary_enriched = condition["condition"] == SUMMARY_CONDITION
        if baseline:
            generation, count = "No generated questions", "0<br><span>0–0/chunk</span>"
            retrieval, generator = (
                "Dense chunk-vector retrieval + BM25 chunk RRF",
                "None",
            )
        elif combined:
            generation = (
                "<code>5–20 per chunk + 5–20 per whole article</code>"
                if condition["condition"] == CONDITION
                else "<code>unbounded per chunk + unbounded per whole article</code>"
            ) + ", based on deduplicated atomic facts"
            count = (
                f"{q['generated_questions']}<br><span>{q['chunk_questions']} chunk + "
                f"{q['article_questions']} article questions</span>"
            )
            retrieval = (
                "Equal 1/3 chunk, chunk-question, article-question "
                "vector fusion + BM25 chunk RRF"
            )
            generator = A.MODEL
        elif summary_enriched:
            generation = (
                "<code>5–20 adaptive questions per chunk + one short "
                "LLM summary per whole article</code>"
            )
            count = (
                f"{q['chunk_questions']}<br><span>chunk questions + "
                f"{q['article_summaries']} article summaries</span>"
            )
            retrieval = (
                "Equal 1/3 chunk, chunk-question, article-summary "
                "vector fusion + BM25 chunk RRF"
            )
            generator = A.MODEL
        else:
            generation = (
                f"<code>{condition['question_rule']}</code>, based on "
                "deduplicated atomic facts per chunk"
            )
            count = (
                f"{q['generated_questions']}<br><span>"
                f"{q['questions_per_chunk_min']}–{q['questions_per_chunk_max']}/chunk</span>"
            )
            retrieval = "0.5/0.5 chunk-question vector fusion + BM25 chunk RRF"
            generator = A.MODEL
        rows += (
            f'<tr><td class="left"><b>{condition["condition"]}</b></td>'
            f'<td class="left">{generation}</td><td>{count}</td><td>1024 / 128</td>'
            f'<td>{condition["stored_vectors"]}</td><td class="left">{retrieval}</td>'
            f"<td>{generator}</td>{cells}</tr>"
        )
    p = payload["protocol"]
    has_summary = any(
        row["condition"] == SUMMARY_CONDITION for row in payload["conditions"]
    )
    summary_note = (
        " Whole-article summary generation also covers all 609 articles."
        if has_summary
        else ""
    )
    combined_quality = next(
        (
            row["quality"]
            for row in payload["conditions"]
            if row["condition"] in (CONDITION, UNBOUNDED_CONDITION)
        ),
        {},
    )
    articles_with_questions = combined_quality.get("articles_with_questions", p["articles"])
    articles_without_questions = combined_quality.get("articles_without_questions", 0)
    chunk_note = (
        f"Chunk-level adaptive generation covers all {p['chunks']} chunks."
    )
    article_note = (
        f"Whole-article question generation succeeded for {articles_with_questions} "
        f"of {p['articles']} articles"
        + (
            f"; {articles_without_questions} article uses chunk-level fallback."
            if articles_without_questions
            else "."
        )
    )
    REPORT.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>609-article baseline and adaptive generated questions</title><style>body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:20px}}main{{max-width:2100px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:9px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}span,.note{{color:#9aa5b3;font-size:11px}}code{{overflow-wrap:anywhere}}</style></head><body><main><h1>609-article MultiHop-RAG — baseline and adaptive generated questions</h1><p class="note">Same {p['articles']} articles, {p['eligible_queries']} eligible queries, {p['chunks']} chunks, gold mapping, Iris embeddings, BM25, and RRF settings for all {len(payload["conditions"])} conditions.</p><p class="note">{chunk_note} {article_note}{summary_note}</p><table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions / enrichment</th><th>Chunk / overlap</th><th>Stored vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>"""
    )


def run() -> dict:
    chunks = read_jsonl(DATA / "chunks.jsonl")
    queries = read_jsonl(DATA / "queries.jsonl")
    chunk_generated = read_jsonl(DATA / "adaptive_generations.jsonl")
    existing_by_chunk = {row["chunk_id"]: row for row in chunk_generated}
    chunk_generated = [
        existing_by_chunk.get(
            chunk["chunk_id"],
            {
                "chunk_id": chunk["chunk_id"],
                "facts": [],
                "bounded_questions": [],
                "unbounded_questions": [],
                "generation_skipped": True,
            },
        )
        for chunk in chunks
    ]
    articles = article_inputs(chunks, chunk_generated)
    article_generated = generate_articles(articles)
    condition, rankings = evaluate(chunks, queries, chunk_generated, article_generated)

    payload = json.loads((RESULTS / "metrics.json").read_text())
    payload["conditions"] = [
        row for row in payload["conditions"] if row["condition"] != CONDITION
    ] + [condition]
    payload["protocol"]["article_question_generation"] = {
        "articles": len(articles),
        "rule": "clamp(5, 20, round(article facts * 0.5))",
        "dense_fusion": "equal 1/3 chunk + chunk-question + article-question",
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    ranking_payload = json.loads((RESULTS / "rankings.json").read_text())
    ranking_payload[CONDITION] = rankings
    (RESULTS / "rankings.json").write_text(json.dumps(ranking_payload))
    render(payload)
    print(json.dumps(condition, indent=2))
    return payload


if __name__ == "__main__":
    run()
