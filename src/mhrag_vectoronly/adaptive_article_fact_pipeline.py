"""Adaptive article-first fact analysis against the locked LargeChunk-BM25 baseline."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import numpy as np

import controlled_article_first as A
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

CACHE = A.DATA / "adaptive_article_facts_questions.jsonl"
METRICS = R.RESULTS / "metrics_adaptive_article_facts.json"
RANKINGS = R.RESULTS / "rankings_adaptive_article_facts.json"

FACT_PROMPT = """You analyze a complete article for retrieval-question coverage.
Extract the meaningful atomic facts explicitly supported by the article. Then
group duplicates and closely related restatements so each returned item is a
distinct retrievable fact. Do not add outside knowledge.

For every retained fact provide:
- a concise atomic factual statement;
- a short verbatim evidence quote from the article;
- importance from 1 to 5, where 5 is central to the article;
- distinctiveness from 1 to 5, where 5 contains unusually discriminative
  names, dates, numbers, comparisons, causes, or outcomes.

Ignore headings, boilerplate, vague opinion, and unsupported inference.
Return valid JSON only:
{"facts":[{"fact":"...","evidence":"...","importance":1,
"distinctiveness":1}]}"""

QUESTION_PROMPT = """You generate grounded retrieval questions from an
analyzed complete article and its deduplicated atomic facts. Generate exactly
the requested number of questions. Prioritize facts with high importance and
distinctiveness while maintaining broad coverage. Important or complex facts
may receive more than one genuinely different question, but do not create
near-duplicates. Every question must be answerable from the article.

For each question return its answer, a short verbatim evidence quote copied
from the article, and the zero-based IDs of the source facts it covers.
Return valid JSON only:
{"questions":[{"question":"...","short_answer":"...",
"evidence":"verbatim quote","source_fact_ids":[0]}]}"""


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _deduplicate_facts(raw: list[dict]) -> list[dict]:
    facts: list[dict] = []
    normalized: list[str] = []
    for item in raw:
        fact = str(item.get("fact", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        key = _normalize(fact)
        if not fact or not evidence or not key:
            continue
        if any(
            key == old or SequenceMatcher(None, key, old).ratio() >= 0.90
            for old in normalized
        ):
            continue
        normalized.append(key)
        facts.append(
            {
                "fact": fact,
                "evidence": evidence,
                "importance": min(5, max(1, int(item.get("importance", 1)))),
                "distinctiveness": min(5, max(1, int(item.get("distinctiveness", 1)))),
            }
        )
    return facts


def _budget(n_facts: int) -> int:
    return min(20, max(5, round(n_facts * 0.5)))


def _generate_article(article: dict, chunks: list[dict]) -> dict:
    analysis_user = f'''Complete article:
"""
{article["cleaned_body"]}
"""'''
    facts: list[dict] = []
    for _ in range(C.GEN_MAX_RETRIES):
        facts = _deduplicate_facts(G._call(FACT_PROMPT, analysis_user).get("facts", []))
        if facts:
            break
    if not facts:
        raise RuntimeError(f"No facts extracted for {article['title']}")

    count = _budget(len(facts))
    fact_block = "\n".join(
        f"[{index}] importance={fact['importance']} "
        f"distinctiveness={fact['distinctiveness']} | {fact['fact']}"
        for index, fact in enumerate(facts)
    )
    question_user = f'''Complete article:
"""
{article["cleaned_body"]}
"""

Deduplicated scored facts:
{fact_block}

Generate exactly {count} questions.'''
    questions: list[dict] = []
    for _ in range(C.GEN_MAX_RETRIES):
        raw = G._call(QUESTION_PROMPT, question_user).get("questions", [])
        questions, seen = [], []
        for item in raw:
            text = str(item.get("question", "")).strip()
            key = _normalize(text)
            if (
                not text
                or not key
                or any(SequenceMatcher(None, key, old).ratio() >= 0.94 for old in seen)
            ):
                continue
            support = A._supporting_chunks(item, chunks, article["article_id"])
            fact_ids = sorted(
                {
                    int(value)
                    for value in item.get("source_fact_ids", [])
                    if str(value).lstrip("-").isdigit() and 0 <= int(value) < len(facts)
                }
            )
            if not support or not fact_ids:
                continue
            seen.append(key)
            questions.append(
                {
                    "question_id": (
                        f"{article['article_key']}::adaptiveq{len(questions)}"
                    ),
                    "question": text,
                    "short_answer": str(item.get("short_answer", "")).strip(),
                    "evidence": str(item.get("evidence", "")).strip(),
                    "source_fact_ids": fact_ids,
                    "supporting_chunk_ids": support,
                    "document_id": article["article_id"],
                }
            )
        if len(questions) >= count:
            questions = questions[:count]
            break
    if len(questions) != count:
        raise RuntimeError(
            f"Generated {len(questions)}/{count} valid questions for {article['title']}"
        )
    return {
        "document_id": article["article_id"],
        "document_title": article["title"],
        "facts": facts,
        "n_distinct_facts": len(facts),
        "question_budget": count,
        "questions": questions,
        "valid": True,
    }


def generate() -> list[dict]:
    articles = H.read_jsonl(H.ARTICLES_PATH)
    chunks = G.read_jsonl(G.CHUNKS_1024)
    cache = (
        {row["document_id"]: row for row in G.read_jsonl(CACHE)}
        if CACHE.exists()
        else {}
    )
    for position, article in enumerate(articles, 1):
        existing = cache.get(article["article_id"], {})
        if existing.get("valid") and len(existing.get("questions", [])) == existing.get(
            "question_budget"
        ):
            continue
        cache[article["article_id"]] = _generate_article(article, chunks)
        G.write_jsonl(
            CACHE,
            [cache[a["article_id"]] for a in articles if a["article_id"] in cache],
        )
        row = cache[article["article_id"]]
        print(
            f"[adaptive-article] {position}/{len(articles)} "
            f"facts={row['n_distinct_facts']} questions={row['question_budget']}",
            flush=True,
        )
    return G.read_jsonl(CACHE)


def _question_rows(cache: list[dict]) -> list[dict]:
    return [
        {
            "id": question["question_id"],
            "text": question["question"],
            "supporting_chunk_ids": question["supporting_chunk_ids"],
            "document_id": article["document_id"],
        }
        for article in cache
        for question in article["questions"]
    ]


def _quality(cache: list[dict]) -> dict:
    covered = 0
    total_facts = 0
    duplicate_pairs = 0
    possible_pairs = 0
    counts = []
    for article in cache:
        facts = article["facts"]
        questions = article["questions"]
        total_facts += len(facts)
        covered += len(
            {
                fact_id
                for question in questions
                for fact_id in question["source_fact_ids"]
            }
        )
        keys = [_normalize(question["question"]) for question in questions]
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                possible_pairs += 1
                duplicate_pairs += int(
                    SequenceMatcher(None, keys[left], keys[right]).ratio() >= 0.94
                )
        counts.append(len(questions))
    return {
        "articles": len(cache),
        "distinct_facts": total_facts,
        "covered_facts": covered,
        "fact_coverage": covered / total_facts,
        "generated_questions": sum(counts),
        "questions_per_article_min": min(counts),
        "questions_per_article_mean": float(np.mean(counts)),
        "questions_per_article_max": max(counts),
        "near_duplicate_pairs": duplicate_pairs,
        "near_duplicate_pair_rate": (
            duplicate_pairs / possible_pairs if possible_pairs else 0.0
        ),
    }


def run() -> dict:
    cache = generate()
    chunks = G.read_jsonl(G.CHUNKS_1024)
    questions = _question_rows(cache)
    gold = {row["query_id"]: row for row in G.read_jsonl(G.GOLD_1024)}
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    test = [q for q in queries if split[q["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)
    chunk_vectors, question_vectors = A._index("adaptive_fact1024", chunks, questions)
    rows = []
    for query in test:
        qvec = query_vectors[query["query_id"]]
        chunk_rank, chunk_scores = R._dense_chunks(chunks, chunk_vectors, qvec)
        question_scores = A._question_scores(questions, question_vectors, qvec)
        dense = R._dual(chunk_rank, chunk_scores, question_scores)
        ranking = R._rrf([dense, R._bm25(chunks, query["query"])])
        rows.append(R._metric_row(query, gold[query["query_id"]], ranking, chunk_map))
    quality = _quality(cache)
    payload = {
        "experiment": "Adaptive article fact analysis",
        "condition": "Article-AdaptiveFacts-LargeChunk-BM25",
        "baseline": "Article-E1-LargeChunk-BM25",
        "question_rule": "clamp(5, 20, round(n_distinct_facts * 0.5))",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "generation_model": C.gen_model(),
        "embedding": "Iris dim384",
        "retrieval": "0.5/0.5 dense chunk-question fusion + BM25 RRF",
        "stored_vectors": len(chunks) + len(questions),
        "quality": quality,
        "metrics": R._evaluate(rows),
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(rows))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
