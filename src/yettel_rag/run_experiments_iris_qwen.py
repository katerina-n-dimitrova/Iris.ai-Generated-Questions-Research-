#!/usr/bin/env python3
"""Run the three frozen Yettel experiments with the Iris-hosted Qwen model."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "yettel_rag"))
sys.path.insert(0, str(ROOT / "src" / "mhrag_vectoronly"))

import run_experiments as E  # noqa: E402
from iris_llm_client import (  # noqa: E402
    IRIS_LLM_MAX_TOKENS,
    IRIS_LLM_MODEL,
    stream_iris_chat,
)

SOURCE_DATA = ROOT / "data" / "processed" / "yettel_bg_experiments"
SOURCE_RESULTS = ROOT / "results" / "yettel_bg_experiments"
DATA = ROOT / "data" / "processed" / "yettel_bg_experiments_iris_qwen"
RESULTS = ROOT / "results" / "yettel_bg_experiments_iris_qwen"
REPORT = ROOT / "report" / "yettel_bg_adaptive_questions_iris_qwen.html"


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", cleaned, flags=re.IGNORECASE
    )
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Iris Qwen response contains no JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Iris Qwen response is not a JSON object")
    return value


def iris_call_json(system: str, user: str, seed: int = E.A.SEED) -> dict:
    last_error = None
    for attempt in range(6):
        retry_note = (
            "\nПредишният отговор беше невалиден. Върни само един валиден JSON обект."
            if attempt
            else ""
        )
        try:
            response = stream_iris_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + retry_note},
                ],
                temperature=0.2,
                max_tokens=IRIS_LLM_MAX_TOKENS,
                seed=seed + attempt,
                json_mode=True,
            )
            if not response:
                raise RuntimeError("Iris Qwen returned empty content")
            return extract_json(response)
        except Exception as error:
            last_error = error
            if attempt < 5:
                delay = min(30.0, 2**attempt) + random.uniform(0.0, 1.0)
                print(f"[iris-retry] {error}; sleep={delay:.1f}s", flush=True)
                time.sleep(delay)
    raise RuntimeError("Iris Qwen failed after 6 attempts") from last_error


def configure(batch_size: int) -> None:
    for directory in (DATA, RESULTS, REPORT.parent):
        directory.mkdir(parents=True, exist_ok=True)

    E.DATA, E.RESULTS, E.REPORT = DATA, RESULTS, REPORT
    E.A.DATA, E.A.RESULTS, E.A.REPORT = DATA, RESULTS, REPORT
    E.A.CHUNKS_PATH = DATA / "chunks.jsonl"
    E.A.QUERIES_PATH = DATA / "queries.jsonl"
    E.A.SUMMARY_PATH = DATA / "summary.json"
    E.A.GEN_PATH = DATA / "adaptive_generations.jsonl"
    E.A.CHUNK_VECTORS = RESULTS / "chunk_vectors.json"
    E.A.QUERY_VECTORS = RESULTS / "query_vectors.json"
    E.A.BOUNDED_VECTORS = RESULTS / "bounded_question_vectors.json"
    E.A.UNBOUNDED_VECTORS = RESULTS / "unused_unbounded_question_vectors.json"
    E.A.METRICS = RESULTS / "metrics.json"
    E.A.RANKINGS = RESULTS / "rankings.json"
    E.A.MAX_WORKERS = max(1, batch_size)
    E.A.MAX_RETRIES = 6
    E.A.MODEL = IRIS_LLM_MODEL
    E.A.call_json = iris_call_json

    E.C.DATA, E.C.RESULTS, E.C.REPORT = DATA, RESULTS, REPORT
    E.C.ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
    E.C.ARTICLE_VECTORS = RESULTS / "article_question_vectors.json"
    E.C.ARTICLE_RANKINGS = RESULTS / "article_chunk_question_rankings.jsonl"
    E.C.MAX_WORKERS = max(1, batch_size)
    E.C.ALLOW_PARTIAL_ARTICLES = False
    E.C._call_json_once = iris_call_json

    def generate_bounded(chunk: dict) -> dict:
        user = f'''Source chunk:\n"""\n{chunk["content"]}\n"""'''
        raw = E.A.call_json(E.A.FACT_PROMPT, user).get("facts", [])
        facts = E.A.dedup_facts([x for x in raw if isinstance(x, dict)])
        if not facts:
            raise RuntimeError(f"No facts for {chunk['chunk_id']}")
        budget = min(20, max(5, round(len(facts) * 0.5)))
        questions = E.A.generate_questions(chunk, facts, budget)
        return {
            "chunk_id": chunk["chunk_id"],
            "n_distinct_facts": len(facts),
            "bounded_budget": budget,
            "facts": facts,
            "bounded_questions": questions,
            "unbounded_questions": [],
            "generation_scope": "bounded_only",
        }

    E.A.generate_one = generate_bounded

    for name in ("chunks.jsonl", "queries.jsonl", "summary.json"):
        target = DATA / name
        if not target.exists():
            shutil.copy2(SOURCE_DATA / name, target)
    for name in ("chunk_vectors.json", "query_vectors.json"):
        target = RESULTS / name
        if not target.exists():
            shutil.copy2(SOURCE_RESULTS / name, target)


def run(batch_size: int = 24) -> dict:
    configure(batch_size)
    chunks, queries, summary = E.prepare()
    print(f"[model] {IRIS_LLM_MODEL} [batch] {batch_size}", flush=True)
    print("[1/3] baseline", flush=True)
    baseline = E.baseline(chunks, queries)
    print("[2/3] adaptive generated questions 5-20", flush=True)
    generated = E.generate_until_complete(chunks)
    bounded = E.adaptive_fast(chunks, queries, generated)[0]
    print("[3/3] adaptive chunk + whole-document questions 5-20", flush=True)
    articles = E.article_inputs(chunks, generated)
    article_generated = E.C.generate_articles(articles)
    combined = E.combined_fast(chunks, queries, generated, article_generated)
    payload = {
        "protocol": {
            **summary,
            "generation_model": IRIS_LLM_MODEL,
            "generation_endpoint": "Iris development chat API",
            "embedding": E.embedding_signature(),
            "retrieval": "dense + Unicode Bulgarian BM25 RRF",
            "rrf_k": E.A.RRF_K,
            "selected_conditions": [
                "Baseline",
                "Adaptive generated questions 5–20",
                E.C.CONDITION,
            ],
        },
        "conditions": [baseline, bounded, combined],
    }
    E.A.METRICS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    E.render(payload)
    html = REPORT.read_text(encoding="utf-8")
    html = html.replace(
        "340-document Yettel Bulgaria RAG — baseline and adaptive generated questions",
        "340-document Yettel Bulgaria RAG — Iris Qwen three experiments",
    ).replace(
        "Both were generated by gpt-5.4-mini, however, which can introduce same-model stylistic coupling; these results should therefore be described as a controlled same-model synthetic benchmark.",
        "The evaluation questions are frozen from the personally created Yettel dataset. Enrichment questions were generated independently with the Iris-hosted Qwen model and stored in isolated checkpoints.",
    )
    REPORT.write_text(html, encoding="utf-8")
    print(f"[done] {REPORT}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    run(args.batch)
