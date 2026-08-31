#!/usr/bin/env python3
"""Run all five frozen Yettel retrieval conditions with Groq Qwen enrichment."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "yettel_rag"))
sys.path.insert(0, str(ROOT / "src" / "mhrag_vectoronly"))

import config  # noqa: E402
import run_experiments as E  # noqa: E402

SOURCE_DATA = ROOT / "data" / "processed" / "yettel_bg_experiments"
SOURCE_RESULTS = ROOT / "results" / "yettel_bg_experiments"
DATA = ROOT / "data" / "processed" / "yettel_bg_experiments_qwen36_27b_groq"
RESULTS = ROOT / "results" / "yettel_bg_experiments_qwen36_27b_groq"
REPORT = ROOT / "report" / "yettel_bg_adaptive_questions_qwen36_27b_groq.html"
JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse_json_content(text: str) -> dict:
    raw = THINK_BLOCK.sub("", text or "").strip()
    raw = JSON_FENCE.sub("", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response contains no JSON object")
    return json.loads(raw[start : end + 1])


def qwen_call_json(system: str, user: str, seed: int = E.A.SEED) -> dict:
    last_error = None
    count_matches = re.findall(r"Generate exactly (\d+)", user)
    if count_matches:
        requested_questions = int(count_matches[-1])
        max_output_tokens = min(3072, max(768, 512 + requested_questions * 40))
    else:
        max_output_tokens = 2048
    for attempt in range(6):
        try:
            response = config.get_groq_client().chat.completions.create(
                model=config.GROQ_CHAT_MODEL,
                temperature=0.3,
                seed=seed,
                max_completion_tokens=max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body={"reasoning_effort": "none"},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Groq Qwen returned empty message content")
            return parse_json_content(content)
        except Exception as error:
            last_error = error
            wait = min(32, 2**attempt)
            print(f"[qwen-retry] {error} sleep={wait}s", flush=True)
            time.sleep(wait)
    raise last_error


def configure() -> None:
    for directory in (DATA, RESULTS, REPORT.parent):
        directory.mkdir(parents=True, exist_ok=True)

    E.DATA, E.RESULTS, E.REPORT = DATA, RESULTS, REPORT
    E.UNBOUNDED_ARTICLE_GENERATIONS = (
        DATA / "article_question_generations_unbounded.jsonl"
    )
    E.UNBOUNDED_ARTICLE_VECTORS = RESULTS / "article_question_vectors_unbounded.json"
    E.UNBOUNDED_ARTICLE_RANKINGS = (
        RESULTS / "article_chunk_question_unbounded_rankings.jsonl"
    )

    E.A.DATA, E.A.RESULTS, E.A.REPORT = DATA, RESULTS, REPORT
    E.A.CHUNKS_PATH = DATA / "chunks.jsonl"
    E.A.QUERIES_PATH = DATA / "queries.jsonl"
    E.A.SUMMARY_PATH = DATA / "summary.json"
    E.A.GEN_PATH = DATA / "adaptive_generations.jsonl"
    E.A.CHUNK_VECTORS = RESULTS / "chunk_vectors.json"
    E.A.QUERY_VECTORS = RESULTS / "query_vectors.json"
    E.A.BOUNDED_VECTORS = RESULTS / "bounded_question_vectors.json"
    E.A.UNBOUNDED_VECTORS = RESULTS / "unbounded_question_vectors.json"
    E.A.METRICS = RESULTS / "metrics.json"
    E.A.RANKINGS = RESULTS / "rankings.json"
    # The configured Groq on-demand tier is limited to 8k TPM. Serial calls
    # prevent concurrent reservations from repeatedly exhausting that budget.
    E.A.MAX_WORKERS = 1
    E.A.MAX_RETRIES = 6
    E.A.MODEL = config.GROQ_CHAT_MODEL

    E.C.DATA, E.C.RESULTS, E.C.REPORT = DATA, RESULTS, REPORT
    E.C.ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
    E.C.ARTICLE_VECTORS = RESULTS / "article_question_vectors.json"
    E.C.ARTICLE_RANKINGS = RESULTS / "article_chunk_question_rankings.jsonl"
    E.C.MAX_WORKERS = 1
    E.C._call_json_once = qwen_call_json
    E.A.call_json = E.C.call_json_resilient

    def generate_bounded_only(chunk: dict) -> dict:
        user = f'''Source chunk:\n"""\n{chunk["content"]}\n"""'''
        facts = []
        for _ in range(E.A.MAX_RETRIES):
            facts = E.A.dedup_facts(
                E.A.call_json(E.A.FACT_PROMPT, user).get("facts", [])
            )
            if facts:
                break
        if not facts:
            raise RuntimeError(f"No facts for {chunk['chunk_id']}")
        bounded_budget = min(20, max(5, round(len(facts) * 0.5)))
        bounded = E.A.generate_questions(chunk, facts, bounded_budget)
        return {
            "chunk_id": chunk["chunk_id"],
            "n_distinct_facts": len(facts),
            "bounded_budget": bounded_budget,
            "facts": facts,
            "bounded_questions": bounded,
            "unbounded_questions": [],
            "generation_scope": "bounded_only",
        }

    E.A.generate_one = generate_bounded_only

    for name in ("chunks.jsonl", "queries.jsonl", "summary.json"):
        target = DATA / name
        if not target.exists():
            shutil.copy2(SOURCE_DATA / name, target)
    for name in ("chunk_vectors.json", "query_vectors.json"):
        target = RESULTS / name
        if not target.exists():
            shutil.copy2(SOURCE_RESULTS / name, target)


def run() -> dict:
    configure()
    base_render = E.render

    def render_once(payload: dict) -> None:
        base_render(payload)
        report = REPORT.read_text(encoding="utf-8")
        report = report.replace(
            "340-document Yettel Bulgaria RAG — baseline and adaptive generated questions",
            "340-document Yettel Bulgaria RAG — Groq Qwen 3.6 27B enrichment questions",
        )
        report = report.replace(
            "Both were generated by gpt-5.4-mini, however, which can introduce same-model stylistic coupling; these results should therefore be described as a controlled same-model synthetic benchmark.",
            "Evaluation queries remain frozen from the original gpt-5.4-mini benchmark. All adaptive enrichment questions in this table were independently generated by Groq qwen/qwen3.6-27b using the same prompts and allocation rules, providing a cross-model control for generator coupling.",
        )
        REPORT.write_text(report, encoding="utf-8")

    print(f"[qwen-groq] model={config.GROQ_CHAT_MODEL} data={DATA}", flush=True)
    chunks, queries, summary = E.prepare()
    generated = E.generate_until_complete(chunks)
    print(
        f"[qwen-groq] chunks={len(chunks)} queries={len(queries)} generated={len(generated)}",
        flush=True,
    )
    baseline = E.baseline(chunks, queries)
    print("[done] Baseline", flush=True)
    articles = E.article_inputs(chunks, generated)
    article_generated = E.C.generate_articles(articles)
    combined = E.combined_fast(chunks, queries, generated, article_generated)
    print(f"[done] {E.C.CONDITION}", flush=True)

    from embeddings import get_embedder

    embedder = get_embedder()
    if not embedder.dim:
        sample = json.loads(E.A.CHUNK_VECTORS.read_text())
        embedder.dim = len(sample["vectors"][0])
    payload = {
        "protocol": {
            **summary,
            "generation_model": E.A.MODEL,
            "embedding": E.embedding_signature(),
            "retrieval": "dense + Unicode BM25 RRF",
            "rrf_k": E.A.RRF_K,
            "selected_conditions": ["Baseline", E.C.CONDITION],
        },
        "conditions": [baseline, combined],
    }
    E.A.METRICS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_once(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run()
