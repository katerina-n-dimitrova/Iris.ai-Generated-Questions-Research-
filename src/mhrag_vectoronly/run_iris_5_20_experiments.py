"""Run the three full-corpus experiments with the Iris-hosted Qwen model.

The baseline does not call an LLM. The two enrichment conditions use isolated
Iris checkpoints, so generations from other models cannot be reused.

``--batch`` controls concurrent streaming requests. vLLM continuously batches
those requests internally; streaming prevents Iris's 60-second HTTP gateway
from timing out while a long JSON response is still being generated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adaptive_questions_100 as A
import full_corpus_no_question_baseline as B
from iris_llm_client import IRIS_LLM_MAX_TOKENS, IRIS_LLM_MODEL, stream_iris_chat
from tqdm import tqdm


ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "mhrag_iris_qwen_5_20_full"
RESULTS = ROOT / "results" / "mhrag_iris_qwen_5_20_full"
REPORT = ROOT / "report" / "mhrag_iris_qwen_5_20_full.html"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)


def _extract_json(text: str) -> dict:
    """Accept plain, fenced, or reasoning-prefixed JSON model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", cleaned, flags=re.IGNORECASE
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Iris LLM response contains no JSON object")
        return json.loads(cleaned[start : end + 1])


def iris_call_json(system: str, user: str, seed: int = A.SEED) -> dict:
    """Stream one JSON response, retrying transient or malformed responses."""
    last_error = None
    max_attempts = 6
    for attempt in range(max_attempts):
        retry_note = (
            "\nYour previous response was malformed. Return one valid JSON "
            "object only, with all strings escaped correctly."
            if attempt
            else ""
        )
        try:
            content = stream_iris_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + retry_note},
                ],
                temperature=0.2,
                max_tokens=IRIS_LLM_MAX_TOKENS,
                seed=seed + attempt,
                json_mode=True,
            )
            if not content:
                raise RuntimeError("Iris LLM returned empty content")
            payload = _extract_json(content)
            if not isinstance(payload, dict):
                raise ValueError("Iris LLM response is not a JSON object")
            return payload
        except Exception as error:
            last_error = error
            if attempt < max_attempts - 1:
                # Jitter keeps concurrently retried requests from hitting the
                # gateway again in the same burst.
                delay = min(30.0, 2**attempt) + random.uniform(0.0, 1.0)
                tqdm.write(
                    f"[llm:retry] {error.__class__.__name__}; "
                    f"attempt={attempt + 2}/{max_attempts} in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(f"Iris LLM failed after {max_attempts} attempts") from last_error


def configure_chunk_experiment(batch_size: int) -> None:
    """Point the adaptive pipeline at Iris-specific caches and the full corpus."""
    A.ARTICLE_COUNT = 609
    A.MODEL = IRIS_LLM_MODEL
    # This also controls the question-completion loop. Some otherwise valid
    # responses contain fewer distinct questions than requested.
    A.MAX_RETRIES = 6
    A.MAX_WORKERS = batch_size
    A.DATA = DATA
    A.RESULTS = RESULTS
    A.REPORT = REPORT
    A.CHUNKS_PATH = DATA / "chunks.jsonl"
    A.QUERIES_PATH = DATA / "queries.jsonl"
    A.SUMMARY_PATH = DATA / "summary.json"
    A.GEN_PATH = DATA / "adaptive_generations.jsonl"
    A.CHUNK_VECTORS = RESULTS / "chunk_vectors_iris.json"
    A.QUERY_VECTORS = RESULTS / "query_vectors_iris.json"
    A.BOUNDED_VECTORS = RESULTS / "bounded_question_vectors_iris.json"
    A.UNBOUNDED_VECTORS = RESULTS / "unbounded_question_vectors_iris.json"
    A.METRICS = RESULTS / "metrics.json"
    A.RANKINGS = RESULTS / "rankings.json"
    A.call_json = iris_call_json

    def prepare_full():
        chunks, queries, alignment = B.prepare()
        summary = {
            "articles": alignment["articles"],
            "chunks": len(chunks),
            "eligible_queries": len(queries),
            "unresolved_queries": alignment["queries_excluded_for_unresolved_evidence"],
            "alignment_methods": alignment["alignment_methods"],
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "selection_seed": None,
        }
        if not A.CHUNKS_PATH.exists():
            A.write_jsonl(A.CHUNKS_PATH, chunks)
            A.write_jsonl(A.QUERIES_PATH, queries)
            A.SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
        return chunks, queries, summary

    A.prepare = prepare_full

    # Generate only the bounded 5-20 pool. The evaluator expects both fields,
    # so the same pool is supplied under the unused unbounded key.
    def generate_bounded(chunk: dict) -> dict:
        user = f'''Source chunk:\n"""\n{chunk["content"]}\n"""'''
        raw_facts = A.call_json(A.FACT_PROMPT, user).get("facts", [])
        if not isinstance(raw_facts, list):
            raise ValueError("Iris LLM 'facts' field is not a list")
        facts = A.dedup_facts(
            [item for item in raw_facts if isinstance(item, dict)]
        )
        if not facts:
            raise RuntimeError(f"No facts for {chunk['chunk_id']}")
        budget = min(20, max(5, round(len(facts) * 0.5)))
        questions = A.generate_questions(chunk, facts, budget)
        return {
            "chunk_id": chunk["chunk_id"],
            "n_distinct_facts": len(facts),
            "bounded_budget": budget,
            "unbounded_budget": budget,
            "facts": facts,
            "bounded_questions": questions,
            "unbounded_questions": questions,
        }

    A.generate_one = generate_bounded


def add_baseline_and_keep_requested_condition(payload: dict) -> dict:
    baseline_payload = json.loads(B.METRICS.read_text())
    bounded = next(
        row
        for row in payload["conditions"]
        if row["condition"] == "Adaptive generated questions 5–20"
    )
    baseline = {
        "condition": "Baseline",
        "question_rule": "No generated questions",
        "quality": {
            "generated_questions": 0,
            "questions_per_chunk_min": 0,
            "questions_per_chunk_mean": 0.0,
            "questions_per_chunk_max": 0,
        },
        "stored_vectors": baseline_payload["protocol"]["stored_vectors"],
        "metrics": baseline_payload["metrics"],
    }
    payload["conditions"] = [baseline, bounded]
    A.METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    A.render(payload)
    return payload


def configure_article_experiment(batch_size: int):
    # Import after A is configured so this module captures the Iris call method.
    import article_chunk_questions_full as C

    # adaptive_questions_full (imported by C) changes A's historical cache
    # paths at import time. Restore this run's isolated Iris configuration.
    configure_chunk_experiment(batch_size)

    C.DATA = DATA
    C.RESULTS = RESULTS
    C.REPORT = REPORT
    C.ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
    C.ARTICLE_VECTORS = RESULTS / "article_question_vectors_iris.json"
    C.ARTICLE_RANKINGS = RESULTS / "article_chunk_question_rankings.jsonl"
    C.MAX_WORKERS = batch_size
    # This dedicated runner retries every missing article, then evaluates the
    # successfully generated subset if Iris still cannot serve some requests.
    C.ALLOW_PARTIAL_ARTICLES = True
    return C


def run(batch_size: int = 32, skip_missing_articles: bool = False) -> dict:
    batch_size = max(1, batch_size)
    print(
        f"[model] {IRIS_LLM_MODEL}  "
        f"[inference] streaming_concurrency={batch_size}",
        flush=True,
    )
    print("[1/3] baseline (no LLM; reuse identical completed result)", flush=True)
    if (
        not B.METRICS.exists()
        or not B.CHUNK_VECTORS.exists()
        or not B.QUERY_VECTORS.exists()
    ):
        B.run()

    configure_chunk_experiment(batch_size)
    for source, target in (
        (B.CHUNK_VECTORS, A.CHUNK_VECTORS),
        (B.QUERY_VECTORS, A.QUERY_VECTORS),
    ):
        if not target.exists():
            shutil.copyfile(source, target)

    print("[2/3] adaptive generated questions 5-20", flush=True)
    payload = add_baseline_and_keep_requested_condition(A.run())

    print("[3/3] adaptive chunk + whole-article questions 5-20", flush=True)
    article_experiment = configure_article_experiment(batch_size)
    article_experiment.SKIP_MISSING_ARTICLES = skip_missing_articles
    payload = article_experiment.run()
    print(f"[done] report: {REPORT}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iris Qwen 5-20 full-corpus experiments"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help=(
            "maximum concurrent streaming requests; vLLM batches them "
            "internally (default: 32)"
        ),
    )
    parser.add_argument(
        "--skip-missing-articles",
        action="store_true",
        help="evaluate the successful cached article generations without retrying missing ones",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch_size=args.batch, skip_missing_articles=args.skip_missing_articles)
