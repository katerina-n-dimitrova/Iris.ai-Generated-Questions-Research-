"""
Generate RAG answers from the retrieved context produced by
run_retrieval_experiments.py, and log generation latency.

For each query we feed the assembled top-k context to the chat model and record
the answer alongside the gold answer (when the dataset provides one).

Outputs results/answer_metrics/answers_<dataset>_<condition>.jsonl with:
    query_id, dataset, condition, query_text, context, generated_answer,
    gold_answer, llm_generation_latency_ms, total_rag_latency_ms

`total_rag_latency_ms` adds the retrieval latency logged earlier (looked up by
query_id) to the generation latency, when available.

Usage:
    python src/run_answer_generation.py
    python src/run_answer_generation.py --datasets wikitablequestions chartqa
    python src/run_answer_generation.py --dry-run     # no API calls
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import common
import config

DEFAULT_CONCURRENCY = 8

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context. If the answer is not in the context, say "
    "you don't know. Be concise."
)


def _retrieval_latency_lookup() -> Dict[str, float]:
    """Map query_id -> total_retrieval_latency_ms from the online latency log."""
    path = config.LATENCY_LOG_DIR / "online_query_latency.csv"
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["query_id"]] = float(row["total_retrieval_latency_ms"])
            except (KeyError, ValueError):
                pass
    return out


def _generate(client, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def run_one(
    dataset: str,
    condition: str,
    dry_run: bool,
    max_samples: int,
    retr_latency: Dict[str, float],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    in_path = config.RETRIEVAL_METRICS_DIR / f"retrieved_{dataset}_{condition}.jsonl"
    if not in_path.exists():
        print(f"  {in_path.name} not found; run retrieval first")
        return 0

    client = None if dry_run else config.get_openai_client()
    rows = list(common.read_jsonl(in_path))[:max_samples]

    def worker(r: Dict) -> Dict:
        question = r["query_text"]
        context = r.get("context", "")
        if dry_run:
            answer, gen_ms = "(dry-run: no answer generated)", 0.0
        else:
            t0 = time.perf_counter()
            try:
                answer = _generate(client, question, context)
            except Exception as e:  # noqa: BLE001
                answer = f"(generation error: {e})"
            gen_ms = (time.perf_counter() - t0) * 1000
        retr_ms = retr_latency.get(r["query_id"], 0.0)
        return {
            "query_id": r["query_id"],
            "dataset": dataset,
            "condition": condition,
            "query_text": question,
            "context": context,
            "generated_answer": answer,
            "gold_answer": r.get("gold_answer", ""),
            "llm_generation_latency_ms": round(gen_ms, 4),
            "total_rag_latency_ms": round(retr_ms + gen_ms, 4),
        }

    # Latency-bound API calls -> thread pool. map() preserves input order.
    workers = 1 if dry_run else max(1, concurrency)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out_rows = list(ex.map(worker, rows))

    out_path = config.ANSWER_METRICS_DIR / f"answers_{dataset}_{condition}.jsonl"
    common.write_jsonl(out_path, out_rows)
    print(f"  {dataset}/{condition}: {len(out_rows)} answers -> {out_path.name}")
    return len(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=config.ALL_DATASETS,
        choices=config.ALL_DATASETS,
    )
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=list(config.CONDITIONS),
        choices=config.CONDITIONS,
    )
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; write empty answers (pipeline test).",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Parallel API calls (latency-bound; default 8).",
    )
    args = ap.parse_args()

    retr_latency = _retrieval_latency_lookup()
    for dataset in args.datasets:
        for condition in args.conditions:
            run_one(
                dataset,
                condition,
                args.dry_run,
                args.max_samples,
                retr_latency,
                args.concurrency,
            )


if __name__ == "__main__":
    main()
