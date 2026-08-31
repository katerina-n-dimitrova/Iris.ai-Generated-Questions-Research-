"""Three-condition adaptive-question retrieval study on a FanOutQA page subset.

The dataset repository stores three Arrow datasets under corpus/, queries/, and
query_to_docs/.  Run this file once with --prepare-only to freeze and inspect
the subset before making any model calls; a normal run is resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from rank_bm25 import BM25Okapi

import adaptive_questions_100 as A


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "fanoutqa_retrieval"
ARTICLE_COUNT = int(os.getenv("FANOUTQA_ARTICLE_COUNT", "20"))
RUN_NAME = f"fanoutqa_adaptive_questions_{ARTICLE_COUNT}"
DATA = ROOT / "data" / "processed" / RUN_NAME
RESULTS = ROOT / "results" / RUN_NAME
REPORT = ROOT / "report" / f"{RUN_NAME}.html"

for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

A.DATA = DATA
A.RESULTS = RESULTS
A.REPORT = REPORT
A.ARTICLE_COUNT = ARTICLE_COUNT
A.MAX_WORKERS = int(os.getenv("FANOUTQA_ADAPTIVE_WORKERS", "24"))
A.MAX_RETRIES = 8
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
BASELINE_METRICS = RESULTS / "baseline_metrics.json"
BASELINE_RANKINGS = RESULTS / "baseline_rankings.jsonl"

_call_json_once = A.call_json


def call_json_resilient(system: str, user: str, seed: int = A.SEED) -> dict:
    """Retry transient transport/rate failures without changing model inputs."""
    last_error = None
    for attempt in range(6):
        try:
            return _call_json_once(system, user, seed)
        except Exception as error:
            last_error = error
            time.sleep(min(16, 2**attempt))
    raise last_error


A.call_json = call_json_resilient


def prepare() -> tuple[list[dict], list[dict], dict]:
    if A.CHUNKS_PATH.exists() and A.QUERIES_PATH.exists() and A.SUMMARY_PATH.exists():
        return (
            A.read_jsonl(A.CHUNKS_PATH),
            A.read_jsonl(A.QUERIES_PATH),
            json.loads(A.SUMMARY_PATH.read_text()),
        )
    missing = [
        str(RAW / name)
        for name in ("corpus", "queries", "query_to_docs")
        if not (RAW / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing FanOutQA Arrow datasets: "
            + ", ".join(missing)
            + ". Download JinChao1022/fanoutqa-retrieval with snapshot_download."
        )
    corpus = [dict(row) for row in load_from_disk(str(RAW / "corpus"))]
    raw_queries = [dict(row) for row in load_from_disk(str(RAW / "queries"))]
    mappings = {
        str(row["query_id"]): {str(x) for x in row["doc_ids"]}
        for row in load_from_disk(str(RAW / "query_to_docs"))
    }
    corpus_by_id = {str(row["doc_id"]): row for row in corpus}
    eligible = [
        q
        for q in raw_queries
        if mappings.get(str(q["query_id"]))
        and mappings[str(q["query_id"])] <= corpus_by_id.keys()
    ]
    shuffled = eligible[:]
    random.Random(A.SEED).shuffle(shuffled)
    selected: set[str] = set()
    for query in shuffled:
        required = mappings[str(query["query_id"])]
        if len(selected | required) <= ARTICLE_COUNT:
            selected |= required
        if len(selected) >= ARTICLE_COUNT:
            break
    if len(selected) != ARTICLE_COUNT:
        raise RuntimeError(
            f"Query-first selection produced {len(selected)}, expected {ARTICLE_COUNT}"
        )

    ordered_articles = [corpus_by_id[doc_id] for doc_id in sorted(selected, key=int)]
    chunks = []
    for article_index, article in enumerate(ordered_articles):
        compatible = {
            "url": str(article["doc_id"]),
            "title": article["title"],
            "body": article["text"],
        }
        chunks.extend(A.chunk_article(compatible, article_index))
    by_document: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk["chunk_id"])
    queries = []
    for query in eligible:
        query_id = str(query["query_id"])
        required = mappings[query_id]
        if required <= selected:
            units = [by_document[doc_id] for doc_id in sorted(required, key=int)]
            queries.append(
                {
                    "query_id": query_id,
                    "query": str(query["question"]).strip(),
                    "answer": query.get("answer", ""),
                    "categories": query.get("categories", []),
                    "required_article_ids": sorted(required, key=int),
                    "n_required_documents": len(required),
                    "n_required_evidence_facts": len(units),
                    "evidence_units": units,
                    "gold_chunk_ids": sorted({cid for unit in units for cid in unit}),
                }
            )
    summary = {
        "dataset": "JinChao1022/fanoutqa-retrieval",
        "articles": len(selected),
        "chunks": len(chunks),
        "eligible_queries": len(queries),
        "source_queries": len(raw_queries),
        "mean_gold_documents_per_query": float(
            np.mean([q["n_required_documents"] for q in queries])
        ),
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "selection": "seed-42 query-first complete-gold-document union",
        "selection_seed": A.SEED,
    }
    A.write_jsonl(A.CHUNKS_PATH, chunks)
    A.write_jsonl(A.QUERIES_PATH, queries)
    A.SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print("[data]", summary, flush=True)
    return chunks, queries, summary


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
        q, m = condition["quality"], condition["metrics"]
        cells = "".join(f"<td><b>{m[key]:.3f}</b></td>" for key in keys)
        baseline = condition["condition"] == "Baseline"
        generation = (
            "No generated questions"
            if baseline
            else f"<code>{condition['question_rule']}</code>, based on deduplicated atomic facts per chunk"
        )
        retrieval = (
            "Dense chunk-vector retrieval + BM25 chunk RRF"
            if baseline
            else "0.5/0.5 chunk-question vector fusion + BM25 chunk RRF"
        )
        generator = "None" if baseline else A.MODEL
        rows += (
            f'<tr><td class="left"><b>{condition["condition"]}</b></td>'
            f'<td class="left">{generation}</td><td>{q["generated_questions"]}'
            f"<br><span>{q['questions_per_chunk_min']}–{q['questions_per_chunk_max']}/chunk</span></td>"
            f"<td>1024 / 128</td><td>{condition['stored_vectors']}</td>"
            f'<td class="left">{retrieval}</td><td>{generator}</td>{cells}</tr>'
        )
    p = payload["protocol"]
    condition_count = len(payload["conditions"])
    REPORT.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>FanOutQA {p["articles"]}-article baseline and adaptive generated questions</title><style>body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:20px}}main{{max-width:1900px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:9px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}span,.note{{color:#9aa5b3;font-size:11px}}code{{overflow-wrap:anywhere}}</style></head><body><main><h1>{p["articles"]}-article FanOutQA — baseline and adaptive generated questions</h1><p class="note">Same query-first {p["articles"]} Wikipedia articles, {p["eligible_queries"]} eligible queries, {p["chunks"]} chunks, published query-to-document gold mapping, Iris embeddings, BM25, and RRF settings for all {condition_count} conditions. Each gold document is an evidence unit containing all of its chunks.</p><table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Chunk / overlap</th><th>Stored vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>"""
    )


def generate(chunks: list[dict]) -> list[dict]:
    """Generate with an append-only checkpoint instead of O(n²) rewrites."""
    cache = {row["chunk_id"]: row for row in A.read_jsonl(A.GEN_PATH)}
    todo = [chunk for chunk in chunks if chunk["chunk_id"] not in cache]
    print(f"[generation] cached={len(cache)} todo={len(todo)}", flush=True)
    failures = []
    with A.GEN_PATH.open("a") as checkpoint:
        with ThreadPoolExecutor(max_workers=A.MAX_WORKERS) as executor:
            futures = {executor.submit(A.generate_one, chunk): chunk for chunk in todo}
            for position, future in enumerate(as_completed(futures), 1):
                chunk = futures[future]
                try:
                    row = future.result()
                    cache[chunk["chunk_id"]] = row
                    checkpoint.write(json.dumps(row) + "\n")
                    checkpoint.flush()
                except Exception as error:
                    failures.append(
                        {"chunk_id": chunk["chunk_id"], "error": str(error)}
                    )
                    print(
                        f"[generation:error] {chunk['chunk_id']}: {error}", flush=True
                    )
                    continue
                print(
                    f"[generation] {position}/{len(todo)} {chunk['chunk_id']} "
                    f"facts={row['n_distinct_facts']} bounded={row['bounded_budget']} "
                    f"unbounded={row['unbounded_budget']}",
                    flush=True,
                )
    if failures:
        (DATA / "generation_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError(f"{len(failures)} chunks need another cached retry")
    failure_path = DATA / "generation_failures.json"
    if failure_path.exists():
        failure_path.unlink()
    ordered = [cache[chunk["chunk_id"]] for chunk in chunks]
    A.write_jsonl(A.GEN_PATH, ordered)
    return ordered


def seed_generation_cache(chunks: list[dict]) -> int:
    """Reuse exact-content generations from larger FanOutQA subset runs."""
    existing = {row["chunk_id"]: row for row in A.read_jsonl(A.GEN_PATH)}
    targets = {chunk["chunk_id"]: chunk for chunk in chunks}
    target_by_content = {
        (
            chunk["document_id"],
            hashlib.sha256(chunk["content"].encode()).hexdigest(),
        ): chunk
        for chunk in chunks
        if chunk["chunk_id"] not in existing
    }
    reused = 0
    for donor_count in (30, 50, 100, 300):
        donor_data = (
            ROOT / "data" / "processed" / f"fanoutqa_adaptive_questions_{donor_count}"
        )
        donor_chunks_path = donor_data / "chunks.jsonl"
        donor_generations_path = donor_data / "adaptive_generations.jsonl"
        if not donor_chunks_path.exists() or not donor_generations_path.exists():
            continue
        donor_chunks = {row["chunk_id"]: row for row in A.read_jsonl(donor_chunks_path)}
        for row in A.read_jsonl(donor_generations_path):
            old_chunk = donor_chunks.get(row["chunk_id"])
            if not old_chunk:
                continue
            key = (
                old_chunk["document_id"],
                hashlib.sha256(old_chunk["content"].encode()).hexdigest(),
            )
            target = target_by_content.pop(key, None)
            if target is None or target["chunk_id"] in existing:
                continue
            cloned = json.loads(json.dumps(row))
            cloned["chunk_id"] = target["chunk_id"]
            for field in ("bounded_questions", "unbounded_questions"):
                for question in cloned[field]:
                    question["supporting_chunk_ids"] = [target["chunk_id"]]
            existing[target["chunk_id"]] = cloned
            reused += 1
    if reused:
        A.write_jsonl(
            A.GEN_PATH,
            [existing[c["chunk_id"]] for c in chunks if c["chunk_id"] in existing],
        )
    print(
        f"[generation:reuse] exact-content reused={reused} cached={len(existing)} ",
        f"target={len(targets)}",
        flush=True,
    )
    return reused


def run_baseline(chunks: list[dict], queries: list[dict], summary: dict) -> dict:
    chunk_vectors = A.embed_resumable(A.CHUNK_VECTORS, [c["content"] for c in chunks])
    query_vectors = A.embed_resumable(A.QUERY_VECTORS, [q["query"] for q in queries])
    chunk_ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi([A.tokenize(c["content"]) for c in chunks])
    scores, rankings = [], []
    for qi, query in enumerate(queries):
        dense_order = np.argsort(-A.cosine_scores(chunk_vectors, query_vectors[qi]))
        sparse_order = np.argsort(-bm25.get_scores(A.tokenize(query["query"])))
        rrf = defaultdict(float)
        for order in (dense_order, sparse_order):
            for rank, index in enumerate(order, 1):
                rrf[chunk_ids[index]] += 1 / (A.RRF_K + rank)
        ranking = [
            cid for cid, _ in sorted(rrf.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        values = A.metric_row(query, ranking)
        scores.append(values)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": values,
            }
        )
    keys = list(scores[0])
    condition = {
        "condition": "Baseline",
        "question_rule": "No generated questions",
        "quality": {
            "generated_questions": 0,
            "questions_per_chunk_min": 0,
            "questions_per_chunk_mean": 0.0,
            "questions_per_chunk_max": 0,
        },
        "stored_vectors": len(chunks),
        "metrics": {key: float(np.mean([row[key] for row in scores])) for key in keys},
    }
    payload = {
        "protocol": {
            **summary,
            "generation_model": A.MODEL,
            "embedding": "Iris dim-384",
            "rrf_k": A.RRF_K,
        },
        "condition": condition,
    }
    BASELINE_METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    A.write_jsonl(BASELINE_RANKINGS, rankings)
    return condition


def run() -> dict:
    chunks, queries, summary = prepare()
    seed_generation_cache(chunks)
    generated = generate(chunks)
    payload = A.evaluate(chunks, queries, generated, summary)
    condition = run_baseline(chunks, queries, summary)
    payload["conditions"] = [condition] + payload["conditions"]
    payload["protocol"]["generation_model"] = (
        f"None for Baseline; {A.MODEL} for adaptive conditions"
    )
    A.METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    combined_rankings = json.loads(A.RANKINGS.read_text())
    combined_rankings["Baseline"] = A.read_jsonl(BASELINE_RANKINGS)
    A.RANKINGS.write_text(json.dumps(combined_rankings))
    render(payload)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    A.prepare = prepare
    A.render = render
    if args.prepare_only:
        prepare()
    else:
        run()
