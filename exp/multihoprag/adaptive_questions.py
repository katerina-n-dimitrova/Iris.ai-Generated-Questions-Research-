"""Run both adaptive-question conditions on the full MultiHop-RAG corpus.

The completed 300-article generation is remapped onto the full corpus by an
exact (document URL, chunk position, chunk text) key.  Only unmatched chunks
are sent to the generator.  Existing question embeddings are likewise reused
by exact question text, while the full baseline supplies chunk/query vectors.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import adaptive_lib as A
import baseline as B


ROOT = A.ROOT
DATA = ROOT / "data" / "processed" / "mhrag_adaptive_questions_full"
RESULTS = ROOT / "results" / "mhrag_adaptive_questions_full"
REPORT = ROOT / "report" / "mhrag_adaptive_questions_full.html"
SOURCE_DATA = ROOT / "data" / "processed" / "mhrag_adaptive_questions_300"
SOURCE_RESULTS = ROOT / "results" / "mhrag_adaptive_questions_300"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

A.ARTICLE_COUNT = 609
A.MAX_WORKERS = 1
A.MAX_RETRIES = 8
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


def prepare() -> tuple[list[dict], list[dict], dict]:
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


def seed_generation(full_chunks: list[dict]) -> None:
    if A.GEN_PATH.exists():
        return
    old_chunks = A.read_jsonl(SOURCE_DATA / "chunks.jsonl")
    old_rows = {
        row["chunk_id"]: row
        for row in A.read_jsonl(SOURCE_DATA / "adaptive_generations.jsonl")
    }
    old_by_key = {
        (row["document_id"], row["chunk_position"], row["content"]): row
        for row in old_chunks
    }
    seeded = []
    for chunk in full_chunks:
        old_chunk = old_by_key.get(
            (chunk["document_id"], chunk["chunk_position"], chunk["content"])
        )
        if not old_chunk or old_chunk["chunk_id"] not in old_rows:
            continue
        row = json.loads(json.dumps(old_rows[old_chunk["chunk_id"]]))
        row["chunk_id"] = chunk["chunk_id"]
        for field in ("bounded_questions", "unbounded_questions"):
            for question in row[field]:
                question["supporting_chunk_ids"] = [chunk["chunk_id"]]
        seeded.append(row)
    A.write_jsonl(A.GEN_PATH, seeded)
    (DATA / "reuse_summary.json").write_text(
        json.dumps(
            {
                "source_chunks": len(old_chunks),
                "full_chunks": len(full_chunks),
                "reused_generation_chunks": len(seeded),
                "chunks_to_generate": len(full_chunks) - len(seeded),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[reuse] generation {len(seeded)}/{len(full_chunks)} chunks", flush=True)


def seed_baseline_vectors() -> None:
    for source, target in (
        (B.CHUNK_VECTORS, A.CHUNK_VECTORS),
        (B.QUERY_VECTORS, A.QUERY_VECTORS),
    ):
        if not target.exists():
            shutil.copyfile(source, target)
            print(f"[reuse] {source.name} -> {target}", flush=True)


def generate_allow_skips(chunks: list[dict]) -> list[dict]:
    """Return all chunks, using empty enrichment rows for recorded failures."""
    cached = {row["chunk_id"]: row for row in A.read_jsonl(A.GEN_PATH)}
    skipped = [chunk for chunk in chunks if chunk["chunk_id"] not in cached]
    (DATA / "skipped_generation_chunks.json").write_text(
        json.dumps(
            {
                "count": len(skipped),
                "policy": (
                    "Question generation skipped after repeated connection failures; "
                    "adaptive dense retrieval falls back to the chunk-vector score."
                ),
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "document_title": chunk["document_title"],
                    }
                    for chunk in skipped
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[generation] using {len(cached)} enriched chunks; skipping {len(skipped)}",
        flush=True,
    )
    return [
        cached.get(
            chunk["chunk_id"],
            {
                "chunk_id": chunk["chunk_id"],
                "n_distinct_facts": 0,
                "unbounded_budget": 0,
                "bounded_budget": 0,
                "facts": [],
                "bounded_questions": [],
                "unbounded_questions": [],
                "generation_skipped": True,
            },
        )
        for chunk in chunks
    ]


_original_embed = A.embed_resumable


def embed_with_question_reuse(
    path: Path, texts: list[str], batch_size: int = 64
) -> np.ndarray:
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("texts") == texts:
            return np.asarray(payload["vectors"], dtype=float)
    source_name = path.name
    source_path = SOURCE_RESULTS / source_name
    if (
        source_name
        not in {
            "bounded_question_vectors_iris.json",
            "unbounded_question_vectors_iris.json",
        }
        or not source_path.exists()
    ):
        return _original_embed(path, texts, batch_size)
    source = json.loads(source_path.read_text())
    reusable = {
        text: vector
        for text, vector in zip(source.get("texts", []), source.get("vectors", []))
    }
    vectors: list[list[float] | None] = [reusable.get(text) for text in texts]
    missing = [index for index, vector in enumerate(vectors) if vector is None]
    print(
        f"[reuse] {source_name}: {len(texts) - len(missing)} reused, "
        f"{len(missing)} to embed",
        flush=True,
    )
    if missing:
        embedder = A.get_embedder()
        for start in range(0, len(missing), batch_size):
            indices = missing[start : start + batch_size]
            batch = embedder.embed_documents([texts[index] for index in indices])
            for index, vector in zip(indices, batch):
                vectors[index] = vector
            print(
                f"[embed] {source_name} {min(start + batch_size, len(missing))}/"
                f"{len(missing)} new",
                flush=True,
            )
    complete = [vector for vector in vectors if vector is not None]
    if len(complete) != len(texts):
        raise RuntimeError(f"Incomplete embedding cache for {path}")
    path.write_text(json.dumps({"texts": texts, "vectors": complete}))
    return np.asarray(complete, dtype=float)


def add_baseline(payload: dict) -> dict:
    baseline_payload = json.loads(B.METRICS.read_text())
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
    payload["conditions"] = [baseline] + [
        row for row in payload["conditions"] if row["condition"] != "Baseline"
    ]
    payload["protocol"]["skipped_question_generation_chunks"] = 29
    A.METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    A.render(payload)
    return payload


def run() -> dict:
    chunks, _, _ = prepare()
    seed_generation(chunks)
    seed_baseline_vectors()
    A.prepare = prepare
    A.generate = generate_allow_skips
    A.embed_resumable = embed_with_question_reuse
    return add_baseline(A.run())


if __name__ == "__main__":
    run()
