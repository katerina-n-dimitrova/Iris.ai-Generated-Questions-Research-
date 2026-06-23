"""
Build one persistent ChromaDB collection per (dataset, condition) and log
offline indexing latency + storage cost.

Embeddings are computed with the configured backend (default: the HF Octen
model, see embeddings.py) and added to Chroma as precomputed vectors so the
embedding time and the chroma-add time are measured separately.

Writes results/latency_logs/offline_indexing.csv with columns:
    dataset, condition, num_documents, num_chunks, avg_tokens_per_chunk,
    total_embedding_time_seconds, avg_embedding_time_per_chunk_ms,
    chroma_add_time_seconds, total_indexing_time_seconds, index_size_mb,
    embedding_model

Usage:
    python src/build_chroma_indexes.py
    python src/build_chroma_indexes.py --datasets scifact --conditions baseline
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

import common
import config
import embeddings

OFFLINE_LOG = config.LATENCY_LOG_DIR / "offline_indexing.csv"
OFFLINE_FIELDS = [
    "dataset", "condition", "num_documents", "num_chunks",
    "avg_tokens_per_chunk", "total_embedding_time_seconds",
    "avg_embedding_time_per_chunk_ms", "chroma_add_time_seconds",
    "total_indexing_time_seconds", "index_size_mb", "embedding_model",
]


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 4)


def _index_size_mb() -> float:
    """On-disk index size. Not measurable in cloud mode -> -1."""
    if config.CHROMA_MODE == "cloud":
        return -1.0
    return _dir_size_mb(config.CHROMA_PERSIST_DIR)


def build_collection(client, dataset: str, condition: str, batch_size: int = 256
                     ) -> dict | None:
    spec = config.DATASETS[dataset]
    path = spec.processed_path(condition)
    if not path.exists():
        print(f"  skip {dataset}/{condition}: {path.name} not found")
        return None

    records = list(common.read_jsonl(path))
    if not records:
        print(f"  skip {dataset}/{condition}: no records")
        return None

    ids = [r["id"] for r in records]
    texts = [r["text_for_embedding"] for r in records]
    metadatas = []
    for r in records:
        md = {"dataset": r["dataset"], "condition": r["condition"],
              "input_type": r["input_type"]}
        # Chroma metadata values must be scalar; copy non-null source fields.
        for k, v in r.get("metadata", {}).items():
            if v is not None:
                md[k] = v
        metadatas.append(md)

    num_chunks = len(records)
    num_documents = len({(m.get("source_id") or i) for i, m in enumerate(metadatas)})
    avg_tokens = round(
        sum(common.estimate_tokens(t) for t in texts) / num_chunks, 2)

    embedder = embeddings.get_embedder()
    coll_name = spec.collection_name(condition)

    # fresh collection each run for reproducible latency
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name, metadata={"hnsw:space": "cosine"})

    t0 = time.perf_counter()
    embed_time = 0.0
    add_time = 0.0
    for i in tqdm(range(0, num_chunks, batch_size),
                  desc=f"  {dataset}/{condition}", leave=False):
        batch_texts = texts[i:i + batch_size]
        te = time.perf_counter()
        vecs = embedder.embed_documents(batch_texts)
        embed_time += time.perf_counter() - te

        ta = time.perf_counter()
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=vecs,
            documents=batch_texts,
            metadatas=metadatas[i:i + batch_size],
        )
        add_time += time.perf_counter() - ta
    total_time = time.perf_counter() - t0

    index_size_mb = _index_size_mb()

    row = {
        "dataset": dataset,
        "condition": condition,
        "num_documents": num_documents,
        "num_chunks": num_chunks,
        "avg_tokens_per_chunk": avg_tokens,
        "total_embedding_time_seconds": round(embed_time, 4),
        "avg_embedding_time_per_chunk_ms": round(embed_time / num_chunks * 1000, 4),
        "chroma_add_time_seconds": round(add_time, 4),
        "total_indexing_time_seconds": round(total_time, 4),
        "index_size_mb": index_size_mb,
        "embedding_model": embedder.name,
    }
    print(f"  {dataset}/{condition}: {num_chunks} chunks, "
          f"{row['total_indexing_time_seconds']}s, {index_size_mb} MB")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=config.ALL_DATASETS,
                    choices=config.ALL_DATASETS)
    ap.add_argument("--conditions", nargs="*", default=list(config.CONDITIONS),
                    choices=config.CONDITIONS)
    args = ap.parse_args()

    print(f"Embedding signature: {embeddings.embedding_signature()}")
    print(f"Chroma backend:      {config.chroma_signature()}")
    client = config.get_chroma_client()

    rows: List[dict] = []
    for dataset in args.datasets:
        for condition in args.conditions:
            r = build_collection(client, dataset, condition)
            if r:
                rows.append(r)

    if rows:
        groups = {(r["dataset"], r["condition"]) for r in rows}
        common.upsert_csv(OFFLINE_LOG, OFFLINE_FIELDS, rows,
                          ("dataset", "condition"), groups)
        print(f"\nOffline indexing log -> {OFFLINE_LOG.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
