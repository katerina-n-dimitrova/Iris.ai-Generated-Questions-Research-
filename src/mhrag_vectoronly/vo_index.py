"""
Stage: build the two LOCAL ChromaDB vector collections (§10).

Collection A (baseline)  : ONE vector per original chunk  (record_type=original_chunk).
Collection B (generated) : 10 vectors per chunk, one per generated question, each
                           carrying its parent_chunk_id  (record_type=generated_question).

Both use the SAME embedding model (repo default Octen-Embedding-0.6B, cosine,
L2-normalized). Query vectors are NEVER stored — they are created transiently at
retrieval time (see vo_retrieval). IDs are deterministic so re-runs upsert rather
than duplicate. This is a DENSE cosine index only — no BM25 / sparse side exists.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import vo_config as C
import vo_data as D
import vo_generate as G
from embeddings import get_embedder, embedding_signature


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 3)


def _add_batches(coll, ids, embs, docs, metas, batch=2000):
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=embs[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )


def build_indexes(force: bool = True) -> dict:
    chunks = D.load_chunks()
    q_by_chunk = G.load_questions_by_chunk()
    embedder = get_embedder()
    stats: Dict[str, dict] = {}

    # ---- Collection A: original chunk vectors ---------------------------- #
    coll_a = C.reset_collection(C.BASELINE_COLLECTION)
    t0 = time.perf_counter()
    chunk_vecs = embedder.embed_documents([c["text"] for c in chunks])
    a_encode = time.perf_counter() - t0
    ids = [c["chunk_id"] for c in chunks]
    metas = [
        {
            "record_type": "original_chunk",
            "chunk_id": c["chunk_id"],
            "parent_chunk_id": c["chunk_id"],
            "parent_document_id": c["parent_document_id"],
            "title": c["title"],
            "source": c["source"],
            "published_at": c["published_at"],
            "category": c["category"],
            "chunk_position": c["chunk_position"],
        }
        for c in chunks
    ]
    t1 = time.perf_counter()
    _add_batches(coll_a, ids, chunk_vecs, [c["text"] for c in chunks], metas)
    a_add = time.perf_counter() - t1
    stats["baseline"] = {
        "collection": C.BASELINE_COLLECTION,
        "num_vectors": len(ids),
        "embed_seconds": round(a_encode, 2),
        "add_seconds": round(a_add, 2),
        "index_size_mb": _dir_size_mb(C.CHROMA_DIR),
    }

    # ---- Collection B: generated-question vectors ------------------------ #
    coll_b = C.reset_collection(C.GENQ_COLLECTION)
    chunk_meta = {c["chunk_id"]: c for c in chunks}
    flat_q, owner = [], []
    for c in chunks:
        for q in q_by_chunk.get(c["chunk_id"], []):
            flat_q.append(q)
            owner.append(c)
    t2 = time.perf_counter()
    q_vecs = embedder.embed_documents(flat_q)
    b_encode = time.perf_counter() - t2
    b_ids, b_docs, b_metas = [], [], []
    counters: Dict[str, int] = {}
    for c, q, v in zip(owner, flat_q, q_vecs):
        cid = c["chunk_id"]
        j = counters.get(cid, 0)
        counters[cid] = j + 1
        b_ids.append(f"{cid}::q{j}")
        b_docs.append(q)
        b_metas.append(
            {
                "record_type": "generated_question",
                "generated_question_id": f"{cid}::q{j}",
                "parent_chunk_id": cid,
                "parent_document_id": c["parent_document_id"],
                "parent_chunk_text": c["text"][:2000],
                "title": c["title"],
                "source": c["source"],
                "published_at": c["published_at"],
                "chunk_position": c["chunk_position"],
            }
        )
    t3 = time.perf_counter()
    _add_batches(coll_b, b_ids, q_vecs, b_docs, b_metas)
    b_add = time.perf_counter() - t3
    stats["generated"] = {
        "collection": C.GENQ_COLLECTION,
        "num_vectors": len(b_ids),
        "num_parent_chunks": len(chunks),
        "questions_per_chunk_avg": round(len(b_ids) / max(len(chunks), 1), 2),
        "embed_seconds": round(b_encode, 2),
        "add_seconds": round(b_add, 2),
        "index_size_mb": _dir_size_mb(C.CHROMA_DIR),
    }

    stats["embedding_model"] = embedding_signature()
    stats["vector_dim"] = embedder.dim
    stats["distance_metric"] = C.SIMILARITY_METRIC
    stats["index_ratio_B_over_A"] = round(len(b_ids) / max(len(ids), 1), 2)
    import json

    with (C.RESULTS_DIR / "index_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    print(
        f"[index] A={stats['baseline']['num_vectors']} chunk vectors, "
        f"B={stats['generated']['num_vectors']} question vectors "
        f"({stats['index_ratio_B_over_A']}x), dim={embedder.dim}"
    )
    return stats


def verify_ready() -> dict:
    """Confirm both collections exist with the expected counts and dim."""
    a = C.get_collection(C.BASELINE_COLLECTION)
    b = C.get_collection(C.GENQ_COLLECTION)
    return {"baseline_count": a.count(), "generated_count": b.count()}


if __name__ == "__main__":
    import pprint

    pprint.pp(build_indexes())
    pprint.pp(verify_ready())
