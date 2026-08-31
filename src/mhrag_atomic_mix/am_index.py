"""
Stage: build the two LOCAL ChromaDB collections (§7, §16).

Collection A (baseline)      : one vector per original chunk. Also used by the
                              round-trip / confusion-margin filters as the "parent
                              retrieval" index.
Collection E (mixed)         : one vector per ACCEPTED atomic/chunk-level question,
                              each carrying its parent_chunk_id. Parent chunk text
                              is metadata only — never embedded here.

Same embedding model as the benchmark queries (Octen, cosine, L2-normalized).
Dense only. Deterministic IDs so re-runs upsert rather than duplicate.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import am_config as C
import am_data as D
from embeddings import get_embedder, embedding_signature


def _add_batches(coll, ids, embs, docs, metas, batch=2000):
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=embs[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )


def _dir_size_mb() -> float:
    total = sum(f.stat().st_size for f in C.CHROMA_DIR.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 3)


def build_baseline_index(chunks: List[dict] = None) -> dict:
    chunks = chunks or D.load_chunks()
    embedder = get_embedder()
    coll = C.reset_collection(C.BASELINE_COLLECTION)
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([c["text"] for c in chunks])
    enc = time.perf_counter() - t0
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
    _add_batches(
        coll, [c["chunk_id"] for c in chunks], vecs, [c["text"] for c in chunks], metas
    )
    return {
        "collection": C.BASELINE_COLLECTION,
        "num_vectors": len(chunks),
        "embed_seconds": round(enc, 2),
    }


def build_mixed_index(accepted: List[dict] = None) -> dict:
    accepted = accepted if accepted is not None else D.read_jsonl(C.QUESTIONS_FILTERED)
    embedder = get_embedder()
    coll = C.reset_collection(C.MIXED_COLLECTION)
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([q["question"] for q in accepted])
    enc = time.perf_counter() - t0
    ids, docs, metas = [], [], []
    for q in accepted:
        ids.append(q["question_id"])
        docs.append(q["question"])
        metas.append(
            {
                "record_type": "generated_question",
                "question_type": q["question_type"],
                "question_view": q.get("question_view", ""),
                "atom_id": q.get("atom_id") or "",
                "parent_chunk_id": q["parent_chunk_id"],
                "parent_document_id": q["parent_document_id"],
                "parent_chunk_text": q.get("parent_chunk_text", "")[:2000],
                "title": q.get("title", ""),
                "source": q.get("source", ""),
                "published_at": q.get("published_at", ""),
                "chunk_position": q.get("chunk_position", 0),
            }
        )
    _add_batches(coll, ids, vecs, docs, metas)
    n_atomic = sum(1 for q in accepted if q["question_type"] == "atomic")
    stats = {
        "collection": C.MIXED_COLLECTION,
        "num_vectors": len(accepted),
        "num_atomic": n_atomic,
        "num_chunk_level": len(accepted) - n_atomic,
        "embed_seconds": round(enc, 2),
    }
    return stats


def build_all() -> dict:
    chunks = D.load_chunks()
    a = build_baseline_index(chunks)
    e = build_mixed_index()
    n_chunks = len(chunks)
    stats = {
        "baseline": {**a, "index_size_mb_after": None},
        "mixed": e,
        "embedding_model": embedding_signature(),
        "vector_dim": get_embedder().dim,
        "distance_metric": C.SIMILARITY_METRIC,
        "num_parent_chunks": n_chunks,
        "index_ratio_E_over_A": round(e["num_vectors"] / max(n_chunks, 1), 2),
        "chroma_store_mb": _dir_size_mb(),
    }
    json.dump(stats, open(C.INDEX_STATS, "w"), indent=2)
    print(
        f"[index] A={a['num_vectors']} chunk vecs, E={e['num_vectors']} question vecs "
        f"(atomic {e['num_atomic']}/chunk-level {e['num_chunk_level']}, "
        f"{stats['index_ratio_E_over_A']}×), dim={stats['vector_dim']}"
    )
    return stats


def verify_ready() -> dict:
    return {
        "baseline": C.get_collection(C.BASELINE_COLLECTION).count(),
        "mixed": C.get_collection(C.MIXED_COLLECTION).count(),
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(build_all())
    pprint.pp(verify_ready())
