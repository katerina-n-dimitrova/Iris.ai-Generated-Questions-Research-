"""
ChromaDB index builder — multi-vector doc2query indexing.

For each experiment condition (baseline, q1, q10, q50, q100) we build a separate
Chroma collection:

  * The ORIGINAL CHUNK is always stored with its own embedding
    (record_type="chunk", parent_chunk_id == its own id).
  * Each GENERATED QUESTION is stored as its own embedding, linked back to the
    parent chunk via metadata (record_type="question", parent_chunk_id=<chunk>).
    The question text is stored as the document but at retrieval time we resolve
    every hit to its parent chunk and use the *chunk* as the evidence.

Storage layout: each condition gets its OWN on-disk persist directory
(chroma_indexes/spiqa/<collection>) so Stage 8 can measure per-condition index
size / storage overhead independently (a single shared SQLite DB would make
per-collection size unmeasurable). We force LOCAL persistence here regardless of
the shared .env CHROMA_MODE, because measuring storage requires on-disk files.

Efficiency: chunk embeddings and the full 100-question pool are embedded ONCE;
the smaller conditions reuse slices of those vectors, so we never re-embed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import spiqa_config as C
from embeddings import get_embedder, embedding_signature


def _client(collection: str):
    import chromadb

    path = C.CHROMA_DIR / collection
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _reset_collection(collection: str):
    client = _client(collection)
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    # cosine space to match normalised embeddings
    return client.create_collection(collection, metadata={"hnsw:space": "cosine"})


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 3)


def _add_in_batches(coll, ids, embeddings, documents, metadatas, batch=1000):
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )


def build_condition(
    split: str,
    n_questions: int,
    chunks: List[Dict],
    chunk_vecs: List[List[float]],
    questions_by_chunk: Dict[str, List[str]],
    qvecs_by_chunk: Dict[str, List[List[float]]],
) -> Dict:
    """
    Build one collection for a given condition.

    chunk_vecs      : embedding for chunks[i] (aligned by index).
    questions_by_chunk / qvecs_by_chunk : the FULL question pool + vectors per
                      chunk_id; we slice the first n_questions for this condition.
    """
    collection = C.collection_name(split, n_questions)
    coll = _reset_collection(collection)

    ids: List[str] = []
    embs: List[List[float]] = []
    docs: List[str] = []
    metas: List[Dict] = []

    n_chunk_records = 0
    n_question_records = 0

    for c, cvec in zip(chunks, chunk_vecs):
        cid = c["chunk_id"]
        base_meta = {
            "record_type": "chunk",
            "parent_chunk_id": cid,
            "paper_id": c["paper_id"],
            "split": c["split"],
            "source_type": c["source_type"],
        }
        # Always store the original chunk vector.
        ids.append(cid)
        embs.append(cvec)
        docs.append(c["text"])
        metas.append(dict(base_meta))
        n_chunk_records += 1

        # Store generated-question vectors (first n of the pool) for this chunk.
        if n_questions > 0:
            qs = questions_by_chunk.get(cid, [])[:n_questions]
            qvs = qvecs_by_chunk.get(cid, [])[:n_questions]
            for j, (qtext, qvec) in enumerate(zip(qs, qvs)):
                ids.append(f"{cid}::q{j}")
                embs.append(qvec)
                docs.append(qtext)
                metas.append(
                    {
                        "record_type": "question",
                        "parent_chunk_id": cid,
                        "paper_id": c["paper_id"],
                        "split": c["split"],
                        "source_type": c["source_type"],
                    }
                )
                n_question_records += 1

    t0 = time.perf_counter()
    _add_in_batches(coll, ids, embs, docs, metas)
    add_secs = time.perf_counter() - t0

    return {
        "condition": C.condition_name(n_questions),
        "collection": collection,
        "n_questions_per_chunk": n_questions,
        "n_chunk_records": n_chunk_records,
        "n_question_records": n_question_records,
        "total_records": len(ids),
        "chroma_add_seconds": round(add_secs, 3),
        "index_size_mb": _dir_size_mb(C.CHROMA_DIR / collection),
    }


def build_all(
    split: str,
    chunks: List[Dict],
    all_questions: Dict[str, List[str]],
    conditions: Optional[List[int]] = None,
) -> Dict:
    """
    Embed chunks + question pool once, then build every requested condition.
    Returns {"embedding_model":..., "encode_seconds":..., "conditions":[...]}.
    """
    conditions = conditions if conditions is not None else C.QUESTION_CONDITIONS
    embedder = get_embedder()

    # 1) Embed all chunk texts once.
    t0 = time.perf_counter()
    chunk_vecs = embedder.embed_documents([c["text"] for c in chunks])
    chunk_encode_secs = time.perf_counter() - t0

    # 2) Embed the full question pool once (flattened), keep per-chunk slices.
    max_n = max(conditions) if conditions else 0
    flat_q: List[str] = []
    owner: List[str] = []
    for c in chunks:
        qs = all_questions.get(c["chunk_id"], [])[:max_n]
        flat_q.extend(qs)
        owner.extend([c["chunk_id"]] * len(qs))

    t1 = time.perf_counter()
    flat_qvecs = embedder.embed_documents(flat_q) if flat_q else []
    q_encode_secs = time.perf_counter() - t1

    qvecs_by_chunk: Dict[str, List[List[float]]] = {}
    questions_by_chunk: Dict[str, List[str]] = {}
    for cid, qt, qv in zip(owner, flat_q, flat_qvecs):
        qvecs_by_chunk.setdefault(cid, []).append(qv)
        questions_by_chunk.setdefault(cid, []).append(qt)

    results = []
    for n in conditions:
        r = build_condition(
            split, n, chunks, chunk_vecs, questions_by_chunk, qvecs_by_chunk
        )
        results.append(r)
        print(
            f"  built {r['collection']}: {r['total_records']} records "
            f"({r['n_question_records']} question vecs) "
            f"{r['index_size_mb']} MB",
            flush=True,
        )

    return {
        "split": split,
        "embedding_model": embedding_signature(),
        "chunk_encode_seconds": round(chunk_encode_secs, 2),
        "question_encode_seconds": round(q_encode_secs, 2),
        "num_chunks": len(chunks),
        "num_question_vectors_embedded": len(flat_q),
        "conditions": results,
    }


def get_collection(split: str, n_questions: int):
    """Open an existing collection for querying."""
    collection = C.collection_name(split, n_questions)
    return _client(collection).get_collection(collection)


if __name__ == "__main__":
    import argparse, json
    from spiqa_chunker import load_chunks
    from question_gen import load_questions

    ap = argparse.ArgumentParser(
        description="Build Chroma collections for all conditions."
    )
    ap.add_argument("--split", default=C.DEFAULT_SPLIT)
    ap.add_argument(
        "--conditions", default=None, help="comma list e.g. 0,1,10 (default: config)"
    )
    args = ap.parse_args()
    conds = [int(x) for x in args.conditions.split(",")] if args.conditions else None

    chunks = load_chunks(args.split)
    qs = load_questions(args.split)
    summary = build_all(args.split, chunks, qs, conds)
    print(json.dumps(summary, indent=2))
