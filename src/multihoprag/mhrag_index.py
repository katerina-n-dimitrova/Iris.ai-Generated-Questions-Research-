"""
Index building for an arm: the dense (Chroma) side and the sparse (BM25) side.

Fixed-factor invariant (see mhrag_config)
-----------------------------------------
* Enrichment arms embed ONLY generated-question vectors into the dense index
  (each question a separate vector pointing to its parent chunk). The chunk text
  is NOT embedded. Only B0 (and the Exp-4 variants b/e/f) embed chunk vectors.
* BM25 runs over the chunk text, with the generated questions appended to the
  chunk's BM25 document in enrichment arms (bm25_appends_questions).

A dedicated LOCAL Chroma PersistentClient is used regardless of the parent
project's CHROMA_MODE (which points at Chroma Cloud). ``validate_arm`` guards the
invariant before anything is embedded.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mhrag_config as C
from embeddings import get_embedder, embedding_signature

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Dense index (Chroma, cosine)
# --------------------------------------------------------------------------- #
def _client(collection: str):
    import chromadb

    path = C.CHROMA_DIR / collection
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _reset(collection: str):
    client = _client(collection)
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    return client.create_collection(collection, metadata={"hnsw:space": "cosine"})


def get_dense_collection(arm_name: str):
    name = C.collection_name(arm_name)
    return _client(name).get_collection(name)


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


def build_dense_index(
    arm_name: str, chunks: List[dict], questions: Dict[str, List[str]]
) -> dict:
    """Build the arm's Chroma collection following the fixed-factor invariant.

    For the Exp-4 doc2query variant (e) the chunk vector is the chunk text with
    its questions appended (one combined vector, ``embeds_chunk`` + no separate
    question vectors); handled by the arm flags -- this builder is generic."""
    arm = C.ARMS[arm_name]
    C.validate_arm(arm)
    embedder = get_embedder()
    coll = _reset(C.collection_name(arm_name))

    ids: List[str] = []
    embs: List[List[float]] = []
    docs: List[str] = []
    metas: List[dict] = []

    t0 = time.perf_counter()
    n_chunk_vecs = n_q_vecs = 0

    if arm.embeds_chunk:
        # Exp-4 (e) appends questions to the chunk text before embedding it as ONE
        # vector; all other chunk-embedding arms embed the raw chunk text.
        concat = getattr(arm, "concat_questions_into_chunk", False)
        texts = []
        for c in chunks:
            t = c["text"]
            if concat:
                qs = questions.get(c["chunk_id"], [])[: C.QUESTION_BUDGET]
                if qs:
                    t = t + " " + " ".join(qs)
            texts.append(t)
        vecs = embedder.embed_documents(texts)
        for c, v, t in zip(chunks, vecs, texts):
            ids.append(f"{c['chunk_id']}::chunk")
            embs.append(v)
            docs.append(t)
            metas.append(
                {
                    "record_type": "chunk",
                    "parent_chunk_id": c["chunk_id"],
                    "article_id": c["article_id"],
                }
            )
        n_chunk_vecs = len(vecs)

    if arm.embeds_questions:
        flat_q, owner = [], []
        for c in chunks:
            qs = questions.get(c["chunk_id"], [])[: C.QUESTION_BUDGET]
            flat_q.extend(qs)
            owner.extend([c] * len(qs))
        qvecs = embedder.embed_documents(flat_q) if flat_q else []
        counters: Dict[str, int] = {}
        for c, qt, qv in zip(owner, flat_q, qvecs):
            cid = c["chunk_id"]
            j = counters.get(cid, 0)
            counters[cid] = j + 1
            ids.append(f"{cid}::q{j}")
            embs.append(qv)
            docs.append(qt)
            metas.append(
                {
                    "record_type": "question",
                    "parent_chunk_id": cid,
                    "article_id": c["article_id"],
                }
            )
        n_q_vecs = len(qvecs)

    encode_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    _add_batches(coll, ids, embs, docs, metas)
    add_s = time.perf_counter() - t1

    return {
        "arm": arm_name,
        "embedding_model": embedding_signature(),
        "num_vectors": len(ids),
        "num_chunk_vectors": n_chunk_vecs,
        "num_question_vectors": n_q_vecs,
        "encode_seconds": round(encode_s, 2),
        "chroma_add_seconds": round(add_s, 2),
        "index_size_mb": _dir_size_mb(C.CHROMA_DIR / C.collection_name(arm_name)),
    }


# --------------------------------------------------------------------------- #
# Sparse index (BM25 over chunk text [+ appended questions])
# --------------------------------------------------------------------------- #
def build_bm25(
    arm_name: str, chunks: List[dict], questions: Dict[str, List[str]]
) -> Tuple[object, List[str]]:
    """Return (BM25Okapi, chunk_ids) where chunk_ids[i] owns tokenized_docs[i].

    B0: document = chunk text. Enrichment arms with bm25_appends_questions:
    document = chunk text + its generated questions. (Exp-4 variants may override
    the BM25 document via ``bm25_text_source`` -- generic here.)"""
    from rank_bm25 import BM25Okapi

    arm = C.ARMS[arm_name]
    source = getattr(arm, "bm25_text_source", "chunk")  # 'chunk' | 'questions'
    chunk_ids, corpus = [], []
    for c in chunks:
        qs = questions.get(c["chunk_id"], [])[: C.QUESTION_BUDGET]
        if source == "questions":
            text = " ".join(qs) if qs else c["text"]
        else:
            text = c["text"]
            if arm.bm25_appends_questions and qs:
                text = text + " " + " ".join(qs)
        chunk_ids.append(c["chunk_id"])
        corpus.append(tokenize(text))
    return BM25Okapi(corpus), chunk_ids
