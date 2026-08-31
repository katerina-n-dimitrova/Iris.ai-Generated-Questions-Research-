"""
Index building for an arm: the dense (Chroma) side and the sparse (BM25) side.

Fixed-factor invariant (see qasper_config)
------------------------------------------
* Enrichment arms embed ONLY generated-question vectors into the dense index
  (each question a separate vector pointing to its parent chunk). The chunk text
  is NOT embedded. Only B0 (and the future Exp-4 variant f) embed chunk vectors.
* BM25 runs over the chunk text, with the generated questions appended to the
  chunk's BM25 document in enrichment arms.

The dense vectors are embedded once and reused; ``validate_arm`` guards the
invariant before anything is embedded.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import qasper_config as C
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


# Canonical keyword-generation arm reused by keyword-BM25 placements (Exp 4).
KEYWORD_ARM = "E4b"


def _budget(items):
    return items[: C.QUESTION_BUDGET]


def build_dense_index(
    arm_name: str, chunks: List[dict], questions: Dict[str, List[str]] = None
) -> dict:
    """Build the arm's Chroma dense collection per its dense_mode.

    Modes: chunk (one chunk vector), questions (NL question vectors from the arm's
    source), qa_pairs (Q:..A:.. vectors), concat_single (one vector of chunk+questions,
    doc2query style), chunk+questions (both). Content is loaded from the generation
    caches by arm/source, so the ``questions`` arg is optional/ignored."""
    import qasper_generate as G

    arm = C.ARMS[arm_name]
    C.validate_arm(arm)
    mode = C.dense_mode(arm)
    embedder = get_embedder()
    coll = _reset(C.collection_name(arm_name))
    src_q = G.load_questions(C.source_arm(arm))  # NL questions (may be self or B1)

    ids: List[str] = []
    embs = []
    docs = []
    metas = []
    n_chunk = n_q = 0

    def add_chunk_vectors(text_of):
        nonlocal n_chunk
        texts = [text_of(c) for c in chunks]
        vecs = embedder.embed_documents(texts)
        for c, v, txt in zip(chunks, vecs, texts):
            ids.append(f"{c['chunk_id']}::chunk")
            embs.append(v)
            docs.append(txt)
            metas.append(
                {
                    "record_type": "chunk",
                    "parent_chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                }
            )
        n_chunk = len(vecs)

    def add_multi_vectors(items_by_chunk, rtype):
        nonlocal n_q
        flat, owner = [], []
        for c in chunks:
            its = _budget(items_by_chunk.get(c["chunk_id"], []))
            flat.extend(its)
            owner.extend([c] * len(its))
        vecs = embedder.embed_documents(flat) if flat else []
        cnt = {}
        for c, txt, v in zip(owner, flat, vecs):
            cid = c["chunk_id"]
            j = cnt.get(cid, 0)
            cnt[cid] = j + 1
            ids.append(f"{cid}::{rtype[0]}{j}")
            embs.append(v)
            docs.append(txt)
            metas.append(
                {
                    "record_type": rtype,
                    "parent_chunk_id": cid,
                    "paper_id": c["paper_id"],
                }
            )
        n_q += len(vecs)

    t0 = time.perf_counter()
    if mode == "chunk":
        add_chunk_vectors(lambda c: c["text"])
    elif mode == "questions":
        add_multi_vectors(src_q, "question")
    elif mode == "qa_pairs":
        add_multi_vectors(G.load_questions(arm_name), "qa")
    elif mode == "concat_single":
        add_chunk_vectors(
            lambda c: (
                c["text"] + " " + " ".join(_budget(src_q.get(c["chunk_id"], [])))
            ).strip()
        )
    elif mode == "chunk+questions":
        add_chunk_vectors(lambda c: c["text"])
        add_multi_vectors(src_q, "question")
    else:
        raise ValueError(f"unknown dense_mode {mode!r} for arm {arm_name}")

    encode_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    _add_batches(coll, ids, embs, docs, metas)
    add_s = time.perf_counter() - t1
    return {
        "arm": arm_name,
        "dense_mode": mode,
        "embedding_model": embedding_signature(),
        "num_vectors": len(ids),
        "num_chunk_vectors": n_chunk,
        "num_question_vectors": n_q,
        "encode_seconds": round(encode_s, 2),
        "chroma_add_seconds": round(add_s, 2),
        "index_size_mb": _dir_size_mb(C.CHROMA_DIR / C.collection_name(arm_name)),
    }


# --------------------------------------------------------------------------- #
# Sparse index (BM25) — mode-driven document construction
# --------------------------------------------------------------------------- #
def build_bm25(
    arm_name: str, chunks: List[dict], questions: Dict[str, List[str]] = None
) -> Tuple[object, List[str]]:
    """BM25 over documents per the arm's bm25_mode: chunk (text only),
    chunk+questions (text + source-arm questions), chunk+keywords (text + keyword
    variants from the canonical keyword arm)."""
    from rank_bm25 import BM25Okapi
    import qasper_generate as G

    arm = C.ARMS[arm_name]
    mode = C.bm25_mode(arm)
    src_q = G.load_questions(C.source_arm(arm)) if mode == "chunk+questions" else {}
    kws = G.load_questions(KEYWORD_ARM) if mode == "chunk+keywords" else {}
    chunk_ids, corpus = [], []
    for c in chunks:
        cid = c["chunk_id"]
        text = c["text"]
        if mode == "chunk+questions":
            extra = _budget(src_q.get(cid, []))
            if extra:
                text += " " + " ".join(extra)
        elif mode == "chunk+keywords":
            extra = _budget(kws.get(cid, []))
            if extra:
                text += " " + " ".join(extra)
        chunk_ids.append(cid)
        corpus.append(tokenize(text))
    return BM25Okapi(corpus), chunk_ids
