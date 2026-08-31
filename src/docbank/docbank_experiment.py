"""
DocBank index + retrieval-eval engine.

Reuses the PeerQA doc2query generation (peerqa_experiment.generate_questions,
cache_path + grounded prompt) and the path-agnostic metric/retrieval helpers,
but builds its OWN Chroma collections under an isolated docbank persist dir:

  * baseline   : embed the CHUNK TEXT (one vector/chunk) — reference.
  * q5/q10/q13/q15 : embed ONLY generated questions (n vectors/chunk, no chunk
    vector); each question maps back to parent_chunk_id.
  * fused      : chunk vector + question vectors together (score fusion at query).

Queries are the synthetic eval questions; gold = the single gold_chunk_id.
"""

from __future__ import annotations

import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional

import docbank_config as C
import peerqa_experiment as E  # generation + path-agnostic helpers
from embeddings import get_embedder, embedding_signature

MAXQ = C.MAX_QUESTIONS


def _prompt_grounded(chunk_text: str, n: int):
    system = (
        "You generate retrieval questions for a scientific-document RAG system. "
        "Given a passage (which may be a paragraph, table, equation, or caption "
        "with context), output natural questions each fully answerable from the "
        "passage alone. Ground every question in the passage — never introduce "
        "facts, numbers, symbols, or entities not present; name the specific "
        "entity / quantity / equation / table the question targets; cover different "
        "facts; no duplicates. Output ONLY a numbered list, one question per line."
    )
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Write up to {n} distinct grounded questions. If it genuinely supports "
        f"fewer than {n}, output only as many as it supports."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


QCACHE = C.PROCESSED_DIR / "enrichment_questions.jsonl"


def generate_enrichment(chunks: List[Dict], *, limit: Optional[int] = None) -> Dict:
    return E.generate_questions(
        chunks,
        n=MAXQ,
        cache_path=QCACHE,
        prompt_fn=_prompt_grounded,
        limit=limit,
        max_workers=C.LLM_WORKERS,
    )


def load_enrichment() -> Dict[str, List[str]]:
    return E.load_questions(cache_path=QCACHE)


# --------------------------------------------------------------------------- #
# Chroma (isolated docbank dir)
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


def embed_all(chunks: List[Dict], all_q: Dict[str, List[str]], embedder):
    t0 = time.perf_counter()
    chunk_vecs = embedder.embed_documents([c["text"] for c in chunks])
    cs = time.perf_counter() - t0
    flat_q, owner = [], []
    for c in chunks:
        qs = all_q.get(c["chunk_id"], [])[:MAXQ]
        flat_q.extend(qs)
        owner.extend([c["chunk_id"]] * len(qs))
    t1 = time.perf_counter()
    qv = embedder.embed_documents(flat_q) if flat_q else []
    qs_s = time.perf_counter() - t1
    qvec_by, qtext_by = {}, {}
    for cid, t, v in zip(owner, flat_q, qv):
        qvec_by.setdefault(cid, []).append(v)
        qtext_by.setdefault(cid, []).append(t)
    return chunk_vecs, qvec_by, qtext_by, round(cs, 2), round(qs_s, 2), len(flat_q)


def build_condition(
    n: int, chunks, chunk_vecs, qtext_by, qvec_by, *, fused=False
) -> Dict:
    name = C.collection_name(n) if not fused else "docbank_fused"
    coll = _reset(name)
    ids, embs, docs, metas = [], [], [], []
    nq_total = 0
    for c, cv in zip(chunks, chunk_vecs):
        cid = c["chunk_id"]
        if n == 0 or fused:  # store chunk vector
            ids.append(cid)
            embs.append(cv)
            docs.append(c["text"])
            metas.append({"record_type": "chunk", "parent_chunk_id": cid})
        if n > 0 or fused:  # store question vectors
            k = MAXQ if fused else n
            for j, (qt, qv) in enumerate(
                zip(qtext_by.get(cid, [])[:k], qvec_by.get(cid, [])[:k])
            ):
                ids.append(f"{cid}::q{j}")
                embs.append(qv)
                docs.append(qt)
                metas.append({"record_type": "question", "parent_chunk_id": cid})
                nq_total += 1
    t0 = time.perf_counter()
    if ids:
        E._add_batches(coll, ids, embs, docs, metas)
    add_s = time.perf_counter() - t0
    cond = "fused" if fused else C.condition_name(n)
    return {
        "condition": cond,
        "n_questions_per_chunk": (MAXQ if fused else n),
        "coll": coll,
        "num_embeddings": len(ids),
        "num_questions": nq_total,
        "index_add_s": round(add_s, 3),
        "index_size_mb": E._dir_size_mb(C.CHROMA_DIR / name),
        "record_type": (
            "chunk_text"
            if n == 0 and not fused
            else ("fused" if fused else "generated_questions")
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation (single gold chunk per query)
# --------------------------------------------------------------------------- #
def evaluate(
    coll, queries: List[Dict], *, fused=False, k_values=None, overfetch_factor=30
) -> Dict:
    from chunksize_experiment import _fused_parents

    k_values = k_values or C.K_VALUES
    embedder = get_embedder()
    overfetch = max(max(k_values) * overfetch_factor, 150)
    per_query, emb_ms, search_ms = [], [], []
    for q in queries:
        gold = {q["gold_chunk_id"]}
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        ranked = (
            _fused_parents(coll, qvec, max(k_values), overfetch)
            if fused
            else E._retrieve_parents(coll, qvec, max(k_values), overfetch)
        )
        search_ms.append((time.perf_counter() - t1) * 1000)
        per_query.append(E._query_metrics(ranked, gold, k_values))

    def avg(k):
        return round(sum(x[k] for x in per_query) / max(len(per_query), 1), 4)

    def pct(v, p):
        if not v:
            return 0.0
        v = sorted(v)
        return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 3)

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {
        "metrics": metrics,
        "num_queries": len(queries),
        "per_query": per_query,
        "latency_ms": {
            "query_embed_mean": round(st.mean(emb_ms), 3) if emb_ms else 0,
            "search_mean": round(st.mean(search_ms), 3) if search_ms else 0,
            "search_p95": pct(search_ms, 95),
            "total_p95": round(pct(emb_ms, 95) + pct(search_ms, 95), 3),
        },
    }
