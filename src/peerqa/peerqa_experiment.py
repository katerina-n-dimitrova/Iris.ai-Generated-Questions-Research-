"""
PeerQA generated-questions-only retrieval experiment — engine.

Pipeline
--------
1. Load + chunk PeerQA (peerqa_data).
2. Generate up to MAX (13) questions per chunk with the LLM (cached, resumable).
   Smaller conditions (5, 10) are the first-k subset of that pool.
3. Build one Chroma collection per condition:
     * baseline : embed the CHUNK TEXT (one vector/chunk)  — reference only.
     * q5/q10/q13 : embed ONLY the generated questions (n vectors/chunk, no
       chunk vector); each question links back to its parent_chunk_id.
4. Evaluate: real PeerQA questions as queries; a question hit resolves to its
   parent chunk (the returned evidence). Metrics: Hit@1/5/10, MRR, nDCG@10,
   Recall@k. Also latency, #questions, #embeddings, index size.

Everything is deterministic + cached so re-runs are cheap.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import peerqa_config as C
from embeddings import get_embedder, embedding_signature

import config as base_config

_NUM_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+?)\s*$")


# --------------------------------------------------------------------------- #
# LLM question generation (doc2query), resumable disk cache
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _prompt(chunk_text: str, n: int) -> List[Dict[str, str]]:
    system = (
        "You generate retrieval questions for a scientific-paper RAG system. "
        "Given a passage, output questions that are SPECIFIC and FULLY answerable "
        "using ONLY the information in the passage. Never introduce facts, numbers, "
        "or entities not present in the passage. Prefer diverse questions covering "
        "different facts, methods, results, and definitions. Output ONLY a numbered "
        "list, one question per line, no preamble."
    )
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Write up to {n} distinct questions answerable solely from this passage. "
        f"If it cannot support {n}, output as many high-quality ones as it genuinely supports."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse(text: str) -> List[str]:
    out, seen = [], set()
    for line in text.splitlines():
        m = _NUM_LINE.match(line)
        cand = (m.group(1) if m else line).strip()
        if not cand or "?" not in cand:
            continue
        cand = cand[: cand.rindex("?") + 1].strip()
        if cand.lower() in seen:
            continue
        seen.add(cand.lower())
        out.append(cand)
    return out


def _questions_path() -> Path:
    return C.PROCESSED_DIR / "questions.jsonl"


def _load_qcache(path: Path) -> Dict[str, dict]:
    cache = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[row["chunk_id"]] = row
    return cache


def generate_questions(
    chunks: List[Dict],
    *,
    n: int = C.MAX_QUESTIONS,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
    cache_path: Optional[Path] = None,
    prompt_fn=None,
) -> Dict:
    """Generate up to n questions per chunk (cached, resumable).

    cache_path : override the JSONL cache location (default questions.jsonl).
    prompt_fn  : (chunk_text, n) -> messages; defaults to the module _prompt.
    """
    client = base_config.get_openai_client()
    path = cache_path or _questions_path()
    prompt_fn = prompt_fn or _prompt
    cache = _load_qcache(path)
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    total = Usage()
    gen_secs = 0.0
    t_wall = time.perf_counter()

    def work(c):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=C.LLM_MODEL,
            messages=prompt_fn(c["text"], n),
            temperature=C.LLM_TEMPERATURE,
            max_tokens=min(4000, max(256, n * 22)),
        )
        secs = time.perf_counter() - t0
        txt = resp.choices[0].message.content or ""
        u = resp.usage
        return (
            c["chunk_id"],
            _parse(txt)[:n],
            secs,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
        )

    fh = path.open("a", encoding="utf-8")
    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futures):
                cid, qs, secs, pt, ct = fut.result()
                total.prompt_tokens += pt
                total.completion_tokens += ct
                gen_secs += secs
                row = {
                    "chunk_id": cid,
                    "n_requested": n,
                    "n_generated": len(qs),
                    "questions": qs,
                    "gen_seconds": round(secs, 3),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[cid] = row
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(
                        f"  [gen] {done}/{len(todo)} (cache {len(cache)})", flush=True
                    )
    finally:
        fh.close()

    wall = time.perf_counter() - t_wall
    per_chunk = [
        cache[c["chunk_id"]]["n_generated"] for c in chunks if c["chunk_id"] in cache
    ]
    return {
        "requested_per_chunk": n,
        "chunks_total": len(chunks),
        "chunks_newly_generated": len(todo),
        "questions_generated_total": sum(per_chunk),
        "avg_questions_per_chunk": round(sum(per_chunk) / max(len(per_chunk), 1), 2),
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "gen_seconds_sum": round(gen_secs, 1),
        "wall_seconds": round(wall, 1),
        "estimated_cost_usd": _estimate_cost(
            total.prompt_tokens, total.completion_tokens
        ),
    }


def load_questions(cache_path: Optional[Path] = None) -> Dict[str, List[str]]:
    path = cache_path or _questions_path()
    out: Dict[str, List[str]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = row["questions"]
    return out


def _estimate_cost(pt: int, ct: int, model: str = C.LLM_MODEL) -> float:
    price = C.PRICE_PER_1M.get(model)
    if not price:
        return -1.0
    return round(pt / 1e6 * price["input"] + ct / 1e6 * price["output"], 4)


# --------------------------------------------------------------------------- #
# Index building (per-condition on-disk Chroma collections)
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


def get_collection(n_questions: int):
    name = C.collection_name(n_questions)
    return _client(name).get_collection(name)


def build_all(
    chunks: List[Dict], all_questions: Dict[str, List[str]], conditions: List[int]
) -> Dict:
    """Embed chunk texts + question pool ONCE, then build each condition."""
    embedder = get_embedder()

    # Baseline chunk-text vectors (only needed if 0 in conditions).
    chunk_vecs = None
    chunk_encode_s = 0.0
    if 0 in conditions:
        t0 = time.perf_counter()
        chunk_vecs = embedder.embed_documents([c["text"] for c in chunks])
        chunk_encode_s = time.perf_counter() - t0

    # Question pool (flattened) — embed once, keep per-chunk slices.
    max_n = max([n for n in conditions if n > 0], default=0)
    flat_q, owner = [], []
    for c in chunks:
        qs = all_questions.get(c["chunk_id"], [])[:max_n]
        flat_q.extend(qs)
        owner.extend([c["chunk_id"]] * len(qs))
    t1 = time.perf_counter()
    flat_qvecs = embedder.embed_documents(flat_q) if flat_q else []
    q_encode_s = time.perf_counter() - t1

    qvecs_by_chunk: Dict[str, List[List[float]]] = {}
    qtext_by_chunk: Dict[str, List[str]] = {}
    for cid, qt, qv in zip(owner, flat_q, flat_qvecs):
        qvecs_by_chunk.setdefault(cid, []).append(qv)
        qtext_by_chunk.setdefault(cid, []).append(qt)

    results = []
    for n in conditions:
        name = C.collection_name(n)
        coll = _reset(name)
        ids, embs, docs, metas = [], [], [], []
        if n == 0:
            for c, cv in zip(chunks, chunk_vecs):
                cid = c["chunk_id"]
                ids.append(cid)
                embs.append(cv)
                docs.append(c["text"])
                metas.append(
                    {
                        "record_type": "chunk",
                        "parent_chunk_id": cid,
                        "paper_id": c["paper_id"],
                    }
                )
        else:
            for c in chunks:
                cid = c["chunk_id"]
                qs = qtext_by_chunk.get(cid, [])[:n]
                qvs = qvecs_by_chunk.get(cid, [])[:n]
                for j, (qt, qv) in enumerate(zip(qs, qvs)):
                    ids.append(f"{cid}::q{j}")
                    embs.append(qv)
                    docs.append(qt)
                    metas.append(
                        {
                            "record_type": "question",
                            "parent_chunk_id": cid,
                            "paper_id": c["paper_id"],
                        }
                    )
        t0 = time.perf_counter()
        _add_batches(coll, ids, embs, docs, metas)
        add_s = time.perf_counter() - t0
        r = {
            "condition": C.condition_name(n),
            "n_questions_per_chunk": n,
            "num_embeddings": len(ids),
            "record_type": "chunk_text" if n == 0 else "generated_questions",
            "chroma_add_seconds": round(add_s, 3),
            "index_size_mb": _dir_size_mb(C.CHROMA_DIR / name),
        }
        results.append(r)
        print(
            f"  built {name}: {len(ids)} embeddings ({r['record_type']}) "
            f"{r['index_size_mb']} MB",
            flush=True,
        )

    return {
        "embedding_model": embedding_signature(),
        "chunk_encode_seconds": round(chunk_encode_s, 2),
        "question_encode_seconds": round(q_encode_s, 2),
        "num_chunks": len(chunks),
        "num_question_vectors_embedded": len(flat_q),
        "conditions": results,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _query_metrics(ranked: List[str], gold: Set[str], k_values: List[int]) -> Dict:
    rels = [1 if cid in gold else 0 for cid in ranked]
    out: Dict[str, float] = {}
    for k in k_values:
        topk = rels[:k]
        out[f"hit@{k}"] = 1.0 if any(topk) else 0.0
        out[f"recall@{k}"] = (sum(topk) / len(gold)) if gold else 0.0
    mrr = 0.0
    for i, r in enumerate(rels):
        if r:
            mrr = 1.0 / (i + 1)
            break
    out["mrr"] = mrr
    ideal = sorted(rels, reverse=True)[:10]
    idcg = _dcg(ideal)
    out["ndcg@10"] = (_dcg(rels[:10]) / idcg) if idcg > 0 else 0.0
    return out


def _retrieve_parents(coll, qvec, k: int, overfetch: int) -> List[str]:
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec], n_results=min(overfetch, n), include=["metadatas"]
    )
    metas = res["metadatas"][0] if res["metadatas"] else []
    seen, seen_set = [], set()
    for m in metas:
        parent = m.get("parent_chunk_id")
        if parent and parent not in seen_set:
            seen_set.add(parent)
            seen.append(parent)
            if len(seen) >= k:
                break
    return seen


def evaluate_condition(
    n_questions: int,
    queries: List[Dict],
    *,
    k_values: Optional[List[int]] = None,
    overfetch_factor: int = 30,
) -> Dict:
    k_values = k_values or C.K_VALUES
    coll = get_collection(n_questions)
    embedder = get_embedder()
    overfetch = max(max(k_values) * overfetch_factor, 150)

    per_query, emb_ms, search_ms = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        ranked = _retrieve_parents(coll, qvec, max(k_values), overfetch)
        search_ms.append((time.perf_counter() - t1) * 1000)
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update({"query_id": q["query_id"], "n_gold": len(q["gold_chunk_ids"])})
        per_query.append(m)

    def _avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def _pct(vals, p):
        if not vals:
            return 0.0
        vals = sorted(vals)
        return round(vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))], 3)

    agg = {}
    for k_ in k_values:
        agg[f"hit@{k_}"] = _avg(f"hit@{k_}")
        agg[f"recall@{k_}"] = _avg(f"recall@{k_}")
    agg["mrr"] = _avg("mrr")
    agg["ndcg@10"] = _avg("ndcg@10")

    return {
        "condition": C.condition_name(n_questions),
        "n_questions_per_chunk": n_questions,
        "num_queries": len(queries),
        "metrics": agg,
        "latency_ms": {
            "query_embed_mean": round(st.mean(emb_ms), 3) if emb_ms else 0,
            "search_mean": round(st.mean(search_ms), 3) if search_ms else 0,
            "search_p50": _pct(search_ms, 50),
            "search_p95": _pct(search_ms, 95),
            "total_mean": round((st.mean(emb_ms) + st.mean(search_ms)), 3)
            if emb_ms
            else 0,
        },
        "per_query": per_query,
    }
