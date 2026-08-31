"""
Experiment 4 — Hybrid Doc2Query-style generated-question retrieval (SPIQA test-B).

Hypothesis: generated questions may help SPARSE/lexical retrieval (BM25) more
than dense embedding text, because they add lexical anchors and likely user
phrasings (the doc2query / docT5query idea) — reducing vocabulary mismatch.

We compare dense-only, BM25-only, dense-append, and hybrid (dense + BM25 fused
with Reciprocal Rank Fusion) using generated questions as *document expansion*.
No fine-tuning. Same LLM (question cache from Experiment 1), same embedding
model, same split / chunking (416/100) / eval queries / top-k. Generated
questions always map back to their parent chunk; only the original chunk is ever
returned as evidence.

RRF: score(chunk) = 1/(60+dense_rank) + 1/(60+bm25_rank), 1-based ranks, missing
list contributes nothing. Results collapsed by parent_chunk_id (1 vector/chunk).

Conditions
----------
1 baseline_dense              dense over original chunks.
2 baseline_hybrid             dense(orig) + BM25(orig), RRF.
3 q10_dense_append            dense over (chunk + 10 questions).
4 q10_bm25_expand             BM25 over (chunk + 10 questions).
5 q10_hybrid_expand           dense(orig) + BM25(chunk+10q), RRF.
6 q50_hybrid_expand           dense(orig) + BM25(chunk+50q), RRF.
7 q10_filtered_hybrid_expand  dense(orig) + BM25(chunk+filtered10q), RRF.
8 q50_filtered_hybrid_expand  dense(orig) + BM25(chunk+filtered50q), RRF.

Outputs -> results/spiqa/test-B/hybrid_doc2query_expansion/
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore")

from rank_bm25 import BM25Okapi

import spiqa_config as C
from embeddings import get_embedder, embedding_signature
from spiqa_loader import load_papers
from spiqa_chunker import load_chunks, build_chunks
from spiqa_eval import build_gold, _query_metrics
import question_gen as QG

OUT_DIR = C.RESULTS_DIR / "test-B" / "hybrid_doc2query_expansion"
EXP_CHROMA_DIR = C.CHROMA_DIR / "hybrid_doc2query"
RRF_K = 60
TOP_K = C.TOP_K
ROUNDTRIP_K = int(__import__("os").getenv("SPIQA_ROUNDTRIP_K", "10"))

_TOK = re.compile(r"[a-z0-9]+")


def _san(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _tok(text: str) -> List[str]:
    return _TOK.findall((text or "").lower())


def _enriched(chunk_text: str, questions: List[str]) -> str:
    if not questions:
        return chunk_text
    return f"{chunk_text}\n" + "\n".join(questions)


def _embed(texts):
    emb = get_embedder()
    return (
        _san(np.asarray(emb.embed_documents(texts), dtype=np.float32))
        if texts
        else np.zeros((0, emb.dim), np.float32)
    )


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return rank[i] = 1-based rank of item i (1 = highest score)."""
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# --------------------------------------------------------------------------- #
# Round-trip filter (dense): keep questions retrieving own parent in top-k
# --------------------------------------------------------------------------- #
def roundtrip_keep(chunk_ids, C_vecs, questions_by_chunk, n, k=ROUNDTRIP_K, batch=4096):
    idx_of = {cid: i for i, cid in enumerate(chunk_ids)}
    flat_q, owner = [], []
    for cid in chunk_ids:
        for q in questions_by_chunk.get(cid, [])[:n]:
            flat_q.append(q)
            owner.append(idx_of[cid])
    kept_by_chunk = {cid: [] for cid in chunk_ids}
    if not flat_q:
        return kept_by_chunk, 0, 0
    owner = np.asarray(owner)
    n_kept = 0
    for b in range(0, len(flat_q), batch):
        qv = _embed(flat_q[b : b + batch])
        sims = qv @ C_vecs.T
        kk = min(k, C_vecs.shape[0] - 1)
        topk = np.argpartition(-sims, kth=kk, axis=1)[:, :k]
        for j in range(sims.shape[0]):
            gi = b + j
            if owner[gi] in topk[j]:
                kept_by_chunk[chunk_ids[owner[gi]]].append(flat_q[gi])
                n_kept += 1
    return kept_by_chunk, len(flat_q), n_kept


# --------------------------------------------------------------------------- #
# Retrieval building blocks
# --------------------------------------------------------------------------- #
class DenseIndex:
    def __init__(self, chunk_ids, vecs):
        self.chunk_ids = chunk_ids
        self.vecs = vecs

    def ranks(self, qv):
        return _ranks_from_scores(self.vecs @ qv)  # aligned to chunk_ids order

    def scores(self, qv):
        return self.vecs @ qv


class BM25Index:
    def __init__(self, chunk_ids, texts):
        self.chunk_ids = chunk_ids
        self.corpus = [_tok(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus)

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(_tok(query)), dtype=np.float32)

    def ranks(self, query):
        return _ranks_from_scores(self.scores(query))

    def size_mb(self, name):
        path = OUT_DIR / "_bm25" / f"{name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"corpus": self.corpus, "bm25": self.bm25}, f)
        return round(path.stat().st_size / 1048576, 3)


def _dense_index_mb(name, chunk_ids, vecs, paper_of):
    import chromadb, shutil

    path = EXP_CHROMA_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    coll = chromadb.PersistentClient(path=str(path)).create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
    metas = [{"parent_chunk_id": c, "paper_id": paper_of[c]} for c in chunk_ids]
    for b in range(0, len(chunk_ids), 2000):
        coll.add(
            ids=chunk_ids[b : b + 2000],
            embeddings=vecs[b : b + 2000].tolist(),
            metadatas=metas[b : b + 2000],
        )
    return round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 3
    )


# --------------------------------------------------------------------------- #
# Per-condition evaluation
# --------------------------------------------------------------------------- #
def eval_condition(
    name,
    chunk_ids,
    queries,
    k_values,
    *,
    dense: Optional[DenseIndex] = None,
    bm25: Optional[BM25Index] = None,
):
    """RRF-fuse whichever of dense/bm25 are provided (one or both)."""
    emb = get_embedder()
    per_query, lat = [], []
    maxk = max(k_values)
    for q in queries:
        t0 = time.perf_counter()
        d_ranks = b_ranks = None
        rrf = np.zeros(len(chunk_ids), dtype=np.float64)
        if dense is not None:
            qv = _san(np.asarray(emb.embed_query(q["question"]), np.float32))
            d_ranks = dense.ranks(qv)
            rrf += 1.0 / (RRF_K + d_ranks)
        if bm25 is not None:
            b_ranks = bm25.ranks(q["question"])
            rrf += 1.0 / (RRF_K + b_ranks)
        order = np.argsort(-rrf)[:maxk]
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = [chunk_ids[i] for i in order]
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update(
            {
                "query_id": q["query_id"],
                "retrieved": ranked,
                "gold": list(q["gold_chunk_ids"]),
                "paper_id": q["paper_id"],
            }
        )
        # store gold ranks for failure analysis
        gi = [i for i, c in enumerate(chunk_ids) if c in q["gold_chunk_ids"]]
        m["dense_rank_of_gold"] = (
            int(min(d_ranks[gi])) if (d_ranks is not None and gi) else None
        )
        m["bm25_rank_of_gold"] = (
            int(min(b_ranks[gi])) if (b_ranks is not None and gi) else None
        )
        m["rrf_rank_of_gold"] = int(_ranks_from_scores(rrf)[min(gi)]) if gi else None
        per_query.append(m)

    def avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def p95(v):
        v = sorted(v)
        return (
            round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 3) if v else 0.0
        )

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics.update({f"recall@{k}": avg(f"recall@{k}") for k in k_values})
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {
        "condition": name,
        "metrics": metrics,
        "per_query": per_query,
        "search_p95_ms": p95(lat),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run(max_papers=50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})
    papers = load_papers("test-B", max_papers)
    keep = {p.paper_id for p in papers}
    chunks = [c for c in load_chunks("test-B") if c["paper_id"] in keep]
    chunk_ids = [c["chunk_id"] for c in chunks]
    text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    paper_of = {c["chunk_id"]: c["paper_id"] for c in chunks}
    queries = build_gold(papers, chunks)
    print(
        f"papers={len(papers)} chunks={len(chunks)} queries={len(queries)} "
        f"({embedding_signature()})",
        flush=True,
    )

    # questions from the Experiment-1 cache (same chunking -> same chunk_ids)
    all_q = QG.load_questions("test-B")
    q10 = {cid: all_q.get(cid, [])[:10] for cid in chunk_ids}
    q50 = {cid: all_q.get(cid, [])[:50] for cid in chunk_ids}
    n_gen10 = sum(len(v) for v in q10.values())
    n_gen50 = sum(len(v) for v in q50.values())

    # dense vectors: original chunks, and enriched (q10 append)
    orig_vecs = _embed([text_by_id[c] for c in chunk_ids])
    print("  embedded original chunks", flush=True)
    append10_vecs = _embed([_enriched(text_by_id[c], q10[c]) for c in chunk_ids])
    dense_orig = DenseIndex(chunk_ids, orig_vecs)
    dense_append10 = DenseIndex(chunk_ids, append10_vecs)

    # round-trip filtering (dense) for filtered conditions
    kept10, gen10, keep10 = roundtrip_keep(chunk_ids, orig_vecs, q10, 10)
    kept50, gen50, keep50 = roundtrip_keep(chunk_ids, orig_vecs, q50, 50)
    print(f"  filter q10 kept {keep10}/{gen10}; q50 kept {keep50}/{gen50}", flush=True)

    # BM25 indexes
    bm25_orig = BM25Index(chunk_ids, [text_by_id[c] for c in chunk_ids])
    bm25_q10 = BM25Index(
        chunk_ids, [_enriched(text_by_id[c], q10[c]) for c in chunk_ids]
    )
    bm25_q50 = BM25Index(
        chunk_ids, [_enriched(text_by_id[c], q50[c]) for c in chunk_ids]
    )
    bm25_q10f = BM25Index(
        chunk_ids, [_enriched(text_by_id[c], kept10[c]) for c in chunk_ids]
    )
    bm25_q50f = BM25Index(
        chunk_ids, [_enriched(text_by_id[c], kept50[c]) for c in chunk_ids]
    )

    # index sizes
    dmb_orig = _dense_index_mb("dense_orig", chunk_ids, orig_vecs, paper_of)
    dmb_app10 = _dense_index_mb("dense_append10", chunk_ids, append10_vecs, paper_of)
    bmb_orig = bm25_orig.size_mb("orig")
    bmb_q10 = bm25_q10.size_mb("q10")
    bmb_q50 = bm25_q50.size_mb("q50")
    bmb_q10f = bm25_q10f.size_mb("q10f")
    bmb_q50f = bm25_q50f.size_mb("q50f")

    # condition registry: (name, dense, bm25, dense_mb, bm25_mb, gen, kept)
    C_ = [
        ("baseline_dense", dense_orig, None, dmb_orig, 0, 0, 0),
        ("baseline_hybrid", dense_orig, bm25_orig, dmb_orig, bmb_orig, 0, 0),
        ("q10_dense_append", dense_append10, None, dmb_app10, 0, n_gen10, n_gen10),
        ("q10_bm25_expand", None, bm25_q10, 0, bmb_q10, n_gen10, n_gen10),
        (
            "q10_hybrid_expand",
            dense_orig,
            bm25_q10,
            dmb_orig,
            bmb_q10,
            n_gen10,
            n_gen10,
        ),
        (
            "q50_hybrid_expand",
            dense_orig,
            bm25_q50,
            dmb_orig,
            bmb_q50,
            n_gen50,
            n_gen50,
        ),
        (
            "q10_filtered_hybrid_expand",
            dense_orig,
            bm25_q10f,
            dmb_orig,
            bmb_q10f,
            n_gen10,
            keep10,
        ),
        (
            "q50_filtered_hybrid_expand",
            dense_orig,
            bm25_q50f,
            dmb_orig,
            bmb_q50f,
            n_gen50,
            keep50,
        ),
    ]

    base_storage = dmb_orig
    results, table = {}, []
    for name, dense, bm25, dmb, bmb, gen, kept in C_:
        res = eval_condition(name, chunk_ids, queries, k_values, dense=dense, bm25=bm25)
        results[name] = res
        m = res["metrics"]
        table.append(
            {
                "condition": name,
                "hit@1": m["hit@1"],
                "hit@5": m["hit@5"],
                "hit@10": m["hit@10"],
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "generated_questions": gen,
                "kept_questions": kept,
                "filtered_%": round(100 * (gen - kept) / gen, 1) if gen else 0.0,
                "dense_index_MB": dmb,
                "bm25_index_MB": bmb,
                "storage_x": round((dmb + bmb) / base_storage, 2)
                if base_storage
                else None,
                "p95_latency": res["search_p95_ms"],
            }
        )
        print(
            f"  {name:30} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"p95={res['search_p95_ms']}ms",
            flush=True,
        )

    _write_table(table)
    _analyses(
        results,
        table,
        chunk_ids,
        paper_of,
        kept10,
        kept50,
        {"gen10": gen10, "keep10": keep10, "gen50": gen50, "keep50": keep50},
    )
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return table


def _paper_of_chunk(cid):
    parts = cid.split("::")
    return parts[1] if len(parts) > 1 else None


def _write_table(rows):
    cols = [
        "condition",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10",
        "generated_questions",
        "kept_questions",
        "filtered_%",
        "dense_index_MB",
        "bm25_index_MB",
        "storage_x",
        "p95_latency",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT_DIR / "hybrid_comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    import csv

    with (OUT_DIR / "hybrid_comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines), flush=True)


def _analyses(results, table, chunk_ids, paper_of, kept10, kept50, filt):
    by = {r["condition"]: r for r in table}

    # failure: baseline_dense vs best hybrid generated-question condition (by nDCG@10)
    hybrid_names = [
        n for n in results if n not in ("baseline_dense",) and n != "baseline_hybrid"
    ]
    best = max(hybrid_names, key=lambda n: by[n]["ndcg@10"])
    base_pq = {x["query_id"]: x for x in results["baseline_dense"]["per_query"]}
    best_pq = {x["query_id"]: x for x in results[best]["per_query"]}

    def hit(pq, k=TOP_K):
        g = set(pq["gold"])
        return any(c in g for c in pq["retrieved"][:k])

    imp, hurt, unch = [], [], []
    for qid, s in best_pq.items():
        b = base_pq[qid]
        b_ok, s_ok = hit(b), hit(s)
        top = s["retrieved"][0] if s["retrieved"] else None
        same_paper_wrong = bool(
            top and _paper_of_chunk(top) == s["paper_id"] and top not in set(s["gold"])
        )
        rec = {
            "eval_query": s["query_id"],
            "gold_paper_id": s["paper_id"],
            "gold_chunk_ids": s["gold"],
            "baseline_dense_top_chunks": b["retrieved"][:5],
            "hybrid_top_chunks": s["retrieved"][:5],
            "dense_rank_of_gold": s.get("dense_rank_of_gold"),
            "bm25_rank_of_gold": s.get("bm25_rank_of_gold"),
            "rrf_rank_of_gold": s.get("rrf_rank_of_gold"),
            "retrieved_correct_paper_wrong_chunk": same_paper_wrong,
        }
        if s_ok and not b_ok:
            dr, br = s.get("dense_rank_of_gold"), s.get("bm25_rank_of_gold")
            rec["possible_reason"] = (
                "lexical_match_helped"
                if (br is not None and (dr is None or br < dr))
                else "other"
            )
            imp.append(rec)
        elif b_ok and not s_ok:
            rec["possible_reason"] = (
                "same_paper_wrong_chunk"
                if same_paper_wrong
                else "generated_question_noise"
            )
            hurt.append(rec)
        else:
            unch.append(
                {"eval_query": s["query_id"], "both_correct": bool(b_ok and s_ok)}
            )
    _save(
        OUT_DIR / "improved_by_hybrid_generated_questions.json",
        {"vs_condition": best, "count": len(imp), "queries": imp},
    )
    _save(
        OUT_DIR / "hurt_by_hybrid_generated_questions.json",
        {"vs_condition": best, "count": len(hurt), "queries": hurt},
    )
    _save(
        OUT_DIR / "unchanged_by_hybrid_generated_questions.json",
        {"vs_condition": best, "count": len(unch), "queries": unch},
    )

    # bm25 vs dense analysis
    def nd(n):
        return by[n]["ndcg@10"]

    bm25_vs_dense = {
        "dense_append_q10_ndcg": nd("q10_dense_append"),
        "bm25_expand_q10_ndcg": nd("q10_bm25_expand"),
        "bm25_beats_dense_append": nd("q10_bm25_expand") > nd("q10_dense_append"),
        "baseline_dense_ndcg": nd("baseline_dense"),
        "baseline_hybrid_ndcg": nd("baseline_hybrid"),
        "hybrid_expand_q10_ndcg": nd("q10_hybrid_expand"),
        "hybrid_beats_dense_only": nd("q10_hybrid_expand") > nd("baseline_dense"),
        "q50_hybrid_ndcg": nd("q50_hybrid_expand"),
        "q50_beats_q10": nd("q50_hybrid_expand") > nd("q10_hybrid_expand"),
        "best_condition": best,
    }
    _save(OUT_DIR / "bm25_vs_dense_analysis.json", bm25_vs_dense)

    filtering = {
        "q10": {
            "generated": filt["gen10"],
            "kept": filt["keep10"],
            "filtered_pct": round(
                100 * (filt["gen10"] - filt["keep10"]) / max(filt["gen10"], 1), 1
            ),
            "unfiltered_hybrid_ndcg": nd("q10_hybrid_expand"),
            "filtered_hybrid_ndcg": nd("q10_filtered_hybrid_expand"),
            "filtering_helped": nd("q10_filtered_hybrid_expand")
            > nd("q10_hybrid_expand"),
        },
        "q50": {
            "generated": filt["gen50"],
            "kept": filt["keep50"],
            "filtered_pct": round(
                100 * (filt["gen50"] - filt["keep50"]) / max(filt["gen50"], 1), 1
            ),
            "unfiltered_hybrid_ndcg": nd("q50_hybrid_expand"),
            "filtered_hybrid_ndcg": nd("q50_filtered_hybrid_expand"),
            "filtering_helped": nd("q50_filtered_hybrid_expand")
            > nd("q50_hybrid_expand"),
        },
    }
    _save(OUT_DIR / "filtering_effect_analysis.json", filtering)

    _save(
        OUT_DIR / "storage_latency_analysis.json",
        {
            r["condition"]: {
                "dense_index_MB": r["dense_index_MB"],
                "bm25_index_MB": r["bm25_index_MB"],
                "storage_x": r["storage_x"],
                "p95_latency": r["p95_latency"],
            }
            for r in table
        },
    )

    _write_summary(table, by, best, bm25_vs_dense, filtering)


def _write_summary(table, by, best, bvd, filt):
    def nd(n):
        return by[n]["ndcg@10"]

    base = nd("baseline_dense")
    L = [
        "# Hybrid Doc2Query-style generated-question retrieval — summary\n",
        "SPIQA test-B (50 papers). Generated questions used as **document expansion** "
        "for BM25 vs as dense embedding text; hybrid = dense(original chunks) + "
        "BM25(enriched) fused with RRF (k=60). Questions map back to parent chunk; "
        "only original chunks are returned. Same LLM/embedder, no fine-tuning.\n",
        "## Comparison table\n",
    ]
    cols = [
        "condition",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10",
        "generated_questions",
        "kept_questions",
        "filtered_%",
        "dense_index_MB",
        "bm25_index_MB",
        "storage_x",
        "p95_latency",
    ]
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in table:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    L.append("\n## Findings\n")
    L.append(
        f"1. **BM25-expand vs dense-append (q10):** BM25 nDCG@10 {nd('q10_bm25_expand')} vs "
        f"dense-append {nd('q10_dense_append')} → generated questions "
        f"{'help sparse MORE than dense' if bvd['bm25_beats_dense_append'] else 'do NOT help sparse more than dense'}."
    )
    L.append(
        f"2. **Hybrid vs dense-only:** hybrid_expand_q10 {nd('q10_hybrid_expand')} vs "
        f"baseline_dense {base} → hybrid {'improves' if bvd['hybrid_beats_dense_only'] else 'does not improve'} retrieval. "
        f"(baseline_hybrid, no questions = {nd('baseline_hybrid')}.)"
    )
    L.append(
        f"3. **q10 vs q50 (hybrid):** {nd('q10_hybrid_expand')} vs {nd('q50_hybrid_expand')} → "
        f"{'q50 helps more' if bvd['q50_beats_q10'] else 'q50 adds little/noise over q10'}."
    )
    L.append(
        f"4. **Round-trip filtering:** q10 {'helped' if filt['q10']['filtering_helped'] else 'did not help'} "
        f"({filt['q10']['filtered_pct']}% removed); q50 "
        f"{'helped' if filt['q50']['filtering_helped'] else 'did not help'} "
        f"({filt['q50']['filtered_pct']}% removed)."
    )
    L.append(
        f"5. **Questions as lexical expansion vs dense text:** "
        + (
            "YES — BM25 expansion beats dense append, supporting the doc2query hypothesis."
            if bvd["bm25_beats_dense_append"]
            else "Not clearly — dense append is at least as good as BM25 expansion here."
        )
    )
    best_row = by[best]
    gain = round(best_row["ndcg@10"] - base, 4)
    L.append(
        f"6. **Cost/benefit:** best generated-question condition `{best}` = "
        f"{best_row['ndcg@10']} nDCG@10 ({'+' if gain >= 0 else ''}{gain} vs dense baseline) at "
        f"{best_row['storage_x']}× storage, {best_row['p95_latency']}ms p95. "
        + (
            "Worth it."
            if gain > 0.01
            else "Marginal — hard to justify the extra index."
        )
    )
    L.append(
        f"\n## Recommendation\n**{best if by[best]['ndcg@10'] > base else 'baseline_dense'}** "
        + (
            "— hybrid BM25 expansion with generated questions is the best configuration."
            if by[best]["ndcg@10"] > base
            else "— generated-question expansion did not beat plain dense retrieval on this sample."
        )
    )
    (OUT_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=50)
    args = ap.parse_args()
    run(args.max_papers)


if __name__ == "__main__":
    main()
