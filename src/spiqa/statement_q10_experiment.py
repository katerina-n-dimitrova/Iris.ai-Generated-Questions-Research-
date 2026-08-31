"""
Statement-level q10 generated-question ENRICHMENT experiment (SPIQA test-B).

Generated questions are a SECONDARY retrieval signal that always maps back to
its parent chunk — never independent evidence. Same LLM, same embedding model,
same chunking / split / queries / top-k across every condition; only the
retrieval/enrichment strategy changes. No fine-tuning.

Conditions
----------
1. baseline                              original chunk vectors only.
2. q10_statement_raw_append              chunk + up-to-10 questions -> ONE vector.
3. q10_statement_roundtrip_filter_append round-trip+anchor filter, then append.
4. q10_statement_score_fusion            0.7*orig_chunk_sim + 0.3*best_question_sim
                                         (questions searched separately, hits
                                         collapsed to parent_chunk_id).
5. q10_statement_filtered_score_fusion   same, but only filtered questions score.
6. q10_statement_filtered_fusion_rerank  fusion(filtered) -> top-20 parents ->
                                         rerank by original chunk text vs query.

Round-trip filter = a question is KEPT iff it (a) retrieves its own parent chunk
within top-k of the original-chunk index AND (b) carries >=1 concrete anchor.

Outputs -> results/spiqa/test-B/statement_q10_enrichment/
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# cosine over normalised vectors; empty-text rows are sanitised via _san, so
# silence the spurious Accelerate/BLAS matmul warnings on degenerate rows.
np.seterr(divide="ignore", invalid="ignore", over="ignore")

import spiqa_config as C
from embeddings import get_embedder, embedding_signature
from spiqa_loader import load_papers
from spiqa_chunker import load_chunks
from spiqa_eval import build_gold, _query_metrics
import statement_gen as SG

OUT_DIR = C.RESULTS_DIR / "test-B" / "statement_q10_enrichment"
EXP_CHROMA_DIR = C.CHROMA_DIR / "statement_q10exp"
ROUNDTRIP_K = int(__import__("os").getenv("SPIQA_ROUNDTRIP_K", "10"))
FUSION_W_ORIG, FUSION_W_Q = 0.7, 0.3
TOP_K = C.TOP_K
RERANK_POOL = 20
HIT_CUTOFF = 10  # "retrieved the correct chunk" = gold in top-N (failure buckets)

CONDITIONS = [
    "baseline",
    "q10_statement_raw_append",
    "q10_statement_roundtrip_filter_append",
    "q10_statement_score_fusion",
    "q10_statement_filtered_score_fusion",
    "q10_statement_filtered_fusion_rerank",
]


# --------------------------------------------------------------------------- #
# numpy helpers
# --------------------------------------------------------------------------- #
def _san(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _embed(texts: List[str]) -> Tuple[np.ndarray, float]:
    emb = get_embedder()
    t0 = time.perf_counter()
    v = (
        _san(np.asarray(emb.embed_documents(texts), dtype=np.float32))
        if texts
        else np.zeros((0, emb.dim), dtype=np.float32)
    )
    return v, time.perf_counter() - t0


def _seg_max(vals: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf, dtype=np.float32)
    if len(vals):
        np.maximum.at(out, owner, vals)
    return out


def _enriched_text(chunk_text: str, questions: List[str]) -> str:
    if not questions:
        return chunk_text
    return f"{chunk_text}\n\nQuestions this passage answers:\n" + "\n".join(
        f"- {q}" for q in questions
    )


# --------------------------------------------------------------------------- #
# Build all shared vectors / masks once
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, chunks, stmt_q):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.idx_of = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
        self.paper_of = {c["chunk_id"]: c["paper_id"] for c in chunks}
        self.N = len(chunks)

        # flatten questions (up to 10 per chunk)
        self.flat_q: List[str] = []
        self.flat_owner: List[int] = []
        self.flat_meta: List[dict] = []  # the structured question obj
        self.q_by_chunk: Dict[str, List[dict]] = {}
        for c in chunks:
            objs = stmt_q.get(c["chunk_id"], [])[:10]
            self.q_by_chunk[c["chunk_id"]] = objs
            for o in objs:
                self.flat_q.append(o["question"])
                self.flat_owner.append(self.idx_of[c["chunk_id"]])
                self.flat_meta.append(o)
        self.owner = np.asarray(self.flat_owner, dtype=np.int64)
        self.has_anchor = np.asarray(
            [len(o.get("required_anchors", [])) > 0 for o in self.flat_meta], dtype=bool
        )
        self.M = len(self.flat_q)

        self.C_vecs = None
        self.Q_vecs = None
        self.kept_mask = None
        self.encode_seconds = 0.0

    def embed_all(self):
        t = 0.0
        self.C_vecs, dt = _embed([self.text_by_id[cid] for cid in self.chunk_ids])
        t += dt
        print(f"  embedded {self.N} original chunks in {dt:.0f}s", flush=True)
        self.Q_vecs, dt = _embed(self.flat_q)
        t += dt
        print(f"  embedded {self.M} generated questions in {dt:.0f}s", flush=True)
        self.encode_seconds = t

    def roundtrip_filter(self, k: int = ROUNDTRIP_K, batch: int = 4096):
        """
        Keep a question iff it retrieves its own parent chunk in top-k of the
        original-chunk index AND carries >=1 anchor. Also records ``rt_pass``
        (the round-trip test alone, ignoring the anchor gate) so the anchor
        analysis can compare anchored vs non-anchored retrieval behaviour.
        """
        rt_pass = np.zeros(self.M, dtype=bool)
        for b in range(0, self.M, batch):
            sims = self.Q_vecs[b : b + batch] @ self.C_vecs.T  # (B,N)
            kk = min(k, self.N - 1)
            topk = np.argpartition(-sims, kth=kk, axis=1)[:, :k]
            for j in range(sims.shape[0]):
                gi = b + j
                if self.owner[gi] in topk[j]:
                    rt_pass[gi] = True
        self.rt_pass = rt_pass
        self.kept_mask = rt_pass & self.has_anchor
        return self.kept_mask


# --------------------------------------------------------------------------- #
# Per-condition scoring
# --------------------------------------------------------------------------- #
class Scorer:
    """Returns (ranked_chunk_ids, source_of_top) for a query vector."""

    def __init__(
        self, data: Data, name: str, enriched_mat: Optional[np.ndarray] = None
    ):
        self.d = data
        self.name = name
        self.enriched_mat = enriched_mat

    def _q_parent_best(self, q_sim: np.ndarray, filtered: bool):
        if self.d.M == 0:
            return np.zeros(self.d.N, dtype=np.float32)
        qs = q_sim.copy()
        if filtered:
            qs = np.where(self.d.kept_mask, qs, -np.inf)
        pb = _seg_max(qs, self.d.owner, self.d.N)
        return np.where(np.isfinite(pb), pb, 0.0).astype(np.float32)

    def score(self, qv, k):
        d = self.d
        C_sim = d.C_vecs @ qv
        if self.name == "baseline":
            scores, src = C_sim, "original_chunk"
        elif self.name in (
            "q10_statement_raw_append",
            "q10_statement_roundtrip_filter_append",
        ):
            scores, src = self.enriched_mat @ qv, "enriched_chunk"
        else:
            filtered = self.name != "q10_statement_score_fusion"
            pb = self._q_parent_best(d.Q_vecs @ qv, filtered)
            scores = FUSION_W_ORIG * C_sim + FUSION_W_Q * pb
            src = "fusion"

        order = np.argsort(-scores)
        if self.name == "q10_statement_filtered_fusion_rerank":
            pool = order[:RERANK_POOL]
            pool = pool[np.argsort(-C_sim[pool])]  # rerank by original chunk sim
            order = np.concatenate([pool, order[RERANK_POOL:]])
            src = "fusion+rerank"
        top = order[:k]
        ranked = [d.chunk_ids[i] for i in top]
        # attribute the source of the #1 hit for fusion conditions
        if src == "fusion":
            i0 = top[0]
            pb0 = self._q_parent_best(
                d.Q_vecs @ qv, self.name != "q10_statement_score_fusion"
            )[i0]
            src = (
                "generated_question"
                if FUSION_W_Q * pb0 > FUSION_W_ORIG * C_sim[i0]
                else "original_chunk"
            )
        return ranked, src


def build_enriched(data: Data, filtered: bool) -> np.ndarray:
    texts = []
    for cid in data.chunk_ids:
        objs = data.q_by_chunk[cid]
        if filtered:
            # map back to global mask
            qs = [
                o["question"] for o, gi in _iter_owned(data, cid) if data.kept_mask[gi]
            ]
        else:
            qs = [o["question"] for o in objs]
        texts.append(_enriched_text(data.text_by_id[cid], qs))
    mat, _ = _embed(texts)
    return mat


def _iter_owned(data: Data, cid: str):
    ci = data.idx_of[cid]
    for gi in range(data.M):
        if data.owner[gi] == ci:
            yield data.flat_meta[gi], gi


# --------------------------------------------------------------------------- #
# Chroma (index size / #embeddings reporting)
# --------------------------------------------------------------------------- #
def _write_chroma(name: str, data: Data, enriched_mat: Optional[np.ndarray]) -> Dict:
    import chromadb, shutil

    path = EXP_CHROMA_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    coll = chromadb.PersistentClient(path=str(path)).create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
    ids, embs, metas = [], [], []
    fusion = "fusion" in name
    for i, cid in enumerate(data.chunk_ids):
        if enriched_mat is not None:  # append conditions: one enriched vec
            ids.append(cid)
            embs.append(enriched_mat[i].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "enriched_chunk"})
        else:  # baseline + fusion: original chunk vec
            ids.append(cid)
            embs.append(data.C_vecs[i].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "chunk"})
    if fusion:  # + question vectors (filtered where applicable)
        filtered = name != "q10_statement_score_fusion"
        for gi in range(data.M):
            if filtered and not data.kept_mask[gi]:
                continue
            cid = data.chunk_ids[data.owner[gi]]
            ids.append(f"{cid}::q{gi}")
            embs.append(data.Q_vecs[gi].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "generated_question"})
    for b in range(0, len(ids), 2000):
        coll.add(
            ids=ids[b : b + 2000],
            embeddings=embs[b : b + 2000],
            metadatas=metas[b : b + 2000],
        )
    size = round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 3
    )
    return {"n_embeddings": len(ids), "index_size_mb": size}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    scorer: Scorer, data: Data, queries: List[Dict], k_values: List[int]
) -> Dict:
    emb = get_embedder()
    per_query = []
    embed_ms, search_ms = [], []
    maxk = max(k_values)
    for q in queries:
        t0 = time.perf_counter()
        qv = _san(np.asarray(emb.embed_query(q["question"]), dtype=np.float32))
        embed_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        ranked, src = scorer.score(qv, maxk)
        search_ms.append((time.perf_counter() - t1) * 1000)
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update(
            {
                "query_id": q["query_id"],
                "retrieved": ranked,
                "gold": list(q["gold_chunk_ids"]),
                "source": src,
            }
        )
        per_query.append(m)

    def avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def p95(v):
        if not v:
            return 0.0
        v = sorted(v)
        return round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 3)

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics.update({f"recall@{k}": avg(f"recall@{k}") for k in k_values})
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {
        "metrics": metrics,
        "per_query": per_query,
        "search_p95_ms": p95(search_ms),
        "query_embed_mean_ms": round(st.mean(embed_ms), 3) if embed_ms else 0,
    }


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def _hit(pq, cutoff):
    gold = set(pq["gold"])
    return any(cid in gold for cid in pq["retrieved"][:cutoff])


def correct_paper_wrong_chunk(per_query, paper_of, queries) -> Dict:
    q_by_id = {q["query_id"]: q for q in queries}
    n = same_paper_wrong = 0
    for pq in per_query:
        gold = set(pq["gold"])
        if not pq["retrieved"]:
            continue
        top = pq["retrieved"][0]
        n += 1
        if top not in gold and paper_of.get(top) == q_by_id[pq["query_id"]]["paper_id"]:
            same_paper_wrong += 1
    return {
        "queries": n,
        "correct_paper_wrong_chunk": same_paper_wrong,
        "rate": round(same_paper_wrong / max(n, 1), 4),
    }


def retrieval_source_of_wrong(per_query) -> Dict:
    cnt = Counter()
    for pq in per_query:
        gold = set(pq["gold"])
        if pq["retrieved"] and pq["retrieved"][0] not in gold:
            cnt[pq["source"]] += 1
    return dict(cnt)


def duplicate_analysis(data: Data, thresh: float = 0.95) -> Dict:
    dup = 0
    exact = 0
    seen_exact = set()
    for cid in data.chunk_ids:
        gis = [gi for gi in range(data.M) if data.owner[gi] == data.idx_of[cid]]
        for a in range(len(gis)):
            qa = re.sub(r"\s+", " ", data.flat_meta[gis[a]]["question"].lower()).strip()
            if qa in seen_exact:
                exact += 1
            seen_exact.add(qa)
            for b in range(a):
                sim = float(data.Q_vecs[gis[a]] @ data.Q_vecs[gis[b]])
                if sim >= thresh:
                    dup += 1
                    break
    return {
        "total_questions": data.M,
        "near_duplicate_questions": dup,
        "near_duplicate_pct": round(100 * dup / max(data.M, 1), 2),
        "exact_duplicate_questions_global": exact,
        "cosine_threshold": thresh,
    }


def question_type_analysis(data: Data, best_eval: Dict, queries: List[Dict]) -> Dict:
    # per-type counts + round-trip keep rate
    per_type = defaultdict(lambda: {"count": 0, "kept": 0, "rt_pass": 0, "anchored": 0})
    for gi in range(data.M):
        t = data.flat_meta[gi].get("question_type", "other")
        per_type[t]["count"] += 1
        per_type[t]["kept"] += int(data.kept_mask[gi])
        per_type[t]["rt_pass"] += int(data.rt_pass[gi])
        per_type[t]["anchored"] += int(data.has_anchor[gi])
    for t, d in per_type.items():
        d["keep_rate"] = round(d["kept"] / max(d["count"], 1), 3)
        d["rt_pass_rate"] = round(d["rt_pass"] / max(d["count"], 1), 3)

    # which type most often provides the "winning" question on queries whose gold
    # chunk was retrieved by the best fusion condition
    emb = get_embedder()
    winning = Counter()
    q_by_id = {q["query_id"]: q for q in queries}
    for pq in best_eval["per_query"]:
        gold = set(pq["gold"])
        hit_gold = next((c for c in pq["retrieved"][:HIT_CUTOFF] if c in gold), None)
        if not hit_gold:
            continue
        qv = _san(
            np.asarray(
                emb.embed_query(q_by_id[pq["query_id"]]["question"]), dtype=np.float32
            )
        )
        best_t, best_s = None, -1e9
        for o, gi in _iter_owned(data, hit_gold):
            s = float(data.Q_vecs[gi] @ qv)
            if s > best_s:
                best_s, best_t = s, o.get("question_type", "other")
        if best_t:
            winning[best_t] += 1
    return {
        "per_type": dict(per_type),
        "winning_type_on_retrieved_gold": dict(winning.most_common()),
    }


def anchor_analysis(data: Data) -> Dict:
    anc = data.has_anchor
    rt = data.rt_pass
    n_anc = int(anc.sum())
    n_no = int((~anc).sum())
    rt_anc = int((rt & anc).sum())
    rt_no = int((rt & ~anc).sum())
    return {
        "questions_with_anchor": n_anc,
        "questions_without_anchor": n_no,
        "roundtrip_pass_rate_with_anchor": round(rt_anc / max(n_anc, 1), 3),
        "roundtrip_pass_rate_without_anchor": round(rt_no / max(n_no, 1), 3),
        "anchor_helps": (rt_anc / max(n_anc, 1)) > (rt_no / max(n_no, 1)),
    }


def filtering_examples(data: Data, n: int = 8) -> Dict:
    kept, removed = [], []
    for gi in range(data.M):
        o = data.flat_meta[gi]
        rec = {
            "chunk_id": data.chunk_ids[data.owner[gi]],
            "question": o["question"],
            "question_type": o.get("question_type"),
            "anchors": o.get("required_anchors"),
            "rt_pass": bool(data.rt_pass[gi]),
            "has_anchor": bool(data.has_anchor[gi]),
        }
        if data.kept_mask[gi] and len(kept) < n:
            kept.append(rec)
        elif not data.kept_mask[gi] and len(removed) < n:
            reason = (
                "no_anchor"
                if not data.has_anchor[gi]
                else "did_not_retrieve_own_parent"
            )
            rec["removed_reason"] = reason
            removed.append(rec)
        if len(kept) >= n and len(removed) >= n:
            break
    return {"kept_examples": kept, "removed_examples": removed}


def failure_analysis(
    base_eval, cond_eval, cond_name, data: Data, queries, cutoff=HIT_CUTOFF
):
    emb = get_embedder()
    base_pq = {x["query_id"]: x for x in base_eval["per_query"]}
    q_by_id = {q["query_id"]: q for q in queries}
    improved, hurt, unchanged = [], [], []

    for cpq in cond_eval["per_query"]:
        qid = cpq["query_id"]
        bpq = base_pq[qid]
        q = q_by_id[qid]
        gold = set(cpq["gold"])
        b_hit, c_hit = _hit(bpq, cutoff), _hit(cpq, cutoff)
        top = cpq["retrieved"][0] if cpq["retrieved"] else None
        top_paper = data.paper_of.get(top)
        same_paper = top_paper == q["paper_id"]

        # top matching generated questions for the top retrieved chunk
        top_qs = []
        if top in data.q_by_chunk:
            qv = _san(np.asarray(emb.embed_query(q["question"]), dtype=np.float32))
            scored = sorted(
                ((float(data.Q_vecs[gi] @ qv), o) for o, gi in _iter_owned(data, top)),
                key=lambda x: -x[0],
            )[:3]
            top_qs = [
                {
                    "question": o["question"],
                    "type": o.get("question_type"),
                    "anchors": o.get("required_anchors"),
                    "sim": round(s, 3),
                }
                for s, o in scored
            ]

        rec = {
            "eval_query": q["question"],
            "gold_chunk_ids": list(gold),
            "gold_paper_id": q["paper_id"],
            "baseline_top_chunks": bpq["retrieved"][:5],
            "generated_method_top_chunks": cpq["retrieved"][:5],
            "was_correct_paper_retrieved": any(
                data.paper_of.get(c) == q["paper_id"] for c in cpq["retrieved"][:cutoff]
            ),
            "was_correct_chunk_retrieved": c_hit,
            "retrieved_correct_paper_wrong_chunk": bool(same_paper and top not in gold),
            "retrieval_source": cpq["source"],
            "top_matching_generated_questions": top_qs,
        }
        if c_hit and not b_hit:
            improved.append(rec)
        elif b_hit and not c_hit:
            if not same_paper:
                rec["possible_failure_reason"] = "wrong_parent_chunk"
            elif top not in gold:
                rec["possible_failure_reason"] = "same_paper_wrong_chunk"
            elif top_qs and not top_qs[0]["anchors"]:
                rec["possible_failure_reason"] = "missing_anchor"
            else:
                rec["possible_failure_reason"] = "other"
            hurt.append(rec)
        else:
            unchanged.append(
                {"eval_query": q["question"], "both_hit": bool(b_hit and c_hit)}
            )
    return improved, hurt, unchanged


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run(
    split="test-B", conditions=None, max_papers=None, gen_limit=None, skip_gen=False
):
    conditions = conditions or CONDITIONS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})

    print(f"=== statement-level q10 experiment | split={split} ===", flush=True)
    papers = load_papers(split, max_papers)
    chunks = load_chunks(split)
    if max_papers:
        keep = {p.paper_id for p in papers}
        chunks = [c for c in chunks if c["paper_id"] in keep]
    queries = build_gold(papers, chunks)
    print(f"chunks={len(chunks)} queries={len(queries)}", flush=True)

    # statement-level generation (cached, resumable)
    stmt_q = SG.load()
    missing = [c for c in chunks if c["chunk_id"] not in stmt_q]
    gen_summary = {}
    if missing and not skip_gen:
        print(
            f"Generating statement questions for {len(missing)} chunks...", flush=True
        )
        gen_summary = SG.generate(chunks, limit=gen_limit)
        print("gen:", json.dumps(gen_summary), flush=True)
        stmt_q = SG.load()

    data = Data(chunks, stmt_q)
    data.embed_all()
    data.roundtrip_filter()
    print(
        f"round-trip filter: kept {int(data.kept_mask.sum())}/{data.M} "
        f"({round(100 * (data.M - int(data.kept_mask.sum())) / max(data.M, 1), 1)}% filtered)",
        flush=True,
    )

    # enriched matrices for the two append conditions (built once, reused)
    enriched = {}
    if "q10_statement_raw_append" in conditions:
        enriched["q10_statement_raw_append"] = build_enriched(data, filtered=False)
    if "q10_statement_roundtrip_filter_append" in conditions:
        enriched["q10_statement_roundtrip_filter_append"] = build_enriched(
            data, filtered=True
        )

    evals, table_rows = {}, []
    n_gen = data.M
    n_kept = int(data.kept_mask.sum())
    for name in conditions:
        em = enriched.get(name)
        chroma = _write_chroma(name, data, em)
        scorer = Scorer(data, name, em)
        ev = evaluate(scorer, data, queries, k_values)
        evals[name] = ev
        m = ev["metrics"]
        is_filt = ("filter" in name) or ("filtered" in name)
        gq = 0 if name == "baseline" else n_gen
        kq = 0 if name == "baseline" else (n_kept if is_filt else n_gen)
        table_rows.append(
            {
                "condition": name,
                "hit@1": m["hit@1"],
                "hit@5": m["hit@5"],
                "hit@10": m["hit@10"],
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "generated_questions": gq,
                "kept_questions": kq,
                "filtered_pct": round(100 * (gq - kq) / gq, 1) if gq else 0.0,
                "embeddings": chroma["n_embeddings"],
                "index_mb": chroma["index_size_mb"],
                "search_p95_ms": ev["search_p95_ms"],
            }
        )
        print(
            f"  {name:40} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"p95={ev['search_p95_ms']}ms",
            flush=True,
        )

    base_mb = next(
        (r["index_mb"] for r in table_rows if r["condition"] == "baseline"), None
    )
    for r in table_rows:
        r["storage_x"] = round(r["index_mb"] / base_mb, 2) if base_mb else None
    _write_table(table_rows)

    # failure analysis vs the best generated-question condition (by nDCG@10)
    gq_rows = [r for r in table_rows if r["condition"] != "baseline"]
    best = max(gq_rows, key=lambda r: r["ndcg@10"])["condition"] if gq_rows else None
    if best:
        imp, hurt, unch = failure_analysis(
            evals["baseline"], evals[best], best, data, queries
        )
        _save(
            OUT_DIR / "improved_queries.json",
            {"vs_condition": best, "count": len(imp), "queries": imp},
        )
        _save(
            OUT_DIR / "hurt_queries.json",
            {"vs_condition": best, "count": len(hurt), "queries": hurt},
        )
        _save(
            OUT_DIR / "unchanged_queries.json",
            {"vs_condition": best, "count": len(unch), "queries": unch},
        )
        print(
            f"failure analysis vs '{best}': +{len(imp)}/-{len(hurt)}/={len(unch)}",
            flush=True,
        )

    # extra analyses
    extra = {
        "correct_paper_wrong_chunk": {
            n: correct_paper_wrong_chunk(evals[n]["per_query"], data.paper_of, queries)
            for n in conditions
        },
        "retrieval_source_of_wrong": {
            n: retrieval_source_of_wrong(evals[n]["per_query"]) for n in conditions
        },
    }
    _save(OUT_DIR / "extra_analysis.json", extra)
    _save(OUT_DIR / "duplicate_question_analysis.json", duplicate_analysis(data))
    _save(OUT_DIR / "anchor_analysis.json", anchor_analysis(data))
    if best:
        _save(
            OUT_DIR / "question_type_analysis.json",
            question_type_analysis(data, evals[best], queries),
        )
    _save(OUT_DIR / "filtered_questions_examples.json", filtering_examples(data))
    # question generation samples
    samples = []
    for cid in data.chunk_ids[:3]:
        samples.append(
            {
                "chunk_id": cid,
                "paper_id": data.paper_of[cid],
                "chunk_text": data.text_by_id[cid][:600],
                "questions": data.q_by_chunk[cid],
            }
        )
    _save(OUT_DIR / "question_generation_samples.json", samples)

    _write_summary(table_rows, best, evals, data, gen_summary)
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return table_rows


def _write_table(rows):
    import csv

    cols = [
        "condition",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10",
        "generated_questions",
        "kept_questions",
        "filtered_pct",
        "embeddings",
        "index_mb",
        "storage_x",
        "search_p95_ms",
    ]
    with (OUT_DIR / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT_DIR / "comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


def _write_summary(rows, best, evals, data: Data, gen_summary):
    by = {r["condition"]: r for r in rows}
    base = by.get("baseline", {})

    def dnd(name):
        if name not in by or not base:
            return None
        return round(by[name]["ndcg@10"] - base["ndcg@10"], 4)

    L = [
        "# Statement-level q10 generated-question enrichment — summary\n",
        "SPIQA **test-B**. Generated questions are statement-level, anchor-heavy, "
        "type-tagged, and used only as a **secondary** signal that maps back to the "
        "parent chunk (never independent evidence). Same LLM + same embedding model "
        f"(`{embedding_signature()}`) across all conditions. No fine-tuning.\n",
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
        "filtered_pct",
        "embeddings",
        "index_mb",
        "storage_x",
        "search_p95_ms",
    ]
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    L.append("\n## Findings\n")
    L.append(
        f"- **Best generated-question condition:** `{best}` "
        f"(nDCG@10 {by[best]['ndcg@10'] if best else 'n/a'}, "
        f"{'+' if (dnd(best) or 0) >= 0 else ''}{dnd(best)} vs baseline)."
    )
    # did statement-level help at all
    L.append(
        f"- **Did statement-level questions help retrieval?** "
        + (
            "Yes — the best condition beats baseline on nDCG@10."
            if (dnd(best) or 0) > 0
            else "Not meaningfully — no condition clearly beats baseline."
        )
    )
    # round-trip filtering
    if (
        "q10_statement_raw_append" in by
        and "q10_statement_roundtrip_filter_append" in by
    ):
        rf = round(
            by["q10_statement_roundtrip_filter_append"]["ndcg@10"]
            - by["q10_statement_raw_append"]["ndcg@10"],
            4,
        )
        L.append(
            f"- **Round-trip + anchor filtering:** {'helped' if rf > 0 else 'did not help'} "
            f"({'+' if rf >= 0 else ''}{rf} nDCG@10 for the append variant; "
            f"{by['q10_statement_roundtrip_filter_append']['filtered_pct']}% removed)."
        )
    # score fusion
    if "q10_statement_score_fusion" in by and "q10_statement_raw_append" in by:
        sf = round(
            by["q10_statement_score_fusion"]["ndcg@10"]
            - by["q10_statement_raw_append"]["ndcg@10"],
            4,
        )
        L.append(
            f"- **Score fusion (0.7·orig + 0.3·best-question):** "
            f"{'helped' if sf > 0 else 'did not help'} "
            f"({'+' if sf >= 0 else ''}{sf} nDCG@10 vs plain append)."
        )
    # rerank
    if (
        "q10_statement_filtered_fusion_rerank" in by
        and "q10_statement_filtered_score_fusion" in by
    ):
        rr = round(
            by["q10_statement_filtered_fusion_rerank"]["ndcg@10"]
            - by["q10_statement_filtered_score_fusion"]["ndcg@10"],
            4,
        )
        L.append(
            f"- **Rerank top-20 by original chunk text:** "
            f"{'helped precision' if rr > 0 else 'did not help'} "
            f"({'+' if rr >= 0 else ''}{rr} nDCG@10 vs fusion without rerank)."
        )
    # anchor + type
    aa = anchor_analysis(data)
    L.append(
        f"- **Anchors:** anchored questions round-trip to their own chunk "
        f"{aa['roundtrip_pass_rate_with_anchor'] * 100:.0f}% of the time vs "
        f"{aa['roundtrip_pass_rate_without_anchor'] * 100:.0f}% for non-anchored "
        f"→ anchor-heavy questions {'DO' if aa['anchor_helps'] else 'do NOT'} retrieve better."
    )
    # storage/latency justification
    if best:
        b = by[best]
        gain = dnd(best)
        L.append(
            f"- **Worth the storage/latency?** `{best}` uses {b['storage_x']}× baseline "
            f"storage ({b['embeddings']} vs {base.get('embeddings')} embeddings), "
            f"{b['search_p95_ms']}ms p95, for {'+' if (gain or 0) >= 0 else ''}{gain} nDCG@10. "
            + (
                "**Justified** at this modest overhead."
                if (gain or 0) > 0.01
                else "**Hard to justify** — the gain is within noise."
            )
        )
    if gen_summary:
        L.append(
            f"\n*Statement-question generation: ${gen_summary.get('estimated_cost_usd')} / "
            f"{gen_summary.get('wall_seconds')}s.*"
        )
    L.append(
        "\nSee `question_type_analysis.json`, `anchor_analysis.json`, "
        "`duplicate_question_analysis.json`, `filtered_questions_examples.json`, "
        "and `improved/hurt/unchanged_queries.json` for detail."
    )
    (OUT_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="test-B")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--gen-limit", type=int, default=None)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()
    conds = args.conditions.split(",") if args.conditions else None
    run(args.split, conds, args.max_papers, args.gen_limit, args.skip_gen)


if __name__ == "__main__":
    main()
