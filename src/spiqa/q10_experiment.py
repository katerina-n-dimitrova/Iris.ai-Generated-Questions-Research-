"""
Controlled q10 generated-question ENRICHMENT experiment (SPIQA test-B).

Unlike the earlier multi-vector study (each question = an independent index
vector), here generated questions are ALWAYS folded back into their parent
chunk. A retrieved chunk is returned as the evidence together with its
questions; questions are never independent evidence. No fine-tuning / training —
retrieval + indexing + enrichment only.

Conditions
----------
1. baseline                        : original chunk vector only (control).
2. q10_raw_append                  : chunk_text + 10 questions -> ONE embedding.
3. q10_roundtrip_filter_append     : keep only questions that retrieve their own
                                     parent chunk in top-k, append survivors,
                                     ONE embedding.
4. q10_score_fusion                : keep original chunk vector AND an enriched
                                     (chunk+questions) vector; rank by
                                     0.7*orig_sim + 0.3*enriched_sim.
5. q10_roundtrip_filter_score_fusion: round-trip filter + append + score fusion.
6. q10_diverse_filtered_fusion     : 10 TYPED, anchored, non-generic questions
                                     (freshly generated) + filter + score fusion.

Retrieval is exact cosine over normalised embeddings (brute force in numpy) so
that the two-vector score fusion is well defined and every condition is scored
identically. Chroma collections are still built per condition to report index
size / #embeddings. Every condition returns the ORIGINAL parent chunk as
evidence.

Outputs -> results/spiqa/test-B/q10_enrichment_experiment/
  results_table.csv / .md, results_summary.md, per_condition/<cond>.json,
  improved_queries.json, hurt_queries.json, unchanged_queries.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import spiqa_config as C
from embeddings import get_embedder, embedding_signature
from llm_adapter import get_llm, Usage
from question_gen import _parse_questions, estimate_cost, load_questions
from spiqa_loader import load_papers
from spiqa_chunker import load_chunks
from spiqa_eval import build_gold, _query_metrics

# --------------------------------------------------------------------------- #
# Config for this experiment
# --------------------------------------------------------------------------- #
N_QUESTIONS = 10
ROUNDTRIP_K = int(__import__("os").getenv("SPIQA_ROUNDTRIP_K", "10"))
FUSION_ORIG_W = 0.7
FUSION_ENRICHED_W = 0.3
TOP_K = C.TOP_K  # retrieval depth for eval
HIT_CUTOFF = 10  # "retrieved the correct chunk" = gold in top-N (failure buckets)

OUT_DIR = C.RESULTS_DIR / "test-B" / "q10_enrichment_experiment"
PERCOND_DIR = OUT_DIR / "per_condition"
DIVERSE_CACHE = C.PROCESSED_DIR / "test-B_questions_diverse.jsonl"
EXP_CHROMA_DIR = C.CHROMA_DIR / "q10exp"

CONDITIONS = [
    "baseline",
    "q10_raw_append",
    "q10_roundtrip_filter_append",
    "q10_score_fusion",
    "q10_roundtrip_filter_score_fusion",
    "q10_diverse_filtered_fusion",
]

DIVERSE_TYPES = [
    "method",
    "dataset",
    "metric/result",
    "comparison",
    "table/evidence",
    "figure/image",
    "definition/background",
    "limitation/error",
    "numeric/detail",
    "conclusion/finding",
]


# --------------------------------------------------------------------------- #
# Diverse question generation (typed, anchored, non-generic) — for condition 6
# --------------------------------------------------------------------------- #
def _diverse_prompt(chunk_text: str):
    system = (
        "You generate high-quality retrieval questions for a scientific-paper RAG "
        "system. Every question must be SPECIFIC to the passage and FULLY "
        "answerable from it — never introduce facts not present. Include concrete "
        "anchors from the passage when available: dataset names, method/model "
        "names, metric names, numbers, table/figure names, scientific entities, or "
        "section-specific terms. Do NOT output generic questions such as 'What did "
        "the paper find?', 'What method was used?', or 'What is the main idea?'. "
        "Output ONLY a numbered list of 10 questions, one per line."
    )
    types = "\n".join(f"{i + 1}. a {t} question" for i, t in enumerate(DIVERSE_TYPES))
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Write exactly 10 DISTINCT, anchored questions. When the passage supports "
        f"it, cover these different angles (skip an angle only if the passage truly "
        f"cannot support it, and replace it with another specific question):\n{types}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_diverse(
    chunks: List[Dict],
    *,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    """Generate (and cache, resumable) 10 diverse typed questions per chunk."""
    cache = {}
    if DIVERSE_CACHE.exists():
        for line in DIVERSE_CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["chunk_id"]] = r
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    llm = get_llm()
    total = Usage()
    t_wall = time.perf_counter()
    fh = DIVERSE_CACHE.open("a", encoding="utf-8")

    def work(c):
        t0 = time.perf_counter()
        text, usage = llm.chat(
            _diverse_prompt(c["text"]), temperature=0.5, max_tokens=600
        )
        return (
            c["chunk_id"],
            _parse_questions(text)[:N_QUESTIONS],
            usage,
            time.perf_counter() - t0,
        )

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futs):
                cid, qs, usage, secs = fut.result()
                total += usage
                row = {
                    "chunk_id": cid,
                    "n_generated": len(qs),
                    "questions": qs,
                    "gen_seconds": round(secs, 3),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[cid] = row
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"  [diverse] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()
    return {
        "chunks_newly_generated": len(todo),
        "chunks_cached_now": len(cache),
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "estimated_cost_usd": estimate_cost(
            total.prompt_tokens, total.completion_tokens
        ),
        "wall_seconds": round(time.perf_counter() - t_wall, 1),
    }


def load_diverse() -> Dict[str, List[str]]:
    out = {}
    if DIVERSE_CACHE.exists():
        for line in DIVERSE_CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["chunk_id"]] = r["questions"]
    return out


# --------------------------------------------------------------------------- #
# Embedding + retrieval primitives (exact cosine over normalised vectors)
# --------------------------------------------------------------------------- #
def _san(arr: np.ndarray) -> np.ndarray:
    """Zero out NaN/inf so empty-text chunks (zero-norm -> NaN after
    normalisation) can't corrupt cosine ranking."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _embed(texts: List[str]) -> Tuple[np.ndarray, float]:
    emb = get_embedder()
    t0 = time.perf_counter()
    vecs = _san(np.asarray(emb.embed_documents(texts), dtype=np.float32))
    return vecs, time.perf_counter() - t0


def _enriched_text(chunk_text: str, questions: List[str]) -> str:
    if not questions:
        return chunk_text
    q = "\n".join(f"- {x}" for x in questions)
    return f"{chunk_text}\n\nQuestions this passage answers:\n{q}"


def roundtrip_filter(
    chunk_ids: List[str],
    orig_mat: np.ndarray,
    questions_by_chunk: Dict[str, List[str]],
    *,
    k: int = ROUNDTRIP_K,
    batch: int = 2048,
):
    """
    Keep only questions that retrieve their OWN parent chunk within top-k of the
    original-chunk index. Returns (kept_by_chunk, stats).
    """
    idx_of = {cid: i for i, cid in enumerate(chunk_ids)}
    # flatten
    flat_q, owner_idx = [], []
    for cid in chunk_ids:
        for qtext in questions_by_chunk.get(cid, [])[:N_QUESTIONS]:
            flat_q.append(qtext)
            owner_idx.append(idx_of[cid])
    kept_by_chunk: Dict[str, List[str]] = {cid: [] for cid in chunk_ids}
    if not flat_q:
        return kept_by_chunk, {"generated": 0, "kept": 0}

    emb = get_embedder()
    n_generated = len(flat_q)
    n_kept = 0
    for b in range(0, len(flat_q), batch):
        qtexts = flat_q[b : b + batch]
        owners = owner_idx[b : b + batch]
        qv = _san(np.asarray(emb.embed_documents(qtexts), dtype=np.float32))
        sims = qv @ orig_mat.T  # (B, N)
        # indices of the top-k chunks per question
        topk = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for row, owner, qtext in zip(topk, owners, qtexts):
            if owner in row:
                kept_by_chunk[chunk_ids[owner]].append(qtext)
                n_kept += 1
    stats = {
        "generated": n_generated,
        "kept": n_kept,
        "filtered_pct": round(100 * (n_generated - n_kept) / max(n_generated, 1), 1),
    }
    return kept_by_chunk, stats


# --------------------------------------------------------------------------- #
# Per-condition representation builder
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    name: str
    chunk_ids: List[str]
    orig_mat: np.ndarray  # always present
    enriched_mat: Optional[np.ndarray]  # None for baseline
    fusion: bool  # score fusion vs single-vector
    questions_used: Dict[str, List[str]]  # questions folded into each chunk
    stats: Dict  # generated/kept/filtered
    encode_seconds: float

    def index_matrix(self) -> np.ndarray:
        """The matrix actually used to rank when NOT fusing."""
        return self.orig_mat if self.enriched_mat is None else self.enriched_mat

    def score(self, qv: np.ndarray) -> np.ndarray:
        if self.fusion and self.enriched_mat is not None:
            return FUSION_ORIG_W * (self.orig_mat @ qv) + FUSION_ENRICHED_W * (
                self.enriched_mat @ qv
            )
        return self.index_matrix() @ qv


def build_condition(
    name: str,
    chunks: List[Dict],
    orig_mat: np.ndarray,
    base_questions: Dict[str, List[str]],
    diverse_questions: Dict[str, List[str]],
) -> Condition:
    chunk_ids = [c["chunk_id"] for c in chunks]
    text_by_id = {c["chunk_id"]: c["text"] for c in chunks}

    if name == "baseline":
        return Condition(
            name,
            chunk_ids,
            orig_mat,
            None,
            False,
            {},
            {"generated": 0, "kept": 0, "filtered_pct": 0.0},
            0.0,
        )

    # choose the question pool
    pool = (
        diverse_questions if name == "q10_diverse_filtered_fusion" else base_questions
    )
    pool = {cid: pool.get(cid, [])[:N_QUESTIONS] for cid in chunk_ids}
    n_gen = sum(len(v) for v in pool.values())

    # round-trip filtering where required
    if "roundtrip_filter" in name or name == "q10_diverse_filtered_fusion":
        used, fstats = roundtrip_filter(chunk_ids, orig_mat, pool)
        stats = {
            "generated": n_gen,
            "kept": fstats["kept"],
            "filtered_pct": fstats["filtered_pct"],
        }
    else:
        used = pool
        stats = {"generated": n_gen, "kept": n_gen, "filtered_pct": 0.0}

    # enriched embeddings (chunk + questions)
    enriched_texts = [
        _enriched_text(text_by_id[cid], used.get(cid, [])) for cid in chunk_ids
    ]
    enriched_mat, enc_secs = _embed(enriched_texts)

    fusion = "score_fusion" in name
    return Condition(
        name, chunk_ids, orig_mat, enriched_mat, fusion, used, stats, enc_secs
    )


# --------------------------------------------------------------------------- #
# Chroma collection (for index size / #embeddings reporting)
# --------------------------------------------------------------------------- #
def _write_chroma(cond: Condition) -> Dict:
    import chromadb

    path = EXP_CHROMA_DIR / cond.name
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    coll = client.create_collection(cond.name, metadata={"hnsw:space": "cosine"})

    ids, embs, metas = [], [], []
    for i, cid in enumerate(cond.chunk_ids):
        if cond.fusion:
            # both original and enriched vectors are stored
            ids.append(f"{cid}::orig")
            embs.append(cond.orig_mat[i].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "orig"})
            ids.append(f"{cid}::enriched")
            embs.append(cond.enriched_mat[i].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "enriched"})
        else:
            ids.append(cid)
            embs.append(cond.index_matrix()[i].tolist())
            metas.append({"parent_chunk_id": cid, "record_type": "chunk"})
    for b in range(0, len(ids), 2000):
        coll.add(
            ids=ids[b : b + 2000],
            embeddings=embs[b : b + 2000],
            metadatas=metas[b : b + 2000],
        )
    size_mb = round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024), 3
    )
    return {"n_embeddings": len(ids), "index_size_mb": size_mb}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(cond: Condition, queries: List[Dict], k_values: List[int]) -> Dict:
    emb = get_embedder()
    idx_of = {cid: i for i, cid in enumerate(cond.chunk_ids)}
    per_query = []
    embed_ms, search_ms = [], []
    maxk = max(k_values)
    for q in queries:
        t0 = time.perf_counter()
        qv = _san(np.asarray(emb.embed_query(q["question"]), dtype=np.float32))
        embed_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        scores = cond.score(qv)
        top = np.argpartition(-scores, kth=min(maxk, len(scores) - 1))[:maxk]
        top = top[np.argsort(-scores[top])]
        ranked = [cond.chunk_ids[i] for i in top]
        search_ms.append((time.perf_counter() - t1) * 1000)
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update(
            {
                "query_id": q["query_id"],
                "retrieved": ranked,
                "gold": list(q["gold_chunk_ids"]),
            }
        )
        per_query.append(m)

    def avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def p95(vals):
        if not vals:
            return 0.0
        v = sorted(vals)
        return round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 3)

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics.update({f"recall@{k}": avg(f"recall@{k}") for k in k_values})
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {
        "metrics": metrics,
        "per_query": per_query,
        "search_p95_ms": p95(search_ms),
        "search_mean_ms": round(st.mean(search_ms), 3) if search_ms else 0,
        "query_embed_mean_ms": round(st.mean(embed_ms), 3) if embed_ms else 0,
    }


# --------------------------------------------------------------------------- #
# Failure analysis (baseline vs a chosen q10 condition)
# --------------------------------------------------------------------------- #
def failure_analysis(
    baseline_eval: Dict,
    cond_eval: Dict,
    cond_name: str,
    queries: List[Dict],
    cond: Condition,
    paper_of_chunk: Dict[str, str],
    cutoff: int = HIT_CUTOFF,
):
    base_pq = {x["query_id"]: x for x in baseline_eval["per_query"]}
    cond_pq = {x["query_id"]: x for x in cond_eval["per_query"]}
    q_by_id = {q["query_id"]: q for q in queries}

    def hit(pq):
        gold = set(pq["gold"])
        return any(cid in gold for cid in pq["retrieved"][:cutoff])

    improved, hurt, unchanged = [], [], []
    for qid, cpq in cond_pq.items():
        bpq = base_pq[qid]
        q = q_by_id[qid]
        b_hit, c_hit = hit(bpq), hit(cpq)
        top_ret = cpq["retrieved"][0] if cpq["retrieved"] else None
        gold_ids = set(cpq["gold"])
        gold_paper = q["paper_id"]
        ret_paper = paper_of_chunk.get(top_ret)
        rec = {
            "query_id": qid,
            "query": q["question"],
            "gold_chunk_ids": cpq["gold"],
            "top_retrieved_chunk_id": top_ret,
            "retrieved_paper_id": ret_paper,
            "gold_paper_id": gold_paper,
            "same_paper": (ret_paper == gold_paper),
            "baseline_top1": bpq["retrieved"][0] if bpq["retrieved"] else None,
            "top_generated_questions": cond.questions_used.get(top_ret, [])[:5],
        }
        if c_hit and not b_hit:
            improved.append(rec)
        elif b_hit and not c_hit:
            # add an explanation heuristic for hurt cases
            if not rec["same_paper"]:
                rec["explanation"] = (
                    "Enriched retrieval pulled a chunk from the WRONG paper to the top."
                )
            elif top_ret not in gold_ids:
                rec["explanation"] = (
                    "Same paper but a non-evidence chunk outranked the gold chunk after enrichment."
                )
            else:
                rec["explanation"] = (
                    f"Gold chunk fell out of top-{cutoff} after enrichment."
                )
            hurt.append(rec)
        else:
            unchanged.append(
                {
                    "query_id": qid,
                    "query": q["question"],
                    "both_hit": bool(b_hit and c_hit),
                }
            )
    return improved, hurt, unchanged


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run(
    split: str = "test-B",
    conditions: Optional[List[str]] = None,
    max_papers: Optional[int] = None,
    gen_limit: Optional[int] = None,
    skip_diverse_gen: bool = False,
):
    conditions = conditions or CONDITIONS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PERCOND_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})

    print(
        f"=== q10 enrichment experiment | split={split} | conditions={conditions} ==="
    )
    papers = load_papers(split, max_papers)
    chunks = load_chunks(split)
    if max_papers:  # keep only chunks from the loaded papers
        keep = {p.paper_id for p in papers}
        chunks = [c for c in chunks if c["paper_id"] in keep]
    paper_of_chunk = {c["chunk_id"]: c["paper_id"] for c in chunks}
    queries = build_gold(papers, chunks)
    print(f"chunks={len(chunks)} queries={len(queries)}")

    # base q10 questions from the existing cache (first 10 per chunk)
    base_questions = {
        cid: qs[:N_QUESTIONS] for cid, qs in load_questions(split).items()
    }

    # diverse questions (generate if needed for condition 6)
    diverse_questions: Dict[str, List[str]] = {}
    diverse_gen_summary = {}
    if "q10_diverse_filtered_fusion" in conditions:
        have = load_diverse()
        missing = [c for c in chunks if c["chunk_id"] not in have]
        if missing and not skip_diverse_gen:
            print(f"Generating diverse questions for {len(missing)} chunks...")
            diverse_gen_summary = generate_diverse(chunks, limit=gen_limit)
            print("diverse gen:", json.dumps(diverse_gen_summary))
        diverse_questions = load_diverse()

    # embed the ORIGINAL chunk texts once (shared by every condition)
    orig_mat, orig_encode_s = _embed([c["text"] for c in chunks])
    print(
        f"embedded {len(chunks)} original chunks in {orig_encode_s:.1f}s "
        f"({embedding_signature()})",
        flush=True,
    )

    evals: Dict[str, Dict] = {}
    conds: Dict[str, Condition] = {}
    table_rows = []
    for name in conditions:
        t0 = time.perf_counter()
        cond = build_condition(
            name, chunks, orig_mat, base_questions, diverse_questions
        )
        chroma = _write_chroma(cond)
        ev = evaluate(cond, queries, k_values)
        build_s = round(time.perf_counter() - t0, 1)
        conds[name] = cond
        evals[name] = ev

        _save(
            PERCOND_DIR / f"{name}.json",
            {
                "condition": name,
                "metrics": ev["metrics"],
                "questions_generated": cond.stats["generated"],
                "questions_kept": cond.stats["kept"],
                "filtered_pct": cond.stats["filtered_pct"],
                "n_embeddings": chroma["n_embeddings"],
                "index_size_mb": chroma["index_size_mb"],
                "search_p95_ms": ev["search_p95_ms"],
                "search_mean_ms": ev["search_mean_ms"],
                "query_embed_mean_ms": ev["query_embed_mean_ms"],
                "encode_seconds": round(cond.encode_seconds, 1),
                "build_seconds": build_s,
            },
        )
        m = ev["metrics"]
        table_rows.append(
            {
                "condition": name,
                "hit@1": m["hit@1"],
                "hit@5": m["hit@5"],
                "hit@10": m["hit@10"],
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "generated_questions": cond.stats["generated"],
                "kept_questions": cond.stats["kept"],
                "filtered_pct": cond.stats["filtered_pct"],
                "embeddings": chroma["n_embeddings"],
                "index_mb": chroma["index_size_mb"],
                "search_p95_ms": ev["search_p95_ms"],
            }
        )
        print(
            f"  {name:36} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"kept={cond.stats['kept']}/{cond.stats['generated']} "
            f"({cond.stats['filtered_pct']}% filt) p95={ev['search_p95_ms']}ms",
            flush=True,
        )

    # storage multiplier vs baseline
    base_mb = next(
        (r["index_mb"] for r in table_rows if r["condition"] == "baseline"), None
    )
    for r in table_rows:
        r["storage_x"] = round(r["index_mb"] / base_mb, 2) if base_mb else None
    _write_table(table_rows)

    # failure analysis vs the BEST q10 condition (by nDCG@10)
    q10_rows = [r for r in table_rows if r["condition"] != "baseline"]
    best = max(q10_rows, key=lambda r: r["ndcg@10"])["condition"] if q10_rows else None
    if best and "baseline" in evals:
        imp, hurt, unch = failure_analysis(
            evals["baseline"], evals[best], best, queries, conds[best], paper_of_chunk
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
            f"Failure analysis vs '{best}': +{len(imp)} improved / -{len(hurt)} hurt / {len(unch)} unchanged"
        )

    _write_summary(table_rows, best, diverse_gen_summary)
    print(f"\nAll outputs under {OUT_DIR}")
    return table_rows


def _write_table(rows: List[Dict]):
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
    with (OUT_DIR / "results_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT_DIR / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _write_summary(rows: List[Dict], best: Optional[str], diverse_gen: Dict):
    by = {r["condition"]: r for r in rows}
    base = by.get("baseline", {})

    def cmp(name, metric="ndcg@10"):
        if name not in by or not base:
            return None
        return round(by[name][metric] - base.get(metric, 0), 4)

    lines = [
        "# q10 generated-question enrichment — results summary\n",
        f"SPIQA **test-B** · {int(base.get('generated_questions', 0)) if base else 0} baseline "
        "control · questions are always folded back into the parent chunk "
        "(append and/or 0.7·orig + 0.3·enriched score fusion). No fine-tuning.\n",
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
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")

    # narrative
    def d(name):
        v = cmp(name)
        return (
            f"{'+' if v and v >= 0 else ''}{v} nDCG@10 vs baseline"
            if v is not None
            else "n/a"
        )

    lines.append("## Findings\n")
    lines.append(
        f"- **Best q10 method:** `{best}` "
        f"({by[best]['ndcg@10'] if best else 'n/a'} nDCG@10, {d(best) if best else ''})."
    )
    # round-trip filtering effect: raw_append vs roundtrip_filter_append
    if "q10_raw_append" in by and "q10_roundtrip_filter_append" in by:
        rf = round(
            by["q10_roundtrip_filter_append"]["ndcg@10"]
            - by["q10_raw_append"]["ndcg@10"],
            4,
        )
        lines.append(
            f"- **Round-trip filtering:** {'helped' if rf > 0 else 'did not help'} "
            f"({'+' if rf >= 0 else ''}{rf} nDCG@10, append with-vs-without filter; "
            f"filtered {by['q10_roundtrip_filter_append']['filtered_pct']}% of questions)."
        )
    # score fusion effect: raw_append vs score_fusion
    if "q10_raw_append" in by and "q10_score_fusion" in by:
        sf = round(
            by["q10_score_fusion"]["ndcg@10"] - by["q10_raw_append"]["ndcg@10"], 4
        )
        lines.append(
            f"- **Score fusion (0.7·orig+0.3·enriched):** "
            f"{'helped' if sf > 0 else 'did not help'} "
            f"({'+' if sf >= 0 else ''}{sf} nDCG@10 vs plain append)."
        )
    # diverse effect
    if (
        "q10_diverse_filtered_fusion" in by
        and "q10_roundtrip_filter_score_fusion" in by
    ):
        dv = round(
            by["q10_diverse_filtered_fusion"]["ndcg@10"]
            - by["q10_roundtrip_filter_score_fusion"]["ndcg@10"],
            4,
        )
        lines.append(
            f"- **Diverse typed generation:** {'helped' if dv > 0 else 'did not help'} "
            f"({'+' if dv >= 0 else ''}{dv} nDCG@10, diverse+filter+fusion vs "
            f"raw+filter+fusion)."
        )
    # cost/benefit
    if best and best in by:
        b = by[best]
        gain = cmp(best)
        lines.append(
            f"- **Worth the cost?** Best method uses {b['storage_x']}× baseline storage "
            f"({b['embeddings']} vs {base.get('embeddings')} embeddings) and "
            f"{b['search_p95_ms']}ms p95 search, for {'+' if gain and gain >= 0 else ''}{gain} "
            f"nDCG@10 over baseline. "
            + (
                "**Justified** — a real quality gain at modest (≤2×) storage."
                if (gain or 0) > 0.01 and (b["storage_x"] or 9) <= 2.5
                else "**Marginal** — weigh the gain against the overhead for your use case."
            )
        )
    if diverse_gen:
        lines.append(
            f"\n*Diverse-question generation cost: "
            f"${diverse_gen.get('estimated_cost_usd')} / "
            f"{diverse_gen.get('wall_seconds')}s wall.*"
        )
    (OUT_DIR / "results_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="test-B")
    ap.add_argument(
        "--conditions",
        default=None,
        help="comma list; default all. e.g. baseline,q10_raw_append",
    )
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument(
        "--gen-limit",
        type=int,
        default=None,
        help="cap diverse-question generation to first N uncached chunks",
    )
    ap.add_argument(
        "--skip-diverse-gen",
        action="store_true",
        help="use whatever diverse questions are already cached; don't call the LLM",
    )
    args = ap.parse_args()
    conds = args.conditions.split(",") if args.conditions else None
    run(args.split, conds, args.max_papers, args.gen_limit, args.skip_diverse_gen)


if __name__ == "__main__":
    main()
