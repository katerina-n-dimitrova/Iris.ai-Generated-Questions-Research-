"""
Orchestrator for the PeerQA chunk-size × question-enrichment experiment.

    python run_chunksize.py --num-papers 15                 # full run
    python run_chunksize.py --num-papers 15 --gen-limit 6   # cheap smoke test

Writes results/peerqa/chunksize_results.json (consumed by chunksize_html.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import peerqa_config as C
import peerqa_data as D
import peerqa_experiment as E
import chunksize_experiment as X
from embeddings import embedding_signature


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _tokens_stats(chunks) -> Dict:
    import statistics as s

    t = [c.n_tokens for c in chunks]
    return (
        {
            "min": min(t),
            "max": max(t),
            "mean": round(s.mean(t), 1),
            "median": round(s.median(t), 1),
        }
        if t
        else {}
    )


def _row(condition, size, overlap, q_per_chunk, ev, idx, gen_wall, enc):
    m = ev["metrics"]
    lat = ev["latency_ms"]
    return {
        "condition": condition,
        "chunk_size": size,
        "overlap": overlap,
        "q_per_chunk": q_per_chunk,
        "hit@1": m.get("hit@1"),
        "hit@5": m.get("hit@5"),
        "hit@10": m.get("hit@10"),
        "mrr": m.get("mrr"),
        "ndcg@10": m.get("ndcg@10"),
        "recall@10": m.get("recall@10"),
        "num_embeddings": idx.get("num_embeddings"),
        "num_questions": idx.get("num_questions", 0),
        "index_size_mb": idx.get("index_size_mb"),
        "index_add_s": idx.get("index_add_s"),
        "gen_wall_s": gen_wall,
        "chunk_encode_s": enc[0],
        "q_encode_s": enc[1],
        "query_embed_ms": lat["query_embed_mean"],
        "search_p95": lat["search_p95"],
        "total_p95_ms": lat["total_p95"],
        "num_queries": ev["num_queries"],
    }


_T3_COLS = [
    "condition",
    "chunk_size",
    "overlap",
    "q_per_chunk",
    "index_content",
    "hit@1",
    "hit@5",
    "hit@10",
    "mrr",
    "ndcg@10",
    "num_embeddings",
    "index_size_mb",
    "search_p95",
    "total_p95_ms",
]


def _build_table3(all_rows: List[Dict]) -> List[Dict]:
    """Distinct, decision-relevant trade-off picks (dedup so no two rows are the
    same condition)."""

    def nd(r):
        return r.get("ndcg@10") or 0

    best = max(all_rows, key=nd)
    enrich = [
        r
        for r in all_rows
        if r.get("index_content") in ("generated_questions", "fused")
    ]
    best_enrich = max(enrich, key=nd) if enrich else None
    band = nd(best) - 0.02
    strong = [r for r in all_rows if nd(r) >= band]
    cheapest_strong = min(strong, key=lambda r: r.get("index_size_mb") or 1e9)
    fastest = min(all_rows, key=lambda r: r.get("search_p95") or 1e9)

    picks = [
        ("best nDCG@10 (overall)", best),
        ("best generated-question setup", best_enrich),
        ("cheapest strong", cheapest_strong),
        ("fastest", fastest),
    ]
    table3, seen = [], set()
    for tag, r in picks:
        if not r:
            continue
        key = (tag, r["condition"])
        if key in seen:
            continue
        seen.add(key)
        table3.append({"selection": tag, **{k: r.get(k) for k in _T3_COLS}})
    return table3


def rows_from_results(d: Dict) -> List[Dict]:
    """Reconstruct the full per-condition row list from a saved results dict."""
    rows = list(d.get("table1", [])) + list(d.get("table2", []))
    for s in d.get("per_size", {}).values():
        if s.get("baseline"):
            rows.append(s["baseline"])
    return rows


def run(args) -> Dict:
    embedder = E.get_embedder()
    papers = D.load_dataset(args.num_papers)
    print(f"\n=== PeerQA chunk-size experiment | papers={len(papers)} ===")

    per_size: Dict[str, Dict] = {}
    table1: List[Dict] = []
    all_rows: List[Dict] = []  # every condition for Table 3

    # ---------- Table 1: fixed sizes × fixed counts ---------- #
    for size, overlap in X.FIXED_SIZES:
        chunk_objs = X.build_fixed_chunks(papers, size, overlap)
        chunks = [c.__dict__ for c in chunk_objs]
        queries = D.build_gold(papers, chunk_objs)
        print(f"\n[size {size}] {len(chunks)} chunks, {len(queries)} queries")

        gen = {}
        if not args.skip_generation:
            gen = E.generate_questions(
                chunks,
                n=X.MAXQ,
                cache_path=X.QCACHE,
                prompt_fn=X._prompt_grounded,
                limit=args.gen_limit,
                max_workers=args.workers,
            )
        all_q = E.load_questions(cache_path=X.QCACHE)

        cvecs, qvec_by, qtext_by, cs_s, q_s, nqv = X.embed_set(chunks, all_q, embedder)
        enc = (cs_s, q_s)
        gen_wall = gen.get("wall_seconds", 0)

        size_rec = {
            "overlap": overlap,
            "num_chunks": len(chunks),
            "num_eval_queries": len(queries),
            "tokens": _tokens_stats(chunk_objs),
            "gen": gen,
            "chunk_encode_s": cs_s,
            "q_encode_s": q_s,
            "num_question_vectors": nqv,
            "counts": {},
        }

        # baseline (chunk text)
        coll, idx = X.build_baseline(f"s{size}_base", chunks, cvecs)
        ev = X.eval_collection(coll, queries)
        base_row = _row("baseline", size, overlap, 0, ev, idx, gen_wall, enc)
        base_row["index_content"] = "chunk_text"
        size_rec["baseline"] = base_row
        all_rows.append(base_row)
        print(
            f"  baseline      nDCG@10={ev['metrics']['ndcg@10']:.3f} "
            f"Hit@5={ev['metrics']['hit@5']:.3f}"
        )

        # fixed counts
        for nq in X.FIXED_COUNTS:
            coll, idx = X.build_questions(
                f"s{size}_q{nq}", chunks, qtext_by, qvec_by, nq
            )
            ev = X.eval_collection(coll, queries)
            r = _row(
                f"chunk_size_{size}_q{nq}", size, overlap, nq, ev, idx, gen_wall, enc
            )
            r["index_content"] = "generated_questions"
            r["strategy"] = "fixed"
            size_rec["counts"][f"q{nq}"] = r
            table1.append(r)
            all_rows.append(r)
            print(
                f"  q{nq:<3}          nDCG@10={ev['metrics']['ndcg@10']:.3f} "
                f"Hit@5={ev['metrics']['hit@5']:.3f} #emb={idx['num_embeddings']}"
            )
        per_size[str(size)] = size_rec

    # ---------- Table 2: adaptive strategies on a variable chunking ---------- #
    print("\n[variable chunking] building section-aware variable-size chunks")
    var_objs = X.build_variable_chunks(papers)
    var_chunks = [c.__dict__ for c in var_objs]
    var_queries = D.build_gold(papers, var_objs)
    print(
        f"  {len(var_chunks)} chunks, {len(var_queries)} queries, tokens={_tokens_stats(var_objs)}"
    )

    gen = {}
    if not args.skip_generation:
        gen = E.generate_questions(
            var_chunks,
            n=X.MAXQ,
            cache_path=X.QCACHE,
            prompt_fn=X._prompt_grounded,
            limit=args.gen_limit,
            max_workers=args.workers,
        )
    all_q = E.load_questions(cache_path=X.QCACHE)
    cvecs, qvec_by, qtext_by, cs_s, q_s, nqv = X.embed_set(var_chunks, all_q, embedder)
    enc = (cs_s, q_s)
    gen_wall = gen.get("wall_seconds", 0)

    # per-chunk adaptive allocations (capped by #generated)
    len_nq = {
        c["chunk_id"]: min(
            X.length_to_nq(c["n_tokens"]), len(all_q.get(c["chunk_id"], [])) or X.MAXQ
        )
        for c in var_chunks
    }
    den_nq = X.density_to_nq_map(var_objs, all_q)
    avg_len = round(sum(len_nq.values()) / max(len(len_nq), 1), 2)
    avg_den = round(sum(den_nq.values()) / max(len(den_nq), 1), 2)

    var_rec = {
        "num_chunks": len(var_chunks),
        "num_eval_queries": len(var_queries),
        "tokens": _tokens_stats(var_objs),
        "gen": gen,
        "chunk_encode_s": cs_s,
        "q_encode_s": q_s,
        "adapt_length_avg_q": avg_len,
        "adapt_density_avg_q": avg_den,
        "conditions": {},
    }
    table2: List[Dict] = []

    def add_var(cond_key, label, build_call, q_desc, index_content, fused=False):
        coll, idx = build_call()
        ev = X.eval_collection(coll, var_queries, fused=fused)
        r = _row(label, "variable", X.VAR_OVERLAP, q_desc, ev, idx, gen_wall, enc)
        r["index_content"] = index_content
        var_rec["conditions"][cond_key] = r
        table2.append(r)
        all_rows.append(r)
        print(
            f"  {label:22} nDCG@10={ev['metrics']['ndcg@10']:.3f} "
            f"Hit@5={ev['metrics']['hit@5']:.3f} #emb={idx['num_embeddings']}"
        )
        return r

    add_var(
        "baseline",
        "variable_baseline",
        lambda: X.build_baseline("svar_base", var_chunks, cvecs),
        0,
        "chunk_text",
    )
    add_var(
        "fixed_q10",
        "variable_fixed_q10",
        lambda: X.build_questions("svar_q10", var_chunks, qtext_by, qvec_by, 10),
        10,
        "generated_questions",
    )
    add_var(
        "adapt_length",
        "variable_adaptive_length",
        lambda: X.build_questions(
            "svar_adaptlen", var_chunks, qtext_by, qvec_by, len_nq
        ),
        f"~{avg_len} (by length)",
        "generated_questions",
    )
    add_var(
        "adapt_density",
        "variable_adaptive_density",
        lambda: X.build_questions(
            "svar_adaptden", var_chunks, qtext_by, qvec_by, den_nq
        ),
        f"~{avg_den} (by density)",
        "generated_questions",
    )
    add_var(
        "fused_density",
        "variable_fused(chunk+density_q)",
        lambda: X.build_fused(
            "svar_fused", var_chunks, cvecs, qtext_by, qvec_by, den_nq
        ),
        f"~{avg_den}+chunk",
        "fused",
        fused=True,
    )

    # ---------- Table 3: trade-offs ---------- #
    table3 = _build_table3(all_rows)

    out = {
        "num_papers": len(papers),
        "embedding_model": embedding_signature(),
        "llm_model": C.LLM_MODEL,
        "fixed_sizes": X.FIXED_SIZES,
        "fixed_counts": X.FIXED_COUNTS,
        "per_size": per_size,
        "variable": var_rec,
        "table1": table1,
        "table2": table2,
        "table3": table3,
    }
    _save(X.RESULTS, out)
    print("\nSaved -> results/peerqa/chunksize_results.json")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-papers", type=int, default=C.NUM_PAPERS)
    ap.add_argument("--gen-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
