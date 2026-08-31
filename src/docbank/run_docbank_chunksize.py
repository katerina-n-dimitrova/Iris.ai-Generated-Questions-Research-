"""
Orchestrator for the DocBank chunk-size × question-enrichment experiment.

    python run_docbank_chunksize.py                 # full run
    python run_docbank_chunksize.py --gen-limit 5   # smoke test

Reuses the existing 118 synthetic eval questions (results/docbank/
docbank_15docs_eval_qa.json); each question's gold is remapped onto every
chunking by its evidence text. Writes results/docbank/docbank_chunksize_results.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as stx
import time
from pathlib import Path
from typing import Dict, List

import docbank_config as C
import docbank_loader as L
import docbank_chunker as CH
import docbank_qa as QA
import docbank_experiment as DX
import docbank_chunksize_experiment as X
from embeddings import embedding_signature


def _save(p: Path, o):
    json.dump(o, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _tokstats(chunks):
    t = [c["n_tokens"] for c in chunks]
    return (
        {
            "min": min(t),
            "max": max(t),
            "mean": round(stx.mean(t), 1),
            "median": round(stx.median(t), 1),
        }
        if t
        else {}
    )


def _row(condition, size, overlap, qpc, ev, idx, enc, gen_wall):
    m, lat = ev["metrics"], ev["latency_ms"]
    return {
        "condition": condition,
        "chunk_size": size,
        "overlap": overlap,
        "q_per_chunk": qpc,
        "index_content": idx.get("record_type", ""),
        "hit@1": m["hit@1"],
        "hit@5": m["hit@5"],
        "hit@10": m["hit@10"],
        "mrr": m["mrr"],
        "ndcg@10": m["ndcg@10"],
        "num_embeddings": idx["num_embeddings"],
        "num_questions": idx["num_questions"],
        "index_size_mb": idx["index_size_mb"],
        "chunk_encode_s": enc[0],
        "q_encode_s": enc[1],
        "gen_wall_s": gen_wall,
        "query_embed_ms": lat["query_embed_mean"],
        "search_p95": lat["search_p95"],
        "total_p95_ms": lat["total_p95"],
        "num_queries": ev["num_queries"],
    }


def _build_table3(all_rows):
    def nd(r):
        return r.get("ndcg@10") or 0

    best = max(all_rows, key=nd)
    enrich = [
        r for r in all_rows if r["index_content"] in ("generated_questions", "fused")
    ]
    best_enrich = max(enrich, key=nd) if enrich else None
    band = nd(best) - 0.02
    cheapest = min(
        [r for r in all_rows if nd(r) >= band],
        key=lambda r: r.get("index_size_mb") or 1e9,
    )
    fastest = min(all_rows, key=lambda r: r.get("search_p95") or 1e9)
    cols = [
        "condition",
        "chunk_size",
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
    ]
    picks = [
        ("best nDCG@10 (overall)", best),
        ("best generated-question setup", best_enrich),
        ("cheapest strong", cheapest),
        ("fastest", fastest),
    ]
    out, seen = [], set()
    for tag, r in picks:
        if r and (tag, r["condition"]) not in seen:
            seen.add((tag, r["condition"]))
            out.append({"selection": tag, **{k: r.get(k) for k in cols}})
    return out


def run(args) -> Dict:
    embedder = DX.get_embedder()
    docs = L.load_documents(args.num_docs)
    qa_rows = QA.load_eval_qa()
    if not qa_rows:
        raise SystemExit("No eval QA found — run run_docbank.py first.")
    qa_by_id = {q["question_id"]: q for q in qa_rows}
    print(f"docs={len(docs)} eval_qa={len(qa_rows)}")

    # ---- build all 5 chunkings + gold maps ---- #
    chunkings: Dict[str, Dict] = {}
    for size, ov in X.FIXED_SIZES:
        objs = CH.build_chunks(docs, size=size, cap=size, overlap=ov, tag=f"s{size}")
        chunkings[f"s{size}"] = {
            "size": size,
            "overlap": ov,
            "variable": False,
            "chunks": [c.to_row() for c in objs],
        }
    vobjs = CH.build_chunks(
        docs,
        size=X.VAR_CAP,
        cap=X.VAR_CAP,
        overlap=X.VAR_OVERLAP,
        tag="svar",
        section_break=True,
    )
    chunkings["svar"] = {
        "size": "variable",
        "overlap": X.VAR_OVERLAP,
        "variable": True,
        "chunks": [c.to_row() for c in vobjs],
    }

    goldmaps = {
        k: X.gold_for_chunking(qa_rows, v["chunks"]) for k, v in chunkings.items()
    }
    common = set.intersection(*[set(g.keys()) for g in goldmaps.values()])
    common = sorted(common)
    print(
        f"gold coverage per chunking: "
        + ", ".join(f"{k}={len(g)}" for k, g in goldmaps.items())
        + f" | common={len(common)}"
    )

    def queries_for(k):
        return [
            {"question": qa_by_id[qid]["question"], "gold_chunk_ids": goldmaps[k][qid]}
            for qid in common
        ]

    # ---- per chunking: generate, embed, build + eval ---- #
    all_rows: List[Dict] = []
    per_size, table1 = {}, []
    for size, ov in X.FIXED_SIZES:
        key = f"s{size}"
        rec = chunkings[key]
        chunks = rec["chunks"]
        gen = {}
        if not args.skip_generation:
            gen = _gen(chunks, args.gen_limit)
        all_q = E_load()
        cvecs, qvec_by, qtext_by, cs_s, qs_s, nqv = X.embed_set(chunks, all_q, embedder)
        enc = (cs_s, qs_s)
        gw = gen.get("wall_seconds", 0)
        q = queries_for(key)
        # baseline
        coll, idx = X.build_baseline(f"{key}_base", chunks, cvecs)
        idx["record_type"] = "chunk_text"
        br = _row("baseline", size, ov, 0, X.evaluate(coll, q), idx, enc, gw)
        all_rows.append(br)
        srec = {
            "size": size,
            "overlap": ov,
            "num_chunks": len(chunks),
            "tokens": _tokstats(chunks),
            "gen": gen,
            "baseline": br,
            "counts": {},
        }
        for nq in X.FIXED_COUNTS:
            coll, idx = X.build_questions(f"{key}_q{nq}", chunks, qtext_by, qvec_by, nq)
            idx["record_type"] = "generated_questions"
            r = _row(
                f"chunk_size_{size}_q{nq}",
                size,
                ov,
                nq,
                X.evaluate(coll, q),
                idx,
                enc,
                gw,
            )
            srec["counts"][f"q{nq}"] = r
            table1.append(r)
            all_rows.append(r)
        per_size[str(size)] = srec
        print(
            f"[s{size}] {len(chunks)} chunks · baseline nDCG {br['ndcg@10']:.3f} · "
            f"best q "
            f"{max(srec['counts'].values(), key=lambda r: r['ndcg@10'])['q_per_chunk']}"
        )

    # ---- variable chunking + adaptive ---- #
    rec = chunkings["svar"]
    chunks = rec["chunks"]
    gen = {} if args.skip_generation else _gen(chunks, args.gen_limit)
    all_q = E_load()
    cvecs, qvec_by, qtext_by, cs_s, qs_s, nqv = X.embed_set(chunks, all_q, embedder)
    enc = (cs_s, qs_s)
    gw = gen.get("wall_seconds", 0)
    q = queries_for("svar")
    len_nq = {
        c["chunk_id"]: min(
            X.length_to_nq(c["n_tokens"]), len(all_q.get(c["chunk_id"], [])) or X.MAXQ
        )
        for c in chunks
    }
    den_nq = X.density_to_nq_map(chunks, all_q)
    avg_len = round(sum(len_nq.values()) / max(len(len_nq), 1), 2)
    avg_den = round(sum(den_nq.values()) / max(len(den_nq), 1), 2)
    var_rec = {
        "num_chunks": len(chunks),
        "tokens": _tokstats(chunks),
        "gen": gen,
        "adapt_length_avg_q": avg_len,
        "adapt_density_avg_q": avg_den,
        "conditions": {},
    }
    table2 = []

    def add_var(k, label, builder, qdesc, content, fused=False):
        coll, idx = builder()
        idx["record_type"] = content
        r = _row(
            label,
            "variable",
            X.VAR_OVERLAP,
            qdesc,
            X.evaluate(coll, q, fused=fused),
            idx,
            enc,
            gw,
        )
        var_rec["conditions"][k] = r
        table2.append(r)
        all_rows.append(r)
        print(f"  {label:26} nDCG {r['ndcg@10']:.3f} #emb {r['num_embeddings']}")
        return r

    add_var(
        "baseline",
        "variable_baseline",
        lambda: X.build_baseline("svar_base", chunks, cvecs),
        0,
        "chunk_text",
    )
    add_var(
        "fixed_q10",
        "variable_fixed_q10",
        lambda: X.build_questions("svar_q10", chunks, qtext_by, qvec_by, 10),
        10,
        "generated_questions",
    )
    add_var(
        "adapt_length",
        "variable_adaptive_length",
        lambda: X.build_questions("svar_alen", chunks, qtext_by, qvec_by, len_nq),
        f"~{avg_len} (by length)",
        "generated_questions",
    )
    add_var(
        "adapt_density",
        "variable_adaptive_density",
        lambda: X.build_questions("svar_aden", chunks, qtext_by, qvec_by, den_nq),
        f"~{avg_den} (by density)",
        "generated_questions",
    )
    add_var(
        "fused_density",
        "variable_fused(chunk+density_q)",
        lambda: X.build_fused("svar_fused", chunks, cvecs, qtext_by, qvec_by, den_nq),
        f"~{avg_den}+chunk",
        "fused",
        fused=True,
    )

    out = {
        "num_documents": len(docs),
        "embedding_model": embedding_signature(),
        "llm_model": C.LLM_MODEL,
        "fixed_sizes": X.FIXED_SIZES,
        "fixed_counts": X.FIXED_COUNTS,
        "num_common_queries": len(common),
        "num_eval_qa_total": len(qa_rows),
        "gold_coverage": {k: len(g) for k, g in goldmaps.items()},
        "per_size": per_size,
        "variable": var_rec,
        "table1": table1,
        "table2": table2,
        "table3": _build_table3(all_rows),
    }
    _save(C.RESULTS_DIR / "docbank_chunksize_results.json", out)
    print("\nSaved -> results/docbank/docbank_chunksize_results.json")
    return out


def _gen(chunks, limit):
    import peerqa_experiment as E

    return E.generate_questions(
        chunks,
        n=X.MAXQ,
        cache_path=X.QCACHE,
        prompt_fn=DX._prompt_grounded,
        limit=limit,
        max_workers=C.LLM_WORKERS,
    )


def E_load():
    import peerqa_experiment as E

    return E.load_questions(cache_path=X.QCACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=C.NUM_DOCS)
    ap.add_argument("--gen-limit", type=int, default=None)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
