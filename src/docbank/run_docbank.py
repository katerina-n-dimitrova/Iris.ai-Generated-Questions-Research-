"""
End-to-end orchestrator for the DocBank generated-question-enrichment experiment.

    python run_docbank.py                       # full run (15 docs)
    python run_docbank.py --gen-limit 5         # cheap smoke test
    python run_docbank.py --skip-generation     # reuse cached questions

Writes results/docbank/docbank_results.json (+ the eval QA json/csv).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import docbank_config as C
import docbank_loader as L
import docbank_chunker as CH
import docbank_qa as QA
import docbank_experiment as X
from embeddings import embedding_signature


def _save(path: Path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _row(cond_rec, ev):
    m, lat = ev["metrics"], ev["latency_ms"]
    return {
        "condition": cond_rec["condition"],
        "n_questions_per_chunk": cond_rec["n_questions_per_chunk"],
        "index_content": cond_rec["record_type"],
        "hit@1": m.get("hit@1"),
        "hit@5": m.get("hit@5"),
        "hit@10": m.get("hit@10"),
        "mrr": m.get("mrr"),
        "ndcg@10": m.get("ndcg@10"),
        "num_embeddings": cond_rec["num_embeddings"],
        "num_questions": cond_rec["num_questions"],
        "index_size_mb": cond_rec["index_size_mb"],
        "index_add_s": cond_rec["index_add_s"],
        "query_embed_ms": lat["query_embed_mean"],
        "search_p95": lat["search_p95"],
        "total_p95_ms": lat["total_p95"],
        "num_queries": ev["num_queries"],
    }


def run(args) -> Dict:
    embedder = X.get_embedder()

    # Stage 1: load + inspect
    docs = L.load_documents(args.num_docs)
    L.save_documents(docs)
    ds_summary = L.dataset_summary(docs)
    print(
        "Dataset:",
        json.dumps(
            {
                k: ds_summary[k]
                for k in ("num_documents", "num_pages_total", "num_blocks_total")
            }
        ),
    )

    # Stage 3: chunk
    chunk_objs = CH.build_chunks(docs)
    CH.save_chunks(chunk_objs)
    chunks = [c.to_row() for c in chunk_objs]
    cstats = CH.chunk_stats(chunk_objs)
    print("Chunks:", cstats["num_chunks"], "avg/doc", cstats["avg_chunks_per_doc"])

    # Stage 2: synthetic eval QA (separate from enrichment)
    qa_rows = QA.load_eval_qa()
    if not qa_rows and not args.skip_qa:
        by_doc: Dict[str, List[Dict]] = defaultdict(list)
        for c in chunks:
            by_doc[c["doc_id"]].append(c)
        print("Generating synthetic eval QA...")
        qa = QA.generate_eval_qa(dict(by_doc), n=args.qa_per_doc)
        qa_rows = qa["rows"]
        QA.save_eval_qa(qa_rows)
        qa_summary = qa["summary"]
        _save(C.RESULTS_DIR / "qa_summary.json", qa_summary)
    else:
        qa_summary = (
            json.load(open(C.RESULTS_DIR / "qa_summary.json"))
            if (C.RESULTS_DIR / "qa_summary.json").exists()
            else {"num_qa": len(qa_rows)}
        )
    # keep only eval queries whose gold chunk still exists
    valid_ids = {c["chunk_id"] for c in chunks}
    queries = [q for q in qa_rows if q["gold_chunk_id"] in valid_ids]
    print(f"Eval QA: {len(qa_rows)} generated, {len(queries)} with a live gold chunk")

    # Stage 4: enrichment questions (doc2query), cached
    gen = {}
    if not args.skip_generation:
        print(f"Generating up to {X.MAXQ} enrichment questions/chunk...")
        gen = X.generate_enrichment(chunks, limit=args.gen_limit)
        print("Enrichment gen:", json.dumps(gen))
    all_q = X.load_enrichment()

    # embed once
    cvecs, qvec_by, qtext_by, cs_s, qs_s, nqv = X.embed_all(chunks, all_q, embedder)

    # Stage 5: build + evaluate conditions
    conditions = C.QUESTION_CONDITIONS
    cond_recs, table, eval_store = [], [], []
    for n in conditions:
        rec = X.build_condition(n, chunks, cvecs, qtext_by, qvec_by)
        ev = X.evaluate(rec["coll"], queries)
        rec.pop("coll")
        cond_recs.append(rec)
        table.append(_row(rec, ev))
        eval_store.append(
            {
                "condition": rec["condition"],
                "metrics": ev["metrics"],
                "latency_ms": ev["latency_ms"],
            }
        )
        m = ev["metrics"]
        print(
            f"  {rec['condition']:9} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"#emb={rec['num_embeddings']}"
        )
    # fused
    frec = X.build_condition(0, chunks, cvecs, qtext_by, qvec_by, fused=True)
    fev = X.evaluate(frec["coll"], queries, fused=True)
    frec.pop("coll")
    cond_recs.append(frec)
    table.append(_row(frec, fev))
    m = fev["metrics"]
    print(
        f"  {'fused':9} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
        f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f}"
    )

    # embeddings/storage vs baseline
    base = next((r for r in table if r["condition"] == "baseline"), None)
    for r in table:
        if base and base["num_embeddings"]:
            r["embeddings_x_baseline"] = round(
                r["num_embeddings"] / base["num_embeddings"], 2
            )
        if base and base["index_size_mb"]:
            r["storage_x_baseline"] = round(
                (r["index_size_mb"] or 0) / base["index_size_mb"], 2
            )

    out = {
        "num_documents": len(docs),
        "embedding_model": embedding_signature(),
        "llm_model": C.LLM_MODEL,
        "dataset_summary": ds_summary,
        "chunk_stats": cstats,
        "qa_summary": qa_summary,
        "qa_rows": qa_rows,
        "num_eval_queries": len(queries),
        "enrichment_gen": gen,
        "encode": {
            "chunk_encode_s": cs_s,
            "question_encode_s": qs_s,
            "num_question_vectors": nqv,
        },
        "conditions": cond_recs,
        "eval": eval_store,
        "table": table,
    }
    _save(C.RESULTS_DIR / "docbank_results.json", out)
    print("\nSaved -> results/docbank/docbank_results.json")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-docs", type=int, default=C.NUM_DOCS)
    ap.add_argument("--qa-per-doc", type=int, default=C.QA_PER_DOC)
    ap.add_argument("--gen-limit", type=int, default=None)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
