"""
End-to-end orchestrator for the PeerQA generated-questions-only experiment.

    python run_peerqa.py --num-papers 15                 # full subset run
    python run_peerqa.py --num-papers 15 --gen-limit 20  # cheap smoke test
    python run_peerqa.py --skip-generation --skip-index  # eval only (reuse cache)

Writes results/peerqa/peerqa_results.json (consumed by peerqa_html.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import peerqa_config as C
import peerqa_data as D
import peerqa_experiment as E


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def build_table(
    conditions: List[int], index_summary: Dict, eval_results: List[Dict]
) -> List[Dict]:
    idx_by_cond = {c["condition"]: c for c in index_summary.get("conditions", [])}
    rows = []
    base = None
    for res in eval_results:
        cond = res["condition"]
        m = res["metrics"]
        idx = idx_by_cond.get(cond, {})
        row = {
            "condition": cond,
            "n_questions_per_chunk": res["n_questions_per_chunk"],
            "index_content": idx.get("record_type"),
            "hit@1": m.get("hit@1"),
            "hit@5": m.get("hit@5"),
            "hit@10": m.get("hit@10"),
            "mrr": m.get("mrr"),
            "ndcg@10": m.get("ndcg@10"),
            "recall@10": m.get("recall@10"),
            "num_embeddings": idx.get("num_embeddings"),
            "index_size_mb": idx.get("index_size_mb"),
            "index_add_s": idx.get("chroma_add_seconds"),
            "query_embed_ms": res["latency_ms"]["query_embed_mean"],
            "search_ms_mean": res["latency_ms"]["search_mean"],
            "search_ms_p95": res["latency_ms"]["search_p95"],
            "num_queries": res["num_queries"],
        }
        rows.append(row)
        if cond == "baseline":
            base = row
    for r in rows:
        if base and base.get("num_embeddings"):
            r["embeddings_x_baseline"] = round(
                (r["num_embeddings"] or 0) / base["num_embeddings"], 2
            )
        if base and base.get("index_size_mb"):
            r["storage_x_baseline"] = round(
                (r["index_size_mb"] or 0) / base["index_size_mb"], 2
            )
    return rows


def run(args) -> Dict:
    conditions = (
        [int(x) for x in args.conditions.split(",")]
        if args.conditions
        else C.QUESTION_CONDITIONS
    )
    max_n = max([n for n in conditions if n > 0], default=0)
    print(
        f"\n=== PeerQA questions-only experiment | papers={args.num_papers} "
        f"| conditions={conditions} ==="
    )

    # Stage 1: load + chunk
    papers = D.load_dataset(args.num_papers)
    chunk_objs = D.build_chunks(papers)
    D.save_chunks(chunk_objs)
    chunks = [c.__dict__ for c in chunk_objs]
    summ = D.dataset_summary(papers, chunk_objs)
    queries = D.build_gold(papers, chunk_objs)
    summ["num_eval_queries"] = len(queries)
    summ["already_chunked_note"] = (
        "PeerQA ships paper text pre-segmented at the SENTENCE level "
        "(papers.jsonl: idx/pidx/sidx/type/content/last_heading). Sentences are "
        "~10-30 tokens (far below a RAG chunk) with NO overlap. We pack "
        "consecutive sentences into ~500-token chunks (cap 600) with 100-token "
        "overlap; gold sentence idxs map onto parent chunks."
    )
    print("Dataset summary:", json.dumps(summ, indent=2))
    _save(C.RESULTS_DIR / "dataset_summary.json", summ)
    print(f"  {len(queries)} eval queries over {len(chunks)} chunks")

    # Stage 2: generate questions (cached, resumable)
    gen_summary = {}
    if not args.skip_generation and max_n > 0:
        print(f"Generating up to {max_n} questions/chunk (cached)...")
        gen_summary = E.generate_questions(chunks, n=max_n, limit=args.gen_limit)
        print("Generation summary:", json.dumps(gen_summary, indent=2))
    all_questions = E.load_questions()

    # Stage 3: build indexes
    idx_path = C.RESULTS_DIR / "index_summary.json"
    if not args.skip_index:
        print("Building Chroma collections...")
        index_summary = E.build_all(chunks, all_questions, conditions)
    else:
        index_summary = (
            json.load(open(idx_path)) if idx_path.exists() else {"conditions": []}
        )
    _save(idx_path, index_summary)

    # Stage 4: evaluate
    print("Evaluating retrieval...")
    eval_results = []
    for n in conditions:
        res = E.evaluate_condition(n, queries, k_values=C.K_VALUES)
        eval_results.append(res)
        m = res["metrics"]
        print(
            f"  {res['condition']:9} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"search_p95={res['latency_ms']['search_p95']}ms"
        )

    table = build_table(conditions, index_summary, eval_results)
    out = {
        "num_papers": args.num_papers,
        "conditions": conditions,
        "dataset_summary": summ,
        "generation": gen_summary,
        "index": index_summary,
        "eval": [
            {k: v for k, v in r.items() if k != "per_query"} for r in eval_results
        ],
        "table": table,
        "embedding_model": index_summary.get("embedding_model"),
        "llm_model": C.LLM_MODEL,
    }
    _save(C.RESULTS_DIR / "peerqa_results.json", out)
    print("\nSaved -> results/peerqa/peerqa_results.json")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-papers", type=int, default=C.NUM_PAPERS)
    ap.add_argument("--conditions", default=None, help="comma list e.g. 0,5,10,13")
    ap.add_argument("--gen-limit", type=int, default=None)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
