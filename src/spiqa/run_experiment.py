"""
End-to-end orchestrator for the SPIQA "number of generated questions per chunk"
experiment (Stages 1-8).

Usage examples
--------------
    # Stage 1 environment check only (verifies key, embedder, Chroma):
    python run_experiment.py check-env
    python run_experiment.py check-env --live-llm   # + 1 tiny real LLM call

    # Small smoke run end-to-end (cheap): 4 papers, all conditions
    python run_experiment.py run --split test-B --max-papers 4

    # Full test-B run:
    python run_experiment.py run --split test-B

    # Retrieval/eval only, reusing cached questions + indexes:
    python run_experiment.py run --split test-B --skip-generation --skip-index

Every important parameter is a CLI flag or SPIQA_* env var (see spiqa_config.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import spiqa_config as C


# --------------------------------------------------------------------------- #
# Stage 1 — environment verification
# --------------------------------------------------------------------------- #
def check_env(live_llm: bool = False) -> Dict:
    report: Dict[str, str] = {}

    # LLM / API key (never printed; only presence + optional live call)
    from llm_adapter import check_llm

    try:
        report["llm"] = check_llm(live=live_llm)
    except Exception as e:
        report["llm"] = f"FAIL: {type(e).__name__}: {e}"

    # HF embedder
    try:
        from embeddings import get_embedder, embedding_signature

        emb = get_embedder()
        v = emb.embed_query("hello world")
        report["embedder"] = f"OK {embedding_signature()} dim={len(v)}"
    except Exception as e:
        report["embedder"] = f"FAIL: {type(e).__name__}: {e}"

    # Chroma (local persistent, isolated dir)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(C.CHROMA_DIR / "_healthcheck"))
        client.get_or_create_collection("healthcheck")
        report["chroma"] = f"OK local:{C.CHROMA_DIR}"
    except Exception as e:
        report["chroma"] = f"FAIL: {type(e).__name__}: {e}"

    # Libs
    libs = {}
    for mod in [
        "openai",
        "litellm",
        "datasets",
        "huggingface_hub",
        "chromadb",
        "sentence_transformers",
        "transformers",
        "tiktoken",
        "pandas",
        "numpy",
        "tqdm",
    ]:
        try:
            __import__(mod)
            libs[mod] = "ok"
        except Exception as e:
            libs[mod] = f"MISSING ({e})"
    report["libraries"] = libs
    return report


# --------------------------------------------------------------------------- #
# Stages 3-8 — full pipeline
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    from spiqa_loader import load_papers, save_samples, split_stats
    from spiqa_chunker import build_chunks, save_chunks, load_chunks
    from question_gen import (
        generate_questions,
        load_questions,
        questions_for_condition,
        estimate_cost,
    )
    from spiqa_index import build_all
    from spiqa_eval import build_gold, evaluate_condition

    split = args.split
    conditions = (
        [int(x) for x in args.conditions.split(",")]
        if args.conditions
        else C.QUESTION_CONDITIONS
    )
    max_n = max(conditions)

    print(f"\n=== SPIQA experiment | split={split} | conditions={conditions} ===")

    # Stage 2/3: load + chunk
    papers = load_papers(split, args.max_papers)
    print("Stage 2 stats:", split_stats(papers))
    save_samples(papers)
    chunk_objs = build_chunks(papers, size=args.chunk_size, overlap=args.overlap)
    save_chunks(chunk_objs, split)
    chunks = [c.to_row() for c in chunk_objs]  # dict rows for downstream stages
    print(f"Stage 3: {len(chunks)} chunks saved.")

    # Stage 5: generate MAX questions (cached, resumable)
    gen_summary = {}
    if not args.skip_generation and max_n > 0:
        print(f"Stage 5: generating up to {max_n} questions/chunk (cached)...")
        gen_summary = generate_questions(chunks, split, n=max_n, limit=args.gen_limit)
        gen_summary["estimated_cost_usd"] = estimate_cost(
            gen_summary["prompt_tokens_this_run"],
            gen_summary["completion_tokens_this_run"],
        )
        print("Stage 5 summary:", json.dumps(gen_summary, indent=2))
    all_questions = load_questions(split)

    # Stage 4/6: build indexes
    if not args.skip_index:
        print("Stage 4/6: building Chroma collections...")
        index_summary = build_all(split, chunks, all_questions, conditions)
    else:
        index_summary = (
            json.load(open(_index_summary_path(split)))
            if _index_summary_path(split).exists()
            else {"conditions": []}
        )
    _save(_index_summary_path(split), index_summary)

    # Stage 7: evaluate
    print("Stage 7: evaluating retrieval...")
    queries = build_gold(papers, chunks)
    print(
        f"  {len(queries)} eval queries "
        f"({sum(1 for q in queries if q['relevance_mode'] == 'chunk')} chunk-level, "
        f"{sum(1 for q in queries if q['relevance_mode'] == 'paper')} paper-level fallback)"
    )
    eval_results = []
    for n in conditions:
        res = evaluate_condition(split, n, queries, k=args.top_k)
        eval_results.append(res)
        m = res["metrics"]
        print(
            f"  {res['condition']:9} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m.get('hit@10', float('nan')):.3f} MRR={m['mrr']:.3f} "
            f"nDCG@10={m['ndcg@10']:.3f} search_p50={res['latency_ms']['chroma_search_p50']}ms"
        )

    # Stage 8: results table
    table = build_results_table(
        split, conditions, index_summary, eval_results, gen_summary, all_questions
    )
    _save(
        C.RESULTS_DIR / f"{split}_results.json",
        {
            "split": split,
            "conditions": conditions,
            "generation": gen_summary,
            "index": index_summary,
            "eval": [
                {k: v for k, v in r.items() if k != "per_query"} for r in eval_results
            ],
            "table": table,
        },
    )
    write_table_csv_md(split, table)
    return {"table": table}


def _index_summary_path(split: str) -> Path:
    return C.RESULTS_DIR / f"{split}_index_summary.json"


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Stage 8 — results table assembly
# --------------------------------------------------------------------------- #
def build_results_table(
    split, conditions, index_summary, eval_results, gen_summary, all_questions
) -> List[Dict]:
    idx_by_cond = {c["condition"]: c for c in index_summary.get("conditions", [])}
    from question_gen import estimate_cost

    rows = []
    for res in eval_results:
        cond = res["condition"]
        n = res["n_questions_per_chunk"]
        m = res["metrics"]
        idx = idx_by_cond.get(cond, {})
        # generation cost attributable to this condition's question vectors:
        # tokens are only paid once (for the max pool); we report the shared
        # total under the max condition and 0 marginal for the subsets.
        rows.append(
            {
                "condition": cond,
                "n_questions_per_chunk": n,
                "hit@1": m.get("hit@1"),
                "hit@5": m.get("hit@5"),
                "hit@10": m.get("hit@10"),
                "recall@5": m.get("recall@5"),
                "recall@10": m.get("recall@10"),
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "num_queries": res["num_queries"],
                "chunk_records": idx.get("n_chunk_records"),
                "question_records": idx.get("n_question_records"),
                "total_embeddings": idx.get("total_records"),
                "index_size_mb": idx.get("index_size_mb"),
                "chroma_add_s": idx.get("chroma_add_seconds"),
                "query_embed_ms": res["latency_ms"]["query_embed_mean"],
                "search_ms_mean": res["latency_ms"]["chroma_search_mean"],
                "search_ms_p95": res["latency_ms"]["chroma_search_p95"],
            }
        )
    # storage / embedding overhead vs baseline
    base = next((r for r in rows if r["condition"] == "baseline"), None)
    for r in rows:
        if base and base["total_embeddings"]:
            r["embeddings_x_baseline"] = round(
                (r["total_embeddings"] or 0) / base["total_embeddings"], 2
            )
        if base and base["index_size_mb"]:
            r["storage_x_baseline"] = round(
                (r["index_size_mb"] or 0) / base["index_size_mb"], 2
            )
    # attach shared generation cost/time to the table meta (once)
    if gen_summary:
        for r in rows:
            r["gen_cost_usd_shared"] = gen_summary.get("estimated_cost_usd")
            r["gen_wall_s_shared"] = gen_summary.get("wall_seconds_this_run")
    return rows


def write_table_csv_md(split, table: List[Dict]):
    import csv

    if not table:
        return
    csv_path = C.RESULTS_DIR / f"{split}_results_table.csv"
    cols = list(table[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    # compact markdown
    md_cols = [
        "condition",
        "n_questions_per_chunk",
        "hit@1",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
        "ndcg@10",
        "total_embeddings",
        "index_size_mb",
        "embeddings_x_baseline",
        "storage_x_baseline",
        "search_ms_p95",
    ]
    md_cols = [c for c in md_cols if c in table[0]]
    md_path = C.RESULTS_DIR / f"{split}_results_table.md"
    lines = [
        "| " + " | ".join(md_cols) + " |",
        "| " + " | ".join("---" for _ in md_cols) + " |",
    ]
    for r in table:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in md_cols) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nStage 8: wrote {csv_path.name} and {md_path.name}")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ce = sub.add_parser("check-env", help="Stage 1 environment verification.")
    ce.add_argument(
        "--live-llm",
        action="store_true",
        help="Make one tiny real LLM call to prove connectivity.",
    )

    r = sub.add_parser("run", help="Run the full pipeline (stages 2-8).")
    r.add_argument("--split", default=C.DEFAULT_SPLIT, choices=list(C.SPLIT_FILES))
    r.add_argument("--max-papers", type=int, default=None)
    r.add_argument("--chunk-size", type=int, default=C.CHUNK_SIZE_TOKENS)
    r.add_argument("--overlap", type=int, default=C.CHUNK_OVERLAP_TOKENS)
    r.add_argument("--top-k", type=int, default=C.TOP_K)
    r.add_argument(
        "--conditions",
        default=None,
        help="comma list e.g. 0,1,10,50,100 (default: config)",
    )
    r.add_argument(
        "--gen-limit",
        type=int,
        default=None,
        help="cap number of chunks to generate questions for (smoke test)",
    )
    r.add_argument("--skip-generation", action="store_true")
    r.add_argument("--skip-index", action="store_true")

    args = ap.parse_args()
    if args.cmd == "check-env":
        print(json.dumps(check_env(args.live_llm), indent=2))
    elif args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
