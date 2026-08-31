"""
Run the context-enrichment METHOD comparison.

For each dataset and each condition (baseline, 3 methods, combined_best) this:
  1. builds the enriched documents (enrichment_methods.build_dataset),
  2. indexes them in a dedicated Chroma collection (logs encoding latency + tokens),
  3. retrieves top-k per query (logs query latency, computes retrieval metrics),
  4. generates an answer from the retrieved context (logs token usage + cost),
  5. grades the answer with the LangChain LLM judge (answer-quality metrics).

Outputs (results/enrichment_method_tests/):
  retrieval_metrics_by_method.csv
  answer_quality_by_method.csv
  latency_by_method.csv
  token_cost_by_method.csv
  full_results_by_query.jsonl
  context_enrichment_summary.md   (via generate_summary.py)

Start small to debug:
  python src/run_enrichment_experiments.py --max-samples 25
  python src/run_enrichment_experiments.py --max-samples 25 --use-llm
  python src/run_enrichment_experiments.py --datasets scifact --max-samples 25
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import common
import config
import embeddings
import enrichment_methods as em
import evaluate_retrieval as er
import evaluate_answers as ea

OUT_DIR = config.RESULTS_DIR / "enrichment_method_tests"
PRICES = {  # USD per 1M tokens (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
}


def _price(model: str):
    return PRICES.get(model, (0.15, 0.60))


def _gen_answer(oai, question: str, context: str):
    """Return (answer, prompt_tokens, completion_tokens, latency_ms)."""
    t0 = time.perf_counter()
    r = oai.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        temperature=0.0,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": "Answer the question using ONLY the provided context. "
                "If the answer is not in the context, say you don't know. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
    )
    ms = (time.perf_counter() - t0) * 1000
    u = r.usage
    return (
        r.choices[0].message.content.strip(),
        u.prompt_tokens,
        u.completion_tokens,
        ms,
    )


def run_condition(
    dataset: str,
    cond: str,
    recs: List[Dict],
    queries: List[Dict],
    client,
    embedder,
    oai,
    judge,
    top_k: int,
    retrieve_k: int,
    concurrency: int,
    do_answers: bool,
) -> Dict[str, Any]:
    coll_name = f"{dataset}_{cond}"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

    ids = [r["id"] for r in recs]
    texts = [r["text_for_embedding"] for r in recs]
    metas = [
        {"source_id": r["metadata"]["source_id"], "dataset": dataset, "condition": cond}
        for r in recs
    ]
    n_chunks = len(recs)
    avg_tokens = round(
        sum(common.estimate_tokens(t) for t in texts) / max(n_chunks, 1), 2
    )

    # ---- index (encoding latency) ----
    embed_t = add_t = 0.0
    B = 256
    for i in range(0, n_chunks, B):
        te = time.perf_counter()
        vecs = embedder.embed_documents(texts[i : i + B])
        embed_t += time.perf_counter() - te
        ta = time.perf_counter()
        collection.add(
            ids=ids[i : i + B],
            embeddings=vecs,
            documents=texts[i : i + B],
            metadatas=metas[i : i + B],
        )
        add_t += time.perf_counter() - ta

    # ---- per-query: retrieve (+ answer + grade) ----
    def worker(q):
        t0 = time.perf_counter()
        qv = embedder.embed_query(q["text"])
        emb_ms = (time.perf_counter() - t0) * 1000
        ts = time.perf_counter()
        res = collection.query(
            query_embeddings=[qv],
            n_results=retrieve_k,
            include=["documents", "metadatas"],
        )
        srch_ms = (time.perf_counter() - ts) * 1000
        rids = (res.get("ids") or [[]])[0]
        rmetas = (res.get("metadatas") or [[]])[0]
        rdocs = (res.get("documents") or [[]])[0]
        ranked = [str((m or {}).get("source_id")) for m in rmetas]
        context = "\n\n---\n\n".join(rdocs[:top_k])
        out = {
            "query_id": q["query_id"],
            "ranked": ranked,
            "gold": [str(g) for g in q.get("gold_source_ids", [])],
            "emb_ms": emb_ms,
            "srch_ms": srch_ms,
            "total_ms": emb_ms + srch_ms,
            "gold_answer": q.get("gold_answer", ""),
            "question": q["text"],
            "answer": "",
            "ptok": 0,
            "ctok": 0,
            "gen_ms": 0.0,
            "faithfulness": None,
            "answer_relevance": None,
            "citation_accuracy": None,
            "answer_correctness": None,
        }
        if do_answers:
            try:
                ans, pt, ct, gms = _gen_answer(oai, q["text"], context)
            except Exception as e:  # noqa: BLE001
                ans, pt, ct, gms = f"(gen error: {e})", 0, 0, 0.0
            out.update(answer=ans, ptok=pt, ctok=ct, gen_ms=gms)
            g = ea._grade_one(judge, q["text"], context, ans, q.get("gold_answer", ""))
            out.update(g)
        return out

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(worker, queries))

    # ---- aggregate retrieval metrics (document-level, dedup) ----
    def dedup(r):
        seen, o = set(), []
        for s in r:
            if s not in seen:
                seen.add(s)
                o.append(s)
        return o

    rm = {
        k: 0.0
        for k in ("Recall@5", "Recall@10", "Precision@5", "MRR", "nDCG@10", "Hit@5")
    }
    scored = [r for r in results if r["gold"]]
    for r in scored:
        ranked = dedup(r["ranked"])
        gold = set(r["gold"])
        rm["Recall@5"] += er._recall_at_k(ranked, gold, 5)
        rm["Recall@10"] += er._recall_at_k(ranked, gold, 10)
        rm["Precision@5"] += er._precision_at_k(ranked, gold, 5)
        rm["MRR"] += er._mrr(ranked, gold)
        rm["nDCG@10"] += er._ndcg_at_k(ranked, gold, 10)
        rm["Hit@5"] += er._hit_at_k(ranked, gold, 5)
    nq = max(len(scored), 1)
    retrieval = {k: round(v / nq, 4) for k, v in rm.items()}

    # ---- latency aggregates ----
    embms = [r["emb_ms"] for r in results]
    srch = [r["srch_ms"] for r in results]
    tot = [r["total_ms"] for r in results]
    latency = {
        "num_chunks": n_chunks,
        "avg_tokens_per_chunk": avg_tokens,
        "encode_ms_per_chunk": round(embed_t / max(n_chunks, 1) * 1000, 4),
        "total_embedding_time_seconds": round(embed_t, 4),
        "chroma_add_time_seconds": round(add_t, 4),
        "total_index_time_seconds": round(embed_t + add_t, 4),
        "query_embedding_latency_ms": round(float(np.mean(embms)), 4),
        "chroma_search_latency_ms": round(float(np.mean(srch)), 4),
        "total_retrieval_latency_ms": round(float(np.mean(tot)), 4),
        "p50_latency_ms": round(float(np.percentile(tot, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(tot, 95)), 4),
    }

    # ---- answer quality + token/cost ----
    answer = {
        "num_answers": 0,
        "faithfulness": None,
        "citation_accuracy": None,
        "answer_relevance": None,
        "exact_match": None,
        "token_f1": None,
        "answer_correctness": None,
    }
    pin, cout = (sum(r["ptok"] for r in results), sum(r["ctok"] for r in results))
    cost = {
        "prompt_tokens": pin,
        "completion_tokens": cout,
        "total_tokens": pin + cout,
        "estimated_cost_usd": 0.0,
    }
    if do_answers:
        pi, po = _price(config.OPENAI_CHAT_MODEL)
        cost["estimated_cost_usd"] = round(pin / 1e6 * pi + cout / 1e6 * po, 6)

        def avg(key, rows):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        em_vals = [
            ea.exact_match(r["answer"], r["gold_answer"])
            for r in results
            if r["gold_answer"]
        ]
        f1_vals = [
            ea.token_f1(r["answer"], r["gold_answer"])
            for r in results
            if r["gold_answer"]
        ]
        answer = {
            "num_answers": len(results),
            "faithfulness": avg("faithfulness", results),
            "citation_accuracy": avg("citation_accuracy", results),
            "answer_relevance": avg("answer_relevance", results),
            "exact_match": round(sum(em_vals) / len(em_vals), 4) if em_vals else None,
            "token_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
            "answer_correctness": avg("answer_correctness", results),
        }
    return {
        "retrieval": retrieval,
        "latency": latency,
        "answer": answer,
        "cost": cost,
        "per_query": results,
        "num_queries": len(results),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=config.ALL_DATASETS,
        choices=config.ALL_DATASETS,
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Number of eval queries per dataset (debug small first).",
    )
    ap.add_argument(
        "--distractors",
        type=int,
        default=60,
        help="Extra non-gold docs for cross-doc datasets.",
    )
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the chat model for LLM-backed enrichment methods.",
    )
    ap.add_argument(
        "--no-answers",
        action="store_true",
        help="Skip answer generation/grading (retrieval+latency only).",
    )
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieve_k = max(10, args.top_k)
    embedder = embeddings.get_embedder()
    client = config.get_chroma_client()
    do_answers = not args.no_answers
    oai = config.get_openai_client() if do_answers else None
    judge = ea._get_langchain_llm() if do_answers else None

    print(f"Embedding: {embeddings.embedding_signature()}")
    print(f"Chroma:    {config.chroma_signature()}")
    print(
        f"Datasets:  {args.datasets} | queries/ds={args.max_samples} | "
        f"use_llm={args.use_llm} | answers={do_answers}\n"
    )

    ret_rows, ans_rows, lat_rows, cost_rows, pq_rows = [], [], [], [], []

    for dataset in args.datasets:
        print(f"[{dataset}] building conditions…")
        conditions, queries = em.build_dataset(
            dataset, args.max_samples, args.distractors, args.use_llm
        )
        for cond, recs in conditions.items():
            if not recs:
                print(f"  {dataset}/{cond}: no records, skip")
                continue
            r = run_condition(
                dataset,
                cond,
                recs,
                queries,
                client,
                embedder,
                oai,
                judge,
                args.top_k,
                retrieve_k,
                args.concurrency,
                do_answers,
            )
            tag = f"{dataset}/{cond}"
            print(
                f"  {tag:<48} nDCG@10={r['retrieval']['nDCG@10']:.3f} "
                f"faith={r['answer']['faithfulness']} "
                f"idx={r['latency']['total_index_time_seconds']}s "
                f"${r['cost']['estimated_cost_usd']}"
            )
            ret_rows.append(
                {
                    "dataset": dataset,
                    "method": cond,
                    "num_queries": r["num_queries"],
                    **r["retrieval"],
                }
            )
            ans_rows.append({"dataset": dataset, "method": cond, **r["answer"]})
            lat_rows.append({"dataset": dataset, "method": cond, **r["latency"]})
            cost_rows.append({"dataset": dataset, "method": cond, **r["cost"]})
            for q in r["per_query"]:
                pq_rows.append({"dataset": dataset, "method": cond, **q})

    # upsert by (dataset, method) so re-running a subset of datasets keeps the rest
    def write(name, rows):
        if not rows:
            return
        groups = {(r["dataset"], r["method"]) for r in rows}
        common.upsert_csv(
            OUT_DIR / name, list(rows[0].keys()), rows, ("dataset", "method"), groups
        )
        print(f"  wrote {name}")

    print("\nWriting result files:")
    write("retrieval_metrics_by_method.csv", ret_rows)
    write("answer_quality_by_method.csv", ans_rows)
    write("latency_by_method.csv", lat_rows)
    write("token_cost_by_method.csv", cost_rows)

    # merge per-query jsonl: keep lines from datasets not re-run, append new ones
    pq_path = OUT_DIR / "full_results_by_query.jsonl"
    processed = set(args.datasets)
    kept = []
    if pq_path.exists():
        for line in pq_path.open(encoding="utf-8"):
            try:
                if json.loads(line).get("dataset") not in processed:
                    kept.append(line.rstrip("\n"))
            except Exception:
                pass
    with pq_path.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
        for r in pq_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  wrote full_results_by_query.jsonl")
    print(f"\nDone -> {OUT_DIR.relative_to(config.PROJECT_ROOT)}")
    print("Next: python src/generate_summary.py")


if __name__ == "__main__":
    main()
