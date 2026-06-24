"""
Evaluate retrieval quality from the retrieved_<dataset>_<condition>.jsonl files
produced by run_retrieval_experiments.py.

Relevance is judged at the *source document* level: a retrieved chunk counts as
relevant if its metadata.source_id is in the query's gold_source_ids. This lets
multi-chunk documents (SciFact sentences, NFCorpus chunks) be scored fairly.

Metrics (per dataset x condition, averaged over queries):
    Recall@5, Recall@10, Precision@5, MRR, nDCG@10, Hit@5

Writes results/retrieval_metrics/retrieval_scores.csv and prints a summary
table comparing baseline vs enriched.

Usage:
    python src/evaluate_retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Dict, List

import common
import config

RETRIEVAL_SCORES = config.RETRIEVAL_METRICS_DIR / "retrieval_scores.csv"
SCORE_FIELDS = [
    "dataset", "condition", "num_queries",
    "Recall@5", "Recall@10", "Precision@5", "MRR", "nDCG@10", "Hit@5",
]


def _ranked_source_ids(retrieved: List[Dict]) -> List[str]:
    """
    Collapse retrieved chunks to unique source documents, keeping the best
    (first) rank for each. Many chunks can share a source_id (table rows,
    abstract sentences, doc chunks), so we evaluate at the document level to
    avoid double-counting the same gold doc.
    """
    seen = set()
    ordered: List[str] = []
    for r in retrieved:
        sid = str(r.get("source_id"))
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered


def _recall_at_k(ranked: List[str], gold: set, k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(ranked[:k]) & gold)
    return hits / len(gold)


def _precision_at_k(ranked: List[str], gold: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(ranked[:k]) & gold)
    return hits / k


def _hit_at_k(ranked: List[str], gold: set, k: int) -> float:
    return 1.0 if set(ranked[:k]) & gold else 0.0


def _mrr(ranked: List[str], gold: set) -> float:
    for i, sid in enumerate(ranked, start=1):
        if sid in gold:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(ranked: List[str], gold: set, k: int) -> float:
    # binary relevance DCG / ideal DCG
    dcg = 0.0
    for i, sid in enumerate(ranked[:k], start=1):
        if sid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_file(path) -> Dict | None:
    rows = list(common.read_jsonl(path))
    rows = [r for r in rows if r.get("gold_source_ids")]
    if not rows:
        print(f"  {path.name}: no queries with gold labels; skipped")
        return None

    agg = {m: 0.0 for m in
           ("Recall@5", "Recall@10", "Precision@5", "MRR", "nDCG@10", "Hit@5")}
    for r in rows:
        ranked = _ranked_source_ids(r["retrieved"])
        gold = {str(g) for g in r["gold_source_ids"]}
        agg["Recall@5"] += _recall_at_k(ranked, gold, 5)
        agg["Recall@10"] += _recall_at_k(ranked, gold, 10)
        agg["Precision@5"] += _precision_at_k(ranked, gold, 5)
        agg["MRR"] += _mrr(ranked, gold)
        agg["nDCG@10"] += _ndcg_at_k(ranked, gold, 10)
        agg["Hit@5"] += _hit_at_k(ranked, gold, 5)

    n = len(rows)
    out = {"dataset": rows[0]["dataset"], "condition": rows[0]["condition"],
           "num_queries": n}
    out.update({m: round(v / n, 4) for m, v in agg.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=config.ALL_DATASETS,
                    choices=config.ALL_DATASETS)
    ap.add_argument("--conditions", nargs="*", default=list(config.CONDITIONS),
                    choices=config.CONDITIONS)
    args = ap.parse_args()

    results: List[Dict] = []
    for dataset in args.datasets:
        for condition in args.conditions:
            path = (config.RETRIEVAL_METRICS_DIR
                    / f"retrieved_{dataset}_{condition}.jsonl")
            if not path.exists():
                continue
            r = evaluate_file(path)
            if r:
                results.append(r)

    if not results:
        print("No retrieval outputs found. Run run_retrieval_experiments.py first.")
        return

    groups = {(r["dataset"], r["condition"]) for r in results}
    common.upsert_csv(RETRIEVAL_SCORES, SCORE_FIELDS, results,
                      ("dataset", "condition"), groups)

    # pretty print
    print(f"\n{'dataset':<20}{'cond':<10}{'R@5':>8}{'R@10':>8}"
          f"{'P@5':>8}{'MRR':>8}{'nDCG@10':>10}{'Hit@5':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<20}{r['condition']:<10}"
              f"{r['Recall@5']:>8}{r['Recall@10']:>8}{r['Precision@5']:>8}"
              f"{r['MRR']:>8}{r['nDCG@10']:>10}{r['Hit@5']:>8}")
    print(f"\nSaved -> {RETRIEVAL_SCORES.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
