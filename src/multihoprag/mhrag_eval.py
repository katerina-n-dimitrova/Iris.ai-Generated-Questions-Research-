"""
Evaluation: retrieval metrics with bootstrap confidence intervals.

MultiHop-RAG queries are MULTI-EVIDENCE (2-4 gold chunks each), so the PRIMARY
metric is Evidence Recall@k = fraction of a query's gold evidence chunks that
appear in the top k (averaged over queries), reported at k = 2, 5, 10. Also
reported: Hit@k (>=1 gold chunk in top k), MRR@10, nDCG@10.

Relevance is binary: a retrieved chunk is relevant iff it is in the query's
gold_chunk_ids. Everything is computed from the saved rankings on disk, so
metrics can be recomputed / re-bootstrapped without re-running retrieval.
Reported per arm: overall, per MultiHop-RAG query type (inference / comparison /
temporal), and for each retrieval mode (dense / bm25 / hybrid). 95% CIs come from
a 1,000-resample bootstrap over queries; arm-vs-arm significance uses a paired
bootstrap (same resampled queries for both arms).
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

import mhrag_config as C
import mhrag_retrieval as R

METRIC_KEYS = (
    [f"evidence_recall@{k}" for k in C.K_VALUES]
    + [f"hit@{k}" for k in C.K_VALUES]
    + [f"mrr@{C.MRR_K}", f"ndcg@{C.NDCG_K}"]
)
# The headline metric used for "best arm" highlighting and significance.
PRIMARY_METRIC = f"evidence_recall@{C.K_VALUES[1]}"  # evidence_recall@5


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #
def _dcg(rels: Sequence[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def per_query_metrics(ranked: List[str], gold: Sequence[str]) -> Dict[str, float]:
    gold_set = set(gold)
    rels = [1 if cid in gold_set else 0 for cid in ranked]
    out: Dict[str, float] = {}
    for k in C.K_VALUES:
        # Evidence Recall@k: fraction of gold chunks retrieved in the top k.
        out[f"evidence_recall@{k}"] = (
            (sum(rels[:k]) / len(gold_set)) if gold_set else 0.0
        )
        # Hit@k: at least one gold chunk in the top k.
        out[f"hit@{k}"] = 1.0 if any(rels[:k]) else 0.0
    # MRR@10 (rank of the first gold chunk)
    mrr = 0.0
    for i, r in enumerate(rels[: C.MRR_K]):
        if r:
            mrr = 1.0 / (i + 1)
            break
    out[f"mrr@{C.MRR_K}"] = mrr
    # nDCG@10 (binary gains; ideal = all gold at the top)
    rels_k = rels[: C.NDCG_K]
    idcg = _dcg([1] * min(len(gold_set), C.NDCG_K))
    out[f"ndcg@{C.NDCG_K}"] = (_dcg(rels_k) / idcg) if idcg > 0 else 0.0
    return out


def score_rows(rows: List[dict]) -> List[dict]:
    """Attach per-query metrics to saved ranking rows."""
    scored = []
    for r in rows:
        m = per_query_metrics(r["ranked"], r["gold_chunk_ids"])
        scored.append({"query_id": r["query_id"], "query_type": r["query_type"], **m})
    return scored


# --------------------------------------------------------------------------- #
# Aggregation + bootstrap
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    mean = float(values.mean()) if len(values) else 0.0
    if len(values) < 2:
        return {
            "mean": round(mean, 4),
            "ci_low": round(mean, 4),
            "ci_high": round(mean, 4),
            "n": int(len(values)),
        }
    idx = rng.integers(0, len(values), size=(C.BOOTSTRAP_N, len(values)))
    boot_means = values[idx].mean(axis=1)
    alpha = (1 - C.BOOTSTRAP_CI) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        "mean": round(mean, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n": int(len(values)),
    }


def aggregate(scored: List[dict], query_type: Optional[str] = None) -> Dict[str, dict]:
    """Mean + bootstrap CI for each metric over (optionally type-filtered) queries."""
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    subset = (
        scored
        if query_type is None
        else [s for s in scored if s["query_type"] == query_type]
    )
    out = {}
    for key in METRIC_KEYS:
        vals = np.array([s[key] for s in subset], dtype=float)
        out[key] = _bootstrap_ci(vals, rng)
    return out


def paired_delta(
    scored_a: List[dict], scored_b: List[dict], metric: str
) -> Dict[str, float]:
    """Paired bootstrap of mean(A) - mean(B) on matched queries (A vs B).

    Significant (95%) iff the delta CI excludes 0."""
    by_id_b = {s["query_id"]: s for s in scored_b}
    a, b = [], []
    for s in scored_a:
        t = by_id_b.get(s["query_id"])
        if t is not None:
            a.append(s[metric])
            b.append(t[metric])
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2:
        d = float(a.mean() - b.mean()) if len(a) else 0.0
        return {
            "delta": round(d, 4),
            "ci_low": round(d, 4),
            "ci_high": round(d, 4),
            "significant": False,
            "n": int(len(a)),
        }
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    idx = rng.integers(0, len(a), size=(C.BOOTSTRAP_N, len(a)))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1 - C.BOOTSTRAP_CI) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    delta = float(a.mean() - b.mean())
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "delta": round(delta, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "significant": bool(lo > 0 or hi < 0),
        "p_value": round(float(min(p, 1.0)), 4),
        "n": int(len(a)),
    }


# --------------------------------------------------------------------------- #
# Evaluate an arm (all available modes) from saved rankings
# --------------------------------------------------------------------------- #
def evaluate_arm(arm_name: str) -> dict:
    """Compute overall + per-query-type metrics for every saved mode, from disk."""
    result = {"arm": arm_name, "config": C.run_config_signature(), "modes": {}}
    modes = R.modes_for_arm(C.ARMS[arm_name])
    for mode in modes:
        rows = R.load_rankings(arm_name, mode)
        scored = score_rows(rows)
        mode_out = {"overall": aggregate(scored, None), "by_query_type": {}}
        for qtype in C.QUERY_TYPES:
            mode_out["by_query_type"][qtype] = aggregate(scored, qtype)
        result["modes"][mode] = mode_out
    with C.metrics_path(arm_name).open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def load_scored(arm_name: str, mode: str) -> List[dict]:
    """Per-query scored rows for an arm/mode (for paired significance tests)."""
    return score_rows(R.load_rankings(arm_name, mode))
