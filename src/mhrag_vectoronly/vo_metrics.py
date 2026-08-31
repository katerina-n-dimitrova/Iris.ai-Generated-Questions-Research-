"""
Stage: retrieval evaluation (§15), paired comparison (§18), bootstrap CIs.

A retrieved chunk is RELEVANT iff it contains >=1 gold evidence fact for the
query (i.e. it is in the query's gold_chunk_ids). Evidence coverage is computed
against evidence_units — each gold fact is one unit represented by the set of
(overlapping) chunks that contain it, so an overlap-duplicated fact counts once.

Metrics @k (k in top_k_values): Hit, Evidence Recall, Precision, MRR, MAP, nDCG,
All-Evidence Hit, Document Recall, plus First-Relevant-Rank and Complete-Evidence
-Rank (rank-based, over the full depth-10 ranking).

Reported overall / by question type / by required-document count, with 1000-
resample bootstrap CIs and paired-bootstrap significance (generated - baseline).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

import vo_config as C
import vo_data as D
import vo_retrieval as R

KS = C.TOP_K_VALUES
RATE_METRICS = [
    "hit",
    "evidence_recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
    "all_evidence_hit",
    "doc_recall",
]


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #
def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def per_query(row: dict) -> dict:
    ranked = [r["chunk_id"] for r in row["ranked"]]
    parents = [r["parent_document_id"] for r in row["ranked"]]
    gold = set(row["gold_chunk_ids"])
    units = [set(u) for u in row["evidence_units"]]
    req_docs = set(row["required_article_ids"])
    n_units = len(units)

    out = {
        "query_id": row["query_id"],
        "question_type": row["question_type"],
        "n_required_documents": row["n_required_documents"],
        "n_required_evidence_facts": row["n_required_evidence_facts"],
    }

    # rank-based (over full ranking)
    first_rel = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
    out["first_relevant_rank"] = first_rel
    complete_rank = None
    seen: set = set()
    for i, c in enumerate(ranked, 1):
        seen.add(c)
        if all(u & seen for u in units):
            complete_rank = i
            break
    out["complete_evidence_rank"] = complete_rank

    for k in KS:
        topk = ranked[:k]
        topk_set = set(topk)
        rel_flags = [1 if c in gold else 0 for c in topk]
        n_rel = sum(rel_flags)
        covered = sum(1 for u in units if u & topk_set)
        docs_hit = len(req_docs & set(parents[:k]))

        out[f"hit@{k}"] = 1.0 if n_rel > 0 else 0.0
        out[f"evidence_recall@{k}"] = covered / n_units if n_units else 0.0
        out[f"precision@{k}"] = n_rel / k
        # MRR@k
        fr = next((i + 1 for i, c in enumerate(topk) if c in gold), None)
        out[f"mrr@{k}"] = 1.0 / fr if fr else 0.0
        # MAP@k (average precision over relevant positions, normalized by min(gold,k))
        hits = 0
        ap = 0.0
        for i, c in enumerate(topk, 1):
            if c in gold:
                hits += 1
                ap += hits / i
        denom = min(len(gold), k) if gold else 1
        out[f"map@{k}"] = ap / denom
        # nDCG@k (binary gain)
        idcg = _dcg([1] * min(len(gold), k))
        out[f"ndcg@{k}"] = (_dcg(rel_flags) / idcg) if idcg > 0 else 0.0
        out[f"all_evidence_hit@{k}"] = 1.0 if (n_units and covered == n_units) else 0.0
        out[f"doc_recall@{k}"] = docs_hit / len(req_docs) if req_docs else 0.0
    return out


# --------------------------------------------------------------------------- #
# Aggregation + bootstrap
# --------------------------------------------------------------------------- #
def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _bootstrap_ci(vals: List[float], n: int, seed: int) -> List[float]:
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return [
        round(float(np.percentile(means, 2.5)), 4),
        round(float(np.percentile(means, 97.5)), 4),
    ]


def _paired_p(diffs: List[float], n: int, seed: int) -> float:
    """Two-sided paired bootstrap p-value that mean diff != 0."""
    if not diffs:
        return 1.0
    rng = np.random.default_rng(seed)
    arr = np.array(diffs)
    obs = arr.mean()
    centered = arr - obs
    boot = np.array(
        [centered[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    )
    p = (np.sum(np.abs(boot) >= abs(obs)) + 1) / (n + 1)
    return round(float(p), 4)


def aggregate(per_q: Dict[str, List[dict]]) -> dict:
    """per_q: condition -> list of per-query metric dicts (aligned by query)."""
    out: Dict[str, dict] = {}
    for cond, rows in per_q.items():
        agg = {}
        for m in RATE_METRICS:
            for k in KS:
                key = f"{m}@{k}"
                vals = [r[key] for r in rows]
                agg[key] = {
                    "mean": round(_mean(vals), 4),
                    "ci": _bootstrap_ci(vals, C.BOOTSTRAP_RESAMPLES, C.SEED),
                }
        frr = [r["first_relevant_rank"] for r in rows if r["first_relevant_rank"]]
        cer = [r["complete_evidence_rank"] for r in rows if r["complete_evidence_rank"]]
        agg["first_relevant_rank_mean"] = round(_mean(frr), 3) if frr else None
        agg["complete_evidence_rank_mean"] = round(_mean(cer), 3) if cer else None
        agg["complete_evidence_achieved_rate"] = round(len(cer) / len(rows), 4)
        out[cond] = agg
    return out


def paired(per_a: List[dict], per_b: List[dict]) -> dict:
    """Paired generated - baseline per metric@k. Same query order in both."""
    by_id_a = {r["query_id"]: r for r in per_a}
    result = {}
    for m in RATE_METRICS:
        for k in KS:
            key = f"{m}@{k}"
            diffs, imp, unch, harm = [], 0, 0, 0
            for rb in per_b:
                ra = by_id_a[rb["query_id"]]
                d = rb[key] - ra[key]
                diffs.append(d)
                if d > 1e-9:
                    imp += 1
                elif d < -1e-9:
                    harm += 1
                else:
                    unch += 1
            mean_d = _mean(diffs)
            result[key] = {
                "abs_improvement": round(mean_d, 4),
                "rel_pct": (
                    round(
                        100
                        * mean_d
                        / _mean([by_id_a[rb["query_id"]][key] for rb in per_b]),
                        1,
                    )
                    if _mean([by_id_a[rb["query_id"]][key] for rb in per_b]) > 1e-9
                    else None
                ),
                "improved": imp,
                "unchanged": unch,
                "harmed": harm,
                "paired_p": _paired_p(diffs, C.BOOTSTRAP_RESAMPLES, C.SEED),
            }
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _subset_means(rows: List[dict], keyfn) -> dict:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[str(keyfn(r))].append(r)
    return groups


def run_metrics() -> dict:
    base = R.load_rankings(C.COND_A)
    gen = R.load_rankings(C.COND_B)
    # align order
    base.sort(key=lambda r: r["query_id"])
    gen.sort(key=lambda r: r["query_id"])

    per_a = [per_query(r) for r in base]
    per_b = [per_query(r) for r in gen]

    overall = aggregate({C.COND_A: per_a, C.COND_B: per_b})
    pair = paired(per_a, per_b)

    # ---- breakdowns: by question type, by required-doc count ------------- #
    def breakdown(keyfn):
        ga = _subset_means(per_a, keyfn)
        gb = _subset_means(per_b, keyfn)
        rows = {}
        for grp in sorted(set(ga) | set(gb)):
            rows[grp] = {
                "n": len(gb.get(grp, [])),
                C.COND_A: {
                    f"{m}@{k}": round(
                        _mean([r[f"{m}@{k}"] for r in ga.get(grp, [])]), 4
                    )
                    for m in RATE_METRICS
                    for k in KS
                },
                C.COND_B: {
                    f"{m}@{k}": round(
                        _mean([r[f"{m}@{k}"] for r in gb.get(grp, [])]), 4
                    )
                    for m in RATE_METRICS
                    for k in KS
                },
            }
        return rows

    by_type = breakdown(lambda r: r["question_type"])
    by_docs = breakdown(lambda r: r["n_required_documents"])

    # ---- write CSVs ------------------------------------------------------ #
    _write_per_query_csv(per_a, per_b)
    _write_overall_csv(overall)
    _write_breakdown_csv(C.METRICS_BY_TYPE, by_type, "question_type")
    _write_breakdown_csv(C.METRICS_BY_DOCCOUNT, by_docs, "n_required_documents")

    rollup = {
        "k_values": KS,
        "n_queries": len(per_b),
        "overall": overall,
        "paired": pair,
        "by_question_type": by_type,
        "by_document_count": by_docs,
    }
    with C.PAIRED_JSON.open("w") as fh:
        json.dump(pair, fh, indent=2)
    with C.METRICS_JSON.open("w") as fh:
        json.dump(rollup, fh, indent=2)

    er5a = overall[C.COND_A]["evidence_recall@5"]["mean"]
    er5b = overall[C.COND_B]["evidence_recall@5"]["mean"]
    print(
        f"[metrics] EvidenceRecall@5  A={er5a}  B={er5b}  "
        f"Δ={round(er5b - er5a, 4)} (p={pair['evidence_recall@5']['paired_p']})"
    )
    return rollup


def _write_per_query_csv(per_a, per_b):
    cols = [
        "query_id",
        "question_type",
        "n_required_documents",
        "n_required_evidence_facts",
        "condition",
    ]
    metric_cols = [f"{m}@{k}" for m in RATE_METRICS for k in KS] + [
        "first_relevant_rank",
        "complete_evidence_rank",
    ]
    with C.PER_QUERY_METRICS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + metric_cols)
        for cond, rows in ((C.COND_A, per_a), (C.COND_B, per_b)):
            for r in rows:
                w.writerow(
                    [
                        r["query_id"],
                        r["question_type"],
                        r["n_required_documents"],
                        r["n_required_evidence_facts"],
                        cond,
                    ]
                    + [r.get(mc) for mc in metric_cols]
                )


def _write_overall_csv(overall):
    with C.OVERALL_METRICS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "metric",
                "k",
                C.COND_A,
                f"{C.COND_A}_ci_lo",
                f"{C.COND_A}_ci_hi",
                C.COND_B,
                f"{C.COND_B}_ci_lo",
                f"{C.COND_B}_ci_hi",
                "delta",
            ]
        )
        for m in RATE_METRICS:
            for k in KS:
                key = f"{m}@{k}"
                a = overall[C.COND_A][key]
                b = overall[C.COND_B][key]
                w.writerow(
                    [
                        m,
                        k,
                        a["mean"],
                        a["ci"][0],
                        a["ci"][1],
                        b["mean"],
                        b["ci"][0],
                        b["ci"][1],
                        round(b["mean"] - a["mean"], 4),
                    ]
                )


def _write_breakdown_csv(path, rows, label):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [label, "n", "condition"] + [f"{m}@{k}" for m in RATE_METRICS for k in KS]
        )
        for grp, d in rows.items():
            for cond in (C.COND_A, C.COND_B):
                w.writerow(
                    [grp, d["n"], cond]
                    + [d[cond][f"{m}@{k}"] for m in RATE_METRICS for k in KS]
                )


if __name__ == "__main__":
    run_metrics()
