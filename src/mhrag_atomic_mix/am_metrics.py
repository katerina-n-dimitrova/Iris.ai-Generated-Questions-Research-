"""
Stage: retrieval evaluation + paired comparison (§20, §21).

Reuses the validated metric math from the 15-article harness (`vo_metrics`):
per-query Hit / Evidence Recall / All-Evidence Hit / Precision / MRR / MAP / nDCG /
Doc Recall + first-relevant & complete-evidence rank, plus bootstrap CIs and
paired-bootstrap significance. Primary metric = Evidence Recall@5. Only the paths,
condition labels (A vs E), and CSV writers are experiment-specific here.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import Dict, List

import am_config as C
import am_retrieval as R
import vo_metrics as VM  # reuse per_query / aggregate / paired / bootstrap

KS = C.TOP_K_VALUES
RATE = VM.RATE_METRICS
# keep vo_metrics' condition keys in sync with this experiment (baseline/generated)
VM.KS = KS
VM.C.SEED = C.SEED
VM.C.BOOTSTRAP_RESAMPLES = C.BOOTSTRAP_RESAMPLES


def run_metrics() -> dict:
    base = sorted(R.load_rankings(C.COND_A), key=lambda r: r["query_id"])
    mixed = sorted(R.load_rankings(C.COND_E), key=lambda r: r["query_id"])
    per_a = [VM.per_query(r) for r in base]
    per_e = [VM.per_query(r) for r in mixed]

    overall = VM.aggregate({C.COND_A: per_a, C.COND_E: per_e})
    pair = VM.paired(per_a, per_e)

    def breakdown(keyfn):
        ga, gb = defaultdict(list), defaultdict(list)
        for r in per_a:
            ga[str(keyfn(r))].append(r)
        for r in per_e:
            gb[str(keyfn(r))].append(r)
        rows = {}
        for grp in sorted(set(ga) | set(gb)):
            rows[grp] = {
                "n": len(gb.get(grp, [])),
                C.COND_A: {
                    f"{m}@{k}": round(
                        _mean([r[f"{m}@{k}"] for r in ga.get(grp, [])]), 4
                    )
                    for m in RATE
                    for k in KS
                },
                C.COND_E: {
                    f"{m}@{k}": round(
                        _mean([r[f"{m}@{k}"] for r in gb.get(grp, [])]), 4
                    )
                    for m in RATE
                    for k in KS
                },
            }
        return rows

    by_type = breakdown(lambda r: r["question_type"])
    by_docs = breakdown(lambda r: r["n_required_documents"])

    _per_query_csv(per_a, per_e)
    _overall_csv(overall, pair)
    _breakdown_csv(C.METRICS_BY_TYPE, by_type, "question_type")
    _breakdown_csv(C.METRICS_BY_DOCCOUNT, by_docs, "n_required_documents")

    rollup = {
        "k_values": KS,
        "n_queries": len(per_e),
        "overall": overall,
        "paired": pair,
        "by_question_type": by_type,
        "by_document_count": by_docs,
    }
    json.dump(rollup, open(C.METRICS_JSON, "w"), indent=2)
    er5a = overall[C.COND_A]["evidence_recall@5"]["mean"]
    er5e = overall[C.COND_E]["evidence_recall@5"]["mean"]
    print(
        f"[metrics] EvidenceRecall@5  A={er5a}  E={er5e}  Δ={round(er5e - er5a, 4)} "
        f"(p={pair['evidence_recall@5']['paired_p']})"
    )
    return rollup


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def _per_query_csv(per_a, per_e):
    metric_cols = [f"{m}@{k}" for m in RATE for k in KS] + [
        "first_relevant_rank",
        "complete_evidence_rank",
    ]
    with C.PER_QUERY_METRICS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "query_id",
                "question_type",
                "n_required_documents",
                "n_required_evidence_facts",
                "condition",
            ]
            + metric_cols
        )
        for cond, rows in ((C.COND_A, per_a), (C.COND_E, per_e)):
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


def _overall_csv(overall, pair):
    with C.OVERALL_METRICS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "metric",
                "k",
                "A_baseline",
                "A_ci_lo",
                "A_ci_hi",
                "E_mixed",
                "E_ci_lo",
                "E_ci_hi",
                "delta_E_minus_A",
                "paired_p",
            ]
        )
        for m in RATE:
            for k in KS:
                key = f"{m}@{k}"
                a = overall[C.COND_A][key]
                e = overall[C.COND_E][key]
                w.writerow(
                    [
                        m,
                        k,
                        a["mean"],
                        a["ci"][0],
                        a["ci"][1],
                        e["mean"],
                        e["ci"][0],
                        e["ci"][1],
                        round(e["mean"] - a["mean"], 4),
                        pair[key]["paired_p"],
                    ]
                )


def _breakdown_csv(path, rows, label):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([label, "n", "condition"] + [f"{m}@{k}" for m in RATE for k in KS])
        for grp, d in rows.items():
            for cond in (C.COND_A, C.COND_E):
                w.writerow(
                    [grp, d["n"], cond]
                    + [d[cond][f"{m}@{k}"] for m in RATE for k in KS]
                )


if __name__ == "__main__":
    run_metrics()
