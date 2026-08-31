"""
Stage: qualitative failure analysis, Condition A vs E (§24).

Buckets each query by how E compares to A at k=5 (evidence coverage) and diagnoses
the mixed-question failure modes: generic chunk-level false positives, atomic
questions too narrow, gold present below top-k, top match to a non-gold chunk with
a lower-ranked question hitting gold. Exports representative contrast examples.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import List

import am_config as C
import am_data as D
import am_retrieval as R

K = 5


def _covered(units, chunk_ids):
    s = set(chunk_ids)
    return sum(1 for u in units if set(u) & s)


def run_failure_analysis() -> dict:
    base = {r["query_id"]: r for r in R.load_rankings(C.COND_A)}
    mixed = {r["query_id"]: r for r in R.load_rankings(C.COND_E)}
    queries = D.load_eligible_queries()

    buckets, diag = Counter(), Counter()
    examples: List[dict] = []
    for q in queries:
        qid = q["query_id"]
        a, e = base[qid], mixed[qid]
        gold = set(q["gold_chunk_ids"])
        units = [set(u) for u in e["evidence_units"]]
        n_units = len(units)
        a_top = [r["chunk_id"] for r in a["ranked"][:K]]
        e_top = [r["chunk_id"] for r in e["ranked"][:K]]
        a_cov, e_cov = _covered(units, a_top), _covered(units, e_top)

        buckets[
            "improved_by_E"
            if e_cov > a_cov
            else ("hurt_by_E" if e_cov < a_cov else "unchanged")
        ] += 1
        if n_units > 1:
            buckets[
                "complete_multihop_E"
                if e_cov == n_units
                else ("partial_multihop_E" if e_cov > 0 else "no_evidence_E")
            ] += 1

        e_full = e["ranked"]
        if e_full:
            if e_full[0]["chunk_id"] not in gold:
                diag["top_match_nongold"] += 1
                if e_full[0].get("best_question_type") == "chunk_level":
                    diag["top_nongold_from_chunk_level"] += 1
                else:
                    diag["top_nongold_from_atomic"] += 1
                if any(r["chunk_id"] in gold for r in e_full[1:]):
                    diag["lower_ranked_hits_gold"] += 1
            else:
                diag["top_match_gold"] += 1
        e_all = {r["chunk_id"] for r in e_full}
        if (gold & e_all) and not (gold & set(e_top)):
            diag["gold_present_below_topk"] += 1

        if len(examples) < 14 and e_cov != a_cov:
            examples.append(
                {
                    "query_id": qid,
                    "query": q["query"],
                    "question_type": q["question_type"],
                    "n_required_evidence_facts": n_units,
                    "gold_answer": q["gold_answer"],
                    "gold_chunk_ids": sorted(gold),
                    "baseline_ranked": [
                        {
                            "rank": r["rank"],
                            "chunk_id": r["chunk_id"],
                            "score": r["score"],
                            "is_gold": r["chunk_id"] in gold,
                        }
                        for r in a["ranked"][:K]
                    ],
                    "mixed_ranked": [
                        {
                            "rank": r["rank"],
                            "chunk_id": r["chunk_id"],
                            "score": r["score"],
                            "is_gold": r["chunk_id"] in gold,
                            "best_question": r.get("best_question", ""),
                            "best_question_type": r.get("best_question_type", ""),
                        }
                        for r in e["ranked"][:K]
                    ],
                    "baseline_evidence_covered": a_cov,
                    "mixed_evidence_covered": e_cov,
                    "verdict": "E better" if e_cov > a_cov else "A better",
                }
            )

    D._write_jsonl(C.FAILURE_JSONL, examples)
    out = {
        "buckets": dict(buckets),
        "diagnostics": dict(diag),
        "n_examples": len(examples),
        "examples": examples,
    }
    json.dump(out, open(C.RESULTS_DIR / "failure_rollup.json", "w"), indent=2)
    print(f"[failure] {dict(buckets)}")
    print(f"[failure] {dict(diag)}")
    return out


if __name__ == "__main__":
    run_failure_analysis()
