"""
Stage: qualitative failure analysis (§19).

Buckets each query by how Condition B (generated-question retrieval) compares to
Condition A (chunk retrieval) at the answer k, and diagnoses the generated-
question-specific failure modes: does the top synthetic question point to a
non-gold parent chunk while a lower-ranked one points to the correct chunk?
partial vs complete multi-hop coverage? Exports representative examples.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List

import vo_config as C
import vo_data as D
import vo_retrieval as R

K = 5


def _covered(units, chunk_ids) -> int:
    s = set(chunk_ids)
    return sum(1 for u in units if set(u) & s)


def run_failure_analysis() -> dict:
    base = {r["query_id"]: r for r in R.load_rankings(C.COND_A)}
    gen = {r["query_id"]: r for r in R.load_rankings(C.COND_B)}
    queries = D.load_eligible_queries()

    buckets = Counter()
    diag = Counter()
    examples: List[dict] = []

    for q in queries:
        qid = q["query_id"]
        a, b = base[qid], gen[qid]
        gold = set(q["gold_chunk_ids"])
        units = [set(u) for u in b["evidence_units"]]
        n_units = len(units)

        a_top = [r["chunk_id"] for r in a["ranked"][:K]]
        b_top = [r["chunk_id"] for r in b["ranked"][:K]]
        a_cov, b_cov = _covered(units, a_top), _covered(units, b_top)

        # improved / hurt / unchanged (by evidence coverage @K)
        if b_cov > a_cov:
            buckets["improved_by_generated"] += 1
        elif b_cov < a_cov:
            buckets["hurt_by_generated"] += 1
        else:
            buckets["unchanged"] += 1

        # multi-hop completeness under generated retrieval
        if n_units > 1:
            if b_cov == n_units:
                buckets["complete_multihop_generated"] += 1
            elif b_cov > 0:
                buckets["partial_multihop_generated"] += 1
            else:
                buckets["no_evidence_generated"] += 1

        # generated-question-specific diagnostics
        b_full = b["ranked"]
        if b_full:
            top_parent = b_full[0]["chunk_id"]
            if top_parent not in gold:
                diag["top_question_points_to_nongold_parent"] += 1
                # did a lower-ranked question point to a gold chunk?
                if any(r["chunk_id"] in gold for r in b_full[1:]):
                    diag["lower_ranked_question_hits_gold"] += 1
            else:
                diag["top_question_points_to_gold_parent"] += 1
        # correct chunk retrieved but ranked low (present in depth-10 but not top-K)
        b_all = {r["chunk_id"] for r in b_full}
        if (gold & b_all) and not (gold & set(b_top)):
            diag["gold_present_but_below_topk"] += 1

        # collect a few representative contrast examples
        if len(examples) < 12 and (b_cov != a_cov):
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
                    "generated_ranked": [
                        {
                            "rank": r["rank"],
                            "chunk_id": r["chunk_id"],
                            "score": r["score"],
                            "is_gold": r["chunk_id"] in gold,
                            "best_question": r.get("best_question", ""),
                        }
                        for r in b["ranked"][:K]
                    ],
                    "baseline_evidence_covered": a_cov,
                    "generated_evidence_covered": b_cov,
                    "verdict": "generated better"
                    if b_cov > a_cov
                    else "baseline better",
                }
            )

    D._write_jsonl(C.FAILURE_JSONL, examples)

    lines = [
        "# Failure analysis — generated-question vs chunk retrieval (@k=5)\n",
        f"Queries: {len(queries)}\n",
        "## Query buckets\n",
    ]
    for k, v in buckets.most_common():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Generated-question diagnostics\n")
    for k, v in diag.most_common():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Representative contrast examples\n")
    for ex in examples[:6]:
        lines.append(
            f"### {ex['query_id']} ({ex['question_type']}, "
            f"{ex['n_required_evidence_facts']} facts) — {ex['verdict']}"
        )
        lines.append(f"> {ex['query']}")
        lines.append(
            f"- gold answer: `{ex['gold_answer']}`  | gold chunks: {ex['gold_chunk_ids']}"
        )
        lines.append(
            f"- baseline covered {ex['baseline_evidence_covered']} / "
            f"generated covered {ex['generated_evidence_covered']}"
        )
        top_bq = ex["generated_ranked"][0].get("best_question", "")
        lines.append(f"- B top match question: _{top_bq}_\n")
    C.FAILURE_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "buckets": dict(buckets),
        "diagnostics": dict(diag),
        "n_examples": len(examples),
        "examples": examples,
    }
    with (C.RESULTS_DIR / "failure_rollup.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[failure] {dict(buckets)}")
    print(f"[failure] {dict(diag)}")
    return result


if __name__ == "__main__":
    run_failure_analysis()
