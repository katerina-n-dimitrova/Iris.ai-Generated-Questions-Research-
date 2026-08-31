"""
Stage: atomic vs chunk-level question-type diagnostics (§22).

The two question types are POOLED for ranking; this stage measures their separate
contributions after the fact: which type won the match on relevant (gold) vs
non-gold retrieved chunks, which type reached gold at lower ranks, and how each
type behaves across benchmark question types. Diagnostic only — never used to
build a separate retrieval condition.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import am_config as C
import am_data as D
import am_retrieval as R

K = 5


def run_diagnostics() -> dict:
    filt = D.read_jsonl(C.QUESTIONS_FILTERED)
    n_atomic = sum(1 for q in filt if q["question_type"] == "atomic")
    mixed = R.load_rankings(C.COND_E)

    top1_type = Counter()
    gold_match_type = Counter()  # winning type on gold chunks in top-K
    fp_match_type = Counter()  # winning type on non-gold chunks in top-K
    lower_gold_type = Counter()  # gold chunks matched at rank>1, by type
    by_qtype_gold = defaultdict(Counter)  # benchmark type -> winning type on gold

    for row in mixed:
        gold = set(row["gold_chunk_ids"])
        ranked = row["ranked"][:K]
        if ranked:
            top1_type[ranked[0].get("best_question_type", "?")] += 1
        for r in ranked:
            t = r.get("best_question_type", "?")
            if r["chunk_id"] in gold:
                gold_match_type[t] += 1
                by_qtype_gold[row["question_type"]][t] += 1
                if r["rank"] > 1:
                    lower_gold_type[t] += 1
            else:
                fp_match_type[t] += 1

    out = {
        "accepted_atomic_questions": n_atomic,
        "accepted_chunk_level_questions": len(filt) - n_atomic,
        "avg_accepted_per_chunk": round(len(filt) / len(D.load_chunks()), 2),
        "top1_match_type": dict(top1_type),
        "gold_match_winning_type": dict(gold_match_type),
        "false_positive_winning_type": dict(fp_match_type),
        "lower_ranked_gold_winning_type": dict(lower_gold_type),
        "gold_match_type_by_benchmark_qtype": {
            k: dict(v) for k, v in by_qtype_gold.items()
        },
    }
    json.dump(out, open(C.DIAGNOSTICS_JSON, "w"), indent=2)

    with C.DIAGNOSTICS_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["diagnostic", "atomic", "chunk_level"])
        for label, d in [
            ("top1_match_type", top1_type),
            ("gold_match_winning_type", gold_match_type),
            ("false_positive_winning_type", fp_match_type),
            ("lower_ranked_gold_winning_type", lower_gold_type),
        ]:
            w.writerow([label, d.get("atomic", 0), d.get("chunk_level", 0)])
    print(
        f"[diag] gold matches: atomic {gold_match_type['atomic']} / "
        f"chunk-level {gold_match_type['chunk_level']} | "
        f"false positives: atomic {fp_match_type['atomic']} / chunk-level {fp_match_type['chunk_level']}"
    )
    return out


if __name__ == "__main__":
    import pprint

    pprint.pp(run_diagnostics())
