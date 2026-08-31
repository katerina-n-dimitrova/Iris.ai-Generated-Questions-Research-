"""
Experiment 1 analysis — the semantic-type cross-tab.

Question: does having a generated question OF THE SAME SEMANTIC TYPE as the user
query, sitting on the gold evidence chunk, make that query more likely to be
retrieved? We classify both the real queries and the E1 generated questions with
the same Cao & Wang ontology, then cross-tabulate, per query type:

    retrieval success (hit@10, E1 hybrid)  ×  same-type generated question
                                              present on a gold chunk (yes/no)

Everything is computed from cached classifications + saved rankings; no retrieval
or generation runs here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import qasper_config as C
import qasper_data as D
import qasper_generate as G
import qasper_ontology as O
import qasper_retrieval as R

SUCCESS_K = 10  # hit@10 defines "retrieval success" for the cross-tab


def _hit(ranked: List[str], gold: List[str], k: int) -> int:
    gs = set(gold)
    return int(any(c in gs for c in ranked[:k]))


def compute_crosstab(arm: str = "B1", mode: str = "hybrid") -> dict:
    """Cross-tab on ``arm``: presence of a same-type generated question on the
    gold chunk vs that arm's retrieval success. Meaningful on the naive arm (B1),
    where type coverage varies; on the type-stratified arm (E1) presence is ~always
    true by construction (that IS the point)."""
    queries = D.load_queries()
    query_types = O.classify_queries(queries)  # query_id -> type
    qtypes_by_chunk = O.classify_arm_questions(
        arm, G.load_questions(arm)
    )  # chunk -> [types]
    rankings = {r["query_id"]: r for r in R.load_rankings(arm, mode)}

    # per query: query type, success, whether a same-type gen q sits on a gold chunk
    cells: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: {"present": [], "absent": []}
    )
    overall = {"present": [], "absent": []}
    per_query = []
    for q in queries:
        qid = q["query_id"]
        tq = query_types.get(qid, "CONCEPT")
        r = rankings.get(qid)
        if r is None:
            continue
        success = _hit(r["ranked"], q["gold_chunk_ids"], SUCCESS_K)
        present = any(
            tq in set(qtypes_by_chunk.get(g, [])) for g in q["gold_chunk_ids"]
        )
        bucket = "present" if present else "absent"
        cells[tq][bucket].append(success)
        overall[bucket].append(success)
        per_query.append(
            {
                "query_id": qid,
                "query_type": tq,
                "success": success,
                "same_type_on_gold": present,
            }
        )

    def _rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    table = {}
    for t in O.TYPES:
        pres, abst = cells[t]["present"], cells[t]["absent"]
        table[t] = {
            "n": len(pres) + len(abst),
            "n_present": len(pres),
            "success_present": _rate(pres),
            "n_absent": len(abst),
            "success_absent": _rate(abst),
        }
    result = {
        "arm": arm,
        "mode": mode,
        "success_metric": f"hit@{SUCCESS_K}",
        "by_query_type": table,
        "overall": {
            "success_present": _rate(overall["present"]),
            "n_present": len(overall["present"]),
            "success_absent": _rate(overall["absent"]),
            "n_absent": len(overall["absent"]),
        },
        "per_query": per_query,
    }
    return result


def _coverage(arm: str) -> dict:
    """Fraction of queries whose gold chunk carries a same-type generated question
    in ``arm`` (i.e. how well the arm's questions cover the query's type)."""
    queries = D.load_queries()
    query_types = O.classify_queries(queries)
    qtypes_by_chunk = O.classify_arm_questions(arm, G.load_questions(arm))
    present = 0
    for q in queries:
        tq = query_types.get(q["query_id"], "CONCEPT")
        if any(tq in set(qtypes_by_chunk.get(g, [])) for g in q["gold_chunk_ids"]):
            present += 1
    return {
        "arm": arm,
        "coverage": round(present / len(queries), 3),
        "n_present": present,
        "n": len(queries),
    }


def compute_exp1() -> dict:
    """Full Experiment-1 analysis: the B1 mechanism cross-tab + E1 vs B1 coverage."""
    b1 = compute_crosstab("B1", "hybrid")  # mechanism: presence varies
    out = {
        "crosstab_arm": "B1",
        "crosstab": b1,
        "coverage": {"B1": _coverage("B1"), "E1": _coverage("E1")},
    }
    with (C.RESULTS_DIR / "exp1_crosstab.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    o = compute_exp1()
    r = o["crosstab"]
    print("Same-type-on-gold coverage (query's type present on a gold chunk):")
    print(f"  B1 naive        : {o['coverage']['B1']['coverage'] * 100:.0f}%")
    print(f"  E1 type-stratified: {o['coverage']['E1']['coverage'] * 100:.0f}%")
    print("\nB1 cross-tab — hit@10 by query type × same-type question on gold:")
    print(
        f"  overall present: {r['overall']['success_present']} (n={r['overall']['n_present']}) | "
        f"absent: {r['overall']['success_absent']} (n={r['overall']['n_absent']})"
    )
    print(f"  {'type':<13}{'n':>3}{'present(hit)':>16}{'absent(hit)':>16}")
    for t, row in r["by_query_type"].items():
        if row["n"]:
            print(
                f"  {t:<13}{row['n']:>3}"
                f"{f'''{row['n_present']}({row['success_present']})''':>16}"
                f"{f'''{row['n_absent']}({row['success_absent']})''':>16}"
            )
