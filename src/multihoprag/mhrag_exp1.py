"""
Experiment 1 analysis — the semantic-type cross-tab.

The spec asks for a cross-tab against MultiHop-RAG's OWN query-type labels
(inference / comparison / temporal): retrieval success of each query type ×
whether a same-or-related-type generated question existed on the query's gold
chunks. "Same-or-related type" is decided in the Cao & Wang space: we classify
each real query into a Cao & Wang type and check whether any of its gold chunks
carries a generated question of that same Cao & Wang type.

Everything is computed from cached classifications + saved rankings; no retrieval
or generation runs here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import mhrag_config as C
import mhrag_data as D
import mhrag_generate as G
import mhrag_ontology as O
import mhrag_retrieval as R

SUCCESS_K = 5  # hit@5 defines "retrieval success" for the cross-tab


def _hit(ranked: List[str], gold: List[str], k: int) -> int:
    gs = set(gold)
    return int(any(c in gs for c in ranked[:k]))


def compute_crosstab(arm: str = "B1", mode: str = "hybrid") -> dict:
    """Cross-tab on ``arm``, grouped by MultiHop-RAG query type: presence of a
    same-(Cao&Wang)-type generated question on the gold chunk vs the arm's
    retrieval success. Meaningful on the naive arm (B1), where type coverage
    varies; on E1 presence is ~always true by construction (that is the point)."""
    queries = D.load_queries()
    cw_query_types = O.classify_queries(queries)  # query_id -> CW type
    cw_by_chunk = O.classify_arm_questions(arm, G.load_questions(arm))  # chunk -> [CW]
    rankings = {r["query_id"]: r for r in R.load_rankings(arm, mode)}

    cells: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: {"present": [], "absent": []}
    )
    overall = {"present": [], "absent": []}
    per_query = []
    for q in queries:
        qid = q["query_id"]
        mh_type = q["query_type"]  # inference/comparison/temporal
        cw_type = cw_query_types.get(qid, "CONCEPT")
        r = rankings.get(qid)
        if r is None:
            continue
        success = _hit(r["ranked"], q["gold_chunk_ids"], SUCCESS_K)
        present = any(
            cw_type in set(cw_by_chunk.get(g, [])) for g in q["gold_chunk_ids"]
        )
        bucket = "present" if present else "absent"
        cells[mh_type][bucket].append(success)
        overall[bucket].append(success)
        per_query.append(
            {
                "query_id": qid,
                "mh_type": mh_type,
                "cw_type": cw_type,
                "success": success,
                "same_type_on_gold": present,
            }
        )

    def _rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    table = {}
    for t in C.QUERY_TYPES:
        pres, abst = cells[t]["present"], cells[t]["absent"]
        table[t] = {
            "n": len(pres) + len(abst),
            "n_present": len(pres),
            "success_present": _rate(pres),
            "n_absent": len(abst),
            "success_absent": _rate(abst),
        }
    return {
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


def _coverage(arm: str) -> dict:
    """Fraction of queries whose gold chunk carries a same-Cao&Wang-type generated
    question in ``arm`` (how well the arm's questions cover the query's type)."""
    queries = D.load_queries()
    cw_query_types = O.classify_queries(queries)
    cw_by_chunk = O.classify_arm_questions(arm, G.load_questions(arm))
    present = 0
    for q in queries:
        tq = cw_query_types.get(q["query_id"], "CONCEPT")
        if any(tq in set(cw_by_chunk.get(g, [])) for g in q["gold_chunk_ids"]):
            present += 1
    return {
        "arm": arm,
        "coverage": round(present / len(queries), 3),
        "n_present": present,
        "n": len(queries),
    }


def _type_distribution(arm: str) -> Dict[str, int]:
    """How many generated questions of each Cao&Wang type the arm produced."""
    from collections import Counter

    cw_by_chunk = O.classify_arm_questions(arm, G.load_questions(arm))
    c = Counter(t for labels in cw_by_chunk.values() for t in labels)
    return {t: c.get(t, 0) for t in O.TYPES}


def compute_exp1() -> dict:
    """Full Experiment-1 analysis: the B1 mechanism cross-tab + E1/B1 coverage +
    generated-question type distributions."""
    out = {
        "crosstab_arm": "B1",
        "crosstab": compute_crosstab("B1", "hybrid"),
        "coverage": {"B1": _coverage("B1"), "E1": _coverage("E1")},
        "type_distribution": {
            "B1": _type_distribution("B1"),
            "E1": _type_distribution("E1"),
        },
    }
    with (C.RESULTS_DIR / "exp1_crosstab.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    o = compute_exp1()
    r = o["crosstab"]
    print("Same-type-on-gold coverage:")
    print(f"  B1 naive          : {o['coverage']['B1']['coverage'] * 100:.0f}%")
    print(f"  E1 type-stratified: {o['coverage']['E1']['coverage'] * 100:.0f}%")
    print(
        f"\nB1 cross-tab — hit@{r['success_metric']} by MultiHop query type × "
        "same-type question on gold:"
    )
    print(
        f"  overall present: {r['overall']['success_present']} "
        f"(n={r['overall']['n_present']}) | absent: "
        f"{r['overall']['success_absent']} (n={r['overall']['n_absent']})"
    )
    for t, row in r["by_query_type"].items():
        print(
            f"  {t:<12} n={row['n']:>3}  present {row['success_present']} "
            f"(n={row['n_present']})  absent {row['success_absent']} (n={row['n_absent']})"
        )
