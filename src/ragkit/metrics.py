"""Evidence-retrieval metrics shared by every experiment.

Every query carries ``evidence_units``: one set of acceptable chunk ids per
required evidence fact. Recall@k is the fraction of units with at least one
hit in the top k; MRR@10 uses the first rank that hits any unit.
"""

from __future__ import annotations


def metric_row(query: dict, ranking: list[str]) -> dict:
    units = [set(unit) for unit in query["evidence_units"]]

    def recall(k):
        found = set(ranking[:k])
        return sum(bool(unit & found) for unit in units) / len(units)

    first = next(
        (
            rank
            for rank, cid in enumerate(ranking[:10], 1)
            if any(cid in unit for unit in units)
        ),
        None,
    )
    return {
        "evidence_recall@1": recall(1),
        "evidence_recall@5": recall(5),
        "evidence_recall@10": recall(10),
        "all_evidence_hit@5": float(recall(5) == 1),
        "mrr@10": 0 if first is None else 1 / first,
    }
