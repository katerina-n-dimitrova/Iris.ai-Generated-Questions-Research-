"""
Experiment 5 — style-match few-shot exemplars (leakage-free).

Selects 8 REAL MultiHop-RAG queries to use as style exemplars for the few-shot
arm (E5b). To avoid any leakage, an exemplar query must:
  * NOT be in the evaluation query set, AND
  * have ALL of its gold evidence articles OUTSIDE the selected corpus.
Both conditions are asserted here (a violation raises), so the few-shot prompt
can never see a query or article that is scored at eval time.

Cached to data/processed/multihoprag/style_exemplars.json for reproducibility.
"""

from __future__ import annotations

import json
import random
from typing import List

import mhrag_config as C
import mhrag_data as D

N_EXEMPLARS = int(__import__("os").getenv("MHRAG_STYLE_N", "8"))
_PATH = C.PROCESSED_DIR / "style_exemplars.json"


def _select() -> dict:
    corpus = D.load_corpus()
    all_q = D.load_all_queries()
    selected_articles = set(json.load(C.SELECTED_ARTICLES_PATH.open())["article_ids"])
    eval_ids = set(json.load(C.SELECTED_QUERIES_PATH.open()))

    candidates = []
    for q in all_q:
        if q["question_type"] == "null_query" or not q.get("evidence_list"):
            continue
        if q["query_id"] in eval_ids:
            continue
        arts = {e[C.ARTICLE_ID_FIELD] for e in q["evidence_list"]}
        if arts & selected_articles:  # any overlap with the corpus -> skip
            continue
        candidates.append(q)

    rng = random.Random(C.SELECTION_SEED)
    rng.shuffle(candidates)
    chosen = candidates[:N_EXEMPLARS]

    # ---- leakage assertions ---- #
    for q in chosen:
        arts = {e[C.ARTICLE_ID_FIELD] for e in q["evidence_list"]}
        assert not (arts & selected_articles), (
            f"LEAKAGE: exemplar {q['query_id']} cites a selected article"
        )
        assert q["query_id"] not in eval_ids, (
            f"LEAKAGE: exemplar {q['query_id']} is in the eval set"
        )

    out = {
        "n_requested": N_EXEMPLARS,
        "n_selected": len(chosen),
        "seed": C.SELECTION_SEED,
        "num_candidates": len(candidates),
        "leakage_check": "passed: no exemplar cites a selected article or is in eval",
        "query_ids": [q["query_id"] for q in chosen],
        "queries": [q["query"].strip() for q in chosen],
        "types": [q["question_type"] for q in chosen],
    }
    json.dump(out, _PATH.open("w"), indent=2)
    return out


def build(force: bool = False) -> dict:
    if _PATH.exists() and not force:
        return json.load(_PATH.open())
    return _select()


def load_exemplars() -> List[str]:
    """The exemplar query strings (builds + caches on first use)."""
    return build()["queries"]


if __name__ == "__main__":
    import pprint

    pprint.pp(build(force=True))
