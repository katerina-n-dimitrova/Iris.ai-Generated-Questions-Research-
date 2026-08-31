"""Sweep routed-document breadth using the existing verified GEPA questions."""

from __future__ import annotations

import json

import numpy as np

import vo_hierarchical_hybrid as H
from embeddings import get_embedder

DOCQ = H.DATA / "gepa_verified_document_questions.jsonl"
OUT = H.RESULTS / "gepa" / "route_k_sweep.json"


def main():
    doc_rows = H.read_jsonl(DOCQ)
    index = H.build_indexes(doc_rows)
    queries = H.load_queries()
    gold = H.load_gold()
    analyses = {x["query_id"]: x for x in H.query_understanding(False)}
    embedder = get_embedder()
    result = {}
    original_k = H.DOC_ROUTE_K
    try:
        for route_k in (3, 5, 7, 10, 15):
            H.DOC_ROUTE_K = route_k
            values = []
            for query in queries:
                qvec = np.asarray(
                    embedder.embed_query(query["query"]), dtype=np.float64
                )
                routed, _ = H.route_documents(index, qvec, analyses[query["query_id"]])
                ranked, _ = H._chunk_hybrid(
                    index, qvec, analyses[query["query_id"]], set(routed)
                )
                row = H._as_metric_row(
                    query,
                    gold[query["query_id"]],
                    ranked[: H.C.RANK_DEPTH],
                    index.chunk_map,
                )
                values.append(H.VM.per_query(row)["evidence_recall@5"])
            result[str(route_k)] = float(np.mean(values))
            print(f"route_k={route_k} Evidence Recall@5={np.mean(values):.4f}")
    finally:
        H.DOC_ROUTE_K = original_k
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
