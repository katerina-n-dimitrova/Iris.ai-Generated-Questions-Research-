"""
Similarity-gated generated-question routing with Flat-hybrid fallback.

For each query:
  1. Search the verified generated document questions by dense cosine.
  2. If the best question similarity is at least ``--threshold``, retrieve
     chunks only from the documents represented by the best question matches.
  3. Otherwise, fall back to unrestricted Flat-hybrid chunk retrieval
     (BM25 + dense vectors + RRF).

The experiment reuses the exact 512/128 dataset and persisted Iris indexes from
``vo_hierarchical_hybrid.py``. Query vectors are cached after the first run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

import vo_config as C
import vo_hierarchical_hybrid as H
import vo_metrics as VM
from embeddings import get_embedder


TAG = "question_similarity_fallback"
RESULTS = C.RESULTS_DIR / TAG
RESULTS.mkdir(parents=True, exist_ok=True)
QUERY_VECTORS_PATH = RESULTS / "query_vectors.json"
RANKINGS_PATH = RESULTS / "rankings.json"
METRICS_PATH = RESULTS / "metrics.json"
REPORT = C.PROJECT_ROOT / "report" / "mhrag_question_similarity_fallback.html"

DEFAULT_THRESHOLD = 0.50


def _load_persisted_indexes() -> H.Indexes:
    """Load vectors from Chroma without re-embedding or rebuilding indexes."""
    chunk_result = C.get_collection(H.TEXT_COLL).get(
        include=["embeddings", "documents", "metadatas"]
    )
    question_result = C.get_collection(H.DOCQ_COLL).get(
        include=["embeddings", "documents", "metadatas"]
    )
    if len(chunk_result["ids"]) != 93 or len(question_result["ids"]) != 150:
        raise RuntimeError(
            "Expected the existing 93-chunk and 150-question hierarchical "
            "indexes. Run vo_hierarchical_hybrid.py once first."
        )

    source_chunks = {c["chunk_id"]: c for c in H.load_chunks()}
    chunks = [source_chunks[cid] for cid in chunk_result["ids"]]
    questions = [
        {
            "question_id": qid,
            "question": text,
            "document_id": meta["document_id"],
        }
        for qid, text, meta in zip(
            question_result["ids"],
            question_result["documents"],
            question_result["metadatas"],
        )
    ]
    return H.Indexes(
        chunks=chunks,
        chunk_map={c["chunk_id"]: c for c in chunks},
        chunk_vectors=np.asarray(chunk_result["embeddings"], dtype=float),
        chunk_bm25=BM25Okapi([H.tokens(c["content"]) for c in chunks]),
        doc_questions=questions,
        docq_vectors=np.asarray(question_result["embeddings"], dtype=float),
        docq_bm25=BM25Okapi([H.tokens(q["question"]) for q in questions]),
    )


def _query_vectors(queries: List[dict], force: bool = False) -> Dict[str, list]:
    if QUERY_VECTORS_PATH.exists() and not force:
        cached = json.loads(QUERY_VECTORS_PATH.read_text())
        if set(cached) == {q["query_id"] for q in queries}:
            return cached
    embedder = get_embedder()
    vectors = {}
    for pos, q in enumerate(queries, 1):
        vectors[q["query_id"]] = embedder.embed_query(q["query"])
        if pos % 10 == 0 or pos == len(queries):
            print(f"[question-fallback:embed] {pos}/{len(queries)}")
    QUERY_VECTORS_PATH.write_text(json.dumps(vectors))
    return vectors


def _embedding_signature() -> str:
    """Reuse the signature recorded when the persisted indexes were built."""
    if H.METRICS_PATH.exists():
        prior = json.loads(H.METRICS_PATH.read_text())
        if prior.get("embedding_signature"):
            return prior["embedding_signature"]
    return "unknown:persisted-hierarchical-index"


def _question_route(index: H.Indexes, qvec: np.ndarray) -> tuple[List[str], dict]:
    """Rank documents by their maximum generated-question cosine similarity."""
    scores = H._cosine_matrix(index.docq_vectors, qvec)
    order = H._rank_indices(scores)
    documents, best_by_doc = [], {}
    for i in order:
        question = index.doc_questions[i]
        did = question["document_id"]
        if did in best_by_doc:
            continue
        best_by_doc[did] = {
            "similarity": float(scores[i]),
            "question_id": question["question_id"],
            "question": question["question"],
        }
        documents.append(did)
    return documents[: H.DOC_ROUTE_K], {
        "best_similarity": float(scores[order[0]]),
        "best_question_id": index.doc_questions[order[0]]["question_id"],
        "best_question": index.doc_questions[order[0]]["question"],
        "routed_documents": documents[: H.DOC_ROUTE_K],
        "best_question_by_document": best_by_doc,
    }


def retrieve(threshold: float, force_embeddings: bool = False) -> dict:
    queries = H.load_queries()
    gold = H.load_gold()
    analyses = {x["query_id"]: x for x in H.query_understanding()}
    index = _load_persisted_indexes()
    vectors = _query_vectors(queries, force=force_embeddings)
    conditions = {
        tag: []
        for tag in ("A-vector", "Flat-hybrid", "Question-route", "Gated-fallback")
    }
    decisions = {}

    for q in queries:
        qid = q["query_id"]
        qvec = np.asarray(vectors[qid], dtype=float)
        analysis = analyses[qid]
        g = gold[qid]

        dense = H._cosine_matrix(index.chunk_vectors, qvec)
        baseline_ids = [
            index.chunks[i]["chunk_id"] for i in H._rank_indices(dense, k=C.RANK_DEPTH)
        ]
        flat_ids, flat_debug = H._chunk_hybrid(index, qvec, analysis, None)
        routed_docs, question_debug = _question_route(index, qvec)
        routed_ids, routed_debug = H._chunk_hybrid(
            index, qvec, analysis, set(routed_docs)
        )
        fallback = question_debug["best_similarity"] < threshold
        gated_ids = flat_ids if fallback else routed_ids

        for tag, ids in (
            ("A-vector", baseline_ids),
            ("Flat-hybrid", flat_ids),
            ("Question-route", routed_ids),
            ("Gated-fallback", gated_ids),
        ):
            conditions[tag].append(
                H._as_metric_row(q, g, ids[: C.RANK_DEPTH], index.chunk_map)
            )
        decisions[qid] = {
            "query": q["query"],
            "threshold": threshold,
            "fallback_to_flat_hybrid": fallback,
            **question_debug,
            "flat_chunk_retrieval": flat_debug,
            "routed_chunk_retrieval": routed_debug,
        }

    payload = {
        "threshold": threshold,
        "conditions": conditions,
        "decisions": decisions,
    }
    RANKINGS_PATH.write_text(json.dumps(payload))
    return payload


def _ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (C.BOOTSTRAP_RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _delta_ci(
    base: np.ndarray, arm: np.ndarray, seed: int = 42
) -> tuple[float, float, float, bool]:
    diff = arm - base
    lo, hi = _ci(diff, seed)
    return float(diff.mean()), lo, hi, bool(lo > 0 or hi < 0)


def evaluate(payload: dict) -> dict:
    conditions = payload["conditions"]
    per = {
        tag: {row["query_id"]: VM.per_query(row) for row in rows}
        for tag, rows in conditions.items()
    }
    qids = sorted(per["A-vector"])
    metric_keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    )
    metrics = {}
    for key in metric_keys:
        metrics[key] = {}
        base = np.asarray([per["A-vector"][qid][key] for qid in qids])
        for tag in per:
            values = np.asarray([per[tag][qid][key] for qid in qids])
            lo, hi = _ci(values)
            item = {"mean": float(values.mean()), "ci_low": lo, "ci_high": hi}
            if tag != "A-vector":
                delta, dlo, dhi, significant = _delta_ci(base, values)
                item.update(
                    delta=delta,
                    delta_low=dlo,
                    delta_high=dhi,
                    significant=significant,
                )
            metrics[key][tag] = item

    decisions = payload["decisions"]
    similarities = np.asarray([row["best_similarity"] for row in decisions.values()])
    fallback_count = sum(row["fallback_to_flat_hybrid"] for row in decisions.values())
    gated_comparisons = {}
    for key in metric_keys:
        gated = np.asarray([per["Gated-fallback"][qid][key] for qid in qids])
        gated_comparisons[key] = {}
        for reference in ("Flat-hybrid", "Question-route"):
            reference_values = np.asarray([per[reference][qid][key] for qid in qids])
            delta, lo, hi, significant = _delta_ci(reference_values, gated)
            gated_comparisons[key][reference] = {
                "delta": delta,
                "delta_low": lo,
                "delta_high": hi,
                "significant": significant,
            }
    result = {
        "n_queries": len(qids),
        "threshold": payload["threshold"],
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(qids),
        "best_question_similarity": {
            "min": float(similarities.min()),
            "p25": float(np.percentile(similarities, 25)),
            "median": float(np.median(similarities)),
            "p75": float(np.percentile(similarities, 75)),
            "max": float(similarities.max()),
        },
        "embedding_signature": _embedding_signature(),
        "metrics": metrics,
        "gated_comparisons": gated_comparisons,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    return result


def _cell(item: dict, baseline: bool) -> str:
    if baseline:
        return (
            f'<div class="v">{item["mean"]:.3f}</div>'
            f'<div class="ci">[{item["ci_low"]:.3f}, {item["ci_high"]:.3f}]</div>'
        )
    delta = item["delta"]
    cls = "good" if delta > 0 else ("bad" if delta < 0 else "flat")
    star = "<b>*</b>" if item["significant"] else ""
    return (
        f'<div class="v">{item["mean"]:.3f}</div>'
        f'<div class="d {cls}" title="Delta 95% CI '
        f'[{item["delta_low"]:+.3f}, {item["delta_high"]:+.3f}]">'
        f"({delta:+.3f}{star})</div>"
    )


def render(result: dict) -> None:
    metric_defs = (
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    )
    labels = {
        "A-vector": "Original chunks · vector-only baseline",
        "Flat-hybrid": "All chunks · BM25 + vector RRF",
        "Question-route": "Generated-question routing · no fallback",
        "Gated-fallback": (
            f"Generated-question routing · Flat-hybrid fallback below "
            f"{result['threshold']:.3f}"
        ),
    }
    body = ""
    for tag, label in labels.items():
        cells = "".join(
            f"<td>{_cell(result['metrics'][key][tag], tag == 'A-vector')}</td>"
            for key, _ in metric_defs
        )
        body += (
            f'<tr class="{"base" if tag == "A-vector" else ""}">'
            f'<td class="nm"><b>{escape(tag)}</b> · {escape(label)}</td>'
            f"{cells}</tr>"
        )
    heads = "".join(f"<th>{label}</th>" for _, label in metric_defs)
    sim = result["best_question_similarity"]
    REPORT.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generated-question similarity fallback</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1250px;margin:auto;padding:34px 22px 70px}}h1{{font-size:24px;margin:0 0 5px}}.cap{{color:var(--muted);font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid var(--line);padding:9px;text-align:center}}th{{background:var(--card)}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}.v{{font-weight:700;font-size:14px}}.ci,.d{{font-size:11px}}.ci,.flat{{color:var(--muted)}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}
</style></head><body><div class="wrap">
<h1>15-article MultiHop-RAG — question similarity with document fallback</h1>
<p class="cap">The query first searches 150 verified generated document
questions. At cosine similarity &ge; {result["threshold"]:.3f}, their top five
documents constrain chunk retrieval; below it, retrieval falls back to
unrestricted Flat-hybrid search over all original chunks.</p>
<table><thead><tr><th>Condition</th>{heads}</tr></thead><tbody>{body}</tbody></table>
<p class="cap">n={result["n_queries"]} queries · fallback
{result["fallback_count"]}/{result["n_queries"]}
({result["fallback_rate"]:.1%}) · best-question similarity:
min {sim["min"]:.3f}, p25 {sim["p25"]:.3f}, median {sim["median"]:.3f},
p75 {sim["p75"]:.3f}, max {sim["max"]:.3f} ·
{escape(result["embedding_signature"])}. Deltas are versus A-vector;
* means paired 95% bootstrap CI excludes zero.</p>
<p class="cap">At this threshold, Gated-fallback versus Flat-hybrid:
Evidence Recall@5
{result["gated_comparisons"]["evidence_recall@5"]["Flat-hybrid"]["delta"]:+.3f};
versus Question-route:
{result["gated_comparisons"]["evidence_recall@5"]["Question-route"]["delta"]:+.3f}.
Similarity is the operational proxy for “not enough relevant information”;
gold labels are used only after retrieval for evaluation.</p>
</div></body></html>""")
    print(f"[question-fallback:report] {REPORT}")


def run(threshold: float = DEFAULT_THRESHOLD, force_embeddings: bool = False) -> dict:
    payload = retrieve(threshold, force_embeddings)
    result = evaluate(payload)
    render(result)
    for key in (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    ):
        print(
            f"{key:<24} "
            + "  ".join(
                f"{tag}={result['metrics'][key][tag]['mean']:.3f}"
                for tag in result["metrics"][key]
            )
        )
    print(
        f"fallback={result['fallback_count']}/{result['n_queries']} "
        f"({result['fallback_rate']:.1%})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1")
    run(args.threshold, args.force_embeddings)


if __name__ == "__main__":
    main()
