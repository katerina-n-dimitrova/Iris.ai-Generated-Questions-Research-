"""
Vision-enrichment experiment (SPIQA test-B, 50 papers).

Question: does adding a vision-model description of each figure/chart/plot image
to its retrieval-unit text improve retrieval — especially for questions whose
gold evidence is a figure/chart/table?

We compare caption-only figure units vs caption+visual-description figure units,
under both dense-only and hybrid (dense + BM25 doc2query-q10, RRF) retrieval, and
report metrics BOTH overall and on the figure-answerable query subset (queries
whose gold includes a figure/table unit). Reuses the retrieval building blocks
from hybrid_doc2query_experiment. Same LLM/embedder, no fine-tuning; vision =
gpt-4o-mini image descriptions (cached).

Outputs -> results/spiqa/test-B/vision_enrichment/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore")

import spiqa_config as C
from embeddings import embedding_signature
from spiqa_loader import load_papers
from spiqa_chunker import load_chunks
from spiqa_eval import build_gold
import question_gen as QG
import hybrid_doc2query_experiment as H
from vision_enrich import load_vision, apply_vision_to_chunks

OUT_DIR = C.RESULTS_DIR / "test-B" / "vision_enrichment"
TOP_K = C.TOP_K


def _prep(chunks):
    ids = [c["chunk_id"] for c in chunks]
    text = {c["chunk_id"]: c["text"] for c in chunks}
    paper = {c["chunk_id"]: c["paper_id"] for c in chunks}
    return ids, text, paper


def _metrics_subset(res, subset_ids):
    """Recompute aggregate metrics over a subset of query_ids from a full eval."""
    pq = [x for x in res["per_query"] if x["query_id"] in subset_ids]
    if not pq:
        return {}

    def avg(k):
        return round(sum(x[k] for x in pq) / len(pq), 4)

    return {
        "hit@1": avg("hit@1"),
        "hit@5": avg("hit@5"),
        "hit@10": avg("hit@10"),
        "mrr": avg("mrr"),
        "ndcg@10": avg("ndcg@10"),
        "n": len(pq),
    }


def run(max_papers=50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})
    papers = load_papers("test-B", max_papers)
    keep = {p.paper_id for p in papers}
    chunks = [c for c in load_chunks("test-B") if c["paper_id"] in keep]
    queries = build_gold(papers, chunks)

    vision = load_vision()
    n_fig = sum(
        1 for c in chunks if c["source_type"] in ("figure_caption", "table_caption")
    )
    n_fig_v = sum(
        1
        for c in chunks
        if c["source_type"] in ("figure_caption", "table_caption")
        and vision.get(c["metadata"].get("fig_id", ""))
    )
    print(
        f"papers={len(papers)} chunks={len(chunks)} queries={len(queries)} "
        f"figure_units={n_fig} with_vision={n_fig_v} ({embedding_signature()})",
        flush=True,
    )

    # figure-answerable queries: gold includes a figure/table unit
    fig_chunk_ids = {
        c["chunk_id"]
        for c in chunks
        if c["source_type"] in ("figure_caption", "table_caption")
    }
    fig_query_ids = {
        q["query_id"]
        for q in queries
        if any(g in fig_chunk_ids for g in q["gold_chunk_ids"])
    }
    print(f"figure-answerable queries: {len(fig_query_ids)}/{len(queries)}", flush=True)

    # questions (Exp-1 cache) — first 10 per chunk, for the doc2query BM25 side
    allq = QG.load_questions("test-B")
    q10 = {cid: allq.get(cid, [])[:10] for cid in [c["chunk_id"] for c in chunks]}

    variants = {"novision": chunks, "vision": apply_vision_to_chunks(chunks, vision)}
    results = {}
    for vname, ch in variants.items():
        ids, text, paper = _prep(ch)
        orig_vecs = H._embed([text[c] for c in ids])
        dense = H.DenseIndex(ids, orig_vecs)
        bm25_q10 = H.BM25Index(ids, [H._enriched(text[c], q10[c]) for c in ids])
        # dense-only and hybrid (dense orig + BM25 chunk+q10)
        results[f"dense_{vname}"] = H.eval_condition(
            f"dense_{vname}", ids, queries, k_values, dense=dense
        )
        results[f"hybrid_q10_{vname}"] = H.eval_condition(
            f"hybrid_q10_{vname}", ids, queries, k_values, dense=dense, bm25=bm25_q10
        )
        print(f"  {vname}: embedded + evaluated dense & hybrid", flush=True)

    # assemble table: overall + figure-subset for each condition
    rows = []
    for cond, res in results.items():
        m = res["metrics"]
        fs = _metrics_subset(res, fig_query_ids)
        rows.append(
            {
                "condition": cond,
                "overall_hit@1": m["hit@1"],
                "overall_hit@5": m["hit@5"],
                "overall_ndcg@10": m["ndcg@10"],
                "overall_mrr": m["mrr"],
                "figQ_hit@1": fs.get("hit@1"),
                "figQ_hit@5": fs.get("hit@5"),
                "figQ_ndcg@10": fs.get("ndcg@10"),
                "figQ_mrr": fs.get("mrr"),
                "figQ_n": fs.get("n"),
                "p95_ms": res["search_p95_ms"],
            }
        )
        print(
            f"  {cond:22} overall nDCG@10={m['ndcg@10']:.3f} | "
            f"figQ nDCG@10={fs.get('ndcg@10')}",
            flush=True,
        )

    _write(rows, results, fig_query_ids, len(queries), n_fig, n_fig_v)
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return rows


def _write(rows, results, fig_query_ids, n_queries, n_fig, n_fig_v):
    cols = [
        "condition",
        "overall_hit@1",
        "overall_hit@5",
        "overall_ndcg@10",
        "overall_mrr",
        "figQ_hit@1",
        "figQ_hit@5",
        "figQ_ndcg@10",
        "figQ_mrr",
        "figQ_n",
        "p95_ms",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT_DIR / "vision_comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    import csv

    with (OUT_DIR / "vision_comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    by = {r["condition"]: r for r in rows}

    def d(a, b, key):
        return round((by[a][key] or 0) - (by[b][key] or 0), 4)

    # failure: which figure queries did vision fix (hybrid_q10_vision vs novision)
    nv = {x["query_id"]: x for x in results["hybrid_q10_novision"]["per_query"]}
    v = {x["query_id"]: x for x in results["hybrid_q10_vision"]["per_query"]}

    def hit(pq, k=TOP_K):
        g = set(pq["gold"])
        return any(c in g for c in pq["retrieved"][:k])

    fixed, broke = [], []
    for qid in fig_query_ids:
        if hit(v[qid]) and not hit(nv[qid]):
            fixed.append(
                {
                    "query_id": qid,
                    "vision_top": v[qid]["retrieved"][:3],
                    "novision_top": nv[qid]["retrieved"][:3],
                    "gold": v[qid]["gold"],
                }
            )
        elif hit(nv[qid]) and not hit(v[qid]):
            broke.append(
                {
                    "query_id": qid,
                    "vision_top": v[qid]["retrieved"][:3],
                    "novision_top": nv[qid]["retrieved"][:3],
                    "gold": v[qid]["gold"],
                }
            )
    json.dump(
        {"fixed_by_vision": fixed, "broken_by_vision": broke},
        open(OUT_DIR / "vision_figure_query_effect.json", "w"),
        indent=2,
    )

    L = [
        "# Vision-enrichment experiment — summary\n",
        f"SPIQA test-B, {n_queries} queries ({len(fig_query_ids)} figure-answerable). "
        f"{n_fig} figure/table units, {n_fig_v} with a gpt-4o-mini visual description. "
        f"Vision text is appended to the figure unit (caption + type + nearby context + "
        f"**visual description**). Retrieval + embedder unchanged; no fine-tuning.\n",
        "## Comparison (overall vs figure-answerable subset)\n",
    ]
    L += lines
    L.append("\n## Findings\n")
    L.append(
        f"- **Vision on dense retrieval:** overall {'+' if d('dense_vision', 'dense_novision', 'overall_ndcg@10') >= 0 else ''}"
        f"{d('dense_vision', 'dense_novision', 'overall_ndcg@10')} nDCG@10; "
        f"on figure questions {'+' if d('dense_vision', 'dense_novision', 'figQ_ndcg@10') >= 0 else ''}"
        f"{d('dense_vision', 'dense_novision', 'figQ_ndcg@10')} nDCG@10."
    )
    L.append(
        f"- **Vision on hybrid retrieval:** overall {'+' if d('hybrid_q10_vision', 'hybrid_q10_novision', 'overall_ndcg@10') >= 0 else ''}"
        f"{d('hybrid_q10_vision', 'hybrid_q10_novision', 'overall_ndcg@10')} nDCG@10; "
        f"on figure questions {'+' if d('hybrid_q10_vision', 'hybrid_q10_novision', 'figQ_ndcg@10') >= 0 else ''}"
        f"{d('hybrid_q10_vision', 'hybrid_q10_novision', 'figQ_ndcg@10')} nDCG@10."
    )
    L.append(
        f"- **Figure queries fixed by vision:** {len(fixed)}; broken: {len(broke)} "
        f"(net {len(fixed) - len(broke)}), see `vision_figure_query_effect.json`."
    )
    fig_gain = d("hybrid_q10_vision", "hybrid_q10_novision", "figQ_ndcg@10")
    L.append(
        f"\n## Recommendation\n"
        + (
            "Vision descriptions **help figure/chart-answerable retrieval** and are worth "
            "adding to figure units (the images are cheap to describe once, offline)."
            if fig_gain > 0.005
            else "Vision descriptions gave **little measurable retrieval lift** here — likely "
            "because captions + generated questions already carry most of the retrievable "
            "signal, and the embedder is the bottleneck. Keep as optional."
        )
    )
    (OUT_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=50)
    args = ap.parse_args()
    run(args.max_papers)


if __name__ == "__main__":
    main()
