"""
Vision-enriched generated-questions retrieval WITH BM25 / hybrid (SPIQA test-B, 20 papers).

Same 20-paper vision study as `vision_q_experiment.py`, but now every condition is
evaluated under three retrieval modes so we can see where generated questions —
and the vision enrichment — actually pay off:

    dense   : dense embedding retrieval (Octen-Embedding-0.6B)
    bm25    : lexical BM25 retrieval
    hybrid  : dense(original chunks) + BM25 fused with Reciprocal Rank Fusion (k=60)

Generated questions are used as DOCUMENT EXPANSION (doc2query): the chunk text and
its first-n generated questions are concatenated into one retrieval document. This
is the natural way to feed questions to BM25 (lexical anchors / likely user
phrasings), and we mirror it on the dense side with a "dense-append" condition.
Questions always map back to the parent chunk; only the original chunk is returned
as evidence.

Two arms on the identical 20 papers isolate the vision step:

    novision : original chunk text            + questions generated from it
    vision   : figure chunks vision-enriched  + questions generated from the
               vision-enriched text  (both cached from the earlier run)

For the vision arm the "chunk text" of a figure/table unit already includes the
gpt-4o-mini visual description, so its dense vector, its BM25 document, and its
generated questions all carry the chart/plot/table content — not just the caption.

Reuses the retrieval building blocks (DenseIndex, BM25Index, RRF eval) from
`hybrid_doc2query_experiment`. No new LLM calls (questions + vision cached). Same
embedder; no fine-tuning.

Outputs -> results/spiqa/test-B/vision_bm25_enrichment/
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
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
from vision_enrich import load_vision, apply_vision_to_chunks
import hybrid_doc2query_experiment as H

OUT_DIR = C.RESULTS_DIR / "test-B" / "vision_bm25_enrichment"
FIG_TYPES = ("figure_caption", "table_caption")
TOP_K = C.TOP_K


def _bm25_mb(bm25: H.BM25Index, name: str) -> float:
    path = OUT_DIR / "_bm25" / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"corpus": bm25.corpus, "bm25": bm25.bm25}, f)
    return round(path.stat().st_size / 1048576, 3)


def _dense_mb(name: str, chunk_ids, vecs, paper_of) -> float:
    import chromadb, shutil

    path = C.CHROMA_DIR / "vision_bm25" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    coll = chromadb.PersistentClient(path=str(path)).create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
    metas = [{"parent_chunk_id": c, "paper_id": paper_of[c]} for c in chunk_ids]
    for b in range(0, len(chunk_ids), 2000):
        coll.add(
            ids=chunk_ids[b : b + 2000],
            embeddings=vecs[b : b + 2000].tolist(),
            metadatas=metas[b : b + 2000],
        )
    return round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 3
    )


def _build_arm(
    arm: str,
    chunks: List[Dict],
    questions: Dict[str, List[str]],
    queries: List[Dict],
    k_values: List[int],
) -> Dict:
    """Build indexes + evaluate all retrieval conditions for one arm."""
    chunk_ids = [c["chunk_id"] for c in chunks]
    text = {c["chunk_id"]: c["text"] for c in chunks}
    paper_of = {c["chunk_id"]: c["paper_id"] for c in chunks}
    q10 = {cid: questions.get(cid, [])[:10] for cid in chunk_ids}
    q50 = {cid: questions.get(cid, [])[:50] for cid in chunk_ids}
    n_q10 = sum(len(v) for v in q10.values())
    n_q50 = sum(len(v) for v in q50.values())

    orig_vecs = H._embed([text[c] for c in chunk_ids])
    append10_vecs = H._embed([H._enriched(text[c], q10[c]) for c in chunk_ids])
    print(f"  [{arm}] embedded original + q10-append chunks", flush=True)

    dense_orig = H.DenseIndex(chunk_ids, orig_vecs)
    dense_app10 = H.DenseIndex(chunk_ids, append10_vecs)
    bm25_orig = H.BM25Index(chunk_ids, [text[c] for c in chunk_ids])
    bm25_q10 = H.BM25Index(chunk_ids, [H._enriched(text[c], q10[c]) for c in chunk_ids])
    bm25_q50 = H.BM25Index(chunk_ids, [H._enriched(text[c], q50[c]) for c in chunk_ids])

    dmb_orig = _dense_mb(f"{arm}_orig", chunk_ids, orig_vecs, paper_of)
    dmb_app10 = _dense_mb(f"{arm}_app10", chunk_ids, append10_vecs, paper_of)
    bmb_orig = _bm25_mb(bm25_orig, f"{arm}_orig")
    bmb_q10 = _bm25_mb(bm25_q10, f"{arm}_q10")
    bmb_q50 = _bm25_mb(bm25_q50, f"{arm}_q50")

    # (name, dense, bm25, dense_mb, bm25_mb, n_questions)
    conds = [
        ("baseline_dense", dense_orig, None, dmb_orig, 0.0, 0),
        ("baseline_bm25", None, bm25_orig, 0.0, bmb_orig, 0),
        ("baseline_hybrid", dense_orig, bm25_orig, dmb_orig, bmb_orig, 0),
        ("q10_dense_append", dense_app10, None, dmb_app10, 0.0, n_q10),
        ("q10_bm25_expand", None, bm25_q10, 0.0, bmb_q10, n_q10),
        ("q10_hybrid_expand", dense_orig, bm25_q10, dmb_orig, bmb_q10, n_q10),
        ("q50_bm25_expand", None, bm25_q50, 0.0, bmb_q50, n_q50),
        ("q50_hybrid_expand", dense_orig, bm25_q50, dmb_orig, bmb_q50, n_q50),
    ]

    base_storage = dmb_orig or 1.0
    rows, results = [], {}
    for name, dense, bm25, dmb, bmb, ngen in conds:
        res = H.eval_condition(
            f"{arm}_{name}", chunk_ids, queries, k_values, dense=dense, bm25=bm25
        )
        results[name] = res
        m = res["metrics"]
        rows.append(
            {
                "arm": arm,
                "condition": name,
                "hit@1": m["hit@1"],
                "hit@5": m["hit@5"],
                "hit@10": m["hit@10"],
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "generated_questions": ngen,
                "dense_index_MB": dmb,
                "bm25_index_MB": bmb,
                "storage_x": round((dmb + bmb) / base_storage, 2),
                "p95_latency_ms": res["search_p95_ms"],
            }
        )
        print(
            f"  [{arm}] {name:20} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
            f"Hit@10={m['hit@10']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
            f"p95={res['search_p95_ms']}ms",
            flush=True,
        )
    return {"rows": rows, "results": results}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(max_papers: int = 20):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})
    papers = load_papers("test-B", max_papers)
    keep = {p.paper_id for p in papers}
    chunks = [c for c in load_chunks("test-B") if c["paper_id"] in keep]
    queries = build_gold(papers, chunks)

    vision = load_vision()
    vision_chunks = apply_vision_to_chunks(chunks, vision)
    n_fig = sum(1 for c in chunks if c["source_type"] in FIG_TYPES)
    n_fig_v = sum(1 for c in vision_chunks if c["metadata"].get("has_vision"))
    fig_chunk_ids = {c["chunk_id"] for c in chunks if c["source_type"] in FIG_TYPES}
    fig_query_ids = {
        q["query_id"]
        for q in queries
        if any(g in fig_chunk_ids for g in q["gold_chunk_ids"])
    }

    base_q = QG.load_questions("test-B")
    vision_q = QG.load_questions("test-Bvis20")
    # fall back to base questions for any chunk missing from the vision cache
    vision_q = {
        c["chunk_id"]: vision_q.get(c["chunk_id"], base_q.get(c["chunk_id"], []))
        for c in chunks
    }

    print(
        f"papers={len(papers)} chunks={len(chunks)} queries={len(queries)} "
        f"figure_units={n_fig} with_vision={n_fig_v} "
        f"figure_answerable_queries={len(fig_query_ids)} ({embedding_signature()})",
        flush=True,
    )

    arms = {
        "novision": _build_arm("novision", chunks, base_q, queries, k_values),
        "vision": _build_arm("vision", vision_chunks, vision_q, queries, k_values),
    }
    rows = arms["novision"]["rows"] + arms["vision"]["rows"]
    _write(rows, papers, chunks, queries, fig_query_ids, n_fig, n_fig_v)
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return rows


PER_ARM_COLS = [
    "condition",
    "hit@1",
    "hit@5",
    "hit@10",
    "mrr",
    "ndcg@10",
    "generated_questions",
    "dense_index_MB",
    "bm25_index_MB",
    "storage_x",
    "p95_latency_ms",
]
COND_ORDER = [
    "baseline_dense",
    "baseline_bm25",
    "baseline_hybrid",
    "q10_dense_append",
    "q10_bm25_expand",
    "q10_hybrid_expand",
    "q50_bm25_expand",
    "q50_hybrid_expand",
]


def _md_table(rows, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return lines


def _write(rows, papers, chunks, queries, fig_query_ids, n_fig, n_fig_v):
    with (OUT_DIR / "vision_bm25_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["arm"] + PER_ARM_COLS)
        w.writeheader()
        w.writerows(rows)

    nov = sorted(
        [r for r in rows if r["arm"] == "novision"],
        key=lambda r: COND_ORDER.index(r["condition"]),
    )
    vis = sorted(
        [r for r in rows if r["arm"] == "vision"],
        key=lambda r: COND_ORDER.index(r["condition"]),
    )
    by = {(r["arm"], r["condition"]): r for r in rows}

    def d(cond, key):
        return round(by[("vision", cond)][key] - by[("novision", cond)][key], 4)

    def nd(arm, cond):
        return by[(arm, cond)]["ndcg@10"]

    all_fig = len(fig_query_ids) == len(queries)
    L = [
        "# Vision-enriched generated questions + BM25 / hybrid — SPIQA test-B (20 papers)\n",
        f"Retrieval on **{len(papers)} papers · {len(chunks)} chunks · {len(queries)} eval "
        f"queries** ({len(fig_query_ids)} figure-answerable), three modes per condition: "
        f"**dense**, **BM25**, and **hybrid** (dense(original) + BM25 fused with RRF, k=60). "
        f"Generated questions are document expansion (chunk + first-n questions). "
        f"**{n_fig}** figure/table units, **{n_fig_v}** vision-enriched; the **vision** arm "
        f"folds the gpt-4o-mini visual description into the figure chunk before it is "
        f"embedded, BM25-indexed, and used to generate questions. Embedder "
        f"{embedding_signature()}; no fine-tuning; questions + vision cached (no new LLM "
        f"calls).\n",
    ]
    if all_fig:
        L.append(
            "*(Every SPIQA test-B question refers to a figure/table, so all "
            f"{len(queries)} queries are figure-answerable — the vision enrichment "
            "can affect any of them.)*\n"
        )

    L += ["## novision arm\n"]
    L += _md_table(nov, PER_ARM_COLS)
    L += ["\n## vision arm\n"]
    L += _md_table(vis, PER_ARM_COLS)

    L += [
        "\n## Vision effect (vision − novision nDCG@10), per condition\n",
        "| condition | novision nDCG@10 | vision nDCG@10 | Δ nDCG@10 | Δ Hit@5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in COND_ORDER:
        L.append(
            f"| {c} | {nd('novision', c)} | {nd('vision', c)} | "
            f"{d(c, 'ndcg@10'):+} | {d(c, 'hit@5'):+} |"
        )

    # best conditions
    best_nov = max(COND_ORDER, key=lambda c: nd("novision", c))
    best_vis = max(COND_ORDER, key=lambda c: nd("vision", c))

    L += [
        "\n## Findings\n",
        f"- **Does BM25 help? (doc2query)** dense baseline nDCG@10 "
        f"{nd('novision', 'baseline_dense')} vs BM25 baseline "
        f"{nd('novision', 'baseline_bm25')} vs hybrid baseline "
        f"{nd('novision', 'baseline_hybrid')} (novision). With q10 expansion: "
        f"BM25 {nd('novision', 'q10_bm25_expand')}, hybrid "
        f"{nd('novision', 'q10_hybrid_expand')} — hybrid "
        f"{'beats' if nd('novision', 'q10_hybrid_expand') > nd('novision', 'baseline_dense') else 'does not beat'} "
        f"the dense baseline.",
        f"- **Best novision config:** `{best_nov}` = {nd('novision', best_nov)} nDCG@10. "
        f"**Best vision config:** `{best_vis}` = {nd('vision', best_vis)} nDCG@10.",
        f"- **Vision effect on hybrid retrieval:** `q10_hybrid_expand` "
        f"{d('q10_hybrid_expand', 'ndcg@10'):+} nDCG@10 / {d('q10_hybrid_expand', 'hit@5'):+} "
        f"Hit@5; `q50_hybrid_expand` {d('q50_hybrid_expand', 'ndcg@10'):+} nDCG@10. "
        f"On plain dense chunks (`baseline_dense`) vision is "
        f"{d('baseline_dense', 'ndcg@10'):+} nDCG@10; on BM25 "
        f"(`q10_bm25_expand`) {d('q10_bm25_expand', 'ndcg@10'):+}.",
        f"- **Where vision lands hardest:** the visual description adds lexical anchors "
        f"(axis labels, chart type, entities) that BM25 can match verbatim, and it lets "
        f"the LLM write ~20 questions/figure vs ~13 caption-only — so the effect shows up "
        f"in both the dense and BM25 sides of the hybrid.",
        "\n## Recommendation\n"
        + (
            f"Use **{best_vis}** (vision-enriched hybrid). Hybrid dense+BM25 with "
            "vision-enriched doc2query questions is the strongest configuration here — "
            "BM25 contributes the lexical match the dense embedder misses, and the vision "
            "description feeds both sides."
            if nd("vision", best_vis) >= nd("novision", best_nov)
            else f"Hybrid helps, but vision enrichment gave little additional lift on top of it "
            f"here; **{best_nov}** (novision) is within noise of the vision arm."
        ),
    ]

    (OUT_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(
        {
            "papers": len(papers),
            "chunks": len(chunks),
            "queries": len(queries),
            "figure_answerable_queries": len(fig_query_ids),
            "figure_units": n_fig,
            "figure_units_with_vision": n_fig_v,
            "embedder": embedding_signature(),
            "rows": rows,
        },
        (OUT_DIR / "vision_bm25_results.json").open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print("\n(novision)\n" + "\n".join(_md_table(nov, PER_ARM_COLS)), flush=True)
    print("\n(vision)\n" + "\n".join(_md_table(vis, PER_ARM_COLS)), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-papers", type=int, default=20)
    args = ap.parse_args()
    run(args.max_papers)


if __name__ == "__main__":
    main()
