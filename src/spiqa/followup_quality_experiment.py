"""
Follow-up experiments: improving GENERATED-QUESTION QUALITY (SPIQA test-B, 20 papers).

The first study showed that simply increasing the *number* of generated questions
saturates (q10 ≈ the sweet spot, q50/q100 add cost without much gain). This study
holds the count at ~10 and instead improves the QUALITY of the questions and how
they are indexed. All experiments run on the SAME 20-paper vision study set
(1026 chunks, 74 figure-answerable eval queries), the SAME embedder
(Octen-Embedding-0.6B) and the SAME LLM (gpt-4o-mini). Cached outputs are reused
wherever they exist (generic questions, diverse questions, vision descriptions).

Experiment groups
-----------------
1. q10_diverse_questions              10 forced question TYPES (cached diverse pool)
2. q10_bm25_aware_questions           questions seeded with extracted lexical anchors
3. q10_filtered_questions             heuristic + round-trip filtering of candidates
   q20_filtered_to_q10_questions      20 candidates filtered down to ~10
4. q10_separate_question_index_rrf    question vectors kept in a SEPARATE index,
   q10_bm25_aware_separate_index_rrf  fused with the original-chunk index via RRF
   q10_filtered_separate_index_rrf    (instead of concatenating into chunk text)
5. q10_hybrid_rerank_top50            dense+BM25+question first stage -> cross-encoder
   q10_bm25_aware_hybrid_rerank_top50
   q10_filtered_hybrid_rerank_top50
6. structured_vision_q10              figure/table chunks -> STRUCTURED record
   structured_vision_q10_bm25_aware   (+ questions from the structured record)
   structured_vision_q10_hybrid

Retrieval primitives (DenseIndex, BM25Index, RRF eval, embedder, tokeniser) are
reused from `hybrid_doc2query_experiment`; question-quality helpers from
`followup_lib`; new cached generation from `followup_gen`.

Outputs -> results/spiqa/test-B/followup_quality/
"""

from __future__ import annotations

import argparse
import csv
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
import followup_lib as L
import followup_gen as G

OUT_DIR = C.RESULTS_DIR / "test-B" / "followup_quality"
FIG_TYPES = ("figure_caption", "table_caption")
TOP_K = C.TOP_K
FU_SPLIT = "test-Bfu20"  # cache key for new (bm25-aware / structured) gen
PREV_JSON = (
    C.RESULTS_DIR / "test-B" / "vision_bm25_enrichment" / "vision_bm25_results.json"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _first_n(qs_by: Dict[str, List[str]], chunk_ids, n) -> Dict[str, List[str]]:
    return {cid: qs_by.get(cid, [])[:n] for cid in chunk_ids}


def _n_q(qs: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in qs.values())


def _row(arm_metrics, name, strategy, setup, ngen, storage_x, p95, notes):
    m = arm_metrics
    return {
        "condition": name,
        "strategy": strategy,
        "setup": setup,
        "hit@1": m["hit@1"],
        "hit@5": m["hit@5"],
        "hit@10": m["hit@10"],
        "mrr": m["mrr"],
        "ndcg@10": m["ndcg@10"],
        "generated_questions": ngen,
        "storage_x": storage_x,
        "p95_latency_ms": p95,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(max_papers: int = 20, gen_limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})

    papers = load_papers("test-B", max_papers)
    keep = {p.paper_id for p in papers}
    chunks = [c for c in load_chunks("test-B") if c["paper_id"] in keep]
    chunk_ids = [c["chunk_id"] for c in chunks]
    text = {c["chunk_id"]: c["text"] for c in chunks}
    paper_of = {c["chunk_id"]: c["paper_id"] for c in chunks}
    queries = build_gold(papers, chunks)
    fig_ids = {c["chunk_id"] for c in chunks if c["source_type"] in FIG_TYPES}
    print(
        f"papers={len(papers)} chunks={len(chunks)} queries={len(queries)} "
        f"fig_chunks={len(fig_ids)} ({embedding_signature()})",
        flush=True,
    )

    # ---- corpus IDF + lexical anchors (for BM25-aware prompting) ----
    idf = L.build_idf([text[c] for c in chunk_ids])
    terms_by = {c["chunk_id"]: L.extract_terms(c, idf) for c in chunks}

    # ---- question pools (reuse caches; generate the two new ones) ----
    generic_all = QG.load_questions("test-B")
    diverse_all = _load_diverse()
    print("  generating BM25-aware questions (cached/resumable)…", flush=True)
    print(
        "   ",
        G.generate_bm25_aware(chunks, terms_by, FU_SPLIT, n=10, limit=gen_limit),
        flush=True,
    )
    bm25aware_all = G.load_bm25_aware(FU_SPLIT)

    q_generic = _first_n(generic_all, chunk_ids, 10)
    q_diverse = _first_n(diverse_all, chunk_ids, 10)
    q_bm25aw = _first_n(bm25aware_all, chunk_ids, 10)

    # ---- dense vectors we reuse everywhere ----
    orig_vecs = H._embed([text[c] for c in chunk_ids])
    dense_orig = H.DenseIndex(chunk_ids, orig_vecs)
    bm25_orig = H.BM25Index(chunk_ids, [text[c] for c in chunk_ids])
    base_dense_mb = H._dense_index_mb("fu_orig", chunk_ids, orig_vecs, paper_of)
    bm25_orig_mb = bm25_orig.size_mb("fu_orig")
    print(f"  base dense index {base_dense_mb} MB, bm25 {bm25_orig_mb} MB", flush=True)

    # ---- filtered pools (heuristic + round-trip) ----
    q_filt10 = _filtered(chunk_ids, text, generic_all, orig_vecs, cand=12, keep=10)
    q_filt20 = _filtered(chunk_ids, text, generic_all, orig_vecs, cand=20, keep=10)

    rows: List[Dict] = []

    # ===================================================================== #
    # 1-3 · question-strategy families under bm25_expand + hybrid_expand
    # ===================================================================== #
    strat = {
        "generic": q_generic,
        "diverse": q_diverse,
        "bm25_aware": q_bm25aw,
        "filtered_q10": q_filt10,
        "filtered_q20to10": q_filt20,
    }
    strat_rows = {}
    for sname, qs in strat.items():
        ngen = _n_q(qs)
        docs = [H._enriched(text[c], qs[c]) for c in chunk_ids]
        bm25_q = H.BM25Index(chunk_ids, docs)
        bm25_mb = bm25_q.size_mb(f"fu_{sname}")
        append_vecs = H._embed(docs)
        dense_app = H.DenseIndex(chunk_ids, append_vecs)
        app_mb = H._dense_index_mb(f"fu_app_{sname}", chunk_ids, append_vecs, paper_of)

        r_bm = H.eval_condition(
            f"{sname}_bm25", chunk_ids, queries, k_values, bm25=bm25_q
        )
        r_hy = H.eval_condition(
            f"{sname}_hybrid",
            chunk_ids,
            queries,
            k_values,
            dense=dense_orig,
            bm25=bm25_q,
        )
        r_app = H.eval_condition(
            f"{sname}_dense_append", chunk_ids, queries, k_values, dense=dense_app
        )
        strat_rows[sname] = {
            "bm25": r_bm,
            "hybrid": r_hy,
            "dense_append": r_app,
            "bm25_mb": bm25_mb,
            "app_mb": app_mb,
            "ngen": ngen,
        }
        print(
            f"  [{sname}] bm25 nDCG@10={r_bm['metrics']['ndcg@10']:.4f}  "
            f"hybrid nDCG@10={r_hy['metrics']['ndcg@10']:.4f}  "
            f"dense_append={r_app['metrics']['ndcg@10']:.4f}",
            flush=True,
        )

    def store_bm25(sname):
        return round(strat_rows[sname]["bm25_mb"] / base_dense_mb, 2)

    def store_hybrid(sname):
        return round((base_dense_mb + strat_rows[sname]["bm25_mb"]) / base_dense_mb, 2)

    # named conditions (headline rows) --------------------------------------
    rows.append(
        _row(
            strat_rows["diverse"]["hybrid"]["metrics"],
            "q10_diverse_questions",
            "10 forced question types",
            "hybrid (dense+BM25 RRF)",
            strat_rows["diverse"]["ngen"],
            store_hybrid("diverse"),
            strat_rows["diverse"]["hybrid"]["search_p95_ms"],
            "diverse vs generic q10 under identical hybrid setup",
        )
    )
    rows.append(
        _row(
            strat_rows["bm25_aware"]["bm25"]["metrics"],
            "q10_bm25_aware_questions",
            "lexical-anchor questions",
            "BM25 doc2query expand",
            strat_rows["bm25_aware"]["ngen"],
            store_bm25("bm25_aware"),
            strat_rows["bm25_aware"]["bm25"]["search_p95_ms"],
            f"anchors from section/caption/metric/IDF terms; hybrid="
            f"{strat_rows['bm25_aware']['hybrid']['metrics']['ndcg@10']}",
        )
    )
    rows.append(
        _row(
            strat_rows["filtered_q10"]["hybrid"]["metrics"],
            "q10_filtered_questions",
            "heuristic+round-trip filtered",
            "hybrid (dense+BM25 RRF)",
            strat_rows["filtered_q10"]["ngen"],
            store_hybrid("filtered_q10"),
            strat_rows["filtered_q10"]["hybrid"]["search_p95_ms"],
            "generic candidates filtered for answerable/specific/non-dup",
        )
    )
    rows.append(
        _row(
            strat_rows["filtered_q20to10"]["hybrid"]["metrics"],
            "q20_filtered_to_q10_questions",
            "20 cand -> best 10",
            "hybrid (dense+BM25 RRF)",
            strat_rows["filtered_q20to10"]["ngen"],
            store_hybrid("filtered_q20to10"),
            strat_rows["filtered_q20to10"]["hybrid"]["search_p95_ms"],
            "more candidates, filtered to a clean 10",
        )
    )

    # ===================================================================== #
    # 4 · separate question index + RRF (A: chunks, B: questions -> parent)
    # ===================================================================== #
    sep_map = {
        "q10_separate_question_index_rrf": ("generic", q_generic),
        "q10_bm25_aware_separate_index_rrf": ("bm25_aware", q_bm25aw),
        "q10_filtered_separate_index_rrf": ("filtered_q10", q_filt10),
    }
    qindexes = {}
    for cond, (sname, qs) in sep_map.items():
        qi = L.QuestionIndex(chunk_ids, qs)
        qindexes[sname] = qi
        res = L.eval_fusion(
            cond,
            chunk_ids,
            queries,
            k_values,
            [L.dense_ranker(dense_orig), L.qindex_ranker(qi)],
        )
        qidx_mb = round(qi.size_mb(), 3)
        storage_x = round((base_dense_mb + qidx_mb) / base_dense_mb, 2)
        rows.append(
            _row(
                res["metrics"],
                cond,
                f"{sname} q10 in separate index",
                "RRF(A:chunks , B:questions→parent)",
                qi.n_questions,
                storage_x,
                res["search_p95_ms"],
                "questions kept OUT of the chunk vector; separate index B",
            )
        )
        print(
            f"  [sep] {cond} nDCG@10={res['metrics']['ndcg@10']:.4f} "
            f"(B={qi.n_questions} qvecs, {qidx_mb} MB)",
            flush=True,
        )

    # ===================================================================== #
    # 5 · reranker after hybrid (dense+BM25+questions -> top50 -> cross-encoder)
    # ===================================================================== #
    reranker = L.CrossEncoderReranker()
    rr_map = {
        "q10_hybrid_rerank_top50": ("generic", q_generic),
        "q10_bm25_aware_hybrid_rerank_top50": ("bm25_aware", q_bm25aw),
        "q10_filtered_hybrid_rerank_top50": ("filtered_q10", q_filt10),
    }
    if reranker.available:
        for cond, (sname, qs) in rr_map.items():
            bm25_q = H.BM25Index(
                chunk_ids, [H._enriched(text[c], qs[c]) for c in chunk_ids]
            )
            qi = qindexes.get(sname) or L.QuestionIndex(chunk_ids, qs)
            first_stage = [
                L.dense_ranker(dense_orig),
                L.bm25_ranker(bm25_q),
                L.qindex_ranker(qi),
            ]
            res = L.eval_rerank(
                cond,
                chunk_ids,
                queries,
                k_values,
                first_stage,
                text,
                reranker,
                first_top_n=50,
            )
            storage_x = (
                store_hybrid(sname)
                if sname in strat_rows
                else round(
                    (base_dense_mb + bm25_q.size_mb(f"fu_rr_{sname}")) / base_dense_mb,
                    2,
                )
            )
            rows.append(
                _row(
                    res["metrics"],
                    cond,
                    f"{sname} q10 + cross-encoder",
                    "hybrid top50 → rerank → top10",
                    _n_q(qs),
                    storage_x,
                    res["search_p95_ms"],
                    f"reranker={reranker.model_name.split('/')[-1]} (recall vs precision)",
                )
            )
            print(
                f"  [rerank] {cond} nDCG@10={res['metrics']['ndcg@10']:.4f} "
                f"p95={res['search_p95_ms']}ms",
                flush=True,
            )
    else:
        print("  [rerank] SKIPPED — cross-encoder unavailable", flush=True)

    # ===================================================================== #
    # 6 · structured figure/table enrichment
    # ===================================================================== #
    vision = load_vision()
    fig_chunks = [c for c in chunks if c["source_type"] in FIG_TYPES]
    nearby_by = {
        c["chunk_id"]: c["text"] for c in fig_chunks
    }  # fig chunk already carries context
    print("  building structured figure/table records (cached/resumable)…", flush=True)
    print(
        "   ",
        G.build_structured(
            fig_chunks, vision, nearby_by, FU_SPLIT, n_questions=10, limit=gen_limit
        ),
        flush=True,
    )
    struct_text, struct_q = G.load_structured(FU_SPLIT)

    # figure/table chunk text -> structured record; text chunks unchanged.
    s_text = {
        c: (struct_text.get(c, text[c]) if c in fig_ids else text[c]) for c in chunk_ids
    }
    # questions: structured questions for fig chunks, generic q10 for text chunks
    s_q = {
        c: (struct_q.get(c, [])[:10] if c in fig_ids else q_generic.get(c, []))
        for c in chunk_ids
    }
    s_docs = [H._enriched(s_text[c], s_q[c]) for c in chunk_ids]

    s_vecs = H._embed([s_text[c] for c in chunk_ids])  # structured dense
    s_dense = H.DenseIndex(chunk_ids, s_vecs)
    s_append_vecs = H._embed(s_docs)
    s_dense_app = H.DenseIndex(chunk_ids, s_append_vecs)
    s_bm25 = H.BM25Index(chunk_ids, s_docs)
    s_dense_mb = H._dense_index_mb("fu_struct", chunk_ids, s_vecs, paper_of)
    s_app_mb = H._dense_index_mb("fu_struct_app", chunk_ids, s_append_vecs, paper_of)
    s_bm25_mb = s_bm25.size_mb("fu_struct")

    r_sd = H.eval_condition(
        "structured_vision_q10", chunk_ids, queries, k_values, dense=s_dense_app
    )
    r_sb = H.eval_condition(
        "structured_vision_q10_bm25_aware", chunk_ids, queries, k_values, bm25=s_bm25
    )
    r_sh = H.eval_condition(
        "structured_vision_q10_hybrid",
        chunk_ids,
        queries,
        k_values,
        dense=s_dense,
        bm25=s_bm25,
    )
    rows.append(
        _row(
            r_sd["metrics"],
            "structured_vision_q10",
            "structured fig record + q10",
            "dense (structured + q10 append)",
            _n_q(s_q),
            round(s_app_mb / base_dense_mb, 2),
            r_sd["search_p95_ms"],
            f"{len(struct_text)} figure/table chunks structured",
        )
    )
    rows.append(
        _row(
            r_sb["metrics"],
            "structured_vision_q10_bm25_aware",
            "structured fig record + q10",
            "BM25 (structured lexical anchors)",
            _n_q(s_q),
            round(s_bm25_mb / base_dense_mb, 2),
            r_sb["search_p95_ms"],
            "axes/values/metrics act as lexical anchors",
        )
    )
    rows.append(
        _row(
            r_sh["metrics"],
            "structured_vision_q10_hybrid",
            "structured fig record + q10",
            "hybrid (dense+BM25 RRF)",
            _n_q(s_q),
            round((s_dense_mb + s_bm25_mb) / base_dense_mb, 2),
            r_sh["search_p95_ms"],
            "best structured config",
        )
    )
    print(
        f"  [structured] dense={r_sd['metrics']['ndcg@10']:.4f} "
        f"bm25={r_sb['metrics']['ndcg@10']:.4f} hybrid={r_sh['metrics']['ndcg@10']:.4f}",
        flush=True,
    )

    # ---- full strategy breakdown (for the report's per-strategy table) ----
    breakdown = []
    for sname in strat:
        for setup in ("bm25", "hybrid", "dense_append"):
            m = strat_rows[sname][setup]["metrics"]
            breakdown.append(
                {
                    "strategy": sname,
                    "setup": setup,
                    **{k: m[k] for k in ("hit@1", "hit@5", "hit@10", "mrr", "ndcg@10")},
                    "p95_latency_ms": strat_rows[sname][setup]["search_p95_ms"],
                }
            )

    prev = _load_prev()
    _write(
        rows,
        breakdown,
        prev,
        papers,
        chunks,
        queries,
        len(fig_ids),
        len(struct_text),
        base_dense_mb,
        reranker.available,
    )
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# filtering (heuristic + round-trip retrieval check)
# --------------------------------------------------------------------------- #
def _filtered(chunk_ids, text, generic_all, orig_vecs, *, cand, keep):
    """Heuristic filter over `cand` candidates, then round-trip retrieval check."""
    heur = {}
    stats = {"generic": 0, "ungrounded": 0, "duplicate": 0, "cand": 0, "after_heur": 0}
    for cid in chunk_ids:
        cands = generic_all.get(cid, [])[:cand]
        stats["cand"] += len(cands)
        f = L.filter_questions(text[cid], cands, keep=keep)
        heur[cid] = f["kept"]
        stats["generic"] += f["dropped_generic"]
        stats["ungrounded"] += f["dropped_ungrounded"]
        stats["duplicate"] += f["dropped_duplicate"]
        stats["after_heur"] += len(f["kept"])
    # round-trip: keep only questions that retrieve their own parent in top-k
    kept_rt, gen_rt, keep_rt = H.roundtrip_keep(chunk_ids, orig_vecs, heur, keep)
    print(
        f"  [filter cand={cand}] heuristic kept {stats['after_heur']}/{stats['cand']} "
        f"(generic {stats['generic']}, ungrounded {stats['ungrounded']}, dup "
        f"{stats['duplicate']}); round-trip kept {keep_rt}/{gen_rt}",
        flush=True,
    )
    return kept_rt


def _load_diverse():
    """Load the pre-generated diverse-question pool (cache from earlier run)."""
    path = C.PROCESSED_DIR / "test-B_questions_diverse.jsonl"
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["chunk_id"]] = r["questions"]
    return out


def _load_prev():
    """Previous novision configs from the report's cached results, for comparison."""
    want = [
        "baseline_dense",
        "baseline_bm25",
        "baseline_hybrid",
        "q10_bm25_expand",
        "q10_hybrid_expand",
        "q50_hybrid_expand",
    ]
    out = {}
    if PREV_JSON.exists():
        data = json.load(PREV_JSON.open())
        for r in data["rows"]:
            if r["arm"] == "novision" and r["condition"] in want:
                out[r["condition"]] = r
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
MAIN_COLS = [
    "condition",
    "strategy",
    "setup",
    "hit@1",
    "hit@5",
    "hit@10",
    "mrr",
    "ndcg@10",
    "generated_questions",
    "storage_x",
    "p95_latency_ms",
    "notes",
]


def _write(
    rows,
    breakdown,
    prev,
    papers,
    chunks,
    queries,
    n_fig,
    n_struct,
    base_dense_mb,
    reranker_ok,
):
    with (OUT_DIR / "followup_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=MAIN_COLS)
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["ndcg@10"])
    payload = {
        "papers": len(papers),
        "chunks": len(chunks),
        "queries": len(queries),
        "figure_chunks": n_fig,
        "structured_figure_chunks": n_struct,
        "embedder": embedding_signature(),
        "base_dense_index_mb": base_dense_mb,
        "reranker_used": reranker_ok,
        "best_new_condition": best["condition"],
        "best_new_ndcg@10": best["ndcg@10"],
        "rows": rows,
        "strategy_breakdown": breakdown,
        "previous_novision": prev,
    }
    json.dump(
        payload,
        (OUT_DIR / "followup_results.json").open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    # markdown table
    L_ = [
        "# Follow-up: improving generated-question quality — SPIQA test-B (20 papers)\n",
        f"{len(papers)} papers · {len(chunks)} chunks · {len(queries)} figure-answerable "
        f"queries · {embedding_signature()} · gpt-4o-mini. All conditions hold count ≈q10 "
        f"and vary question QUALITY / indexing.\n",
        "## New conditions\n",
        "| " + " | ".join(MAIN_COLS) + " |",
        "| " + " | ".join("---" for _ in MAIN_COLS) + " |",
    ]
    for r in rows:
        L_.append("| " + " | ".join(str(r.get(c, "")) for c in MAIN_COLS) + " |")
    L_ += [
        f"\n**Best new condition:** `{best['condition']}` = {best['ndcg@10']} nDCG@10.\n"
    ]
    (OUT_DIR / "results_summary.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    print("\n".join(L_[3:]), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument(
        "--gen-limit",
        type=int,
        default=None,
        help="cap new LLM generations (smoke test)",
    )
    args = ap.parse_args()
    run(args.max_papers, args.gen_limit)


if __name__ == "__main__":
    main()
