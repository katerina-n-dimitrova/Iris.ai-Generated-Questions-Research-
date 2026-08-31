"""
Experiment 3 — chunk size / overlap ablation for generated-question enrichment
(SPIQA test-B).

Research question: do smaller chunks with smaller overlap make generated-question
enrichment more effective than the current chunking?

Only two things vary: (1) chunk size / overlap, (2) number of generated questions
per chunk (0 / 10 / 50). Everything else is held fixed — same LLM, embedding
model, split, eval queries, top-k, metrics, generation prompt, retrieval pipeline,
and table/chart handling.

Chunk configs
-------------
current : 416 tokens / 100 overlap  (the pipeline default)
smaller : 208 tokens /  50 overlap  (~ current/2)

Only regular TEXT chunks shrink. Figure/table units are size-independent
retrieval units (caption + title + nearby context, never split) and are byte-for-
byte identical across both configs — so the smaller cache reuses their cached
questions and generation only runs for the new smaller text chunks.

Enrichment = APPEND (kept deliberately simple/focused): a chunk is embedded as
"original chunk text + its generated questions" as ONE vector. Questions are
never independent evidence; retrieval returns the original parent chunk.

Conditions: {current,smaller} x {baseline(0), q10, q50}. Outputs ->
results/spiqa/test-B/chunk_size_overlap_ablation/
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore")

import spiqa_config as C
from embeddings import get_embedder, embedding_signature
from spiqa_loader import load_papers
from spiqa_chunker import build_chunks
from spiqa_eval import build_gold, _query_metrics
import question_gen as QG

OUT_DIR = C.RESULTS_DIR / "test-B" / "chunk_size_overlap_ablation"
EXP_CHROMA_DIR = C.CHROMA_DIR / "chunk_ablation"

CONFIGS = {  # name -> (chunk_size, overlap, question_cache_label)
    "current": (416, 100, "test-B"),  # reuse Experiment-1 cache
    "smaller": (208, 50, "test-B_smaller_chunkabl"),  # generated fresh (text only)
}
Q_CONDITIONS = [0, 10, 50]
TOP_K = C.TOP_K
MAX_GEN = 50  # generate up to 50 questions/chunk (q50 is the max)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _san(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _embed(texts):
    emb = get_embedder()
    t0 = time.perf_counter()
    v = (
        _san(np.asarray(emb.embed_documents(texts), dtype=np.float32))
        if texts
        else np.zeros((0, emb.dim), np.float32)
    )
    return v, time.perf_counter() - t0


def _enriched(chunk_text, questions):
    if not questions:
        return chunk_text
    return f"{chunk_text}\n\nQuestions this passage answers:\n" + "\n".join(
        f"- {q}" for q in questions
    )


def _dir_size_mb(path: Path):
    return round(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 3
    )


def _write_chroma(
    name: str, ids: List[str], vecs: np.ndarray, paper_of: Dict[str, str]
):
    import chromadb, shutil

    path = EXP_CHROMA_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    coll = chromadb.PersistentClient(path=str(path)).create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
    metas = [{"parent_chunk_id": cid, "paper_id": paper_of[cid]} for cid in ids]
    for b in range(0, len(ids), 2000):
        coll.add(
            ids=ids[b : b + 2000],
            embeddings=vecs[b : b + 2000].tolist(),
            metadatas=metas[b : b + 2000],
        )
    return _dir_size_mb(path)


# --------------------------------------------------------------------------- #
# per-config preparation
# --------------------------------------------------------------------------- #
class Config:
    def __init__(self, cfg_name: str, papers, gen_limit=None):
        self.name = cfg_name
        self.size, self.overlap, self.qlabel = CONFIGS[cfg_name]
        self.chunks = [
            c.to_row()
            for c in build_chunks(papers, size=self.size, overlap=self.overlap)
        ]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.text_by_id = {c["chunk_id"]: c["text"] for c in self.chunks}
        self.paper_of = {c["chunk_id"]: c["paper_id"] for c in self.chunks}
        toks = [c["metadata"].get("n_tokens", 0) for c in self.chunks]
        self.avg_tokens = round(st.mean(toks), 1) if toks else 0
        self.median_tokens = round(st.median(toks), 1) if toks else 0
        self.num_chunks = len(self.chunks)
        self.gen_limit = gen_limit
        self.questions: Dict[str, List[str]] = {}
        self.gen_summary = {}
        self.queries = build_gold(papers, self.chunks)

    def ensure_questions(self):
        # 'current' reuses the Experiment-1 cache (identical chunk_ids); 'smaller'
        # generates fresh, but its figure/table units share chunk_ids with the
        # current cache, so we seed those to avoid re-generating them.
        if self.name == "smaller":
            self._seed_figure_questions()
            missing = [
                c
                for c in self.chunks
                if c["chunk_id"] not in QG.load_questions(self.qlabel)
            ]
            if missing:
                self.gen_summary = QG.generate_questions(
                    self.chunks, self.qlabel, n=MAX_GEN, limit=self.gen_limit
                )
                self.gen_summary["estimated_cost_usd"] = QG.estimate_cost(
                    self.gen_summary["prompt_tokens_this_run"],
                    self.gen_summary["completion_tokens_this_run"],
                )
                print(
                    f"  [{self.name}] gen: {json.dumps(self.gen_summary)}", flush=True
                )
        self.questions = QG.load_questions(self.qlabel)

    def _seed_figure_questions(self):
        """Copy figure/table-unit question rows from the current cache into the
        smaller cache (their chunk_ids are identical across configs)."""
        dst = QG.questions_path(self.qlabel)
        have = QG.load_questions(self.qlabel)
        cur = {}
        src = QG.questions_path("test-B")
        if src.exists():
            for line in src.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    cur[r["chunk_id"]] = r
        fig_ids = [c["chunk_id"] for c in self.chunks if "::fig::" in c["chunk_id"]]
        seed = [cur[cid] for cid in fig_ids if cid in cur and cid not in have]
        if seed:
            with dst.open("a", encoding="utf-8") as f:
                for r in seed:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# retrieval + eval for one (config, n_questions)
# --------------------------------------------------------------------------- #
def run_condition(cfg: Config, n_questions: int, k_values: List[int]) -> Dict:
    # build enriched texts (append) and embed as one vector per chunk
    texts = [
        _enriched(
            cfg.text_by_id[cid],
            cfg.questions.get(cid, [])[:n_questions] if n_questions else [],
        )
        for cid in cfg.chunk_ids
    ]
    vecs, enc_s = _embed(texts)
    cond_name = f"{cfg.name}_chunking_" + (
        "baseline" if n_questions == 0 else f"q{n_questions}"
    )
    index_mb = _write_chroma(cond_name, cfg.chunk_ids, vecs, cfg.paper_of)

    emb = get_embedder()
    maxk = max(k_values)
    per_query, search_ms = [], []
    for q in cfg.queries:
        qv = _san(np.asarray(emb.embed_query(q["question"]), np.float32))
        t1 = time.perf_counter()
        scores = vecs @ qv
        top = np.argpartition(-scores, kth=min(maxk, len(scores) - 1))[:maxk]
        top = top[np.argsort(-scores[top])]
        search_ms.append((time.perf_counter() - t1) * 1000)
        ranked = [cfg.chunk_ids[i] for i in top]
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update(
            {
                "query_id": q["query_id"],
                "retrieved": ranked,
                "gold": list(q["gold_chunk_ids"]),
                "paper_id": q["paper_id"],
            }
        )
        per_query.append(m)

    def avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def p95(v):
        v = sorted(v)
        return (
            round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 3) if v else 0.0
        )

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics.update({f"recall@{k}": avg(f"recall@{k}") for k in k_values})
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    # avg questions actually used per chunk for this condition
    used = (
        [min(len(cfg.questions.get(cid, [])), n_questions) for cid in cfg.chunk_ids]
        if n_questions
        else [0]
    )
    return {
        "condition": cond_name,
        "config": cfg.name,
        "n_questions": n_questions,
        "chunk_size": cfg.size,
        "overlap": cfg.overlap,
        "num_chunks": cfg.num_chunks,
        "avg_chunk_tokens": cfg.avg_tokens,
        "median_chunk_tokens": cfg.median_tokens,
        "avg_questions_used": round(st.mean(used), 1),
        "num_embeddings": len(cfg.chunk_ids),
        "index_mb": index_mb,
        "search_p95_ms": p95(search_ms),
        "metrics": metrics,
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# failure analysis (current vs smaller, at same q level)
# --------------------------------------------------------------------------- #
def _hit(pq, cutoff):
    gold = set(pq["gold"])
    return any(c in gold for c in pq["retrieved"][:cutoff])


def failure(cur_res, sml_res, cutoff=TOP_K):
    cur = {x["query_id"]: x for x in cur_res["per_query"]}
    sml = {x["query_id"]: x for x in sml_res["per_query"]}
    improved, hurt, unchanged = [], [], []
    for qid, s in sml.items():
        c = cur[qid]
        c_ok, s_ok = _hit(c, cutoff), _hit(s, cutoff)
        s_top = s["retrieved"][0] if s["retrieved"] else None
        same_paper_wrong = bool(
            s_top
            and s["paper_id"] == _paper_of(sml_res, s_top)
            and s_top not in set(s["gold"])
        )
        rec = {
            "eval_query": s["query_id"],
            "gold_paper_id": s["paper_id"],
            "current_gold_chunk_ids": c["gold"],
            "smaller_gold_chunk_ids": s["gold"],
            "current_chunking_top_chunks": c["retrieved"][:5],
            "smaller_chunking_top_chunks": s["retrieved"][:5],
            "current_correct": c_ok,
            "smaller_correct": s_ok,
            "smaller_retrieved_correct_paper_wrong_chunk": same_paper_wrong,
        }
        if s_ok and not c_ok:
            rec["possible_reason"] = "smaller_chunk_more_precise"
            improved.append(rec)
        elif c_ok and not s_ok:
            rec["possible_reason"] = (
                "same_paper_wrong_chunk"
                if same_paper_wrong
                else "smaller_chunk_lost_context"
            )
            hurt.append(rec)
        else:
            unchanged.append(
                {"eval_query": s["query_id"], "both_correct": bool(c_ok and s_ok)}
            )
    return improved, hurt, unchanged


def _paper_of(res, chunk_id):
    # chunk_id encodes paper: "<split>::<paper_id>::<kind>::<n>"
    parts = chunk_id.split("::")
    return parts[1] if len(parts) > 1 else None


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def run(max_papers=None, gen_limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = sorted({1, 5, 10, TOP_K})
    papers = load_papers("test-B", max_papers)

    cfgs = {}
    results = {}
    for cname in ("current", "smaller"):
        print(f"=== config {cname} ===", flush=True)
        cfg = Config(cname, papers, gen_limit=gen_limit)
        cfg.ensure_questions()
        cfgs[cname] = cfg
        for n in Q_CONDITIONS:
            r = run_condition(cfg, n, k_values)
            results[r["condition"]] = r
            m = r["metrics"]
            print(
                f"  {r['condition']:32} chunks={r['num_chunks']} "
                f"avgtok={r['avg_chunk_tokens']} Hit@1={m['hit@1']:.3f} "
                f"Hit@5={m['hit@5']:.3f} MRR={m['mrr']:.3f} nDCG@10={m['ndcg@10']:.3f} "
                f"idx={r['index_mb']}MB p95={r['search_p95_ms']}ms",
                flush=True,
            )

    base_mb = results["current_chunking_baseline"]["index_mb"]
    table = []
    for cond, r in results.items():
        m = r["metrics"]
        table.append(
            {
                "condition": cond,
                "chunk_size": r["chunk_size"],
                "overlap": r["overlap"],
                "num_chunks": r["num_chunks"],
                "avg_chunk_tokens": r["avg_chunk_tokens"],
                "median_chunk_tokens": r["median_chunk_tokens"],
                "question_count": r["n_questions"],
                "avg_questions_used": r["avg_questions_used"],
                "hit@1": m["hit@1"],
                "hit@5": m["hit@5"],
                "hit@10": m["hit@10"],
                "mrr": m["mrr"],
                "ndcg@10": m["ndcg@10"],
                "num_embeddings": r["num_embeddings"],
                "index_mb": r["index_mb"],
                "storage_x": round(r["index_mb"] / base_mb, 2) if base_mb else None,
                "search_p95_ms": r["search_p95_ms"],
            }
        )
    _write_table(table)

    # failure analyses
    for qn in (10, 50):
        imp, hurt, unch = failure(
            results[f"current_chunking_q{qn}"], results[f"smaller_chunking_q{qn}"]
        )
        suffix = "" if qn == 10 else "_q50"
        _save(
            OUT_DIR / f"improved_with_smaller_chunks{suffix}.json",
            {
                "comparison": f"current_q{qn} vs smaller_q{qn}",
                "count": len(imp),
                "queries": imp,
            },
        )
        _save(
            OUT_DIR / f"hurt_by_smaller_chunks{suffix}.json",
            {
                "comparison": f"current_q{qn} vs smaller_q{qn}",
                "count": len(hurt),
                "queries": hurt,
            },
        )
        _save(
            OUT_DIR / f"unchanged_by_smaller_chunks{suffix}.json",
            {
                "comparison": f"current_q{qn} vs smaller_q{qn}",
                "count": len(unch),
                "queries": unch,
            },
        )
        print(
            f"  failure q{qn}: +{len(imp)} improved / -{len(hurt)} hurt / {len(unch)} unchanged",
            flush=True,
        )

    # chunk statistics + latency/storage
    _save(
        OUT_DIR / "chunk_statistics.json",
        {
            cname: {
                "chunk_size": cfgs[cname].size,
                "overlap": cfgs[cname].overlap,
                "num_chunks": cfgs[cname].num_chunks,
                "avg_chunk_tokens": cfgs[cname].avg_tokens,
                "median_chunk_tokens": cfgs[cname].median_tokens,
                "num_text_chunks": sum(
                    1 for c in cfgs[cname].chunks if c["source_type"] == "text"
                ),
                "num_figure_table_units": sum(
                    1 for c in cfgs[cname].chunks if c["source_type"] != "text"
                ),
                "generation": cfgs[cname].gen_summary,
            }
            for cname in cfgs
        },
    )
    _save(
        OUT_DIR / "latency_storage_analysis.json",
        {
            r["condition"]: {
                "num_embeddings": r["num_embeddings"],
                "index_mb": r["index_mb"],
                "storage_x": round(r["index_mb"] / base_mb, 2) if base_mb else None,
                "search_p95_ms": r["search_p95_ms"],
            }
            for r in results.values()
        },
    )
    _write_summary(table, results)
    print(f"\nAll outputs under {OUT_DIR}", flush=True)
    return table


def _write_table(rows):
    cols = [
        "condition",
        "chunk_size",
        "overlap",
        "num_chunks",
        "avg_chunk_tokens",
        "question_count",
        "avg_questions_used",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10",
        "num_embeddings",
        "index_mb",
        "storage_x",
        "search_p95_ms",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT_DIR / "chunk_size_comparison_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    import csv

    with (OUT_DIR / "chunk_size_comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines), flush=True)


def _write_summary(table, results):
    by = {r["condition"]: r for r in table}

    def nd(cond):
        return by[cond]["ndcg@10"]

    def h1(cond):
        return by[cond]["hit@1"]

    d_base = round(nd("smaller_chunking_baseline") - nd("current_chunking_baseline"), 4)
    d_q10 = round(nd("smaller_chunking_q10") - nd("current_chunking_q10"), 4)
    d_q50 = round(nd("smaller_chunking_q50") - nd("current_chunking_q50"), 4)
    cur_q_gain = round(nd("current_chunking_q50") - nd("current_chunking_q10"), 4)
    sml_q_gain = round(nd("smaller_chunking_q50") - nd("smaller_chunking_q10"), 4)

    # correct-paper-wrong-chunk at top-1 for the two baselines
    def cpwc(cond):
        pq = results[cond]["per_query"]
        n = wrong = 0
        for x in pq:
            if not x["retrieved"]:
                continue
            n += 1
            top = x["retrieved"][0]
            if (
                top not in set(x["gold"])
                and _paper_of(results[cond], top) == x["paper_id"]
            ):
                wrong += 1
        return round(wrong / max(n, 1), 3)

    L = [
        "# Chunk size / overlap ablation — summary\n",
        f"SPIQA test-B. current = **416 tok / 100 overlap**, smaller = **208 tok / 50 "
        f"overlap**. Enrichment = append (chunk + questions as one vector). Same "
        f"LLM + embedder (`{embedding_signature()}`), split, queries, top-k, prompt. "
        f"Figure/table units are identical across configs (never split).\n",
        "## Comparison table\n",
    ]
    cols = [
        "condition",
        "chunk_size",
        "overlap",
        "num_chunks",
        "avg_chunk_tokens",
        "question_count",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10",
        "index_mb",
        "storage_x",
        "search_p95_ms",
    ]
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in table:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    L.append("\n## Findings\n")
    L.append(
        f"1. **Smaller chunks vs baseline retrieval:** {'improved' if d_base > 0 else 'did NOT improve'} "
        f"({'+' if d_base >= 0 else ''}{d_base} nDCG@10; "
        f"Hit@1 {h1('current_chunking_baseline')}→{h1('smaller_chunking_baseline')})."
    )
    L.append(
        f"2. **Smaller chunks + q10:** {'more effective' if d_q10 > 0 else 'not more effective'} "
        f"({'+' if d_q10 >= 0 else ''}{d_q10} nDCG@10 vs current+q10)."
    )
    L.append(
        f"3. **Smaller chunks + q50:** {'more effective' if d_q50 > 0 else 'not more effective'} "
        f"({'+' if d_q50 >= 0 else ''}{d_q50} nDCG@10 vs current+q50)."
    )
    L.append(
        f"4. **q50 vs q10:** current config q50−q10 = {'+' if cur_q_gain >= 0 else ''}{cur_q_gain}; "
        f"smaller config q50−q10 = {'+' if sml_q_gain >= 0 else ''}{sml_q_gain} nDCG@10. "
        + (
            "Smaller chunks reduce the benefit of adding many questions."
            if sml_q_gain < cur_q_gain
            else "Extra questions still help at least as much with smaller chunks."
        )
    )
    L.append(
        f"5. **Chunk-count blow-up:** {by['current_chunking_baseline']['num_chunks']} → "
        f"{by['smaller_chunking_baseline']['num_chunks']} chunks "
        f"(×{round(by['smaller_chunking_baseline']['num_chunks'] / by['current_chunking_baseline']['num_chunks'], 2)})."
    )
    L.append(
        f"6. **Latency/storage:** smaller baseline index {by['smaller_chunking_baseline']['index_mb']}MB "
        f"(×{by['smaller_chunking_baseline']['storage_x']}) vs current; "
        f"p95 {by['smaller_chunking_baseline']['search_p95_ms']}ms vs "
        f"{by['current_chunking_baseline']['search_p95_ms']}ms."
    )
    L.append(
        f"7. **Exact-chunk vs correct-paper-wrong-chunk:** correct-paper-wrong-chunk@1 = "
        f"{cpwc('current_chunking_baseline')} (current) vs "
        f"{cpwc('smaller_chunking_baseline')} (smaller). "
        + (
            "Smaller chunks reduce wrong-chunk errors."
            if cpwc("smaller_chunking_baseline") < cpwc("current_chunking_baseline")
            else "Smaller chunks do NOT reduce wrong-chunk errors."
        )
    )
    # recommendation
    best = max(table, key=lambda r: r["ndcg@10"])
    L.append(
        f"\n## Recommendation\n**{best['condition']}** gave the best nDCG@10 ({best['ndcg@10']}). "
        + (
            "Smaller chunking is worth it."
            if best["config"] == "smaller" or "smaller" in best["condition"]
            else "Current chunking remains the better default; smaller chunking did not pay off here."
        )
    )
    (OUT_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--gen-limit", type=int, default=None)
    args = ap.parse_args()
    run(args.max_papers, args.gen_limit)


if __name__ == "__main__":
    main()
