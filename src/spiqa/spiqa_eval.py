"""
Retrieval evaluation for the SPIQA generated-questions experiment.

Ground truth
------------
SPIQA gives us, per question:
  * referred_figures_tables -> exact gold figure/table ids (strong labels), and
  * evidential_info / answer.evidence -> verbatim passage excerpts.

We convert those into a set of gold CHUNK ids per query:
  * every figure/table chunk whose fig_id is in referred_figures_tables, and
  * every text chunk that contains a gold evidence sentence (verbatim substring
    match after whitespace normalisation — SPIQA evidence is quoted from the
    passages).

Fallback (documented): if a question yields no matchable chunk-level gold
(no referred figures and no locatable evidence text), we fall back to
PAPER-LEVEL relevance for that query — any chunk from the source paper counts.
Each query records which relevance mode was used so the summary can report the
share of strict vs fallback judgements.

Retrieval (multi-vector aware)
------------------------------
For each query we embed it once, over-fetch candidates from Chroma, then map
every hit back to its parent chunk (via metadata.parent_chunk_id), dedupe
keeping the best rank, and keep the top-K DISTINCT parent chunks. The chunk is
the retrieved evidence — never the generated question.

Metrics: Hit@1/5/10, Recall@k, MRR, nDCG@10. Latency: query-embedding time and
Chroma search time (mean / p50 / p95).
"""

from __future__ import annotations

import math
import re
import statistics as st
import time
from typing import Dict, List, Optional, Tuple

import spiqa_config as C
from embeddings import get_embedder
from spiqa_index import get_collection
from spiqa_loader import load_papers, Paper


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\xa0", " ")).strip().lower()


def _sentences(text: str) -> List[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text or "")
        if len(s.strip()) >= 40
    ]


# --------------------------------------------------------------------------- #
# Gold construction
# --------------------------------------------------------------------------- #
def build_gold(papers: List[Paper], chunks: List[Dict]) -> List[Dict]:
    """
    Return a list of query dicts:
       {query_id, paper_id, question, gold_chunk_ids:set, relevance_mode}
    """
    # index chunks by paper
    by_paper: Dict[str, List[Dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)

    queries: List[Dict] = []
    for paper in papers:
        pchunks = by_paper.get(paper.paper_id, [])
        # pre-normalise this paper's text chunks for evidence matching
        norm_text = [
            (c, _norm(c["text"])) for c in pchunks if c["source_type"] == "text"
        ]
        fig_chunks = [
            c
            for c in pchunks
            if c["source_type"] in ("figure_caption", "table_caption")
        ]

        for qi, qa in enumerate(paper.questions):
            gold: set = set()

            # (a) figure/table gold via referred ids (exact match on fig_id)
            refs = set(qa.referred_figures)
            if refs:
                for c in fig_chunks:
                    if c["metadata"].get("fig_id") in refs:
                        gold.add(c["chunk_id"])

            # (b) text gold via verbatim evidence sentence match
            for sent in _sentences(qa.evidence_text):
                ns = _norm(sent)
                for c, nt in norm_text:
                    if ns in nt:
                        gold.add(c["chunk_id"])

            mode = "chunk"
            if not gold:
                # fallback: paper-level relevance (any chunk of this paper)
                gold = {c["chunk_id"] for c in pchunks}
                mode = "paper"

            if not gold:
                continue  # paper had no chunks at all; skip

            queries.append(
                {
                    "query_id": f"{paper.paper_id}::{qi}",
                    "paper_id": paper.paper_id,
                    "question": qa.question,
                    "gold_chunk_ids": gold,
                    "relevance_mode": mode,
                }
            )
    return queries


# --------------------------------------------------------------------------- #
# Retrieval + per-query metrics
# --------------------------------------------------------------------------- #
def _retrieve_parents(coll, qvec: List[float], k: int, overfetch: int) -> List[str]:
    """Return top-k DISTINCT parent chunk ids, best-rank-first."""
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec], n_results=min(overfetch, n), include=["metadatas"]
    )
    metas = res["metadatas"][0] if res["metadatas"] else []
    seen: List[str] = []
    seen_set = set()
    for m in metas:
        parent = m.get("parent_chunk_id")
        if parent and parent not in seen_set:
            seen_set.add(parent)
            seen.append(parent)
            if len(seen) >= k:
                break
    return seen


def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _query_metrics(ranked: List[str], gold: set, k_values: List[int]) -> Dict:
    rels = [1 if cid in gold else 0 for cid in ranked]
    out: Dict[str, float] = {}
    for k in k_values:
        topk = rels[:k]
        out[f"hit@{k}"] = 1.0 if any(topk) else 0.0
        out[f"recall@{k}"] = (sum(topk) / len(gold)) if gold else 0.0
    # MRR (first relevant rank)
    mrr = 0.0
    for i, r in enumerate(rels):
        if r:
            mrr = 1.0 / (i + 1)
            break
    out["mrr"] = mrr
    # nDCG@10
    ideal = sorted(rels, reverse=True)[:10]
    idcg = _dcg(ideal)
    out["ndcg@10"] = (_dcg(rels[:10]) / idcg) if idcg > 0 else 0.0
    return out


def evaluate_condition(
    split: str,
    n_questions: int,
    queries: List[Dict],
    *,
    k: int = C.TOP_K,
    k_values: Optional[List[int]] = None,
    overfetch_factor: int = 25,
) -> Dict:
    """Run all queries against one condition's collection and aggregate metrics."""
    k_values = k_values or C.K_VALUES
    coll = get_collection(split, n_questions)
    embedder = get_embedder()

    overfetch = max(k * overfetch_factor, 100)
    per_query: List[Dict] = []
    emb_times: List[float] = []
    search_times: List[float] = []

    for q in queries:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        emb_times.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        ranked = _retrieve_parents(coll, qvec, max(k_values), overfetch)
        search_times.append((time.perf_counter() - t1) * 1000)

        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        m.update(
            {
                "query_id": q["query_id"],
                "relevance_mode": q["relevance_mode"],
                "n_gold": len(q["gold_chunk_ids"]),
                "retrieved": ranked[: max(k_values)],
            }
        )
        per_query.append(m)

    def _avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    agg: Dict[str, float] = {}
    for k_ in k_values:
        agg[f"hit@{k_}"] = _avg(f"hit@{k_}")
        agg[f"recall@{k_}"] = _avg(f"recall@{k_}")
    agg["mrr"] = _avg("mrr")
    agg["ndcg@10"] = _avg("ndcg@10")

    def _pct(vals, p):
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))
        return round(vals[idx], 3)

    return {
        "condition": C.condition_name(n_questions),
        "n_questions_per_chunk": n_questions,
        "num_queries": len(queries),
        "num_chunk_level_queries": sum(
            1 for q in queries if q["relevance_mode"] == "chunk"
        ),
        "num_paper_level_fallback_queries": sum(
            1 for q in queries if q["relevance_mode"] == "paper"
        ),
        "metrics": agg,
        "latency_ms": {
            "query_embed_mean": round(st.mean(emb_times), 3) if emb_times else 0,
            "chroma_search_mean": round(st.mean(search_times), 3)
            if search_times
            else 0,
            "chroma_search_p50": _pct(search_times, 50),
            "chroma_search_p95": _pct(search_times, 95),
            "total_mean": round((st.mean(emb_times) + st.mean(search_times)), 3)
            if emb_times
            else 0,
        },
        "per_query": per_query,
    }


if __name__ == "__main__":
    import argparse, json
    from spiqa_chunker import load_chunks

    ap = argparse.ArgumentParser(description="Evaluate retrieval for one condition.")
    ap.add_argument("--split", default=C.DEFAULT_SPLIT)
    ap.add_argument("-n", "--num", type=int, default=0)
    ap.add_argument("--max-papers", type=int, default=None)
    args = ap.parse_args()

    papers = load_papers(args.split, args.max_papers)
    chunks = load_chunks(args.split)
    queries = build_gold(papers, chunks)
    res = evaluate_condition(args.split, args.num, queries)
    res.pop("per_query")
    print(json.dumps(res, indent=2))
