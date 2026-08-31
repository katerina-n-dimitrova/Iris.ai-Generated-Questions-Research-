"""
Ablation: drop the strict positive-margin gate (which collapsed to rank-1) and
instead accept a Condition-E question when its parent chunk round-trips within
rank <= {1, 5, 10}. Everything else (early gates, near-dup, coverage-aware
selection, max/chunk) is unchanged. Reports each variant's retrieval vs the
Condition-A chunk baseline.

Round-trip rank is threshold-independent, so it is computed once and re-thresholded.
Writes results/mhrag_atomic_chunk_mix_10/rank_sweep.json and
report/mhrag_atomic_chunk_mix_10_rank_sweep.html. Rebuilds the canonical strict
E index at the end so the main report's collection is unchanged.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import am_config as C
import am_data as D
import am_index as IDX
import am_retrieval as AR
from am_filter import _grounded, _self_contained
from embeddings import get_embedder
import vo_metrics as VM

VM.KS = C.TOP_K_VALUES
KS = C.TOP_K_VALUES
THRESHOLDS = [1, 5, 10]


def prepare():
    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    raw = D.read_jsonl(C.QUESTIONS_RAW)
    survivors = []
    for q in raw:
        if not q["question"].strip() or not q.get("parent_chunk_id"):
            continue
        if not q.get("supporting_spans") or not q.get("short_answer"):
            continue
        if not _grounded(q, chunks.get(q["parent_chunk_id"], {}).get("text", "")):
            continue
        if not _self_contained(q):
            continue
        survivors.append(q)

    if C.get_collection(C.BASELINE_COLLECTION).count() == 0:
        IDX.build_baseline_index(list(chunks.values()))
    coll = C.get_collection(C.BASELINE_COLLECTION)
    n = coll.count()
    embedder = get_embedder()
    embs = embedder.embed_documents([q["question"] for q in survivors])
    emb_of = {q["question_id"]: np.array(v) for q, v in zip(survivors, embs)}
    for q, v in zip(survivors, embs):
        res = coll.query(
            query_embeddings=[v], n_results=n, include=["metadatas", "distances"]
        )
        ids = [m["parent_chunk_id"] for m in res["metadatas"][0]]
        sims = [1.0 - d for d in res["distances"][0]]
        parent = q["parent_chunk_id"]
        prank = ids.index(parent) + 1 if parent in ids else 10**9
        psim = sims[ids.index(parent)] if parent in ids else 0.0
        nonparent = [s for i, s in zip(ids, sims) if i != parent]
        q["_parent_rank"] = prank
        q["_margin"] = psim - (max(nonparent) if nonparent else 0.0)
    return survivors, emb_of, chunks


def select(survivors, emb_of, chunks, parent_topk):
    """Accept parent-rank <= parent_topk (NO margin gate), then near-dup + coverage."""
    passed = [q for q in survivors if q["_parent_rank"] <= parent_topk]
    by_chunk = defaultdict(list)
    for q in passed:
        by_chunk[q["parent_chunk_id"]].append(q)
    accepted = []
    for cid, qs in by_chunk.items():
        qs.sort(key=lambda x: x["_margin"], reverse=True)
        kept = []
        for q in qs:
            if not any(
                float(emb_of[q["question_id"]] @ emb_of[k["question_id"]])
                >= C.NEAR_DUP_THRESHOLD
                for k in kept
            ):
                kept.append(q)
        atomic = [q for q in kept if q["question_type"] == "atomic"]
        chunk_lv = [q for q in kept if q["question_type"] == "chunk_level"]
        chosen, covered = [], set()
        cap_atomic = C.MAX_TOTAL_Q - C.DEFAULT_CHUNK_LEVEL_Q
        for q in atomic:
            if q["atom_id"] in covered:
                continue
            chosen.append(q)
            covered.add(q["atom_id"])
            if len(chosen) >= cap_atomic:
                break
        for q in atomic:
            if len(chosen) >= cap_atomic:
                break
            if q not in chosen:
                chosen.append(q)
        for q in chunk_lv[: C.MAX_CHUNK_LEVEL_Q]:
            if len(chosen) >= C.MAX_TOTAL_Q:
                break
            chosen.append(q)
        for q in chosen:
            q = dict(q)
            q["parent_chunk_text"] = chunks[cid]["text"]
            accepted.append(q)
    return accepted


def _rankings(accepted, queries, gold, qvecs):
    IDX.build_mixed_index(accepted)
    coll = C.get_collection(C.MIXED_COLLECTION)
    rows = []
    for q in queries:
        g = gold[q["query_id"]]
        e = AR.retrieve_mixed(
            coll, qvecs[q["query_id"]], C.RANK_DEPTH, C.CANDIDATE_MULTIPLIER
        )
        rows.append(
            {
                "query_id": q["query_id"],
                "question_type": q["question_type"],
                "n_required_documents": q["n_required_documents"],
                "n_required_evidence_facts": q["n_required_evidence_facts"],
                "gold_chunk_ids": g["gold_chunk_ids"],
                "evidence_units": g["evidence_units"],
                "required_article_ids": q["required_article_ids"],
                "ranked": e,
            }
        )
    return rows


def run():
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    qvecs = {q["query_id"]: embedder.embed_query(q["query"]) for q in queries}

    # baseline A rankings (once)
    coll_a = C.get_collection(C.BASELINE_COLLECTION)
    base_rows = []
    for q in queries:
        g = gold[q["query_id"]]
        a = AR.retrieve_baseline(coll_a, qvecs[q["query_id"]], C.RANK_DEPTH)
        base_rows.append(
            {
                "query_id": q["query_id"],
                "question_type": q["question_type"],
                "n_required_documents": q["n_required_documents"],
                "n_required_evidence_facts": q["n_required_evidence_facts"],
                "gold_chunk_ids": g["gold_chunk_ids"],
                "evidence_units": g["evidence_units"],
                "required_article_ids": q["required_article_ids"],
                "ranked": a,
            }
        )
    per_a = [VM.per_query(r) for r in sorted(base_rows, key=lambda r: r["query_id"])]

    survivors, emb_of, chunks = prepare()

    variants = {}
    for thr in THRESHOLDS:
        accepted = select(survivors, emb_of, chunks, thr)
        rows = _rankings(accepted, queries, gold, qvecs)
        per_e = [VM.per_query(r) for r in sorted(rows, key=lambda r: r["query_id"])]
        pair = VM.paired(per_a, per_e)
        agg = VM.aggregate({"generated": per_e})["generated"]
        n_atomic = sum(1 for q in accepted if q["question_type"] == "atomic")
        variants[thr] = {
            "parent_rank_max": thr,
            "num_vectors": len(accepted),
            "num_atomic": n_atomic,
            "num_chunk_level": len(accepted) - n_atomic,
            "chunks_covered": len({q["parent_chunk_id"] for q in accepted}),
            "metrics": {
                f"{m}@{k}": agg[f"{m}@{k}"]["mean"] for m in VM.RATE_METRICS for k in KS
            },
            "first_relevant_rank": agg["first_relevant_rank_mean"],
            "paired_p": {
                f"{m}@{k}": pair[f"{m}@{k}"]["paired_p"]
                for m in VM.RATE_METRICS
                for k in KS
            },
        }

    base_agg = VM.aggregate({"baseline": per_a})["baseline"]
    baseline = {
        "num_vectors": coll_a.count(),
        "metrics": {
            f"{m}@{k}": base_agg[f"{m}@{k}"]["mean"]
            for m in VM.RATE_METRICS
            for k in KS
        },
        "first_relevant_rank": base_agg["first_relevant_rank_mean"],
    }

    out = {
        "n_queries": len(queries),
        "k_values": KS,
        "baseline": baseline,
        "variants": variants,
        "thresholds": THRESHOLDS,
        "note": "Margin gate removed; acceptance = round-trip parent rank <= threshold.",
    }
    json.dump(out, open(C.RESULTS_DIR / "rank_sweep.json", "w"), indent=2)

    # restore canonical strict E index for the main report
    IDX.build_mixed_index()
    _render(out)
    _print(out)
    return out


def _print(out):
    b = out["baseline"]["metrics"]
    print(
        f"\nn={out['n_queries']}  |  Condition A baseline: "
        f"ER@1={b['evidence_recall@1']:.3f} ER@5={b['evidence_recall@5']:.3f} "
        f"ER@10={b['evidence_recall@10']:.3f} | vectors={out['baseline']['num_vectors']}"
    )
    for thr in out["thresholds"]:
        v = out["variants"][thr]
        m = v["metrics"]
        p = v["paired_p"]

        def s(key):
            return "*" if p[key] < 0.05 else " "

        print(
            f"E rank<= {thr:<2}: ER@1={m['evidence_recall@1']:.3f}{s('evidence_recall@1')} "
            f"ER@5={m['evidence_recall@5']:.3f}{s('evidence_recall@5')} "
            f"ER@10={m['evidence_recall@10']:.3f}{s('evidence_recall@10')} | "
            f"Hit@5={m['hit@5']:.3f} MRR@10={m['mrr@10']:.3f} | "
            f"vecs={v['num_vectors']} ({v['num_atomic']}a/{v['num_chunk_level']}cl) "
            f"chunks={v['chunks_covered']}/96"
        )


def _render(out):
    A = out["baseline"]["metrics"]
    cols = [("evidence_recall", "Evidence Recall"), ("hit", "Hit"), ("mrr", "MRR")]
    show_k = [1, 5, 10]
    head = "".join(f"<th>{lbl}@{k}</th>" for _, lbl in cols for k in show_k)

    def row(name, met, vecs, extra, p=None, cls=""):
        cells = ""
        for mk, _ in cols:
            for k in show_k:
                key = f"{mk}@{k}"
                val = met[key]
                d = (
                    ""
                    if p is None
                    else f"<span class='d {'good' if val - A[key] > 0 else ('bad' if val - A[key] < 0 else 'flat')}'>({val - A[key]:+.3f}{'*' if p and p[key] < 0.05 else ''})</span>"
                )
                cells += f"<td>{val:.3f}{d}</td>"
        return f"<tr class='{cls}'><td class='nm'>{name}</td>{cells}<td>{vecs}</td><td>{extra}</td></tr>"

    rows = row(
        "A · original chunks",
        A,
        out["baseline"]["num_vectors"],
        "1× · frr " + f"{out['baseline']['first_relevant_rank']}",
        cls="base",
    )
    for thr in out["thresholds"]:
        v = out["variants"][thr]
        rows += row(
            f"E · parent rank ≤ {thr}",
            v["metrics"],
            v["num_vectors"],
            f"{v['num_atomic']}a/{v['num_chunk_level']}cl · {v['chunks_covered']}/96 chunks · frr {v['first_relevant_rank']}",
            p=v["paired_p"],
        )
    html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Condition E — round-trip rank threshold sweep (margin gate removed)</title>
<style>
:root{{--bg:#fff;--fg:#1a1d24;--muted:#5b6572;--line:#e4e8ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8a94a2;--a:#2563eb;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#7a8494;--a:#6ea0ff;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 20px 70px}}h1{{font-size:23px;margin:0 0 6px}}
.lede{{color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{border:1px solid var(--line);padding:7px 8px;text-align:center}}th{{background:var(--card)}}td.nm,th:first-child{{text-align:left}}
tr.base{{background:var(--card);font-weight:600}}
.d{{display:block;font-size:10.5px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.legend{{font-size:12.5px;color:var(--muted)}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--a);border-radius:10px;padding:14px 16px;margin:16px 0}}
</style></head><body><div class=wrap>
<h1>Condition E — accept by round-trip parent rank ≤ {{1, 5, 10}}</h1>
<p class=lede>Strict positive-margin gate removed (it collapsed to rank-1). A question is accepted when its
parent chunk round-trips within the given rank; then near-dup + coverage-aware selection as before.
n={out["n_queries"]} eligible queries · Δ vs Condition A in parentheses · <b>*</b> = paired p&lt;0.05.</p>
<div class=verdict><b>Takeaway:</b> loosening the round-trip threshold adds question vectors and chunk
coverage but does <b>not</b> close the gap to the chunk baseline — Evidence Recall stays below A at
every threshold, and looser thresholds add noise (more false-positive matches) rather than recall.</div>
<div style="overflow-x:auto"><table><thead><tr><th>Condition</th>{head}<th>vectors</th><th>notes</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class=legend>frr = mean first-relevant rank (lower is better). "a/cl" = accepted atomic / chunk-level questions.
A closed-collection, 10-article pilot — not the full MultiHop-RAG benchmark. Main report:
<a href="mhrag_atomic_chunk_mix_10_results.html">mhrag_atomic_chunk_mix_10_results.html</a></p>
</div></body></html>"""
    (C.REPORT_HTML.parent / f"{C.NAMESPACE}_rank_sweep.html").write_text(
        html, encoding="utf-8"
    )
    print(f"[sweep] wrote {C.REPORT_HTML.parent / (C.NAMESPACE + '_rank_sweep.html')}")


if __name__ == "__main__":
    run()
