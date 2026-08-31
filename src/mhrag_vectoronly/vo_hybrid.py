"""
Hybrid (dense + BM25 + RRF) sub-study on the 15-article set, for arms A, B, E1, E3.

Fairness invariant: BM25 runs over the SAME text each arm's dense index uses —
chunk text for A, the generated questions for B/E1/E3 — so only the indexed
representation differs. Per query we take the dense ranking (Chroma) and the BM25
ranking, fuse with Reciprocal Rank Fusion (k=60, fixed, untuned), map questions to
parent chunks, dedup, and evaluate the fused ranked chunks. Reports dense vs hybrid
for every arm plus the delta (paired bootstrap CI).

No new LLM calls — reuses the question sets already generated/indexed. This is a
clearly-labeled hybrid sub-study, separate from the dense-only tables.
"""

from __future__ import annotations

import re
from html import escape

import numpy as np
from rank_bm25 import BM25Okapi

import vo_config as C
import vo_data as D
import vo_retrieval as VR
import vo_metrics as VM
import vo_three_arms as T
from embeddings import get_embedder

VM.KS = C.TOP_K_VALUES
KS = C.TOP_K_VALUES
RRF_K = 60
REPORT = C.PROJECT_ROOT / "report" / "mhrag_15articles_hybrid.html"
_TOK = re.compile(r"[a-z0-9]+")


def tok(s):
    return _TOK.findall((s or "").lower())


def rrf(dense_list, bm25_list, depth):
    sc = {}
    for lst in (dense_list, bm25_list):
        for rank, cid in enumerate(lst, 1):
            sc[cid] = sc.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return [cid for cid, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:depth]]


def bm25_chunk_rank(bm25, ids, query, depth):
    order = np.argsort(-bm25.get_scores(tok(query)))[:depth]
    return [ids[i] for i in order]


def bm25_q_rank(bm25, owners, query, depth):
    order = np.argsort(-bm25.get_scores(tok(query)))
    out = []
    for i in order:
        p = owners[i]
        if p not in out:
            out.append(p)
            if len(out) >= depth:
                break
    return out


def _q_items(source):
    """Return list of (question, parent_chunk_id) for a question arm."""
    if source == "B":
        rows = D.read_jsonl(T.GEN_PATH)
        return [(q, r["chunk_id"]) for r in rows for q in r["questions"]]
    k = 1 if source == "E1" else 3
    rows = D.read_jsonl(T.SENT_PATH)  # 3-questions-per-sentence generation
    return [
        (q, r["chunk_id"])
        for r in rows
        for s in r["sentences"]
        for q in s["questions"][:k]
    ]


def _dci(base, arm, n=1000, seed=42):
    d = arm - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    return d.mean(), lo, hi, (lo > 0 or hi < 0)


def run():
    chunks = D.load_chunks()
    parent = {c["chunk_id"]: c["parent_document_id"] for c in chunks}
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    depth = C.RANK_DEPTH

    # arm spec: (tag, label, kind, dense_collection, bm25_builder)
    arms = [
        ("A", "Original chunks", "chunk", C.BASELINE_COLLECTION, None),
        ("B", "General 10 Q/chunk", "q", T.GEN_COLL, "B"),
        ("E1", "1 Q/sentence", "q", T.SENT1_COLL, "E1"),
        ("E3", "3 Q/sentence", "q", T.SENT3_COLL, "E3"),
    ]

    # build BM25 indices + grab dense collections + vector counts
    setup = {}
    for tag, label, kind, coll_name, src in arms:
        coll = C.get_collection(coll_name)
        if kind == "chunk":
            bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
            setup[tag] = dict(
                label=label,
                kind=kind,
                coll=coll,
                bm25=bm25,
                ids=[c["chunk_id"] for c in chunks],
                vectors=coll.count(),
            )
        else:
            items = _q_items(src)
            bm25 = BM25Okapi([tok(q) for q, _ in items])
            setup[tag] = dict(
                label=label,
                kind=kind,
                coll=coll,
                bm25=bm25,
                owners=[c for _, c in items],
                vectors=coll.count(),
            )

    # per-query rankings -> per_query metrics for dense and hybrid
    rows = {tag: {"dense": [], "hybrid": []} for tag, *_ in arms}
    for q in queries:
        g = gold[q["query_id"]]
        qv = embedder.embed_query(q["query"])
        common = {
            "query_id": q["query_id"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        for tag, label, kind, _cn, _src in arms:
            s = setup[tag]
            if kind == "chunk":
                dense = [
                    r["chunk_id"] for r in VR.retrieve_baseline(s["coll"], qv, depth)
                ]
                bm = bm25_chunk_rank(s["bm25"], s["ids"], q["query"], depth)
            else:
                dense = [
                    r["chunk_id"]
                    for r in VR.retrieve_generated(
                        s["coll"], qv, depth, C.CANDIDATE_MULTIPLIER
                    )
                ]
                bm = bm25_q_rank(s["bm25"], s["owners"], q["query"], depth)
            hyb = rrf(dense, bm, depth)
            for mode, lst in (("dense", dense), ("hybrid", hyb)):
                ranked = [
                    {"rank": i, "chunk_id": cid, "parent_document_id": parent[cid]}
                    for i, cid in enumerate(lst, 1)
                ]
                rows[tag][mode].append(VM.per_query({**common, "ranked": ranked}))

    qids = sorted(r["query_id"] for r in rows["A"]["dense"])

    def vals(pq_list, key):
        idx = {r["query_id"]: r for r in pq_list}
        return np.array([idx[q][key] for q in qids])

    METS = [
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    data = {}
    for tag, *_ in arms:
        data[tag] = {}
        for mk, _ in METS:
            dv = vals(rows[tag]["dense"], mk)
            hv = vals(rows[tag]["hybrid"], mk)
            dm, dlo, dhi, sig = _dci(dv, hv)
            data[tag][mk] = dict(dense=dv.mean(), hybrid=hv.mean(), d=dm, sig=sig)

    _render(arms, setup, data, METS, len(qids))
    print("=== dense -> hybrid (Recall@5) ===")
    for tag, *_ in arms:
        c = data[tag]["evidence_recall@5"]
        print(
            f"  {tag:<3} {c['dense']:.3f} -> {c['hybrid']:.3f}  (Δ{c['d']:+.3f}{'*' if c['sig'] else ''})"
        )


def _render(arms, setup, data, METS, n):
    def cell(tag, mk):
        c = data[tag][mk]
        cls = "good" if c["d"] > 1e-9 else ("bad" if c["d"] < -1e-9 else "flat")
        star = "<b>*</b>" if c["sig"] else ""
        return (
            f'<td><span class="dn">{c["dense"]:.3f}</span> <span class="ar">→</span> '
            f'<span class="hy">{c["hybrid"]:.3f}</span>'
            f'<span class="d {cls}">({c["d"]:+.3f}{star})</span></td>'
        )

    body = ""
    for tag, label, *_ in arms:
        tds = "".join(cell(tag, mk) for mk, _ in METS)
        body += (
            f'<tr class="{"base" if tag == "A" else ""}"><td class="nm"><b>{tag}</b> · {escape(label)}</td>'
            f"{tds}<td>{setup[tag]['vectors']}</td></tr>"
        )
    head = "".join(
        f"<th>{lbl}<span class='sub'>dense → hybrid (Δ)</span></th>" for _, lbl in METS
    )

    # best hybrid arm vs A-hybrid
    a_h = data["A"]["evidence_recall@5"]["hybrid"]
    best = max(
        (t for t, *_ in arms if t != "A"),
        key=lambda t: data[t]["evidence_recall@5"]["hybrid"],
    )
    best_h = data[best]["evidence_recall@5"]["hybrid"]

    html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Hybrid (dense + BM25) — 15-article MultiHop-RAG</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1;--dn:#5b6572;--hy:#2563eb;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1;--dn:#9aa5b3;--hy:#6ea0ff;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:34px 22px 70px}}h1{{font-size:22px;margin:0 0 4px}}.cap{{color:var(--muted);font-size:12.5px;margin:6px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}th,td{{border:1px solid var(--line);padding:8px;text-align:center}}
th{{background:var(--card);font-weight:600}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}tr.base td.nm{{font-weight:700}}
.sub{{display:block;font-weight:400;color:var(--muted);font-size:10px}}.dn{{color:var(--dn)}}.hy{{color:var(--hy);font-weight:700}}.ar{{color:var(--muted)}}
.d{{display:block;font-size:11px;margin-top:1px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--hy);border-radius:10px;padding:14px 16px;margin:16px 0}}
.leg{{font-size:12px;color:var(--muted);margin-top:8px}}
</style></head><body><div class="wrap">
<h1>Hybrid retrieval (dense + BM25 + RRF) — 15-article MultiHop-RAG</h1>
<p class="cap">Each arm's BM25 runs over the same text its dense index uses (chunk text for A; the generated questions for B/E1/E3), fused with Reciprocal Rank Fusion (k={RRF_K}, fixed). n={n} queries · 154 chunks · Octen-0.6B · gpt-5.4-mini questions. Each cell: <span class="dn">dense</span> → <span class="hy">hybrid</span> (Δ = hybrid−dense); green/red = BM25 helped/hurt; <b>*</b> = Δ 95% bootstrap CI excludes 0.</p>
<table><thead><tr><th>Condition</th>{head}<th>Vectors</th></tr></thead><tbody>{body}</tbody></table>
<div class="verdict"><b>Read:</b> the &quot;→ hybrid&quot; column shows what adding BM25 does to each representation. The strong baseline is <b>A hybrid</b> (dense chunks + BM25 chunks) = {a_h:.3f} Recall@5. The best generated arm under hybrid is <b>{best} = {best_h:.3f}</b> — {"above" if best_h > a_h else "still below"} the hybrid-chunks baseline (Δ{best_h - a_h:+.3f}).</div>
<p class="leg">Hybrid sub-study — separate from the dense-only tables. RRF k=60 fixed (untuned). BM25 tokenization = lowercase alphanumerics. Closed 15-article pilot (n={n}) — not full-corpus MultiHop-RAG.</p>
</div></body></html>"""
    REPORT.write_text(html, encoding="utf-8")
    print(f"[hybrid] wrote {REPORT}")


if __name__ == "__main__":
    run()
