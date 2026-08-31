"""
Hybrid (dense + BM25 + RRF) sub-study on the 50-article set, arms A, B, E1, E3.

Same design as the 15-article hybrid run: BM25 over each arm's own text (chunk text
for A; the generated questions for B/E1/E3), RRF-fused (k=60, fixed) with the dense
ranking, mapped to parent chunks. Reuses the BM25/RRF/bootstrap helpers from
vo_hybrid and the 50-article collections/paths from vo_fifty. No new LLM calls.
Writes report/mhrag_50articles_hybrid.html.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from html import escape

import numpy as np
from rank_bm25 import BM25Okapi

import vo_fifty as F
import vo_retrieval as VR
import vo_metrics as VM
from embeddings import get_embedder
from vo_data import read_jsonl
from vo_hybrid import tok, rrf, bm25_chunk_rank, bm25_q_rank, _dci, RRF_K

VM.KS = F.KS
KS = F.KS
REPORT = F.base.PROJECT_ROOT / "report" / "mhrag_50articles_hybrid.html"


def q_items(source):
    if source == "B":
        return [
            (q, r["chunk_id"]) for r in read_jsonl(F.GEN_PATH) for q in r["questions"]
        ]
    k = 1 if source == "E1" else 3
    return [
        (q, r["chunk_id"])
        for r in read_jsonl(F.SENT_PATH)
        for s in r["sentences"]
        for q in s["questions"][:k]
    ]


def run():
    chunks = F.load_chunks()
    parent = {c["chunk_id"]: c["parent_document_id"] for c in chunks}
    queries = F.load_queries()
    gold = F.load_gold()
    embedder = get_embedder()
    depth = F.RANK_DEPTH

    arms = [
        ("A", "Original chunks", "chunk", F.BASE_COLL, None),
        ("B", "General 10 Q/chunk", "q", F.GEN_COLL, "B"),
        ("E1", "1 Q/sentence", "q", F.SENT1_COLL, "E1"),
        ("E3", "3 Q/sentence", "q", F.SENT3_COLL, "E3"),
    ]

    setup = {}
    for tag, label, kind, coll_name, src in arms:
        coll = F.get_collection(coll_name)
        if kind == "chunk":
            setup[tag] = dict(
                label=label,
                kind=kind,
                coll=coll,
                bm25=BM25Okapi([tok(c["text"]) for c in chunks]),
                ids=[c["chunk_id"] for c in chunks],
                vectors=coll.count(),
            )
        else:
            items = q_items(src)
            setup[tag] = dict(
                label=label,
                kind=kind,
                coll=coll,
                bm25=BM25Okapi([tok(q) for q, _ in items]),
                owners=[c for _, c in items],
                vectors=coll.count(),
            )
        print(f"[hybrid50] built {tag} ({setup[tag]['vectors']} vectors)", flush=True)

    rows = {tag: {"dense": [], "hybrid": []} for tag, *_ in arms}
    for n, q in enumerate(queries, 1):
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
                    for r in VR.retrieve_generated(s["coll"], qv, depth, F.CAND_MULT)
                ]
                bm = bm25_q_rank(s["bm25"], s["owners"], q["query"], depth)
            hyb = rrf(dense, bm, depth)
            for mode, lst in (("dense", dense), ("hybrid", hyb)):
                ranked = [
                    {"rank": i, "chunk_id": cid, "parent_document_id": parent[cid]}
                    for i, cid in enumerate(lst, 1)
                ]
                rows[tag][mode].append(VM.per_query({**common, "ranked": ranked}))
        if n % 100 == 0:
            print(f"[hybrid50] {n}/{len(queries)} queries", flush=True)

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
            f'<span class="hy">{c["hybrid"]:.3f}</span><span class="d {cls}">({c["d"]:+.3f}{star})</span></td>'
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
    a_h = data["A"]["evidence_recall@5"]["hybrid"]
    best = max(
        (t for t, *_ in arms if t != "A"),
        key=lambda t: data[t]["evidence_recall@5"]["hybrid"],
    )
    best_h = data[best]["evidence_recall@5"]["hybrid"]
    html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Hybrid (dense + BM25) — 50-article MultiHop-RAG</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1;--dn:#5b6572;--hy:#2563eb;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1;--dn:#9aa5b3;--hy:#6ea0ff;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:34px 22px 70px}}h1{{font-size:22px;margin:0 0 4px}}.cap{{color:var(--muted);font-size:12.5px;margin:6px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}th,td{{border:1px solid var(--line);padding:8px;text-align:center}}
th{{background:var(--card);font-weight:600}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}tr.base td.nm{{font-weight:700}}
.sub{{display:block;font-weight:400;color:var(--muted);font-size:10px}}.dn{{color:var(--dn)}}.hy{{color:var(--hy);font-weight:700}}.ar{{color:var(--muted)}}
.d{{display:block;font-size:11px;margin-top:1px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--hy);border-radius:10px;padding:14px 16px;margin:16px 0}}.leg{{font-size:12px;color:var(--muted);margin-top:8px}}
</style></head><body><div class="wrap">
<h1>Hybrid retrieval (dense + BM25 + RRF) — 50-article MultiHop-RAG</h1>
<p class="cap">Each arm's BM25 runs over the same text its dense index uses (chunk text for A; the generated questions for B/E1/E3), fused with RRF (k={RRF_K}, fixed). n={n} queries · 625 chunks · Octen-0.6B · gpt-5.4-mini. Each cell: <span class="dn">dense</span> → <span class="hy">hybrid</span> (Δ = hybrid−dense); green/red = BM25 helped/hurt; <b>*</b> = Δ 95% bootstrap CI excludes 0.</p>
<table><thead><tr><th>Condition</th>{head}<th>Vectors</th></tr></thead><tbody>{body}</tbody></table>
<div class="verdict"><b>Read:</b> the strong baseline is <b>A hybrid</b> (dense chunks + BM25 chunks) = {a_h:.3f} Recall@5. The best generated arm under hybrid is <b>{best} = {best_h:.3f}</b> — {"above" if best_h > a_h else "still below"} the hybrid-chunks baseline (Δ{best_h - a_h:+.3f}).</div>
<p class="leg">Hybrid sub-study — separate from the pure-vector table. RRF k=60 fixed (untuned). Closed 50-article pilot (n={n}) — not full-corpus MultiHop-RAG.</p>
</div></body></html>"""
    REPORT.write_text(html, encoding="utf-8")
    print(f"[hybrid50] wrote {REPORT}", flush=True)


if __name__ == "__main__":
    run()
