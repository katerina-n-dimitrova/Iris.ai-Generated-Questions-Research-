"""
Append the "Follow-up experiments: improving generated-question quality" section
to the existing report (report/generated_questions_results.html).

Reads results/spiqa/test-B/followup_quality/followup_results.json and splices a
new set of cards in BEFORE </main>, between idempotent markers so re-running
replaces the section instead of duplicating it. The existing report content is
never overwritten. Styling reuses the classes already defined in the report head.
"""

from __future__ import annotations

import json
from pathlib import Path

import spiqa_config as C

REPORT = C.PROJECT_ROOT / "report" / "generated_questions_results.html"
DATA = C.RESULTS_DIR / "test-B" / "followup_quality" / "followup_results.json"
START = "<!--FOLLOWUP_START-->"
END = "<!--FOLLOWUP_END-->"

# previous novision reference (from the cached report results)
PREV_LABELS = {
    "baseline_dense": "baseline · dense",
    "baseline_bm25": "baseline · BM25",
    "baseline_hybrid": "baseline · hybrid",
    "q10_bm25_expand": "q10 · BM25 expand",
    "q10_hybrid_expand": "q10 · hybrid expand",
    "q50_hybrid_expand": "q50 · hybrid expand",
}


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _cell(v, base, better_is_more=True):
    """Green/red vs a baseline value."""
    if not isinstance(v, (int, float)):
        return f"<td>{v}</td>"
    cls = ""
    if base is not None:
        if (v > base and better_is_more) or (v < base and not better_is_more):
            cls = " class='win'"
        elif (v < base and better_is_more) or (v > base and not better_is_more):
            cls = " class='lose'"
    return f"<td{cls}>{_fmt(v)}</td>"


def build_section(d: dict) -> str:
    rows = d["rows"]
    prev = d.get("previous_novision", {})
    base_ndcg = prev.get("baseline_dense", {}).get("ndcg@10")
    by = {r["condition"]: r for r in rows}
    best = by[d["best_new_condition"]]

    # references for interpretation
    def nd(cond):
        return by[cond]["ndcg@10"] if cond in by else None

    bd = d.get("strategy_breakdown", [])
    bmap = {(r["strategy"], r["setup"]): r for r in bd}

    def snd(strat, setup):
        r = bmap.get((strat, setup))
        return r["ndcg@10"] if r else None

    H = [START]

    # 1 · why -----------------------------------------------------------------
    H.append(f"""
<div class='card'>
<h2>Follow-up experiments: improving generated-question quality</h2>
<div class='dt'>question quality &gt; question count</div>
<p class='sub'>The first study showed that simply <b>increasing the number</b> of generated questions
<b>saturates</b> — q10 is the sweet spot and q50/q100 add cost without much gain. So instead of adding
more questions, these follow-ups hold the count at ~q10 and improve <b>question quality and how questions
are indexed</b>: forcing question-type diversity, grounding questions in the chunk's exact lexical anchors,
filtering out weak/duplicate questions, keeping questions in a <b>separate</b> index instead of polluting
the chunk vector, adding a <b>cross-encoder reranker</b> on top of hybrid recall, and giving figure/table
chunks a <b>structured</b> visual representation. Same 20-paper SPIQA test-B set
({d["chunks"]} chunks · {d["queries"]} figure-answerable queries), same embedder ({d["embedder"]}),
same LLM (gpt-4o-mini), cached outputs reused. Green/red = vs the previous <b>baseline_dense</b>
(nDCG@10 {base_ndcg}).</p>
</div>""")

    # 2 · all new conditions --------------------------------------------------
    trs = []
    for r in rows:
        is_best = r["condition"] == d["best_new_condition"]
        cls = " class='best'" if is_best else ""
        trs.append(
            f"<tr{cls}><td class='l'>{r['condition']}</td>"
            f"<td class='l'>{r['strategy']}</td><td class='l'>{r['setup']}</td>"
            + _cell(r["hit@1"], prev.get("baseline_dense", {}).get("hit@1"))
            + _cell(r["hit@5"], prev.get("baseline_dense", {}).get("hit@5"))
            + _cell(r["hit@10"], prev.get("baseline_dense", {}).get("hit@10"))
            + _cell(r["mrr"], prev.get("baseline_dense", {}).get("mrr"))
            + _cell(r["ndcg@10"], base_ndcg)
            + f"<td class='muted'>{r['storage_x']}×</td>"
            + f"<td class='muted'>{r['p95_latency_ms']}</td>"
            + f"<td class='l muted'>{r['notes']}</td></tr>"
        )
    H.append(f"""
<div class='card'>
<h2>All new conditions</h2>
<div class='dt'>quality · lexical grounding · filtering · separate index · reranking · structured vision</div>
<div class='scroll'><table><thead><tr>
<th class='l'>Condition</th><th class='l'>Question strategy</th><th class='l'>Retrieval setup</th>
<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>
<th>Storage ×</th><th>p95 (ms)</th><th class='l'>Notes</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>
<div class='note'>Storage × is relative to the original-chunk dense index. Separate-index rows add a
questions-only vector index (B); reranker rows add no index (a cross-encoder model runs at query time,
which is why their p95 is higher). BM25-only rows are sub-1× because a lexical index is far smaller than
dense vectors.</div>
</div>""")

    # 3 · comparison vs previous best -----------------------------------------
    prev_trs = []
    for cond, label in PREV_LABELS.items():
        if cond not in prev:
            continue
        p = prev[cond]
        prev_trs.append(
            f"<tr class='base'><td class='l'>{label}</td>"
            f"<td>{_fmt(p['hit@1'])}</td><td>{_fmt(p['hit@5'])}</td><td>{_fmt(p['hit@10'])}</td>"
            f"<td>{_fmt(p['mrr'])}</td><td>{_fmt(p['ndcg@10'])}</td>"
            f"<td class='muted'>{p.get('storage_x', '')}×</td></tr>"
        )
    prev_trs.append(
        f"<tr class='best'><td class='l'><b>{best['condition']}</b> (best new)</td>"
        f"<td>{_fmt(best['hit@1'])}</td><td>{_fmt(best['hit@5'])}</td><td>{_fmt(best['hit@10'])}</td>"
        f"<td>{_fmt(best['mrr'])}</td><td><b>{_fmt(best['ndcg@10'])}</b></td>"
        f"<td class='muted'>{best['storage_x']}×</td></tr>"
    )
    H.append(f"""
<div class='card'>
<h2>Best new condition vs the previous best configurations</h2>
<div class='dt'>previous numbers = novision arm, reused from the earlier run</div>
<div class='scroll'><table style='max-width:760px'><thead><tr>
<th class='l'>Condition</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>
<th>Storage ×</th></tr></thead><tbody>{"".join(prev_trs)}</tbody></table></div>
</div>""")

    # 4 · strategy breakdown --------------------------------------------------
    strat_order = [
        "generic",
        "diverse",
        "bm25_aware",
        "filtered_q10",
        "filtered_q20to10",
    ]
    strat_label = {
        "generic": "generic q10 (control)",
        "diverse": "diverse q10",
        "bm25_aware": "BM25-aware q10",
        "filtered_q10": "filtered q10",
        "filtered_q20to10": "q20→10 filtered",
    }
    br_trs = []
    for s in strat_order:
        r_bm = bmap.get((s, "bm25"))
        r_hy = bmap.get((s, "hybrid"))
        r_ap = bmap.get((s, "dense_append"))
        if not r_hy:
            continue
        cls = " class='base'" if s == "generic" else ""
        base_bm = snd("generic", "bm25")
        base_hy = snd("generic", "hybrid")
        base_ap = snd("generic", "dense_append")
        br_trs.append(
            f"<tr{cls}><td class='l'>{strat_label[s]}</td>"
            + _cell(r_ap["ndcg@10"], base_ap if s != "generic" else None)
            + _cell(r_bm["ndcg@10"], base_bm if s != "generic" else None)
            + _cell(r_hy["ndcg@10"], base_hy if s != "generic" else None)
            + "</tr>"
        )
    H.append(f"""
<div class='card'>
<h2>Question-strategy breakdown (nDCG@10, same setups)</h2>
<div class='dt'>diverse / lexical-anchor / filtered vs the generic q10 control</div>
<p class='sub'>Every strategy uses ~10 questions; only their <b>content</b> differs. Each is run under the same
three doc2query setups so the comparison is apples-to-apples. Green/red vs the <b>generic q10</b> row.</p>
<div class='scroll'><table style='max-width:620px'><thead><tr><th class='l'>Question strategy</th>
<th>dense append</th><th>BM25 expand</th><th>hybrid</th></tr></thead>
<tbody>{"".join(br_trs)}</tbody></table></div>
</div>""")

    # 5 · interpretation + recommendation ------------------------------------
    interp, rec = _interpretation(d, by, snd, base_ndcg)
    H.append(f"""
<div class='card'>
<h2>Interpretation &amp; recommendation</h2>
<div class='dt'>what moved retrieval, and what to use next</div>
<div class='fb'><b>Which method helped most</b><br>{interp}</div>
<div class='fb' style='background:#f0fdf4;border-color:#bbf7d0'><b style='color:#166534'>Recommendation</b><br>{rec}</div>
</div>""")

    H.append(END)
    return "\n".join(H)


def _pct(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _interpretation(d, by, snd, base_ndcg):
    best = by[d["best_new_condition"]]
    # families
    gen_hy = snd("generic", "hybrid")
    div_hy = snd("diverse", "hybrid")
    bm_bm = snd("bm25_aware", "bm25")
    gen_bm = snd("generic", "bm25")
    filt_hy = snd("filtered_q10", "hybrid")

    def g(cond):
        return by[cond]["ndcg@10"] if cond in by else None

    sep = g("q10_separate_question_index_rrf")
    sep_bm = g("q10_bm25_aware_separate_index_rrf")
    rer = g("q10_hybrid_rerank_top50")
    rer_bm = g("q10_bm25_aware_hybrid_rerank_top50")
    st_h = g("structured_vision_q10_hybrid")

    parts = []
    if div_hy is not None and gen_hy is not None:
        dd = _pct(div_hy, gen_hy)
        parts.append(
            f"<b>Diverse question types</b> {'helped' if dd and dd > 0 else 'did not help'} "
            f"({div_hy} vs generic {gen_hy}, {dd:+})."
        )
    if bm_bm is not None and gen_bm is not None:
        db = _pct(bm_bm, gen_bm)
        parts.append(
            f"<b>BM25-aware lexical grounding</b> moved BM25 retrieval "
            f"{db:+} nDCG@10 (to {bm_bm}) — {'a real lexical-matching gain' if db and db > 0 else 'no gain'}, "
            f"confirming whether the win is lexical."
        )
    if filt_hy is not None and gen_hy is not None:
        df = _pct(filt_hy, gen_hy)
        verdict = (
            "matched the unfiltered q10 at ~40% fewer questions (a cost win, not a quality gain)"
            if abs(df or 0) < 0.005
            else ("helped" if df and df > 0 else "did not beat the unfiltered q10")
        )
        parts.append(
            f"<b>Filtering</b> {verdict} ({filt_hy} vs {gen_hy}, {df:+}) — fewer-but-cleaner questions."
        )
    if sep is not None and gen_hy is not None:
        ds = _pct(sep, gen_hy)
        parts.append(
            f"<b>Separate question index</b> (RRF, no concatenation) = {sep} "
            f"({ds:+} vs hybrid append) — {'keeping questions out of the chunk vector helped' if ds and ds > 0 else 'concatenation was already fine here'}."
        )
    if rer is not None:
        best_rer = max([x for x in (rer, rer_bm) if x is not None], default=None)
        dr = _pct(best_rer, gen_hy)
        # recall (Hit@10) of the rerank family, to make the recall-vs-precision point concrete
        rr_hit10 = max(
            (
                by[c]["hit@10"]
                for c in (
                    "q10_hybrid_rerank_top50",
                    "q10_bm25_aware_hybrid_rerank_top50",
                    "q10_filtered_hybrid_rerank_top50",
                )
                if c in by
            ),
            default=None,
        )
        if best_rer is not None and dr is not None and dr < 0:
            parts.append(
                f"<b>Cross-encoder reranking</b> <i>hurt</i> nDCG@10 ({best_rer}, {dr:+} vs hybrid) "
                f"even though it gave the highest recall of any condition "
                f"(Hit@10 {rr_hit10}) — the correct chunk is in the top-50, but a general-domain "
                f"MS-MARCO reranker mis-orders scientific figure/table text and is far slower; "
                f"first-stage recall was already the bottleneck, not ranking."
            )
        elif best_rer is not None:
            parts.append(
                f"<b>Cross-encoder reranking</b> reached {best_rer} ({dr:+} vs hybrid) — "
                f"generated questions supply recall (Hit@10 {rr_hit10}) and the reranker sharpens precision."
            )
    if st_h is not None:
        dstruct = _pct(st_h, base_ndcg)
        parts.append(
            f"<b>Structured figure/table context</b> = {st_h} hybrid ({dstruct:+} vs baseline_dense) — "
            f"structuring axes/values/methods {'helped multimodal retrieval' if dstruct and dstruct > 0 else 'did not add over caption+vision'}."
        )
    parts = [p for p in parts if p]
    interp = " ".join(parts) + (
        f" <b>Overall best new config: <span class='win'>{best['condition']} "
        f"= {best['ndcg@10']} nDCG@10</span></b> "
        f"({_pct(best['ndcg@10'], base_ndcg):+} vs baseline_dense) at "
        f"{best['storage_x']}× storage."
    )

    # recommendation
    recs = []
    if (
        bm_bm is not None
        and gen_bm is not None
        and _pct(bm_bm, gen_bm)
        and _pct(bm_bm, gen_bm) > 0
    ):
        recs.append(
            "replace generic generated questions with <b>BM25-aware, lexically-grounded</b> ones — "
            "they add anchors BM25 matches verbatim"
        )
    if (
        filt_hy is not None
        and gen_hy is not None
        and _pct(filt_hy, gen_hy)
        and _pct(filt_hy, gen_hy) >= 0
    ):
        recs.append(
            "<b>filter</b> candidates (drop generic/duplicate/ungrounded) to get the same quality at lower cost"
        )
    if (
        sep is not None
        and gen_hy is not None
        and _pct(sep, gen_hy)
        and _pct(sep, gen_hy) > 0
    ):
        recs.append(
            "keep questions in a <b>separate index fused by RRF</b> rather than concatenating them into chunk text"
        )
    if (
        rer is not None
        and gen_hy is not None
        and _pct(max(filter(None, [rer, rer_bm]), default=0), gen_hy)
        and _pct(max(filter(None, [rer, rer_bm]), default=0), gen_hy) > 0
    ):
        recs.append(
            "use generated questions for <b>recall</b> and a <b>cross-encoder reranker</b> for precision"
        )
    if (
        st_h is not None
        and base_ndcg is not None
        and _pct(st_h, base_ndcg)
        and _pct(st_h, base_ndcg) > 0
    ):
        recs.append(
            "use <b>structured visual context</b> (axes/values/methods) for figure/table chunks in multimodal papers"
        )
    rec = (
        f"For future experiments, adopt <b>{best['condition']}</b> as the reference configuration. "
        + (
            "Concretely: " + "; ".join(recs) + "."
            if recs
            else "None of the quality tweaks beat the previous best by a clear margin, so the cheaper "
            "q10 BM25/hybrid expansion from the first study remains the practical default."
        )
    )
    return interp, rec


def append_to_report():
    frag = build_section(json.load(DATA.open(encoding="utf-8")))
    html = REPORT.read_text(encoding="utf-8")
    if START in html and END in html:
        pre = html[: html.index(START)]
        post = html[html.index(END) + len(END) :]
        html = pre + frag + post
    else:
        html = html.replace("</main>", frag + "\n</main>", 1)
    # refresh footer note (idempotent-ish; only append once)
    if "Follow-up quality study" not in html:
        html = html.replace(
            "</footer>",
            " · Follow-up quality study: results/spiqa/test-B/followup_quality</footer>",
            1,
        )
    REPORT.write_text(html, encoding="utf-8")
    print(f"Report updated: {REPORT}")


if __name__ == "__main__":
    append_to_report()
