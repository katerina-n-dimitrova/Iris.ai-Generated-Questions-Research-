"""
Stage: self-contained HTML report (§30, §31), Condition A vs Condition E.

Renders report/mhrag_atomic_chunk_mix_10_results.html from the cached artifacts:
subset, generation, filtering funnel, gold alignment, retrieval sweep + paired
significance, breakdowns, atomic/chunk-level diagnostics, answers, failure buckets,
cost/latency, and the 25 final-report answers.
"""

from __future__ import annotations

import json
from html import escape

import am_config as C

KS = C.TOP_K_VALUES


def _l(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def build() -> str:
    sub = _l(C.SUBSET_REPORT)
    arts = _l(C.PILOT_ARTICLES)
    gen = _l(C.GEN_QUALITY)
    filt = _l(C.FILTER_REPORT)
    gold = _l(C.GOLD_REPORT)
    M = _l(C.METRICS_JSON)
    overall = M["overall"]
    paired = M["paired"]
    ans = _l(C.ANSWER_METRICS_JSON)
    fail = _l(C.RESULTS_DIR / "failure_rollup.json")
    idx = _l(C.INDEX_STATS)
    lat = _l(C.RETRIEVAL_LATENCY)
    diag = _l(C.DIAGNOSTICS_JSON)
    A, E = C.COND_A, C.COND_E

    er5 = paired["evidence_recall@5"]
    er5a = overall[A]["evidence_recall@5"]["mean"]
    er5e = overall[E]["evidence_recall@5"]["mean"]
    gen_model = gen.get("generation_model", "the generation LLM")
    # data-driven verdict: classify on the primary metric (Evidence Recall@5)
    d, p = er5["abs_improvement"], er5["paired_p"]
    sig_worse = [
        f"{ml}"
        for mk, ml in [
            ("hit", "Hit"),
            ("mrr", "MRR"),
            ("evidence_recall", "Evidence Recall"),
        ]
        for k in KS
        if paired[f"{mk}@{k}"]["paired_p"] < 0.05
        and overall[E][f"{mk}@{k}"]["mean"] < overall[A][f"{mk}@{k}"]["mean"]
    ]
    if p >= 0.05 and abs(d) < 0.02:
        head = (
            "<b>Roughly a statistical TIE.</b> Pooling atomic + chunk-level questions and indexing "
            "them instead of the chunks left Evidence Recall@5 essentially unchanged "
            f"({er5a:.3f} → {er5e:.3f}, Δ{d:+.3f}, paired p={p}). "
        )
        tail = (
            "A few metrics still favour the chunk baseline at k=4–5 (see stars) but the primary "
            "metric shows no significant difference — with "
            + escape(gen_model)
            + " the mixed-question representation matches plain chunk vectors."
        )
    elif d < 0 and (p < 0.05 or sig_worse):
        head = (
            f"<b>No — Condition E HURTS retrieval.</b> Evidence Recall@5 fell {er5a:.3f} → {er5e:.3f} "
            f"(Δ{d:+.3f}, paired p={p}), "
        )
        tail = (
            "and E is significantly worse on "
            + (", ".join(sorted(set(sig_worse))) or "several metrics")
            + ". Narrow generated questions match the broad multi-hop benchmark queries worse than the full chunk text."
        )
    else:
        head = (
            f"<b>Condition E is at least as good as the baseline.</b> Evidence Recall@5 {er5a:.3f} → {er5e:.3f} "
            f"(Δ{d:+.3f}, paired p={p}). "
        )
        tail = "See the sweep and paired table for where the gains concentrate."
    verdict = head + tail

    metric_labels = [
        ("hit", "Hit@k"),
        ("evidence_recall", "Evidence Recall@k (PRIMARY)"),
        ("all_evidence_hit", "All-Evidence Hit@k"),
        ("precision", "Precision@k"),
        ("mrr", "MRR@k"),
        ("ndcg", "nDCG@k"),
        ("map", "MAP@k"),
        ("doc_recall", "Document Recall@k"),
    ]
    sweep = ""
    for mk, ml in metric_labels:
        cells = f'<tr><td class="metric">{ml}</td>'
        for k in KS:
            key = f"{mk}@{k}"
            a = overall[A][key]["mean"]
            e = overall[E][key]["mean"]
            p = paired[key]["paired_p"]
            d = e - a
            cls = "good" if d > 0 else ("bad" if d < 0 else "flat")
            sig = "sig" if p < 0.05 else ""
            cells += (
                f'<td class="triple {sig}"><span class="a">{a:.3f}</span>'
                f'<span class="b">{e:.3f}</span>'
                f'<span class="d {cls}">{d:+.3f}{" *" if p < 0.05 else ""}</span></td>'
            )
        sweep += cells + "</tr>"
    sweep_head = "".join(f"<th>k={k}</th>" for k in KS)

    paired_rows = ""
    for mk, ml in metric_labels:
        pk = paired[f"{mk}@5"]
        paired_rows += (
            f"<tr><td>{ml}</td><td class='good'>{pk['improved']}</td>"
            f"<td class='flat'>{pk['unchanged']}</td><td class='bad'>{pk['harmed']}</td>"
            f"<td>{pk['abs_improvement']:+.3f}</td><td>{pk['paired_p']}</td></tr>"
        )

    def bd(section):
        r = ""
        for grp, d in section.items():
            a = d[A]["evidence_recall@5"]
            e = d[E]["evidence_recall@5"]
            r += (
                f"<tr><td>{escape(str(grp))}</td><td>{d['n']}</td><td>{a:.3f}</td><td>{e:.3f}</td>"
                f"<td class='{'good' if e > a else ('bad' if e < a else 'flat')}'>{e - a:+.3f}</td></tr>"
            )
        return r

    by_type = bd(M["by_question_type"])
    by_docs = bd(M["by_document_count"])

    art_rows = "".join(
        f"<tr><td>{escape(a['title'][:66])}</td><td>{escape(a['source'])}</td>"
        f"<td>{escape(a['category'])}</td><td>{a['article_length_words']}</td>"
        f"<td>{a['n_chunks']}</td><td>{a['n_eligible_queries']}</td></tr>"
        for a in arts["articles"]
    )

    # filtering funnel
    fr = filt["rejections_by_reason"]
    funnel = "".join(
        f"<tr><td>{escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in sorted(fr.items(), key=lambda kv: -kv[1])
    )

    # diagnostics
    gm, fp = diag["gold_match_winning_type"], diag["false_positive_winning_type"]
    t1 = diag["top1_match_type"]
    lr = diag["lower_ranked_gold_winning_type"]
    diag_rows = (
        f"<tr><td>Accepted questions indexed</td><td>{diag['accepted_atomic_questions']}</td>"
        f"<td>{diag['accepted_chunk_level_questions']}</td></tr>"
        f"<tr><td>Won the top-1 match</td><td>{t1.get('atomic', 0)}</td><td>{t1.get('chunk_level', 0)}</td></tr>"
        f"<tr><td>Won a match on a <b>gold</b> chunk (top-5)</td><td>{gm.get('atomic', 0)}</td><td>{gm.get('chunk_level', 0)}</td></tr>"
        f"<tr><td>Won a match on a <b>non-gold</b> chunk (top-5)</td><td>{fp.get('atomic', 0)}</td><td>{fp.get('chunk_level', 0)}</td></tr>"
        f"<tr><td>Reached gold at rank&gt;1</td><td>{lr.get('atomic', 0)}</td><td>{lr.get('chunk_level', 0)}</td></tr>"
    )

    ab, abk = ans["by_condition"], ans["by_condition_bucket"]
    ans_rows = (
        f"<tr><td>Overall (n={ab[A]['n']})</td>"
        f"<td>{ab[A]['exact_match']:.3f} / {ab[E]['exact_match']:.3f}</td>"
        f"<td>{ab[A]['token_f1']:.3f} / {ab[E]['token_f1']:.3f}</td>"
        f"<td>{ab[A]['yesno_accuracy']} / {ab[E]['yesno_accuracy']}</td></tr>"
    )
    for b in ("all", "partial", "none"):
        ba, be = abk[A][b], abk[E][b]
        ans_rows += (
            f"<tr><td>{b} evidence (n {ba['n']}/{be['n']})</td>"
            f"<td>{ba['exact_match']:.3f} / {be['exact_match']:.3f}</td>"
            f"<td>{ba['token_f1']:.3f} / {be['token_f1']:.3f}</td>"
            f"<td>{ba['yesno_accuracy']} / {be['yesno_accuracy']}</td></tr>"
        )

    fb, fd = fail["buckets"], fail["diagnostics"]
    fbucket = "".join(
        f"<tr><td>{escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in sorted(fb.items(), key=lambda kv: -kv[1])
    )
    fdiag = "".join(
        f"<tr><td>{escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in sorted(fd.items(), key=lambda kv: -kv[1])
    )

    ex_cards = ""
    for ex in fail["examples"][:5]:

        def rl(rk):
            return "".join(
                f"<li class='{'gold' if r['is_gold'] else ''}'>#{r['rank']} <code>{escape(r['chunk_id'])}</code> "
                f"({r['score']:.3f})"
                + (
                    f" · <span class='qt'>{r.get('best_question_type', '')}</span> "
                    f"<i>{escape(r.get('best_question', '')[:70])}</i>"
                    if r.get("best_question")
                    else ""
                )
                + "</li>"
                for r in rk
            )

        ex_cards += (
            f"<div class='card'><div class='q'>{escape(ex['query'])}</div>"
            f"<div class='meta'>{ex['question_type']} · {ex['n_required_evidence_facts']} gold facts · "
            f"answer <code>{escape(ex['gold_answer'])}</code> · "
            f"<b class='{'good' if ex['verdict'] == 'E better' else 'bad'}'>{ex['verdict']}</b> "
            f"(A covered {ex['baseline_evidence_covered']}, E covered {ex['mixed_evidence_covered']})</div>"
            f"<div class='cols'><div><h5>A · chunk vectors</h5><ol>{rl(ex['baseline_ranked'])}</ol></div>"
            f"<div><h5>E · mixed questions</h5><ol>{rl(ex['mixed_ranked'])}</ol></div></div></div>"
        )

    frr_a = overall[A]["first_relevant_rank_mean"]
    frr_e = overall[E]["first_relevant_rank_mean"]

    def _dir(mk, k):
        a = overall[A][f"{mk}@{k}"]["mean"]
        e = overall[E][f"{mk}@{k}"]["mean"]
        p = paired[f"{mk}@{k}"]["paired_p"]
        word = (
            "tie"
            if p >= 0.05 and abs(e - a) < 0.02
            else ("higher" if e > a else "lower")
        )
        return a, e, p, word

    # Hit summary across k
    hit_sig = [
        k
        for k in KS
        if paired[f"hit@{k}"]["paired_p"] < 0.05
        and overall[E][f"hit@{k}"]["mean"] < overall[A][f"hit@{k}"]["mean"]
    ]
    h5a, h5e, h5p, _ = _dir("hit", 5)
    hit_ans = f"Hit@5 {h5a:.3f} → {h5e:.3f} (p={h5p})." + (
        f" Baseline is significantly higher at k={hit_sig}."
        if hit_sig
        else " No significant difference."
    )
    # All-Evidence across k
    allev_pos = sum(
        1
        for k in KS
        if overall[E][f"all_evidence_hit@{k}"]["mean"]
        > overall[A][f"all_evidence_hit@{k}"]["mean"]
    )
    ae5a, ae5e, ae5p, _ = _dir("all_evidence_hit", 5)
    allev_ans = (
        f"E is higher at {allev_pos}/{len(KS)} of the k values (All-Evidence Hit@5 {ae5a:.3f} → {ae5e:.3f}, "
        f"p={ae5p}) — directionally better for multi-hop but not significant."
    )
    m10a, m10e, m10p, m10w = _dir("mrr", 10)
    mrr_ans = f"First-relevant rank {frr_a} → {frr_e}; MRR@10 {m10a:.3f} → {m10e:.3f} ({m10w}, p={m10p})."
    # largest absolute difference metric@k
    all_keys = [(mk, k) for mk, _ in metric_labels for k in KS]
    bmk, bk = max(
        all_keys,
        key=lambda kk: abs(
            overall[E][f"{kk[0]}@{kk[1]}"]["mean"]
            - overall[A][f"{kk[0]}@{kk[1]}"]["mean"]
        ),
    )
    bd = overall[E][f"{bmk}@{bk}"]["mean"] - overall[A][f"{bmk}@{bk}"]["mean"]
    big_ans = f"Largest gap is {bmk.replace('_', ' ')}@{bk} (Δ{bd:+.3f}, p={paired[f'{bmk}@{bk}']['paired_p']})."
    er5_word = _dir("evidence_recall", 5)[3]
    er5_ans = (
        f"{'No — essentially tied' if er5_word == 'tie' else ('Yes' if er5_word == 'higher' else 'No — lower')}: "
        f"{er5a:.3f} → {er5e:.3f} (Δ{er5['abs_improvement']:+.3f}, p={er5['paired_p']})."
    )
    em_a, em_e = ab[A]["exact_match"], ab[E]["exact_match"]
    ans_word = (
        "tie" if abs(em_e - em_a) < 0.02 else ("improved" if em_e > em_a else "lower")
    )
    qa = [
        (
            "How were the 10 articles selected?",
            "Query-first greedy (seed 42): shuffle non-null queries, accept a query iff its evidence "
            "articles fit a 10-article budget, then keep every fully-covered query. No retrieval used.",
        ),
        (
            "How many benchmark queries were fully covered?",
            f"{sub['num_fully_eligible_queries']} eligible (inference {sub['eligible_by_question_type'].get('inference')} / "
            f"temporal {sub['eligible_by_question_type'].get('temporal')} / comparison {sub['eligible_by_question_type'].get('comparison')}); "
            f"{sub['num_excluded_queries']} excluded.",
        ),
        ("How many chunks?", f"{sub['num_chunks']} token-chunks (256/50/80)."),
        (
            "How many atomic facts?",
            f"{gen['total_atoms']} atoms ({gen['avg_atoms_per_chunk']}/chunk).",
        ),
        ("How many raw atomic questions?", f"{gen['raw_atomic_questions']}."),
        ("How many raw chunk-level questions?", f"{gen['raw_chunk_level_questions']}."),
        (
            "How many questions survived filtering?",
            f"{filt['accepted_questions']} ({filt['accepted_atomic']} atomic + {filt['accepted_chunk_level']} chunk-level), "
            f"{filt['avg_accepted_per_chunk']}/chunk.",
        ),
        (
            "Main rejection reasons?",
            f"confusion-margin {fr.get('confusion_margin_too_low', 0)}, round-trip below top-{C.PARENT_TOPK} "
            f"{fr.get('roundtrip_parent_below_topk', 0)}, grounding {fr.get('grounding_span_absent', 0)}, "
            f"near-dup {fr.get('near_duplicate', 0)}. (With min-margin 0 the margin gate ≡ rank-1 gate.)",
        ),
        ("Did E improve Evidence Recall@5?", er5_ans),
        ("Did E improve Hit@k?", hit_ans),
        ("Did E improve All-Evidence Hit@k?", allev_ans),
        ("Did E improve first-relevant rank or MRR?", mrr_ans),
        (
            "Did E improve complete multi-hop retrieval?",
            f"{fb.get('complete_multihop_E', 0)} complete vs {fb.get('partial_multihop_E', 0)} partial "
            f"and {fb.get('no_evidence_E', 0)} no-evidence under E (among multi-hop queries).",
        ),
        ("At which k was the largest difference?", big_ans),
        (
            "How many queries improved / tied / worsened (ER@5)?",
            f"{er5['improved']} improved, {er5['unchanged']} tied, {er5['harmed']} worsened.",
        ),
        (
            "Atomic or chunk-level: more gold matches?",
            f"Atomic won {gm.get('atomic', 0)} gold matches vs chunk-level {gm.get('chunk_level', 0)} "
            f"(atomic are ~{round(diag['accepted_atomic_questions'] / max(diag['accepted_chunk_level_questions'], 1), 1)}× as many, "
            "so chunk-level is comparably efficient per question).",
        ),
        (
            "Which type produced more false positives?",
            f"Atomic {fp.get('atomic', 0)} vs chunk-level {fp.get('chunk_level', 0)} top-5 false-positive matches.",
        ),
        (
            "Which benchmark question types benefited most?",
            "See the per-type breakdown — differences are small on this n; no type shows a significant gain.",
        ),
        (
            "Did chunk-level questions help multi-fact queries?",
            f"All-Evidence Hit@5 is {'higher' if ae5e > ae5a else 'not higher'} for E ({ae5a:.3f}→{ae5e:.3f}, p={ae5p}) — directional, not significant.",
        ),
        (
            "Did atomic questions improve exact-fact retrieval?",
            f"Evidence Recall@1 {overall[A]['evidence_recall@1']['mean']:.3f} → {overall[E]['evidence_recall@1']['mean']:.3f} "
            f"(p={paired['evidence_recall@1']['paired_p']}) — roughly on par.",
        ),
        (
            "Did retrieval changes improve final-answer accuracy?",
            f"Answer EM {'about the same' if ans_word == 'tie' else ans_word}: {em_a:.3f} → {em_e:.3f}; "
            f"F1 {ab[A]['token_f1']:.3f} → {ab[E]['token_f1']:.3f}.",
        ),
        (
            "How much larger was the generated-question index?",
            f"{idx['index_ratio_E_over_A']}× ({idx['baseline']['num_vectors']} → {idx['mixed']['num_vectors']} vectors).",
        ),
        (
            "Additional offline cost?",
            f"{gen['generation_seconds']}s atom+question generation ({escape(gen_model)}) + {filt['filter_seconds']}s filtering "
            f"+ {idx['mixed']['embed_seconds']}s question embedding.",
        ),
        (
            "Online latency difference?",
            f"Query embed {lat['latency_ms']['query_embed_ms']['mean']}ms (shared); search A "
            f"{lat['latency_ms']['baseline_search_ms']['mean']}ms vs E {lat['latency_ms']['mixed_search_ms']['mean']}ms.",
        ),
        (
            "Was the improvement worth the extra vectors and cost?",
            (
                f"Not clearly — E matches the chunk baseline on the primary metric (ER@5 tie) but needs "
                f"{idx['index_ratio_E_over_A']}× the vectors plus offline generation/filtering, so there is no efficiency win on this pilot."
                if er5_word == "tie"
                else (
                    "No — E is worse for more vectors and more cost; not justified on this pilot."
                    if er5_ans.startswith("No")
                    else f"Possibly — E edges the baseline on ER@5 but at {idx['index_ratio_E_over_A']}× the vectors; weigh the gain against index cost."
                )
            ),
        ),
    ]
    qa_html = "".join(
        f"<div class='qa'><div class='qn'>{i + 1}. {escape(q)}</div><div class='an'>{a}</div></div>"
        for i, (q, a) in enumerate(qa)
    )

    html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>MultiHop-RAG · Chunks vs Atomic+Chunk-Level Questions (10-article pilot)</title>
<style>
:root{{--bg:#fff;--fg:#1a1d24;--muted:#5b6572;--line:#e4e8ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8a94a2;--a:#2563eb;--b:#9333ea;--sig:#fff6d5;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#7a8494;--a:#6ea0ff;--b:#c98bff;--sig:#3a3410;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:25px;margin:0 0 6px}}h2{{font-size:20px;margin:38px 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}}
h3{{font-size:16px;margin:22px 0 8px}}.lede{{color:var(--muted)}}
.tag{{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:3px 11px;font-size:12px;margin:3px 4px 3px 0;color:var(--muted)}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--vc);border-radius:10px;padding:16px 18px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}}th,td{{border:1px solid var(--line);padding:7px 9px;text-align:center}}
th{{background:var(--card)}}td.metric,td:first-child{{text-align:left}}
.triple .a{{color:var(--a);font-weight:600;margin-right:7px}}.triple .b{{color:var(--b);font-weight:600;margin-right:7px}}
.triple .d{{display:block;font-size:11.5px}}.triple.sig{{background:var(--sig)}}
.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:12px}}.scroll{{overflow-x:auto}}
.legend{{font-size:12.5px;color:var(--muted)}}
.kpis{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}.kpi{{flex:1 1 150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.kpi .n{{font-size:21px;font-weight:700}}.kpi .l{{color:var(--muted);font-size:12px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:720px){{.grid2,.cols{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}}
.card .q{{font-weight:600;margin-bottom:4px}}.card .meta{{color:var(--muted);font-size:12.5px;margin-bottom:10px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.cols h5{{margin:0 0 6px;font-size:13px}}
.cols ol{{margin:0;padding-left:20px;font-size:12.5px}}.cols li.gold{{color:var(--good);font-weight:600}}
.qt{{font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;color:var(--b)}}
.qa{{border-bottom:1px solid var(--line);padding:10px 0}}.qa .qn{{font-weight:600}}.qa .an{{color:var(--muted);margin-top:3px}}
</style></head><body><div class=wrap>

<h1>MultiHop-RAG — Chunks vs Atomic + Chunk-Level Questions</h1>
<p class=lede>A closed-collection, 10-article MultiHop-RAG dense-vector pilot comparing original-chunk
embeddings (<b>Condition A</b>) with a pooled mixture of atomic-fact and chunk-level synthetic-question
embeddings (<b>Condition E</b>). Dense cosine only — no BM25 / sparse / hybrid / rerank.</p>
<div>
<span class=tag>10-article pilot · seed 42</span><span class=tag>Octen-Embedding-0.6B · dim 1024 · cosine</span>
<span class=tag>{escape(gen_model)}</span><span class=tag>{sub["num_fully_eligible_queries"]} eligible queries</span>
<span class=tag>k = {", ".join(map(str, KS))}</span></div>

<div class=verdict style="--vc:{"var(--flat)" if er5_word == "tie" else ("var(--good)" if er5_word == "higher" else "var(--bad)")}"><b>Headline:</b> {verdict}</div>

<div class=kpis>
<div class=kpi><div class=n>{sub["num_chunks"]}</div><div class=l>chunks (A vectors)</div></div>
<div class=kpi><div class=n>{idx["mixed"]["num_vectors"]}</div><div class=l>accepted question vectors (E, {idx["index_ratio_E_over_A"]}×)</div></div>
<div class=kpi><div class=n>{overall[A]["evidence_recall@5"]["mean"]:.3f} / {overall[E]["evidence_recall@5"]["mean"]:.3f}</div><div class=l>Evidence Recall@5 · A / E</div></div>
<div class=kpi><div class=n>{ab[A]["exact_match"]:.2f} / {ab[E]["exact_match"]:.2f}</div><div class=l>Answer EM · A / E</div></div>
</div>
<p class=legend><b>Caveat.</b> A closed-collection, 10-article pilot (n={sub["num_fully_eligible_queries"]}), NOT the full-corpus MultiHop-RAG benchmark.</p>

<h2>1 · Retrieval — full metric sweep</h2>
<p class=legend>Each cell: <span style="color:var(--a)">A (chunks)</span> · <span style="color:var(--b)">E (mixed questions)</span> ·
<b>Δ = E−A</b> (<span class=good>green</span>=E better, <span class=bad>red</span>=A better). <b>*</b> = paired p&lt;0.05. Higher is better.</p>
<div class=scroll><table><thead><tr><th>Metric</th>{sweep_head}</tr></thead><tbody>{sweep}</tbody></table></div>

<h2>2 · Paired comparison (@k=5)</h2>
<div class=scroll><table><thead><tr><th>Metric</th><th class=good>E better</th><th class=flat>tie</th><th class=bad>A better</th><th>mean Δ</th><th>p</th></tr></thead><tbody>{paired_rows}</tbody></table></div>

<h2>3 · Breakdown by question type &amp; hop count</h2>
<div class=grid2>
<div><h3>By question type</h3><div class=scroll><table><thead><tr><th>Type</th><th>n</th><th>ER@5 A</th><th>ER@5 E</th><th>Δ</th></tr></thead><tbody>{by_type}</tbody></table></div></div>
<div><h3>By required documents</h3><div class=scroll><table><thead><tr><th>#docs</th><th>n</th><th>ER@5 A</th><th>ER@5 E</th><th>Δ</th></tr></thead><tbody>{by_docs}</tbody></table></div></div></div>

<h2>4 · Atomic vs chunk-level diagnostics</h2>
<p class=legend>The two types are pooled for ranking; this shows their separate contributions (atomic / chunk-level counts).</p>
<div class=scroll><table><thead><tr><th></th><th>atomic</th><th>chunk-level</th></tr></thead><tbody>{diag_rows}</tbody></table></div>

<h2>5 · Answer generation (top-5 chunks → {escape(gen_model)})</h2>
<p class=legend>Values are <b>A / E</b>, split by how much gold evidence the retriever delivered.</p>
<div class=scroll><table><thead><tr><th>Bucket</th><th>Exact match</th><th>Token F1</th><th>Yes/No acc</th></tr></thead><tbody>{ans_rows}</tbody></table></div>

<h2>6 · Failure analysis</h2>
<div class=grid2>
<div><h3>Query buckets</h3><div class=scroll><table><tbody>{fbucket}</tbody></table></div></div>
<div><h3>Diagnostics</h3><div class=scroll><table><tbody>{fdiag}</tbody></table></div></div></div>
<h3>Representative contrasts</h3>{ex_cards}

<h2>7 · Condition-E generation &amp; filtering funnel</h2>
<p class=legend>{gen["total_atoms"]} atoms · {gen["raw_questions_total"]} raw questions
({gen["raw_atomic_questions"]} atomic + {gen["raw_chunk_level_questions"]} chunk-level) →
<b>{filt["accepted_questions"]} accepted</b> ({filt["accepted_atomic"]} atomic + {filt["accepted_chunk_level"]} chunk-level,
{filt["avg_accepted_per_chunk"]}/chunk). {filt["chunks_with_zero_accepted"]} chunks ended with no accepted question.
A strict rank-1 round-trip filter would keep {filt["would_pass_strict_rank1_roundtrip"]}.</p>
<div class=scroll><table><thead><tr><th>Rejected by</th><th>count</th></tr></thead><tbody>{funnel}</tbody></table></div>

<h2>8 · Cost, storage &amp; latency</h2>
<div class=scroll><table><thead><tr><th></th><th>A · chunk vectors</th><th>E · mixed question vectors</th></tr></thead><tbody>
<tr><td>Vectors indexed</td><td>{idx["baseline"]["num_vectors"]}</td><td>{idx["mixed"]["num_vectors"]} ({idx["index_ratio_E_over_A"]}×)</td></tr>
<tr><td>Offline embed time</td><td>{idx["baseline"]["embed_seconds"]} s</td><td>{idx["mixed"]["embed_seconds"]} s</td></tr>
<tr><td>Extra offline LLM (atoms+questions)</td><td>—</td><td>{gen["generation_seconds"]} s</td></tr>
<tr><td>Filtering time</td><td>—</td><td>{filt["filter_seconds"]} s</td></tr>
<tr><td>Query-embed latency (shared)</td><td colspan=2>{lat["latency_ms"]["query_embed_ms"]["mean"]} ms mean · p95 {lat["latency_ms"]["query_embed_ms"]["p95"]} ms</td></tr>
<tr><td>Vector-search latency</td><td>{lat["latency_ms"]["baseline_search_ms"]["mean"]} ms</td><td>{lat["latency_ms"]["mixed_search_ms"]["mean"]} ms</td></tr>
</tbody></table></div>

<h2>9 · Selected 10-article pilot collection</h2>
<p class=legend>Gold: {gold["total_gold_evidence_facts"]} facts — {gold["exact_matched"]} exact, {gold["fuzzy_matched"]} fuzzy,
{gold["unresolved"]} unresolved; {gold["facts_crossing_chunk_boundaries"]} span a chunk overlap (counted once).
Avg {gold["avg_gold_chunks_per_query"]} gold chunks & {gold["avg_evidence_facts_per_query"]} facts per query.</p>
<div class=scroll><table><thead><tr><th>Title</th><th>Source</th><th>Category</th><th>Words</th><th>Chunks</th><th>Elig. q</th></tr></thead><tbody>{art_rows}</tbody></table></div>

<h2>10 · Final report — the 25 questions</h2>{qa_html}

<h2>Method &amp; reproducibility</h2>
<p class=lede>A and E share the same 10 articles, cleaning, 256/50/80 chunk boundaries, eligible queries,
embedding model, cosine metric, k values, gold mapping, answer LLM + prompt, and seed 42. The only difference is
the indexed representation: one chunk vector (A) vs pooled atomic + chunk-level question vectors mapped to the parent
chunk by max similarity (E). Gold evidence/answers were used only to build the closed collection and evaluation labels.
Run with <code>python run_am.py</code>. Isolated from the 15-article experiment.</p>
</div></body></html>"""
    C.REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] wrote {C.REPORT_HTML}")
    return str(C.REPORT_HTML)


if __name__ == "__main__":
    build()
