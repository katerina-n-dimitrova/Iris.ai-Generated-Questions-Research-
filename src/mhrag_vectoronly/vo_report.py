"""
Stage: assemble the self-contained HTML results report (§25).

Reads every cached json/csv artifact and renders
``report/multihoprag_vectoronly_results.html`` — a theme-aware, dependency-free
page comparing Condition A (original-chunk vectors) with Condition B (10
generated-questions-per-chunk vectors) under DENSE-vector-only retrieval on the
closed 15-article MultiHop-RAG pilot.
"""

from __future__ import annotations

import json
from html import escape
from typing import List

import vo_config as C

KS = C.TOP_K_VALUES


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _delta_cell(d: float, p: float, good_positive=True) -> str:
    sig = p is not None and p < 0.05
    cls = "flat"
    if abs(d) > 1e-9:
        winner_pos = d > 0
        cls = "good" if (winner_pos == good_positive) else "bad"
    star = " *" if sig else ""
    return f'<td class="{cls}">{d:+.3f}{star}</td>'


# --------------------------------------------------------------------------- #
def build() -> str:
    subset = _load(C.PILOT_REPORT)
    articles = _load(C.PILOT_ARTICLES)
    genq = _load(C.GENQ_QUALITY)
    gold = _load(C.GOLD_REPORT)
    metrics = _load(C.METRICS_JSON)
    paired = metrics["paired"]
    overall = metrics["overall"]
    answers = _load(C.ANSWER_METRICS_JSON)
    failure = _load(C.RESULTS_DIR / "failure_rollup.json")
    idx = _load(C.RESULTS_DIR / "index_stats.json")
    lat = _load(C.RESULTS_DIR / "retrieval_latency.json")
    A, B = C.COND_A, C.COND_B

    # ---- headline verdict ------------------------------------------------ #
    er5 = paired["evidence_recall@5"]
    hit5 = paired["hit@5"]
    verdict = (
        "No — under dense-vector-only retrieval, indexing 10 synthetic "
        "questions per chunk does <b>not</b> improve retrieval over "
        "indexing the original chunks on this pilot. The two conditions "
        "are statistically tied on the primary metric (Evidence Recall@5), "
        "with a small, occasionally-significant deficit for generated "
        "questions at k=3–5."
    )

    # ---- metric sweep table --------------------------------------------- #
    metric_labels = [
        ("hit", "Hit@k", "found any gold evidence chunk"),
        (
            "evidence_recall",
            "Evidence Recall@k",
            "fraction of gold facts covered (PRIMARY)",
        ),
        (
            "all_evidence_hit",
            "All-Evidence Hit@k",
            "ALL gold facts covered (multi-hop)",
        ),
        ("precision", "Precision@k", "fraction of retrieved chunks relevant"),
        ("mrr", "MRR@k", "rank of first relevant chunk"),
        ("ndcg", "nDCG@k", "relevant chunks near top"),
        ("map", "MAP@k", "average precision"),
        ("doc_recall", "Document Recall@k", "required source articles covered"),
    ]
    sweep_rows = []
    for mkey, mlabel, mdesc in metric_labels:
        cells = [
            f'<tr><td class="metric">{mlabel}<br><span class="sub">{mdesc}</span></td>'
        ]
        for k in KS:
            key = f"{mkey}@{k}"
            a = overall[A][key]["mean"]
            b = overall[B][key]["mean"]
            p = paired[key]["paired_p"]
            d = b - a
            better = "A" if a > b else ("B" if b > a else "=")
            sig = "sig" if p < 0.05 else ""
            cells.append(
                f'<td class="triple {sig}"><span class="a">{a:.3f}</span>'
                f'<span class="b">{b:.3f}</span>'
                f'<span class="d {"good" if (d > 0) else ("bad" if d < 0 else "flat")}">'
                f"{d:+.3f}{' *' if p < 0.05 else ''}</span></td>"
            )
        cells.append("</tr>")
        sweep_rows.append("".join(cells))
    sweep_head = "".join(f"<th>k={k}</th>" for k in KS)

    # ---- paired improved/unchanged/harmed (@5) --------------------------- #
    paired_rows = []
    for mkey, mlabel, _ in metric_labels:
        pk = paired[f"{mkey}@5"]
        paired_rows.append(
            f"<tr><td>{mlabel}</td><td class='good'>{pk['improved']}</td>"
            f"<td class='flat'>{pk['unchanged']}</td><td class='bad'>{pk['harmed']}</td>"
            f"<td>{pk['abs_improvement']:+.3f}</td><td>{_fmt(pk['paired_p'])}</td></tr>"
        )

    # ---- breakdown by type / doc count ----------------------------------- #
    def breakdown_table(section, label):
        rows = []
        for grp, d in section.items():
            a = d[A]["evidence_recall@5"]
            b = d[B]["evidence_recall@5"]
            a1 = d[A]["hit@1"]
            b1 = d[B]["hit@1"]
            ah = d[A]["all_evidence_hit@5"]
            bh = d[B]["all_evidence_hit@5"]
            rows.append(
                f"<tr><td>{escape(str(grp))}</td><td>{d['n']}</td>"
                f"<td>{a:.3f}</td><td>{b:.3f}</td>"
                f"<td class='{'good' if b > a else ('bad' if b < a else 'flat')}'>{b - a:+.3f}</td>"
                f"<td>{a1:.3f} / {b1:.3f}</td><td>{ah:.3f} / {bh:.3f}</td></tr>"
            )
        return "".join(rows)

    by_type = breakdown_table(metrics["by_question_type"], "Question type")
    by_docs = breakdown_table(metrics["by_document_count"], "Required docs")

    # ---- articles table -------------------------------------------------- #
    art_rows = "".join(
        f"<tr><td>{escape(a['title'][:70])}</td><td>{escape(a['source'])}</td>"
        f"<td>{escape(a['category'])}</td><td>{a['article_length_words']}</td>"
        f"<td>{a['n_chunks']}</td><td>{a['n_eligible_queries']}</td></tr>"
        for a in articles["articles"]
    )

    # ---- answer buckets -------------------------------------------------- #
    ab = answers["by_condition"]
    abk = answers["by_condition_bucket"]
    ans_rows = (
        f"<tr><td>Overall (n=84)</td>"
        f"<td>{ab[A]['exact_match']:.3f} / {ab[B]['exact_match']:.3f}</td>"
        f"<td>{ab[A]['token_f1']:.3f} / {ab[B]['token_f1']:.3f}</td>"
        f"<td>{_fmt(ab[A]['yesno_accuracy'])} / {_fmt(ab[B]['yesno_accuracy'])}</td></tr>"
    )
    for bucket in ("all", "partial", "none"):
        ba, bb = abk[A][bucket], abk[B][bucket]
        ans_rows += (
            f"<tr><td>{bucket} evidence retrieved (n {ba['n']}/{bb['n']})</td>"
            f"<td>{ba['exact_match']:.3f} / {bb['exact_match']:.3f}</td>"
            f"<td>{ba['token_f1']:.3f} / {bb['token_f1']:.3f}</td>"
            f"<td>{_fmt(ba['yesno_accuracy'])} / {_fmt(bb['yesno_accuracy'])}</td></tr>"
        )

    # ---- failure buckets ------------------------------------------------- #
    fb = failure["buckets"]
    fd = failure["diagnostics"]
    fbucket_rows = "".join(
        f"<tr><td>{escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in sorted(fb.items(), key=lambda kv: -kv[1])
    )
    fdiag_rows = "".join(
        f"<tr><td>{escape(k.replace('_', ' '))}</td><td>{v}</td></tr>"
        for k, v in sorted(fd.items(), key=lambda kv: -kv[1])
    )

    # ---- example cards --------------------------------------------------- #
    ex_cards = []
    for ex in failure["examples"][:5]:

        def rank_list(rk):
            return "".join(
                f"<li class='{'gold' if r['is_gold'] else ''}'>#{r['rank']} "
                f"<code>{escape(r['chunk_id'])}</code> ({r['score']:.3f})"
                + (
                    f" — <i>{escape(r.get('best_question', '')[:80])}</i>"
                    if r.get("best_question")
                    else ""
                )
                + "</li>"
                for r in rk
            )

        ex_cards.append(f"""
        <div class="card">
          <div class="q">{escape(ex["query"])}</div>
          <div class="meta">{ex["question_type"]} · {ex["n_required_evidence_facts"]} gold facts ·
            answer <code>{escape(ex["gold_answer"])}</code> ·
            <b class="{"good" if "generated better" in ex["verdict"] else "bad"}">{ex["verdict"]}</b>
            (A covered {ex["baseline_evidence_covered"]}, B covered {ex["generated_evidence_covered"]})</div>
          <div class="cols">
            <div><h5>A · chunk vectors</h5><ol>{rank_list(ex["baseline_ranked"])}</ol></div>
            <div><h5>B · question vectors</h5><ol>{rank_list(ex["generated_ranked"])}</ol></div>
          </div>
        </div>""")

    # ---- final report Q&A ------------------------------------------------ #
    frr_a = overall[A]["first_relevant_rank_mean"]
    frr_b = overall[B]["first_relevant_rank_mean"]
    cer_a = overall[A]["complete_evidence_rank_mean"]
    cer_b = overall[B]["complete_evidence_rank_mean"]
    cer_rate_a = overall[A]["complete_evidence_achieved_rate"]
    cer_rate_b = overall[B]["complete_evidence_achieved_rate"]
    qa = [
        (
            "How were the 15 articles selected?",
            "Query-first greedy selection (seed 42): shuffle non-null queries, accept a "
            "query iff its evidence articles fit a 15-article budget, then keep every query "
            "fully covered by the resulting closed set. Retrieval results were never used.",
        ),
        (
            "How many benchmark queries were fully covered?",
            f"{subset['num_fully_eligible_queries']} eligible "
            f"(inference {metrics['by_question_type'].get('inference', {}).get('n', '?')} / "
            f"temporal {metrics['by_question_type'].get('temporal', {}).get('n', '?')} / "
            f"comparison {metrics['by_question_type'].get('comparison', {}).get('n', '?')}); "
            f"{subset['num_excluded_queries']} excluded (needed an article outside the 15).",
        ),
        (
            "How many chunks / synthetic questions?",
            f"{subset['num_chunks']} token-chunks (256/50/80), and "
            f"{genq['actual_valid_questions']} synthetic questions (exactly 10/chunk).",
        ),
        (
            "Did generated-question retrieval outperform chunk-vector retrieval?",
            "No. Tied on Evidence Recall@5 (Δ=%+.3f, p=%.2f); baseline slightly ahead at k=3–5."
            % (er5["abs_improvement"], er5["paired_p"]),
        ),
        (
            "At which k was the largest difference?",
            "k=1 favours B marginally (Hit@1 +0.036, n.s.); k=4–5 favour A and reach "
            "significance (Hit@4/@5 p≈0.03).",
        ),
        (
            "Did generated questions improve Hit@k?",
            "Only at k=1 (n.s.). At k≥3 baseline is higher; Hit@4/@5 significantly so.",
        ),
        (
            "Evidence Recall@k?",
            "No — baseline ≥ generated at every k>1 (none significant).",
        ),
        (
            "All-Evidence Hit@k (multi-hop)?",
            "Marginally higher for B at k=4/5 (+0.024, n.s.); lower at k=3/10. A wash.",
        ),
        (
            "Did generated questions find the first evidence chunk earlier?",
            f"No — mean first-relevant rank A={frr_a} vs B={frr_b} (baseline earlier).",
        ),
        (
            "Did they help retrieve ALL evidence for multi-hop questions?",
            f"Complete-evidence achieved rate A={cer_rate_a:.3f} vs B={cer_rate_b:.3f}; among "
            f"those achieved, B slightly earlier (rank {cer_b} vs {cer_a}). Net: no.",
        ),
        (
            "Which question types improved / worsened most?",
            "See per-type table: differences are small and within noise on this n.",
        ),
        (
            "How many queries improved / unchanged / worsened (Evidence Recall@5)?",
            f"{er5['improved']} improved, {er5['unchanged']} unchanged, {er5['harmed']} harmed.",
        ),
        (
            "How often did the top synthetic question point to a non-gold chunk?",
            f"{fd.get('top_question_points_to_nongold_parent', 0)} of 84 queries "
            f"({fd.get('top_question_points_to_nongold_parent', 0) / 84 * 100:.0f}%).",
        ),
        (
            "How often did a lower-ranked synthetic question point to the correct chunk?",
            f"{fd.get('lower_ranked_question_hits_gold', 0)} queries — i.e. the gold chunk was "
            "usually reachable but a distractor question outranked it.",
        ),
        (
            "Partial vs complete multi-hop retrieval (B)?",
            f"{fb.get('partial_multihop_generated', 0)} partial vs "
            f"{fb.get('complete_multihop_generated', 0)} complete "
            f"({fb.get('no_evidence_generated', 0)} no-evidence).",
        ),
        (
            "Did retrieval differences change answer accuracy?",
            f"Slightly. EM A={ab[A]['exact_match']:.3f} vs B={ab[B]['exact_match']:.3f}; "
            f"F1 {ab[A]['token_f1']:.3f} vs {ab[B]['token_f1']:.3f} — baseline modestly ahead.",
        ),
        (
            "Index-size increase?",
            f"{idx['index_ratio_B_over_A']}× more vectors "
            f"({idx['baseline']['num_vectors']} → {idx['generated']['num_vectors']}); "
            f"store {idx['baseline']['index_size_mb']}MB → {idx['generated']['index_size_mb']}MB.",
        ),
        (
            "Extra offline generation + embedding cost?",
            f"{genq['generation_seconds']}s of LLM generation (gpt-4o-mini, ~$0.02) plus "
            f"{idx['generated']['embed_seconds']}s to embed questions vs "
            f"{idx['baseline']['embed_seconds']}s for chunks.",
        ),
        (
            "Online latency difference?",
            f"Query embed {lat['latency_ms']['query_embed_ms']['mean']}ms (shared); "
            f"search A {lat['latency_ms']['baseline_search_ms']['mean']}ms vs "
            f"B {lat['latency_ms']['generated_search_ms']['mean']}ms (B fetches 10× candidates + dedup).",
        ),
        (
            "Is the improvement worth ~10× more vectors?",
            "No. There is no retrieval or answer-quality gain to justify the 10× index "
            "and the extra offline generation cost on this pilot.",
        ),
    ]
    qa_html = "".join(
        f"<div class='qa'><div class='qn'>{i + 1}. {escape(q)}</div>"
        f"<div class='an'>{a}</div></div>"
        for i, (q, a) in enumerate(qa)
    )

    # ---- assemble -------------------------------------------------------- #
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MultiHop-RAG · Dense-Vector-Only Pilot: Chunk vs 10-Question Indexing</title>
<style>
:root {{ --bg:#ffffff; --fg:#1a1d24; --muted:#5b6572; --line:#e4e8ee; --card:#f7f9fc;
  --good:#0a7d3f; --bad:#c0392b; --flat:#8a94a2; --accentA:#2563eb; --accentB:#9333ea; --sig:#fff6d5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1319; --fg:#e6eaf0; --muted:#9aa5b3;
  --line:#242b36; --card:#161c25; --good:#38d17a; --bad:#ff6b5e; --flat:#7a8494;
  --accentA:#6ea0ff; --accentB:#c98bff; --sig:#3a3410; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 80px; }}
h1 {{ font-size:26px; margin:0 0 6px; }}
h2 {{ font-size:20px; margin:40px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
h3 {{ font-size:16px; margin:26px 0 10px; }}
.lede {{ color:var(--muted); font-size:15px; }}
.tag {{ display:inline-block; background:var(--card); border:1px solid var(--line); border-radius:20px;
  padding:3px 11px; font-size:12px; margin:3px 4px 3px 0; color:var(--muted); }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accentA);
  border-radius:10px; padding:16px 18px; margin:18px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; margin:10px 0; }}
th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:center; }}
th {{ background:var(--card); font-weight:600; }}
td.metric, td:first-child {{ text-align:left; }}
td.metric .sub, .sub {{ color:var(--muted); font-size:11.5px; font-weight:400; }}
.triple {{ line-height:1.3; }}
.triple .a {{ color:var(--accentA); font-weight:600; margin-right:7px; }}
.triple .b {{ color:var(--accentB); font-weight:600; margin-right:7px; }}
.triple .d {{ display:block; font-size:11.5px; }}
.triple.sig {{ background:var(--sig); }}
.good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .flat {{ color:var(--flat); }}
code {{ background:var(--card); padding:1px 5px; border-radius:4px; font-size:12px; }}
.scroll {{ overflow-x:auto; }}
.legend {{ font-size:12.5px; color:var(--muted); margin:4px 0 0; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }}
.card .q {{ font-weight:600; margin-bottom:4px; }}
.card .meta {{ color:var(--muted); font-size:12.5px; margin-bottom:10px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.cols h5 {{ margin:0 0 6px; font-size:13px; }}
.cols ol {{ margin:0; padding-left:20px; font-size:12.5px; }}
.cols li.gold {{ color:var(--good); font-weight:600; }}
.qa {{ border-bottom:1px solid var(--line); padding:10px 0; }}
.qa .qn {{ font-weight:600; }} .qa .an {{ color:var(--muted); margin-top:3px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:720px) {{ .cols,.grid2 {{ grid-template-columns:1fr; }} }}
.kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }}
.kpi {{ flex:1 1 150px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.kpi .n {{ font-size:22px; font-weight:700; }} .kpi .l {{ color:var(--muted); font-size:12px; }}
</style></head><body><div class="wrap">

<h1>MultiHop-RAG — Dense-Vector-Only Pilot</h1>
<p class="lede">Does indexing <b>10 synthetic questions per chunk</b> (Condition B) improve
retrieval of gold evidence over directly indexing the <b>original chunks</b> (Condition A)?
Vector-search only — cosine similarity in a local ChromaDB. No BM25 / sparse / hybrid / RRF.</p>
<div>
<span class="tag">closed 15-article pilot</span><span class="tag">seed 42</span>
<span class="tag">Octen-Embedding-0.6B · dim 1024 · cosine</span>
<span class="tag">gpt-4o-mini generation</span>
<span class="tag">{subset["num_fully_eligible_queries"]} eligible queries</span>
<span class="tag">k = {", ".join(map(str, KS))}</span>
</div>

<div class="verdict"><b>Headline:</b> {verdict}</div>

<div class="kpis">
  <div class="kpi"><div class="n">{subset["num_chunks"]}</div><div class="l">token-chunks (Condition A vectors)</div></div>
  <div class="kpi"><div class="n">{idx["generated"]["num_vectors"]}</div><div class="l">question vectors (Condition B, {idx["index_ratio_B_over_A"]}×)</div></div>
  <div class="kpi"><div class="n">{overall[A]["evidence_recall@5"]["mean"]:.3f} / {overall[B]["evidence_recall@5"]["mean"]:.3f}</div><div class="l">Evidence Recall@5 · A / B</div></div>
  <div class="kpi"><div class="n">{ab[A]["exact_match"]:.2f} / {ab[B]["exact_match"]:.2f}</div><div class="l">Answer EM · A / B</div></div>
</div>

<p class="legend"><b>Caveat.</b> This is a <b>selected 15-article pilot collection</b>, not the
official full-corpus MultiHop-RAG benchmark. n={subset["num_fully_eligible_queries"]} is small — read
significance stars, not point estimates, and treat non-significant deltas as ties.</p>

<h2>1 · Retrieval results — full metric sweep</h2>
<p class="legend">Each cell: <span class="a" style="color:var(--accentA)">A (chunks)</span> ·
<span class="b" style="color:var(--accentB)">B (questions)</span> · <b>Δ = B−A</b>
(<span class="good">green</span>=B better, <span class="bad">red</span>=A better).
<b>*</b> = paired-bootstrap p&lt;0.05 (yellow row-cell). Higher is better for all metrics.</p>
<div class="scroll"><table>
<thead><tr><th>Metric</th>{sweep_head}</tr></thead>
<tbody>{"".join(sweep_rows)}</tbody></table></div>

<h2>2 · Paired comparison (@k=5) — who wins query-by-query</h2>
<div class="scroll"><table>
<thead><tr><th>Metric</th><th class="good">B better</th><th class="flat">tie</th>
<th class="bad">A better</th><th>mean Δ</th><th>paired p</th></tr></thead>
<tbody>{"".join(paired_rows)}</tbody></table></div>
<p class="legend">Improvements and harms are roughly symmetric on every metric — the net effect is a wash.</p>

<h2>3 · Breakdown by question type &amp; hop count</h2>
<div class="grid2">
<div><h3>By question type</h3><div class="scroll"><table>
<thead><tr><th>Type</th><th>n</th><th>ER@5 A</th><th>ER@5 B</th><th>Δ</th>
<th>Hit@1 A/B</th><th>AllEv@5 A/B</th></tr></thead><tbody>{by_type}</tbody></table></div></div>
<div><h3>By required source documents</h3><div class="scroll"><table>
<thead><tr><th>#docs</th><th>n</th><th>ER@5 A</th><th>ER@5 B</th><th>Δ</th>
<th>Hit@1 A/B</th><th>AllEv@5 A/B</th></tr></thead><tbody>{by_docs}</tbody></table></div></div>
</div>

<h2>4 · Answer generation (top-5 chunks → gpt-4o-mini)</h2>
<p class="legend">Values are <b>A / B</b>. Split by how much gold evidence the retriever
delivered, to separate retrieval failure from reasoning failure.</p>
<div class="scroll"><table>
<thead><tr><th>Bucket</th><th>Exact match</th><th>Token F1</th><th>Yes/No acc</th></tr></thead>
<tbody>{ans_rows}</tbody></table></div>
<p class="legend">Note the extra <b>none-evidence</b> bucket for B (9 vs 2) — generated-question
retrieval more often misses evidence entirely, even though it reaches <i>all</i> evidence on a
couple more queries.</p>

<h2>5 · Failure analysis</h2>
<div class="grid2">
<div><h3>Query buckets</h3><div class="scroll"><table><tbody>{fbucket_rows}</tbody></table></div></div>
<div><h3>Generated-question diagnostics</h3><div class="scroll"><table><tbody>{fdiag_rows}</tbody></table></div></div>
</div>
<h3>Representative contrasts</h3>
{"".join(ex_cards)}

<h2>6 · Cost, storage &amp; latency</h2>
<div class="scroll"><table>
<thead><tr><th></th><th>A · chunk vectors</th><th>B · question vectors</th></tr></thead>
<tbody>
<tr><td>Vectors indexed</td><td>{idx["baseline"]["num_vectors"]}</td><td>{idx["generated"]["num_vectors"]} ({idx["index_ratio_B_over_A"]}×)</td></tr>
<tr><td>Chroma store size</td><td>{idx["baseline"]["index_size_mb"]} MB</td><td>{idx["generated"]["index_size_mb"]} MB</td></tr>
<tr><td>Offline embed time</td><td>{idx["baseline"]["embed_seconds"]} s</td><td>{idx["generated"]["embed_seconds"]} s</td></tr>
<tr><td>Extra offline LLM generation</td><td>—</td><td>{genq["generation_seconds"]} s (~$0.02)</td></tr>
<tr><td>Query-embed latency (shared)</td><td colspan="2">{lat["latency_ms"]["query_embed_ms"]["mean"]} ms mean · p95 {lat["latency_ms"]["query_embed_ms"]["p95"]} ms</td></tr>
<tr><td>Vector-search latency</td><td>{lat["latency_ms"]["baseline_search_ms"]["mean"]} ms</td><td>{lat["latency_ms"]["generated_search_ms"]["mean"]} ms</td></tr>
</tbody></table></div>

<h2>7 · Selected 15-article pilot collection</h2>
<p class="legend">Gold alignment: {gold["total_gold_evidence_facts"]} evidence facts —
{gold["exact_matched"]} exact, {gold["fuzzy_matched"]} fuzzy, {gold["unresolved"]} unresolved;
{gold["facts_crossing_chunk_boundaries"]} span a chunk overlap (counted once).
Avg {gold["avg_gold_chunks_per_query"]} gold chunks &amp; {gold["avg_evidence_facts_per_query"]} evidence facts per query.</p>
<div class="scroll"><table>
<thead><tr><th>Title</th><th>Source</th><th>Category</th><th>Words</th><th>Chunks</th><th>Elig. queries</th></tr></thead>
<tbody>{art_rows}</tbody></table></div>

<h2>8 · Generated-question quality</h2>
<p class="legend">{genq["actual_valid_questions"]} questions · exactly {genq["avg_valid_questions_per_chunk"]:.0f}/chunk ·
duplicate rate {genq["duplicate_rate"] * 100:.1f}% · avg {genq["avg_retries_per_chunk"]} retries/chunk ·
{genq["validation_failure_rate"] * 100:.0f}% of sets tripped a QC flag (mostly non-verbatim supporting spans /
meta-references) but still carry 10 usable questions. Types:
{", ".join(f"{k} {v}" for k, v in list(genq["question_type_distribution"].items())[:6])}.</p>

<h2>9 · Final report — the 20 questions</h2>
{qa_html}

<h2>Method &amp; reproducibility</h2>
<p class="lede">Both conditions share the same 15 articles, cleaning, token-chunk boundaries
(256/50/80), eligible queries, embedding model, cosine metric, k values, gold mapping, answer
LLM &amp; prompt, and seed (42). The <b>only</b> difference is the indexed vector representation:
one chunk vector (A) vs ten generated-question vectors mapped back to the parent chunk via
max-similarity (B). Query vectors are transient and never stored. Gold evidence and gold answers
were used only to build the closed collection and the evaluation labels — never during indexing,
question generation, or retrieval scoring. Run with <code>python run_vo.py</code>.</p>
<p class="legend">Artifacts under <code>data/processed/mhrag_vectoronly/</code> and
<code>results/mhrag_vectoronly/</code>; config in <code>config/mhrag_vectoronly.yaml</code>.</p>

</div></body></html>"""

    C.REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] wrote {C.REPORT_HTML}")
    return str(C.REPORT_HTML)


if __name__ == "__main__":
    build()
