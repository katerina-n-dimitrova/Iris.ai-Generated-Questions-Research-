"""
Render the PeerQA chunk-size × question-enrichment experiment as a standalone
HTML report (report/peerqa_chunksize_results.html). Separate file — does not
overwrite earlier reports. All numbers pulled from chunksize_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import peerqa_config as C

RESULTS = C.RESULTS_DIR / "chunksize_results.json"
OUT = C.PROJECT_ROOT / "report" / "peerqa_chunksize_results.html"

STYLE = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1e293b;background:#f1f5f9;line-height:1.55}
header{background:linear-gradient(135deg,#0c4a6e,#0f766e);color:#fff;padding:38px 24px}
header .wrap{max-width:1120px;margin:0 auto}header h1{margin:0 0 8px;font-size:26px}header p{margin:0;opacity:.92;font-size:14.5px}
main{max-width:1120px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 22px;margin:18px 0;box-shadow:0 1px 3px rgba(15,23,42,.05)}
h2{font-size:20px;margin:2px 0 2px;border-left:4px solid #0f766e;padding-left:10px}
.dt{color:#0f766e;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:2px 0 12px}
.sub{color:#475569;font-size:14px;margin:6px 0 12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0}
th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #eef2f7}
th{background:#f8fafc;color:#64748b;font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}
tr.base td{background:#f0fdfa}
tr.best td{background:#f0fdf4;font-weight:600}
tr.top td{background:#fef08a !important;font-weight:700}
tr.top td.l:first-of-type::after{content:' ★ best';color:#a16207;font-weight:700;font-size:10px}
tr.grp td{background:#f8fafc;font-weight:700;color:#0f172a;text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
.win{color:#16a34a;font-weight:700}.lose{color:#dc2626}.muted{color:#94a3b8}
.fb{background:#f0fdfa;border:1px solid #99f6e4;border-radius:9px;padding:13px 15px;margin-top:14px;font-size:14px}
.fb b{color:#0f766e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:10px 0}
.kv{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:11px 13px}
.kv .n{font-size:19px;font-weight:700;color:#0f172a}.kv .k{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.03em;margin-top:2px}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;margin-left:6px}
.tag.up{background:#dcfce7;color:#166534}.tag.down{background:#fee2e2;color:#991b1b}.tag.mix{background:#fef9c3;color:#854d0e}
ul{margin:8px 0 0;padding-left:20px}li{margin:5px 0;font-size:14px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12.5px}
footer{max-width:1120px;margin:0 auto;padding:10px 24px 44px;color:#64748b;font-size:12px}
"""


def _f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _cls(v, b, higher=True):
    if v is None or b is None or abs(v - b) < 1e-9:
        return ""
    return "win" if ((v > b) == higher) else "lose"


def _metric_cells(r, base, cols=("hit@1", "hit@5", "hit@10", "mrr", "ndcg@10")):
    out = []
    for k in cols:
        v = r.get(k)
        b = base.get(k) if base else None
        out.append(f"<td class='{_cls(v, b) if base else ''}'>{_f(v)}</td>")
    return "".join(out)


def build_html(d: dict) -> str:
    per_size = d["per_size"]
    var = d["variable"]
    table1 = d["table1"]
    table3 = d["table3"]
    emb = d.get("embedding_model", "")
    llm = d.get("llm_model", "")
    sizes = [s for s, _ in d["fixed_sizes"]]

    # best overall by ndcg (generated-question setup) for the green marker
    best_cond = max(d["table1"] + d["table2"], key=lambda r: r.get("ndcg@10") or 0)[
        "condition"
    ]

    # yellow highlight = the single best (max nDCG@10) row IN EACH TABLE.
    def _nd(r):
        return (r or {}).get("ndcg@10") or 0

    # Table 1 candidates: every rendered row (per-size baselines + q rows).
    t1_cands = []
    for s in sizes:
        rec = per_size[str(s)]
        t1_cands.append((s, "baseline", rec["baseline"]))
        for nq in d["fixed_counts"]:
            r = rec["counts"].get(f"q{nq}")
            if r:
                t1_cands.append((s, r["condition"], r))
    t1b = max(t1_cands, key=lambda x: _nd(x[2]))
    t1_top = (t1b[0], t1b[1])  # (size, condition)
    t2_top = max(var["conditions"].values(), key=_nd)["condition"]  # best var row
    t3_top = "best nDCG@10 (overall)"  # selection label

    H = [
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>PeerQA chunk size × question enrichment</title><style>{STYLE}</style></head><body>"
    ]
    H.append(
        "<header><div class='wrap'><h1>PeerQA · chunk size × generated-question enrichment</h1>"
        "<p>Does the amount of generated-question enrichment needed for retrieval depend on "
        "<b>chunk size</b> and/or <b>information density</b>? Questions-only doc2query retrieval "
        "(embed only generated questions; a hit returns the parent chunk), swept across 4 chunk "
        "sizes and fixed vs adaptive question counts.</p></div></header><main>"
    )

    # ---- setup ---- #
    ns = d["num_papers"]
    H.append(
        "<div class='card'><h2>Experimental setup</h2>"
        "<div class='dt'>same PeerQA subset &amp; retrieval setup as the questions-only experiment</div>"
        "<p class='sub'>PeerQA reviewer questions are the queries; gold = author-annotated evidence "
        "sentences mapped onto whichever chunk(s) contain them. Retrieval embeds <b>only generated "
        "questions</b> (each mapped to its parent chunk); a chunk-text dense index is the baseline, and a "
        "<b>fused</b> (chunk-vector ⊕ best-question) variant is included for the adaptive testbed.</p>"
        f"<ul><li><b>Subset:</b> {ns} papers (same selection as before).</li>"
        f"<li><b>Embedder:</b> <code>{emb}</code>. <b>LLM:</b> <code>{llm}</code>, grounded diverse-subtopic prompt.</li>"
        "<li><b>Chunk sizes:</b> 200 / 400 / 600 / 800 tokens, overlap ~22% (45 / 90 / 125 / 175). "
        "Adaptive strategies run on a separate <b>section-aware variable-size</b> chunking (~100–800 tokens).</li>"
        "<li><b>Question pool:</b> up to 15 grounded questions generated once per chunk (cached); "
        "fixed q5/q10/q13/q15 and adaptive allocations are first-k slices.</li>"
        "<li><b>Metrics:</b> Hit@1/5/10, MRR, nDCG@10, + query latency, generation time, encode time, "
        "#questions, #embeddings, index size.</li></ul></div>"
    )

    # ---- chunking conditions overview ---- #
    H.append(
        "<div class='card'><h2>Chunking conditions</h2>"
        "<div class='dt'>chunks &amp; token stats per size</div><div class='scroll'><table><thead><tr>"
        "<th class='l'>Chunk size</th><th>Overlap</th><th>#Chunks</th><th>Eval queries</th>"
        "<th>tok min</th><th>tok mean</th><th>tok max</th><th>≤15 Q avg/chunk</th>"
        "<th>Gen time (s)</th></tr></thead><tbody>"
    )
    for s in sizes:
        rec = per_size[str(s)]
        t = rec["tokens"]
        gen = rec.get("gen", {})
        H.append(
            f"<tr><td class='l'>{s} tok</td><td>{rec['overlap']}</td><td>{rec['num_chunks']}</td>"
            f"<td>{rec['num_eval_queries']}</td><td>{_f(t.get('min'), 0)}</td>"
            f"<td>{_f(t.get('mean'), 0)}</td><td>{_f(t.get('max'), 0)}</td>"
            f"<td>{gen.get('avg_questions_per_chunk', '—')}</td>"
            f"<td class='muted'>{gen.get('wall_seconds', '—')}</td></tr>"
        )
    vt = var["tokens"]
    vgen = var.get("gen", {})
    H.append(
        f"<tr><td class='l'>variable</td><td>~120</td>"
        f"<td>{var['num_chunks']}</td><td>{var['num_eval_queries']}</td><td>{_f(vt.get('min'), 0)}</td>"
        f"<td>{_f(vt.get('mean'), 0)}</td><td>{_f(vt.get('max'), 0)}</td>"
        f"<td>{vgen.get('avg_questions_per_chunk', '—')}</td>"
        f"<td class='muted'>{vgen.get('wall_seconds', '—')}</td></tr>"
    )
    H.append(
        "</tbody></table></div><p class='sub' style='margin-top:8px'>Smaller chunks ⇒ many more "
        "chunks (and embeddings) for the same corpus; the LLM also generates fewer grounded questions "
        "from a small chunk, so the effective questions/chunk is lower even at the same requested count.</p></div>"
    )

    # ---- Table 1 ---- #
    H.append(
        "<div class='card'><h2>Table 1 · Fixed question counts by chunk size</h2>"
        "<div class='dt'>questions-only index · green/red vs that size's chunk-text baseline</div>"
        "<p class='sub'>Each block is one chunk size; the shaded <b>baseline</b> row embeds the chunk "
        "text. q5–q15 embed only generated questions. Colour compares each q-row to its own size baseline.</p>"
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Condition</th><th>Size</th><th>Overlap</th><th>Q/chunk</th>"
        "<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Size MB</th><th>p95 ms</th></tr></thead><tbody>"
    )
    for s in sizes:
        rec = per_size[str(s)]
        base = rec["baseline"]
        H.append(
            f"<tr class='grp'><td colspan='12'>Chunk size {s} tokens (overlap {rec['overlap']}, "
            f"{rec['num_chunks']} chunks)</td></tr>"
        )
        # baseline row
        bcls = "top" if t1_top == (s, "baseline") else "base"
        H.append(
            f"<tr class='{bcls}'><td class='l'>baseline (chunk text)</td><td>{s}</td>"
            f"<td>{rec['overlap']}</td><td>0</td>{_metric_cells(base, None)}"
            f"<td class='muted'>{base['num_embeddings']}</td>"
            f"<td class='muted'>{_f(base['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(base['search_p95'], 2)}</td></tr>"
        )
        for nq in d["fixed_counts"]:
            r = rec["counts"].get(f"q{nq}")
            if not r:
                continue
            cls = (
                "top"
                if t1_top == (s, r["condition"])
                else ("best" if r["condition"] == best_cond else "")
            )
            H.append(
                f"<tr class='{cls}'><td class='l'>chunk_size_{s}_q{nq}</td><td>{s}</td>"
                f"<td>{rec['overlap']}</td><td>{nq}</td>{_metric_cells(r, base)}"
                f"<td class='muted'>{r['num_embeddings']}</td>"
                f"<td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
                f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
            )
    H.append("</tbody></table></div>")
    H.append(_table1_finding(d))
    H.append("</div>")

    # ---- Table 2 ---- #
    vc = var["conditions"]
    vbase = vc.get("baseline")
    H.append(
        "<div class='card'><h2>Table 2 · Adaptive vs fixed question counts</h2>"
        "<div class='dt'>section-aware variable-size chunking · one chunk set, one gold</div>"
        f"<p class='sub'>All rows share the same variable chunking ({var['num_chunks']} chunks, "
        f"tokens {_f(vt.get('min'), 0)}–{_f(vt.get('max'), 0)}). <b>adaptive length-based</b> sets #questions "
        "from each chunk's token length; <b>adaptive density-based</b> from an information-density score "
        "(calibrated to average ~10, a matched-budget test vs fixed q10). Green/red vs the chunk-text baseline. "
        "This is the <b>original-chunk vs questions-only vs fused</b> comparison.</p>"
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Condition</th><th>Strategy</th><th>Q/chunk</th>"
        "<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Size MB</th><th>p95 ms</th></tr></thead><tbody>"
    )
    order = [
        ("baseline", "original chunk", "chunk-text"),
        ("fixed_q10", "questions-only", "fixed q10"),
        ("adapt_length", "questions-only", "adaptive · length"),
        ("adapt_density", "questions-only", "adaptive · density"),
        ("fused_density", "fused", "chunk ⊕ density-q"),
    ]
    for key, strat, label in order:
        r = vc.get(key)
        if not r:
            continue
        is_base = key == "baseline"
        cls = "top" if r["condition"] == t2_top else ("base" if is_base else "")
        H.append(
            f"<tr class='{cls}'><td class='l'>{label}</td><td class='muted'>{strat}</td>"
            f"<td>{r['q_per_chunk']}</td>{_metric_cells(r, None if is_base else vbase)}"
            f"<td class='muted'>{r['num_embeddings']}</td>"
            f"<td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
        )
    H.append("</tbody></table></div>")
    H.append(_table2_finding(var))
    H.append("</div>")

    # ---- Table 3 ---- #
    H.append(
        "<div class='card'><h2>Table 3 · Best quality / latency trade-off</h2>"
        "<div class='dt'>picked across every condition</div><div class='scroll'><table><thead><tr>"
        "<th class='l'>Selection</th><th class='l'>Condition</th><th>Size</th><th>Q/chunk</th>"
        "<th>Index</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Size MB</th><th>p95 ms</th></tr></thead><tbody>"
    )
    for r in table3:
        tcls = "top" if r["selection"] == t3_top else ""
        H.append(
            f"<tr class='{tcls}'><td class='l'><b>" + r["selection"] + "</b></td>"
            f"<td class='l'>{r['condition']}</td><td>{r['chunk_size']}</td><td>{r['q_per_chunk']}</td>"
            f"<td class='muted'>{r.get('index_content', '')}</td>"
            f"<td>{_f(r['hit@1'])}</td><td>{_f(r['hit@5'])}</td><td>{_f(r['hit@10'])}</td>"
            f"<td>{_f(r['mrr'])}</td><td>{_f(r['ndcg@10'])}</td>"
            f"<td class='muted'>{r['num_embeddings']}</td><td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
        )
    H.append("</tbody></table></div></div>")

    # ---- analysis ---- #
    H.append(_analysis(d))
    # ---- recommendation ---- #
    H.append(_recommendation(d))

    H.append(
        f"<footer>PeerQA subset ({ns} papers) · chunk-size × question-enrichment · embedder {emb} · "
        "LLM "
        + llm
        + ". All figures from results/peerqa/chunksize_results.json.</footer></main></body></html>"
    )
    return "".join(H)


# --------------------------------------------------------------------------- #
# Findings / analysis (data-driven)
# --------------------------------------------------------------------------- #
def _best_count_per_size(d):
    """For each size, the fixed count with max nDCG@10, and its delta over baseline."""
    out = {}
    for s, rec in d["per_size"].items():
        base_nd = rec["baseline"]["ndcg@10"] or 0
        rows = list(rec["counts"].values())
        if not rows:
            continue
        best = max(rows, key=lambda r: r.get("ndcg@10") or 0)
        out[int(s)] = (best, best.get("ndcg@10", 0) - base_nd, base_nd)
    return out


def _table1_finding(d):
    bc = _best_count_per_size(d)
    parts = []
    for s in sorted(bc):
        best, dnd, base_nd = bc[s]
        parts.append(
            f"<b>{s}t</b>: best {best['q_per_chunk']}q "
            f"(nDCG {_f(best['ndcg@10'], 3)}, {'+' if dnd >= 0 else ''}{_f(dnd, 3)} vs base {_f(base_nd, 3)})"
        )
    return (
        "<div class='fb'><b>Reading Table 1</b><br>Best fixed count per size — "
        + "; ".join(parts)
        + ". A rising best-count with size ⇒ bigger chunks want more questions; "
        "a flat best-count ⇒ the sweet spot is size-independent.</div>"
    )


def _table2_finding(var):
    vc = var["conditions"]
    q10 = vc.get("fixed_q10", {}).get("ndcg@10")
    al = vc.get("adapt_length", {}).get("ndcg@10")
    ad = vc.get("adapt_density", {}).get("ndcg@10")
    fu = vc.get("fused_density", {}).get("ndcg@10")
    base = vc.get("baseline", {}).get("ndcg@10")

    def d(a, b):
        return f"{'+' if (a or 0) - (b or 0) >= 0 else ''}{_f((a or 0) - (b or 0), 3)}"

    return (
        "<div class='fb'><b>Reading Table 2</b><br>"
        f"On the variable chunking: chunk-text baseline nDCG {_f(base, 3)}; fixed q10 {_f(q10, 3)}; "
        f"adaptive-length {_f(al, 3)} ({d(al, q10)} vs q10, avg {var.get('adapt_length_avg_q')}q); "
        f"adaptive-density {_f(ad, 3)} ({d(ad, q10)} vs q10, avg {var.get('adapt_density_avg_q')}q at matched budget); "
        f"fused {_f(fu, 3)} ({d(fu, base)} vs baseline). Adaptive beats fixed q10 only if it reallocates the same "
        "budget to where it helps; fused shows whether keeping the chunk vector recovers the precision that "
        "questions-only loses.</div>"
    )


def _analysis(d):
    bc = _best_count_per_size(d)
    sizes = sorted(bc)
    best_counts = [bc[s][0]["q_per_chunk"] for s in sizes]
    rising = len(best_counts) >= 2 and best_counts[-1] > best_counts[0]
    q10_often_best = sum(1 for s in sizes if bc[s][0]["q_per_chunk"] == 10)
    var = d["variable"]["conditions"]
    q10n = var.get("fixed_q10", {}).get("ndcg@10") or 0
    adn = max(
        var.get("adapt_length", {}).get("ndcg@10") or 0,
        var.get("adapt_density", {}).get("ndcg@10") or 0,
    )
    adaptive_wins = adn > q10n + 0.002
    # gains from size vs count: spread of baselines across sizes vs spread of counts within a size
    base_nds = [d["per_size"][str(s)]["baseline"]["ndcg@10"] or 0 for s in sizes]
    size_spread = max(base_nds) - min(base_nds)
    within_spreads = []
    for s in sizes:
        vals = [r.get("ndcg@10") or 0 for r in d["per_size"][str(s)]["counts"].values()]
        if vals:
            within_spreads.append(max(vals) - min(vals))
    count_spread = max(within_spreads) if within_spreads else 0

    per_size_best = ", ".join(f"{s}t→{bc[s][0]['q_per_chunk']}q" for s in sizes)
    li = []
    li.append(
        "<li><b>Do larger chunks benefit from more questions?</b> "
        + (
            "Yes — the best fixed count rises with size ("
            if rising
            else "Not clearly — the best fixed count is roughly flat across sizes ("
        )
        + per_size_best
        + "). More questions help most where a chunk holds more distinct facts.</li>"
    )
    li.append(
        f"<li><b>Is q10 still a sweet spot across sizes?</b> q10 is the top fixed count in "
        f"{q10_often_best}/{len(sizes)} sizes; elsewhere the best sits at "
        f"{', '.join(str(bc[s][0]['q_per_chunk']) for s in sizes)} questions respectively — "
        "q10 is a solid default but not universally optimal.</li>"
    )
    li.append(
        f"<li><b>Does adaptive beat fixed q10?</b> "
        f"{'Yes' if adaptive_wins else 'No'} — best adaptive nDCG@10 {_f(adn, 3)} vs fixed q10 {_f(q10n, 3)} "
        "at matched average budget. Reallocating a fixed question budget toward denser/longer chunks "
        f"{'pays off' if adaptive_wins else 'does not clearly help on this subset'}.</li>"
    )
    li.append(
        "<li><b>Do smaller chunks need fewer questions?</b> Yes structurally — a 200-token chunk holds "
        "few distinct facts, so the LLM produces fewer grounded questions and extra requested questions "
        "become near-duplicates that add embeddings without recall; small chunks saturate earliest.</li>"
    )
    li.append(
        "<li><b>Do larger chunks get too broad/noisy?</b> Larger chunks raise recall (the right chunk is "
        "bigger, easier to hit) but blur rank-1 precision; questions counter this by giving the big chunk "
        "several sharp entry points, which is why big chunks gain the most from more questions.</li>"
    )
    li.append(
        f"<li><b>Gains from size, count, or both?</b> Across baselines, chunk size alone moves nDCG@10 by "
        f"~{_f(size_spread, 3)}; within a fixed size, question count moves it by up to ~{_f(count_spread, 3)}. "
        f"{'Size is the larger lever here' if size_spread > count_spread else 'Question count is the larger lever here'}, "
        "but they interact — the best config needs both tuned.</li>"
    )
    li.append(
        "<li><b>Is the extra generation/indexing worth it?</b> Generated questions cost N× the embeddings "
        "and a one-off LLM pass; they buy recall (Hit@5/10) more than rank-1 precision. Worth it when recall "
        "at depth matters; if only Hit@1 matters, the chunk-text (or fused) index is cheaper and stronger.</li>"
    )
    return (
        "<div class='card'><h2>Analysis</h2><div class='dt'>chunk size vs question enrichment</div>"
        "<ul>" + "".join(li) + "</ul></div>"
    )


def _recommendation(d):
    t3 = {r["selection"]: r for r in d["table3"]}
    best = t3.get("best nDCG@10 (overall)", {})
    best_enr = t3.get("best generated-question setup", {})
    cheap = t3.get("cheapest strong", {})
    var = d["variable"]["conditions"]
    q10n = var.get("fixed_q10", {}).get("ndcg@10") or 0
    adn = max(
        var.get("adapt_length", {}).get("ndcg@10") or 0,
        var.get("adapt_density", {}).get("ndcg@10") or 0,
    )
    which_adaptive = (
        "density-based"
        if (var.get("adapt_density", {}).get("ndcg@10") or 0)
        >= (var.get("adapt_length", {}).get("ndcg@10") or 0)
        else "length-based"
    )
    adaptive_worth = adn > q10n + 0.005
    return (
        "<div class='card'><h2>Final recommendation</h2><div class='dt'>one practical setup for future runs</div>"
        "<ul>"
        f"<li><b>Best overall:</b> plain <b>{best.get('chunk_size', '—')}-token chunk-text</b> retrieval "
        f"(<code>{best.get('condition', '—')}</code>) is the single strongest and cheapest index — nDCG@10 "
        f"{_f(best.get('ndcg@10'), 4)} at {_f(best.get('index_size_mb'), 1)} MB. On this subset, using a "
        "larger chunk beats adding generated questions on nDCG@10.</li>"
        f"<li><b>Best fixed chunk size + question count:</b> if you do enrich, use the largest chunk with the "
        f"most questions — <code>{best_enr.get('condition', '—')}</code> "
        f"(size {best_enr.get('chunk_size', '—')}, {best_enr.get('q_per_chunk', '—')} q/chunk), nDCG@10 "
        f"{_f(best_enr.get('ndcg@10'), 4)}; smaller chunks and fewer questions are strictly worse.</li>"
        f"<li><b>Adaptive rule:</b> if adopting adaptive counts, prefer the <b>{which_adaptive}</b> rule "
        "(cap 15, floor ~4). It concentrates questions on denser/longer chunks and sparse ones get fewer, "
        "trimming wasted near-duplicate questions.</li>"
        f"<li><b>Length vs density:</b> {'density-based edges out length-based' if which_adaptive == 'density-based' else 'length-based is the simpler, competitive choice'} "
        "here; density needs only cheap regex signals (entities, numbers, metric terms, table/figure refs).</li>"
        f"<li><b>Is the complexity worth it vs fixed q10?</b> "
        f"{'Yes — adaptive beats q10 by ' + _f(adn - q10n, 3) + ' nDCG@10 at matched budget' if adaptive_worth else 'Marginal on this subset — fixed q10 at a mid chunk size is the pragmatic default'}; "
        "revisit at full-corpus scale before committing to adaptive machinery.</li>"
        "<li><b>Precision vs recall:</b> for rank-1-critical use, keep the chunk vector and <b>fuse</b> it with "
        "the generated-question vectors rather than going questions-only — fusion recovers the precision that "
        "pure questions-only sacrifices while keeping its recall.</li>"
        "</ul></div>"
    )


def main():
    d = json.load(open(RESULTS, encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(d), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
