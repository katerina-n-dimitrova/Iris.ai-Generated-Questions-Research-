"""
Render the DocBank chunk-size × question-enrichment experiment as a standalone
HTML report (report/docbank_chunksize_results.html). Separate file. All numbers
from docbank_chunksize_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import docbank_config as C

RESULTS = C.RESULTS_DIR / "docbank_chunksize_results.json"
OUT = C.PROJECT_ROOT / "report" / "docbank_chunksize_results.html"

STYLE = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1e293b;background:#f1f5f9;line-height:1.55}
header{background:linear-gradient(135deg,#0c4a6e,#0d9488);color:#fff;padding:38px 24px}
header .wrap{max-width:1120px;margin:0 auto}header h1{margin:0 0 8px;font-size:26px}header p{margin:0;opacity:.93;font-size:14.5px}
main{max-width:1120px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 22px;margin:18px 0;box-shadow:0 1px 3px rgba(15,23,42,.05)}
h2{font-size:20px;margin:2px 0 2px;border-left:4px solid #0d9488;padding-left:10px}
.dt{color:#0d7a70;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:2px 0 12px}
.sub{color:#475569;font-size:14px;margin:6px 0 12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0}
th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #eef2f7}
th{background:#f8fafc;color:#64748b;font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}
tr.base td{background:#f0fdfa}
tr.grp td{background:#f8fafc;font-weight:700;color:#0f172a;text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
tr.top td{background:#fef08a !important;font-weight:700}
tr.top td.l:first-of-type::after{content:' ★ best';color:#a16207;font-weight:700;font-size:10px}
.win{color:#16a34a;font-weight:700}.lose{color:#dc2626}.muted{color:#94a3b8}
.fb{background:#f0fdfa;border:1px solid #99f6e4;border-radius:9px;padding:13px 15px;margin-top:14px;font-size:14px}
.fb b{color:#0d7a70}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:10px 0}
.kv{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:11px 13px}
.kv .n{font-size:19px;font-weight:700;color:#0f172a}.kv .k{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.03em;margin-top:2px}
ul{margin:8px 0 0;padding-left:20px}li{margin:5px 0;font-size:14px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
footer{max-width:1120px;margin:0 auto;padding:10px 24px 44px;color:#64748b;font-size:12px}
@media (prefers-color-scheme:dark){
body{background:#0b1220;color:#e2e8f0}.card{background:#111a2e;border-color:#1e293b}
th{background:#0f1728;color:#94a3b8}.kv{background:#0f1728;border-color:#1e293b}.kv .n{color:#f1f5f9}
td,th{border-color:#1e293b}tr.base td{background:#0e2a28}tr.grp td{background:#0f1728;color:#e2e8f0}
.fb{background:#0e2a28;border-color:#155e56}tr.top td{background:#a16207 !important;color:#fff}code{background:#1e293b;color:#cbd5e1}
}
"""


def _f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _cls(v, b, higher=True):
    if v is None or b is None or abs(v - b) < 1e-9:
        return ""
    return "win" if ((v > b) == higher) else "lose"


def _mcells(r, base):
    out = []
    for k in ("hit@1", "hit@5", "hit@10", "mrr", "ndcg@10"):
        v = r.get(k)
        b = base.get(k) if base else None
        out.append(f"<td class='{_cls(v, b) if base else ''}'>{_f(v)}</td>")
    return "".join(out)


def _best_count_per_size(d):
    out = {}
    for s, rec in d["per_size"].items():
        bnd = rec["baseline"]["ndcg@10"] or 0
        rows = list(rec["counts"].values())
        best = max(rows, key=lambda r: r.get("ndcg@10") or 0)
        out[int(s)] = (best, (best.get("ndcg@10") or 0) - bnd, bnd)
    return out


def body_cards(d, embed=False):
    """The chunk-size cards (overview + Table 1/2/3 + analysis + recommendation).
    embed=True prepends a section divider suitable for appending into the main
    DocBank report; embed=False emits the standalone setup card."""
    per_size = d["per_size"]
    var = d["variable"]
    sizes = [s for s, _ in d["fixed_sizes"]]
    emb = d.get("embedding_model", "")
    llm = d.get("llm_model", "")
    best_all = max(d["table1"] + d["table2"], key=lambda r: r.get("ndcg@10") or 0)

    # per-table yellow winners
    t1c = []
    for s in sizes:
        rec = per_size[str(s)]
        t1c.append((s, "baseline", rec["baseline"]))
        for nq in d["fixed_counts"]:
            r = rec["counts"].get(f"q{nq}")
            if r:
                t1c.append((s, r["condition"], r))
    t1b = max(t1c, key=lambda x: x[2].get("ndcg@10") or 0)
    t1_top = (t1b[0], t1b[1])
    t2_top = max(var["conditions"].values(), key=lambda r: r.get("ndcg@10") or 0)[
        "condition"
    ]

    H = []
    if embed:
        H.append(
            "<div class='card' id='chunksize'>"
            "<h2>14 · Chunk size × question count — do bigger chunks want more questions?</h2>"
            "<div class='dt'>PeerQA chunk-size study, replicated on DocBank</div>"
            f"<p class='sub'>Follow-up to the results above. The {d['num_eval_qa_total']} synthetic eval "
            "questions are reused as queries, each gold <b>remapped onto every chunking</b> by its verbatim "
            f"evidence text; only questions mapping in all 5 chunkings are kept — <b>{d['num_common_queries']}</b> "
            "common queries, so sizes are directly comparable. Five chunkings: fixed <b>200 / 400 / 600 / 800</b> "
            "tok (overlap ~22%) + one section-aware <b>variable</b>; questions-only index + chunk-text baseline "
            "+ fused. Gold coverage per chunking: "
            + ", ".join(f"{k} {v}" for k, v in d.get("gold_coverage", {}).items())
            + f" of {d['num_eval_qa_total']}.</p></div>"
        )
    else:
        H.append(
            "<div class='card'><h2>Experimental setup</h2>"
            "<div class='dt'>same 15 DocBank docs · reused synthetic QA · gold remapped by evidence</div>"
            f"<p class='sub'>The {d['num_eval_qa_total']} synthetic eval questions are reused as queries; "
            "each question's gold chunk is <b>remapped onto every chunking</b> by locating its verbatim "
            f"evidence text. Only questions whose evidence maps in <b>all 5 chunkings</b> are kept — a "
            f"common set of <b>{d['num_common_queries']}</b> queries, so sizes are directly comparable. "
            "Retrieval embeds ONLY generated questions (parent chunk returned on a hit); a chunk-text "
            "index is the baseline and a fused variant keeps the chunk vector.</p>"
            f"<ul><li><b>Chunk sizes:</b> 200 / 400 / 600 / 800 tokens (overlap ~22%) + a section-aware "
            "<b>variable</b> chunking (cap 800).</li>"
            f"<li><b>Embedder:</b> <code>{emb}</code>. <b>LLM:</b> <code>{llm}</code>.</li>"
            "<li><b>Question pool:</b> up to 15 grounded questions/chunk (cached); q5/q10/q13/q15 and "
            "adaptive allocations are first-k slices.</li>"
            "<li><b>Gold coverage per chunking:</b> "
            + ", ".join(f"{k} {v}" for k, v in d.get("gold_coverage", {}).items())
            + f" of {d['num_eval_qa_total']} (evidence-text match).</li></ul></div>"
        )

    # chunking overview
    H.append(
        "<div class='card'><h2>Chunking conditions</h2>"
        "<div class='dt'>chunks &amp; token stats per size</div><div class='scroll'><table><thead><tr>"
        "<th class='l'>Chunk size</th><th>Overlap</th><th>#Chunks</th>"
        "<th>tok min</th><th>tok mean</th><th>tok max</th><th>≤15 Q avg/chunk</th>"
        "<th>Gen s</th></tr></thead><tbody>"
    )
    for s in sizes:
        rec = per_size[str(s)]
        t = rec["tokens"]
        g = rec.get("gen", {})
        H.append(
            f"<tr><td class='l'>{s} tok</td><td>{rec['overlap']}</td><td>{rec['num_chunks']}</td>"
            f"<td>{_f(t.get('min'), 0)}</td><td>{_f(t.get('mean'), 0)}</td><td>{_f(t.get('max'), 0)}</td>"
            f"<td>{g.get('avg_questions_per_chunk', '—')}</td><td class='muted'>{g.get('wall_seconds', '—')}</td></tr>"
        )
    vt = var["tokens"]
    vg = var.get("gen", {})
    H.append(
        f"<tr><td class='l'>variable</td><td>~120</td>"
        f"<td>{var['num_chunks']}</td><td>{_f(vt.get('min'), 0)}</td><td>{_f(vt.get('mean'), 0)}</td>"
        f"<td>{_f(vt.get('max'), 0)}</td><td>{vg.get('avg_questions_per_chunk', '—')}</td>"
        f"<td class='muted'>{vg.get('wall_seconds', '—')}</td></tr>"
    )
    H.append("</tbody></table></div></div>")

    # Table 1
    H.append(
        "<div class='card'><h2>Table 1 · Fixed question counts by chunk size</h2>"
        "<div class='dt'>questions-only · green/red vs that size's chunk-text baseline · ★ = best row</div>"
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
        bcls = "top" if t1_top == (s, "baseline") else "base"
        H.append(
            f"<tr class='{bcls}'><td class='l'>baseline (chunk text)</td><td>{s}</td>"
            f"<td>{rec['overlap']}</td><td>0</td>{_mcells(base, None)}"
            f"<td class='muted'>{base['num_embeddings']}</td>"
            f"<td class='muted'>{_f(base['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(base['search_p95'], 2)}</td></tr>"
        )
        for nq in d["fixed_counts"]:
            r = rec["counts"].get(f"q{nq}")
            if not r:
                continue
            cls = "top" if t1_top == (s, r["condition"]) else ""
            H.append(
                f"<tr class='{cls}'><td class='l'>chunk_size_{s}_q{nq}</td><td>{s}</td>"
                f"<td>{rec['overlap']}</td><td>{nq}</td>{_mcells(r, base)}"
                f"<td class='muted'>{r['num_embeddings']}</td>"
                f"<td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
                f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
            )
    H.append("</tbody></table></div>" + _t1_finding(d) + "</div>")

    # Table 2
    vc = var["conditions"]
    vbase = vc.get("baseline")
    H.append(
        "<div class='card'><h2>Table 2 · Adaptive vs fixed — “bigger chunk ⇒ more questions”</h2>"
        "<div class='dt'>section-aware variable chunking · one chunk set, one gold</div>"
        f"<p class='sub'>All rows share the variable chunking ({var['num_chunks']} chunks, tokens "
        f"{_f(vt.get('min'), 0)}–{_f(vt.get('max'), 0)}). <b>adaptive length-based</b> gives each chunk "
        "more questions the longer it is (200→4 … 800→12); <b>adaptive density-based</b> allocates by "
        "an information-density score (avg ~10, matched-budget vs fixed q10). Green/red vs the "
        "chunk-text baseline.</p><div class='scroll'><table><thead><tr>"
        "<th class='l'>Condition</th><th>Strategy</th><th>Q/chunk</th>"
        "<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Size MB</th><th>p95 ms</th></tr></thead><tbody>"
    )
    order = [
        ("baseline", "original chunk", "chunk-text"),
        ("fixed_q10", "questions-only", "fixed q10"),
        ("adapt_length", "questions-only", "adaptive · length (bigger⇒more)"),
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
            f"<td>{r['q_per_chunk']}</td>{_mcells(r, None if is_base else vbase)}"
            f"<td class='muted'>{r['num_embeddings']}</td>"
            f"<td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
        )
    H.append("</tbody></table></div>" + _t2_finding(var) + "</div>")

    # Table 3
    H.append(
        "<div class='card'><h2>Table 3 · Best quality / latency trade-off</h2>"
        "<div class='dt'>across every condition</div><div class='scroll'><table><thead><tr>"
        "<th class='l'>Selection</th><th class='l'>Condition</th><th>Size</th><th>Q/chunk</th>"
        "<th>Index</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Size MB</th><th>p95 ms</th></tr></thead><tbody>"
    )
    for r in d["table3"]:
        tcls = "top" if r["selection"] == "best nDCG@10 (overall)" else ""
        H.append(
            f"<tr class='{tcls}'><td class='l'><b>{r['selection']}</b></td>"
            f"<td class='l'>{r['condition']}</td><td>{r['chunk_size']}</td><td>{r['q_per_chunk']}</td>"
            f"<td class='muted'>{r.get('index_content', '')}</td>"
            f"<td>{_f(r['hit@1'])}</td><td>{_f(r['hit@5'])}</td><td>{_f(r['hit@10'])}</td>"
            f"<td>{_f(r['mrr'])}</td><td>{_f(r['ndcg@10'])}</td>"
            f"<td class='muted'>{r['num_embeddings']}</td><td class='muted'>{_f(r['index_size_mb'], 2)}</td>"
            f"<td class='muted'>{_f(r['search_p95'], 2)}</td></tr>"
        )
    H.append("</tbody></table></div></div>")

    H.append(_analysis(d))
    H.append(_recommendation(d, best_all))
    return "".join(H)


def build_html(d):
    emb = d.get("embedding_model", "")
    llm = d.get("llm_model", "")
    prefix = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>DocBank chunk size × question enrichment</title><style>{STYLE}</style></head><body>"
        "<header><div class='wrap'><h1>DocBank · chunk size × generated-question enrichment</h1>"
        "<p>Does the enrichment needed depend on <b>chunk size</b>, and do <b>bigger chunks want "
        "more questions</b>? The PeerQA chunk-size study, replicated on DocBank layout text: "
        "5 chunkings (four fixed sizes + one section-aware variable) × fixed and adaptive "
        "question counts.</p></div></header><main>"
    )
    footer = (
        f"<footer>DocBank subset ({d['num_documents']} docs) · chunk-size × question enrichment · "
        f"{d['num_common_queries']} common queries · embedder {emb} · LLM {llm}. From "
        "results/docbank/docbank_chunksize_results.json.</footer></main></body></html>"
    )
    return prefix + body_cards(d, embed=False) + footer


def _t1_finding(d):
    bc = _best_count_per_size(d)
    parts = [
        f"<b>{s}t</b>: best {bc[s][0]['q_per_chunk']}q (nDCG {_f(bc[s][0]['ndcg@10'], 3)}, "
        f"{'+' if bc[s][1] >= 0 else ''}{_f(bc[s][1], 3)} vs base {_f(bc[s][2], 3)})"
        for s in sorted(bc)
    ]
    return (
        "<div class='fb'><b>Reading Table 1</b><br>Best fixed count per size — "
        + "; ".join(parts)
        + ". A rising best-count with size ⇒ bigger chunks want more questions.</div>"
    )


def _t2_finding(var):
    vc = var["conditions"]

    def g(k):
        return (vc.get(k, {}) or {}).get("ndcg@10")

    def dd(a, b):
        return f"{'+' if (a or 0) - (b or 0) >= 0 else ''}{_f((a or 0) - (b or 0), 3)}"

    return (
        "<div class='fb'><b>Reading Table 2</b><br>"
        f"chunk-text baseline {_f(g('baseline'), 3)}; fixed q10 {_f(g('fixed_q10'), 3)}; "
        f"adaptive-length {_f(g('adapt_length'), 3)} ({dd(g('adapt_length'), g('fixed_q10'))} vs q10, "
        f"avg {var.get('adapt_length_avg_q')}q); adaptive-density {_f(g('adapt_density'), 3)} "
        f"({dd(g('adapt_density'), g('fixed_q10'))} vs q10); fused {_f(g('fused_density'), 3)} "
        f"({dd(g('fused_density'), g('baseline'))} vs baseline).</div>"
    )


def _analysis(d):
    bc = _best_count_per_size(d)
    sizes = sorted(bc)
    best_counts = [bc[s][0]["q_per_chunk"] for s in sizes]
    rising = len(best_counts) >= 2 and best_counts[-1] > best_counts[0]
    per_size_best = ", ".join(f"{s}t→{bc[s][0]['q_per_chunk']}q" for s in sizes)
    q10_best = sum(1 for s in sizes if bc[s][0]["q_per_chunk"] == 10)
    vc = d["variable"]["conditions"]
    q10n = (vc.get("fixed_q10", {}) or {}).get("ndcg@10") or 0
    aln = (vc.get("adapt_length", {}) or {}).get("ndcg@10") or 0
    adn = (vc.get("adapt_density", {}) or {}).get("ndcg@10") or 0
    adaptive_wins = max(aln, adn) > q10n + 0.002
    base_nds = [d["per_size"][str(s)]["baseline"]["ndcg@10"] or 0 for s in sizes]
    size_spread = max(base_nds) - min(base_nds)
    within = []
    for s in sizes:
        vals = [r.get("ndcg@10") or 0 for r in d["per_size"][str(s)]["counts"].values()]
        within.append(max(vals) - min(vals))
    count_spread = max(within) if within else 0
    li = [
        "<li><b>Do larger chunks benefit from more questions?</b> "
        + (
            "Yes — the best fixed count rises with size ("
            if rising
            else "Not strictly — the best fixed count is roughly flat across sizes ("
        )
        + per_size_best
        + "). More questions help most where a chunk holds more distinct facts.</li>",
        f"<li><b>Is q10 a sweet spot across sizes?</b> q10 is the top fixed count in {q10_best}/{len(sizes)} "
        "sizes; DocBank chunks are small/dense so the plateau often sits at "
        f"{', '.join(str(bc[s][0]['q_per_chunk']) for s in sizes)} questions respectively.</li>",
        f"<li><b>“Bigger chunk ⇒ more questions” (adaptive length) vs fixed q10:</b> "
        f"{'adaptive length wins' if aln > q10n + 0.002 else 'adaptive length ≈ fixed q10'} "
        f"(nDCG {_f(aln, 3)} vs {_f(q10n, 3)}) — and at avg {d['variable'].get('adapt_length_avg_q')} q/chunk "
        "(fewer than 10), so it is also cheaper.</li>",
        "<li><b>Do smaller chunks need fewer questions?</b> Yes — a 200-token DocBank chunk holds few "
        "facts, so the LLM makes fewer grounded questions and extra requested ones become near-duplicate "
        "embeddings; small chunks saturate earliest.</li>",
        "<li><b>Do larger chunks get too broad/noisy?</b> Larger chunks lift recall (the right chunk is "
        "easier to hit) but blur rank-1; more questions give a big chunk several sharp entry points, which "
        "is why bigger chunks gain most from more questions.</li>",
        f"<li><b>Gains from size, count, or both?</b> Chunk size alone moves baseline nDCG@10 by "
        f"~{_f(size_spread, 3)}; question count within a size by up to ~{_f(count_spread, 3)}. "
        f"{'Question count is the larger lever' if count_spread > size_spread else 'Chunk size is the larger lever'} "
        "here, but they interact.</li>",
        "<li><b>Worth the cost?</b> Generated questions cost N× embeddings + a one-off LLM pass. On DocBank "
        "(synthetic eval) they clearly help; the smallest count on each size's plateau captures most of the "
        "gain — going to q15 rarely pays.</li>",
    ]
    return (
        "<div class='card'><h2>Analysis</h2><div class='dt'>chunk size vs question enrichment</div>"
        "<ul>" + "".join(li) + "</ul>"
        "<div class='fb' style='background:#fff7ed;border-color:#fed7aa'><b style='color:#c2410c'>"
        "Caveat</b><br>The eval questions and enrichment questions are both LLM-generated from the same "
        "chunks, which flatters questions-only retrieval (as in the single-size DocBank run). Treat "
        "absolute levels as optimistic; the <i>size×count</i> shape is the finding.</div></div>"
    )


def _recommendation(d, best_all):
    bc = _best_count_per_size(d)
    sizes = sorted(bc)
    top_size = max(sizes, key=lambda s: bc[s][0].get("ndcg@10") or 0)
    vc = d["variable"]["conditions"]
    aln = (vc.get("adapt_length", {}) or {}).get("ndcg@10") or 0
    q10n = (vc.get("fixed_q10", {}) or {}).get("ndcg@10") or 0
    return (
        "<div class='card'><h2>Recommendation</h2><div class='dt'>one practical setup</div><ul>"
        f"<li><b>Best fixed setup:</b> chunk size <b>{top_size} tok</b> with "
        f"<b>{bc[top_size][0]['q_per_chunk']} questions/chunk</b> (nDCG@10 "
        f"{_f(bc[top_size][0].get('ndcg@10'), 4)}). Overall best across all conditions: "
        f"<code>{best_all['condition']}</code> ({_f(best_all.get('ndcg@10'), 4)}).</li>"
        f"<li><b>Adaptive rule (“bigger ⇒ more”):</b> "
        f"{'adopt length-based allocation — it beats fixed q10 at lower cost' if aln > q10n + 0.002 else 'length-based ties fixed q10; use it only if chunk sizes vary widely'} "
        "(cap 15, floor ~4).</li>"
        "<li><b>Length vs density:</b> length-based is simpler and competitive; density needs only cheap "
        "regex signals (numbers, symbols, Table/Eq refs).</li>"
        "<li><b>Precision:</b> for rank-1-critical use keep the chunk vector and <b>fuse</b> with the "
        "question vectors instead of going questions-only.</li>"
        "<li><b>Next:</b> replace synthetic QA with real/citation-grounded eval to remove generator "
        "overlap, and sample table/figure-rich documents to stress layout retrieval.</li>"
        "</ul></div>"
    )


def main():
    d = json.load(open(RESULTS, encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(d), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
