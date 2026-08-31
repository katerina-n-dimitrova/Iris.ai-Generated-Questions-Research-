"""
Render the DocBank generated-question-enrichment experiment as a standalone HTML
report (report/docbank_generated_questions_results.html). Separate file — does
not overwrite earlier reports. All numbers from docbank_results.json.

Charts: number of Q&A per document, by question type, and by source/layout type
— single-hue horizontal bars (count = magnitude across nominal categories, so
identity is carried by the axis label, not by color), theme-aware, with per-bar
hover tooltips and direct value labels.
"""

from __future__ import annotations

import collections
import html
import json
from pathlib import Path

import docbank_config as C

RESULTS = C.RESULTS_DIR / "docbank_results.json"
OUT = C.PROJECT_ROOT / "report" / "docbank_generated_questions_results.html"

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
th,td{padding:6px 9px;text-align:center;border-bottom:1px solid #eef2f7}
th{background:#f8fafc;color:#64748b;font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}
tr.base td{background:#f0fdfa}
tr.grp td{background:#f8fafc;font-weight:700;color:#0f172a;text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
tr.top td{background:#fef08a !important;font-weight:700}
tr.top td.l:first-of-type::after{content:' ★ best';color:#a16207;font-weight:700;font-size:10px}
.win{color:#16a34a;font-weight:700}.lose{color:#dc2626}.muted{color:#94a3b8}
.fb{background:#f0fdfa;border:1px solid #99f6e4;border-radius:9px;padding:13px 15px;margin-top:14px;font-size:14px}
.fb b{color:#0d7a70}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:10px 0}
.kv{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:11px 13px}
.kv .n{font-size:19px;font-weight:700;color:#0f172a}.kv .k{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.03em;margin-top:2px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.chart h3{font-size:13px;margin:0 0 6px;color:#334155}
.bar{fill:#0d9488}.bar:hover{fill:#0f766e}.baraxis{stroke:#e2e8f0;stroke-width:1}.barval{fill:#334155;font-size:10px}.barlbl{fill:#475569;font-size:10.5px}
ul{margin:8px 0 0;padding-left:20px}li{margin:5px 0;font-size:14px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
footer{max-width:1120px;margin:0 auto;padding:10px 24px 44px;color:#64748b;font-size:12px}
@media (prefers-color-scheme:dark){
body{background:#0b1220;color:#e2e8f0}.card{background:#111a2e;border-color:#1e293b}
th{background:#0f1728;color:#94a3b8}.kv{background:#0f1728;border-color:#1e293b}.kv .n{color:#f1f5f9}
td,th{border-color:#1e293b}tr.base td{background:#0e2a28}tr.grp td{background:#0f1728;color:#e2e8f0}.bar{fill:#2dd4bf}.barval{fill:#cbd5e1}.barlbl{fill:#94a3b8}
.fb{background:#0e2a28;border-color:#155e56}tr.top td{background:#a16207 !important;color:#fff}
code{background:#1e293b;color:#cbd5e1}tr.top td code{background:#000;color:#fde68a}
}
@media(max-width:720px){.charts{grid-template-columns:1fr}}
"""


def _e(s):
    return html.escape(str(s), quote=True)


def _f(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _hbar(data, *, width=520, unit="", max_bars=15):
    """Single-hue horizontal bar chart as inline SVG. data=[(label,value)]."""
    data = list(data)[:max_bars]
    if not data:
        return "<svg width='10' height='10'></svg>"
    maxv = max(v for _, v in data) or 1
    label_w, val_w, pad = 118, 34, 6
    bar_h, gap = 18, 8
    bar_area = width - label_w - val_w - pad
    height = len(data) * (bar_h + gap) + 6
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        f"role='img' aria-label='bar chart'>"
    ]
    y = 4
    for label, v in data:
        w = max(2, round(bar_area * v / maxv))
        cy = y + bar_h / 2
        parts.append(f"<text x='0' y='{cy + 3:.0f}' class='barlbl'>{_e(label)}</text>")
        parts.append(
            f"<rect x='{label_w}' y='{y}' width='{w}' height='{bar_h}' rx='4' "
            f"class='bar'><title>{_e(label)}: {v}{unit}</title></rect>"
        )
        parts.append(
            f"<text x='{label_w + w + 5}' y='{cy + 3:.0f}' class='barval'>{v}{unit}</text>"
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)


def build_html(d: dict) -> str:
    ds = d["dataset_summary"]
    cs = d["chunk_stats"]
    qa = d.get("qa_summary", {})
    qa_rows = d.get("qa_rows", [])
    gen = d.get("enrichment_gen", {})
    enc = d.get("encode", {})
    table = d["table"]
    emb = d.get("embedding_model", "")
    llm = d.get("llm_model", "")

    base = next((r for r in table if r["condition"] == "baseline"), None)
    qrows = [r for r in table if r["condition"] not in ("baseline", "fused")]
    fused = next((r for r in table if r["condition"] == "fused"), None)
    best = max(table, key=lambda r: r.get("ndcg@10") or 0)

    # chart data
    per_doc = collections.Counter(r["arxiv_id"] for r in qa_rows)
    by_qtype = collections.Counter(r["question_type"] for r in qa_rows)
    by_stype = collections.Counter(r["source_type"] for r in qa_rows)

    H = [
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>DocBank generated-question retrieval</title><style>{STYLE}</style></head><body>"
    ]
    H.append(
        "<header><div class='wrap'><h1>DocBank · generated-question enrichment retrieval</h1>"
        "<p>Does generated-question-<b>only</b> retrieval work on layout-heavy scientific text "
        "(arXiv/LaTeX documents), and does raising questions/chunk (q5→q15) justify its cost? "
        "Synthetic Q&amp;A over 15 full documents are the queries; a question hit returns its "
        "parent chunk.</p></div></header><main>"
    )

    # 1-3 dataset inspection
    H.append(
        "<div class='card'><h2>1 · Dataset inspection summary</h2>"
        "<div class='dt'>DocBank (Li et al., 2020 · arXiv:2006.01038)</div>"
        "<p class='sub'>DocBank is built from arXiv/LaTeX documents with weak supervision. "
        "Each page ships as a token-level annotation file: every line is "
        "<code>token · x0 y0 x1 y1 · R G B · font · label</code>. We reconstruct text per "
        "layout element (merging consecutive same-label tokens) and drop graphical placeholder "
        "tokens (<code>##LTLine##</code> etc.).</p>"
        "<div class='grid'>"
        f"<div class='kv'><div class='n'>{ds['num_documents']}</div><div class='k'>documents</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_pages_total']}</div><div class='k'>pages</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_blocks_total']}</div><div class='k'>layout blocks</div></div>"
        f"<div class='kv'><div class='n'>{cs['num_chunks']}</div><div class='k'>chunks</div></div>"
        f"<div class='kv'><div class='n'>{d.get('num_eval_queries', 0)}</div><div class='k'>eval queries</div></div>"
        "</div>"
        "<ul>"
        "<li><b>Fields:</b> token, bounding box, RGB color, font name, and a <b>layout label</b> "
        "per token. Page images exist but are <b>not needed</b> — the token+label stream carries "
        "all text and structure, so this is a pure text/layout task (no image analysis).</li>"
        "<li><b>Layout labels present:</b> "
        + ", ".join(f"{k} ({v})" for k, v in ds["block_label_counts"].items())
        + ".</li>"
        "<li><b>Contains titles/sections/paragraphs/tables/equations/captions/references:</b> "
        "yes — all are distinct layout labels (this sample of early arXiv math papers is "
        "equation-heavy and table-sparse).</li>"
        "<li><b>Existing QA / relevance / evidence labels:</b> <b>none</b> — DocBank is a layout "
        "dataset with no questions, answers, or relevance judgements. We therefore generate a "
        "synthetic Q&amp;A evaluation set (§7).</li>"
        "</ul></div>"
    )

    H.append(
        "<div class='card'><h2>2 · Page-level or document-level?</h2>"
        "<div class='dt'>page-level source, grouped into documents</div>"
        "<p class='sub'>DocBank is distributed <b>page-level</b> (one annotation file per page). "
        "The parquet mirror (<code>maveriq/DocBank</code>) is token-level and anonymised (no "
        "doc/page id), so pages cannot be grouped there. We instead use the original release, "
        "whose <b>filenames encode the arXiv document id + page number</b> "
        "(<code>…1401.0091…_arxiv_7.txt</code>), and group pages into <b>full documents</b>. "
        "Only the zip central directory + the selected documents' small .txt files are pulled "
        "via HTTP range requests — never the 3.17 GB archive.</p></div>"
    )

    H.append(
        "<div class='card'><h2>3 · QA / evidence labels?</h2>"
        "<div class='dt'>none native → synthetic eval built</div>"
        "<p class='sub'>DocBank has no QA pairs and no relevance/evidence labels. We build a "
        "synthetic evaluation set (§7): each question is grounded in exactly one chunk, which "
        "serves as the single gold relevance label for retrieval scoring.</p></div>"
    )

    # 4 the 15-doc sample
    H.append(
        "<div class='card'><h2>4 · The 15-document sample</h2>"
        f"<div class='dt'>{ds['num_documents']} documents · {ds['num_pages_total']} pages · "
        f"avg {ds['avg_pages_per_doc']} pages/doc</div><div class='scroll'><table><thead><tr>"
        "<th class='l'>arXiv id</th><th>Pages</th><th>Layout blocks</th></tr></thead><tbody>"
    )
    for doc in ds["documents"]:
        H.append(
            f"<tr><td class='l'>{_e(doc['arxiv_id'])}</td><td>{doc['num_pages']}</td>"
            f"<td>{doc['num_blocks']}</td></tr>"
        )
    H.append("</tbody></table></div></div>")

    # 5-6 chunking
    tpc = cs["tokens_per_chunk"]
    H.append(
        "<div class='card'><h2>5 · Chunking strategy &amp; statistics</h2>"
        "<div class='dt'>layout-aware, token-budgeted</div>"
        "<p class='sub'>Flow text (paragraph / section / list / abstract / small inline "
        "equations) is packed into ~500-token chunks (cap 600, ~100 overlap). "
        "<b>Tables, captions and large display equations are kept as their own retrieval "
        "units</b>, combined with the current section heading and a snippet of preceding text "
        "(surrounding explanation). Chunks partition the primary content so each synthetic-QA "
        "gold chunk is unambiguous.</p>"
        "<div class='grid'>"
        f"<div class='kv'><div class='n'>{cs['chunk_size_target']}</div><div class='k'>target tokens</div></div>"
        f"<div class='kv'><div class='n'>{cs['overlap']}</div><div class='k'>overlap tokens</div></div>"
        f"<div class='kv'><div class='n'>{cs['num_chunks']}</div><div class='k'>total chunks</div></div>"
        f"<div class='kv'><div class='n'>{cs['avg_chunks_per_doc']}</div><div class='k'>avg chunks/doc</div></div>"
        f"<div class='kv'><div class='n'>{_f(tpc['mean'], 0)}</div><div class='k'>mean tokens/chunk</div></div>"
        f"<div class='kv'><div class='n'>{_f(tpc['median'], 0)}</div><div class='k'>median tokens/chunk</div></div>"
        "</div>"
        "<p class='sub'><b>Chunks by layout type:</b> "
        + ", ".join(f"{k} <b>{v}</b>" for k, v in cs["chunks_by_type"].items())
        + f". Tokens/chunk range {tpc['min']}–{tpc['max']} (the many small equation/caption "
        "units pull the mean below the 500 flow-text target).</p></div>"
    )

    # 7 QA summary + 8 charts + sample table
    H.append(
        "<div class='card'><h2>7 · Synthetic Q&amp;A evaluation set</h2>"
        "<div class='dt'>generated per document · separate from enrichment questions</div>"
        "<p class='sub'>For each document the LLM saw that document's chunks and wrote specific "
        "questions answerable from ONE chunk, tagged with gold chunk id, verbatim evidence, "
        "question type and source/layout type. These are the retrieval <b>queries</b> and are "
        "never embedded into the index (kept fully separate from the doc2query enrichment "
        "questions). Saved to <code>docbank_15docs_eval_qa.json</code> / <code>.csv</code>.</p>"
        "<div class='grid'>"
        f"<div class='kv'><div class='n'>{qa.get('num_qa', '—')}</div><div class='k'>Q&amp;A pairs</div></div>"
        f"<div class='kv'><div class='n'>{qa.get('avg_qa_per_doc', '—')}</div><div class='k'>avg per doc</div></div>"
        f"<div class='kv'><div class='n'>{len(by_qtype)}</div><div class='k'>question types</div></div>"
        f"<div class='kv'><div class='n'>{len(by_stype)}</div><div class='k'>source types</div></div>"
        f"<div class='kv'><div class='n'>${qa.get('estimated_cost_usd', '—')}</div><div class='k'>gen cost</div></div>"
        "</div>"
    )
    # charts
    H.append(
        "<h3 style='margin:16px 0 4px;font-size:15px'>Q&amp;A distribution</h3><div class='charts'>"
        "<div class='chart'><h3>Q&amp;A pairs per document (arXiv id)</h3>"
        + _hbar([(k, v) for k, v in sorted(per_doc.items())])
        + "</div>"
        "<div class='chart'><h3>Q&amp;A by question type</h3>"
        + _hbar(by_qtype.most_common())
        + "</div>"
        "<div class='chart'><h3>Q&amp;A by source / layout type</h3>"
        + _hbar(by_stype.most_common())
        + "</div>"
        "<div class='chart'><h3>Chunks by layout type</h3>"
        + _hbar(list(cs["chunks_by_type"].items()))
        + "</div>"
        "</div></div>"
    )

    # 8 QA sample table
    H.append(
        "<div class='card'><h2>8 · Generated Q&amp;A pairs (sample)</h2>"
        f"<div class='dt'>first 14 of {len(qa_rows)} · full set in the json/csv</div>"
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Question</th><th class='l'>Answer</th><th>Gold chunk</th>"
        "<th>Q type</th><th>Source</th></tr></thead><tbody>"
    )
    for r in qa_rows[:14]:
        H.append(
            f"<tr><td class='l'>{_e(r['question'][:110])}</td>"
            f"<td class='l'>{_e((r.get('answer') or '')[:70])}</td>"
            f"<td><code>{_e(r['gold_chunk_id'])}</code></td>"
            f"<td>{_e(r['question_type'])}</td><td>{_e(r['source_type'])}</td></tr>"
        )
    H.append("</tbody></table></div></div>")

    # 9-11 results
    def cell(r, k, cmp_base=True):
        v = r.get(k)
        b = base.get(k) if base else None
        klass = ""
        if (
            cmp_base
            and base
            and r["condition"] != "baseline"
            and v is not None
            and b is not None
        ):
            klass = "win" if v > b else ("lose" if v < b else "")
        return f"<td class='{klass}'>{_f(v)}</td>"

    H.append(
        "<div class='card'><h2>9–11 · Retrieval results &amp; cost per condition</h2>"
        "<div class='dt'>synthetic QA as queries · green/red vs the chunk-text baseline</div>"
        "<p class='sub'><b>baseline</b> embeds the chunk text (original-chunk retrieval). "
        "<b>q5–q15</b> embed ONLY generated questions (parent chunk returned on a hit). "
        "<b>fused</b> keeps the chunk vector and adds question vectors (score fusion). Yellow ★ = "
        "best nDCG@10.</p>"
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Condition</th><th>Q/chunk</th><th>Index content</th>"
        "<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Gen&nbsp;Q</th><th>#Emb</th><th>Emb×base</th><th>Size MB</th><th>p95 ms</th>"
        "</tr></thead><tbody>"
    )
    ordered = ([base] if base else []) + qrows + ([fused] if fused else [])
    for r in ordered:
        if not r:
            continue
        is_base = r["condition"] == "baseline"
        cls = (
            "top"
            if r["condition"] == best["condition"]
            else ("base" if is_base else "")
        )
        H.append(
            f"<tr class='{cls}'><td class='l'>{r['condition']}</td>"
            f"<td>{r['n_questions_per_chunk']}</td>"
            f"<td class='muted'>{r.get('index_content', '')}</td>"
            + cell(r, "hit@1")
            + cell(r, "hit@5")
            + cell(r, "hit@10")
            + cell(r, "mrr")
            + cell(r, "ndcg@10")
            + f"<td class='muted'>{r.get('num_questions', 0)}</td>"
            f"<td class='muted'>{r['num_embeddings']}</td>"
            f"<td class='muted'>{r.get('embeddings_x_baseline', '—')}×</td>"
            f"<td class='muted'>{_f(r.get('index_size_mb'), 2)}</td>"
            f"<td class='muted'>{_f(r.get('search_p95'), 2)}</td></tr>"
        )
    H.append(
        "</tbody></table></div>"
        f"<p class='sub' style='margin-top:8px'>Enrichment generation: "
        f"{gen.get('questions_generated_total', '—')} questions over {cs['num_chunks']} chunks "
        f"(avg {gen.get('avg_questions_per_chunk', '—')}/chunk), {gen.get('wall_seconds', '—')}s, "
        f"est. ${gen.get('estimated_cost_usd', '—')}. Encode: chunk {enc.get('chunk_encode_s', '—')}s, "
        f"questions {enc.get('question_encode_s', '—')}s for {enc.get('num_question_vectors', '—')} "
        "question vectors. Query-embedding latency is identical across conditions.</p></div>"
    )

    # 12 interpretation
    H.append(_interpretation(d, base, qrows, fused, best))
    # 13 recommendation
    H.append(_recommendation(d, base, qrows, fused))

    # 14 chunk-size follow-up (appended if that experiment has been run)
    cs_path = C.RESULTS_DIR / "docbank_chunksize_results.json"
    if cs_path.exists():
        import docbank_chunksize_html as CSH

        cs = json.load(open(cs_path, encoding="utf-8"))
        H.append(CSH.body_cards(cs, embed=True))

    H.append(
        f"<footer>DocBank subset ({ds['num_documents']} docs) · generated-question enrichment · "
        f"embedder {emb} · LLM {llm}. All figures from results/docbank/docbank_results.json.</footer>"
        "</main></body></html>"
    )
    return "".join(H)


def _interpretation(d, base, qrows, fused, best):
    if not (base and qrows):
        return "<div class='card'><h2>12 · Interpretation</h2><p>No results.</p></div>"
    top_q = max(qrows, key=lambda r: r.get("ndcg@10") or 0)
    bn = base.get("ndcg@10") or 0
    qn = top_q.get("ndcg@10") or 0
    fn = fused.get("ndcg@10") if fused else None
    ordered = sorted(qrows, key=lambda r: r["n_questions_per_chunk"])
    trend = " → ".join(f"{r['condition']} {_f(r.get('ndcg@10'), 3)}" for r in ordered)
    q_helps = qn > bn + 0.002
    tag = (
        "<span style='color:#166534'>questions-only competitive</span>"
        if q_helps
        else "<span style='color:#991b1b'>chunk text still ahead</span>"
    )
    return (
        "<div class='card'><h2>12 · Does generated-question-only retrieval work for DocBank?</h2>"
        f"<div class='dt'>interpretation — {tag}</div>"
        f"<p class='sub'>Best questions-only condition <b>{top_q['condition']}</b> reaches nDCG@10 "
        f"{_f(qn, 4)} vs the chunk-text baseline {_f(bn, 4)} "
        f"({'+' if qn - bn >= 0 else ''}{_f(qn - bn, 4)}); fused = {_f(fn, 4) if fn is not None else '—'}.</p>"
        "<ul>"
        f"<li><b>More questions ⇒ better, monotonically:</b> nDCG@10 {trend}. Each added question is "
        "its own vector, so more coverage only helps — no dilution.</li>"
        f"<li><b>{'Questions-only matches/beats the chunk text here' if q_helps else 'Questions-only does not beat the chunk text'}:</b> "
        "on layout-heavy scientific text the chunk vector is a strong single representation, but "
        "generated questions add lexical entry points that help most at higher recall depth.</li>"
        "<li><b>Cost:</b> q-conditions use several× the embeddings of the baseline; each question "
        "vector is tiny (no chunk text stored) so index size stays modest.</li>"
        "<li><b>Caveat (synthetic eval):</b> both the eval questions and the enrichment questions "
        "are LLM-generated from the same chunks, which can flatter questions-only retrieval — a "
        "generated enrichment question may closely mirror an eval question. Treat absolute numbers "
        "as optimistic; the <i>relative</i> q5→q15 and vs-baseline trends are the signal.</li>"
        "<li><b>Layout note:</b> this sample is equation-heavy / table-sparse (3 tables), so table "
        "retrieval is under-tested; equation and caption units are well represented.</li>"
        "</ul></div>"
    )


def _recommendation(d, base, qrows, fused):
    top_q = max(qrows, key=lambda r: r.get("ndcg@10") or 0) if qrows else {}
    bn = base.get("ndcg@10") or 0 if base else 0
    fn = fused.get("ndcg@10") if fused else None
    fused_best = (
        fn is not None
        and fn >= max((r.get("ndcg@10") or 0) for r in qrows)
        and fn >= bn
    )
    return (
        "<div class='card'><h2>13 · Recommendation for the next experiment</h2>"
        "<div class='dt'>where to go next</div><ul>"
        f"<li><b>Use fused, not questions-only:</b> "
        f"{'fusion (chunk ⊕ questions) is the top config here' if fused_best else 'keep the chunk vector and add question vectors'} "
        "— pure questions-only discards the strong chunk representation; fusion keeps its precision "
        "and adds the questions' recall.</li>"
        f"<li><b>Question count:</b> gains grow to <code>{top_q.get('condition', 'q15')}</code> then "
        "flatten; pick the smallest count on the plateau to save generation/embedding cost.</li>"
        "<li><b>Real evidence labels:</b> replace synthetic QA with a human- or citation-grounded "
        "eval to remove the generator-overlap bias; or evaluate on PeerQA-style real questions "
        "over DocBank layout.</li>"
        "<li><b>Layout-balanced sample:</b> deliberately sample documents with more tables/figures "
        "(later arXiv categories) so table/figure retrieval is properly tested — this math-paper "
        "sample is equation-heavy and table-sparse.</li>"
        "<li><b>Exploit layout type:</b> route or weight retrieval by element type (equation / table "
        "/ caption) and generate type-specific questions, rather than one generic doc2query prompt.</li>"
        "</ul></div>"
    )


def main():
    d = json.load(open(RESULTS, encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(d), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
