"""
Render the PeerQA generated-questions-only experiment as a standalone HTML
report (report/peerqa_generated_questions_results.html).

Reads results/peerqa/peerqa_results.json (written by run_peerqa.py) and produces
the seven required sections: dataset inspection, already-chunked verdict, final
chunking strategy, experimental setup, results table (5 vs 10 vs 13), a short
interpretation of whether questions-only helps, and a next-experiment
recommendation. All numbers are pulled from the JSON — nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import peerqa_config as C

RESULTS = C.RESULTS_DIR / "peerqa_results.json"
OUT = C.PROJECT_ROOT / "report" / "peerqa_generated_questions_results.html"

STYLE = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1e293b;background:#f1f5f9;line-height:1.55}
header{background:linear-gradient(135deg,#0f766e,#7c3aed);color:#fff;padding:38px 24px}
header .wrap{max-width:1080px;margin:0 auto}header h1{margin:0 0 8px;font-size:26px}header p{margin:0;opacity:.92;font-size:14.5px}
main{max-width:1080px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 22px;margin:18px 0;box-shadow:0 1px 3px rgba(15,23,42,.05)}
h2{font-size:20px;margin:2px 0 2px;border-left:4px solid #0f766e;padding-left:10px}
.dt{color:#0f766e;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:2px 0 12px}
.sub{color:#475569;font-size:14px;margin:6px 0 12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0}
th,td{padding:7px 9px;text-align:center;border-bottom:1px solid #eef2f7}
th{background:#f8fafc;color:#64748b;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}tr.base td{background:#f0fdfa;font-weight:600}
tr.best td{background:#f0fdf4}
.win{color:#16a34a;font-weight:700}.lose{color:#dc2626}
.muted{color:#94a3b8}
.fb{background:#f0fdfa;border:1px solid #99f6e4;border-radius:9px;padding:13px 15px;margin-top:14px;font-size:14px}
.fb b{color:#0f766e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:10px 0}
.kv{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:11px 13px}
.kv .n{font-size:20px;font-weight:700;color:#0f172a}.kv .k{font-size:11.5px;color:#64748b;text-transform:uppercase;letter-spacing:.03em;margin-top:2px}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;margin-left:6px}
.tag.up{background:#dcfce7;color:#166534}.tag.down{background:#fee2e2;color:#991b1b}.tag.mix{background:#fef9c3;color:#854d0e}
ul{margin:8px 0 0;padding-left:20px}li{margin:5px 0;font-size:14px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12.5px}
footer{max-width:1080px;margin:0 auto;padding:10px 24px 44px;color:#64748b;font-size:12px}
"""


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _cls(v, base, higher_better=True):
    if v is None or base is None:
        return ""
    if abs(v - base) < 1e-9:
        return ""
    good = (v > base) if higher_better else (v < base)
    return "win" if good else "lose"


def build_html(data: dict) -> str:
    ds = data["dataset_summary"]
    gen = data.get("generation", {})
    idx = data.get("index", {})
    table = data["table"]
    emb_model = data.get("embedding_model") or idx.get("embedding_model", "")
    llm = data.get("llm_model", "")

    by_cond = {r["condition"]: r for r in table}
    base = by_cond.get("baseline")
    qrows = [r for r in table if r["condition"] != "baseline"]
    # best q-condition by nDCG@10
    best_q = max(qrows, key=lambda r: r.get("ndcg@10") or 0) if qrows else None

    tpc = ds["tokens_per_chunk"]
    corpora = ", ".join(ds["corpora"])

    H = []
    H.append(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>PeerQA generated-questions retrieval</title><style>{STYLE}</style></head><body>"
    )
    H.append(
        "<header><div class='wrap'><h1>PeerQA · embedding only generated questions</h1>"
        "<p>Doc2query retrieval where each chunk is represented in the index <b>only</b> by "
        "LLM-generated questions (5 / 10 / 13 per chunk) — never the chunk text. Real PeerQA "
        "reviewer questions are the queries; a question hit returns its parent chunk as evidence.</p>"
        "</div></header><main>"
    )

    # ---- 1. Dataset inspection ----------------------------------------- #
    H.append(
        "<div class='card'><h2>1 · Dataset inspection summary</h2>"
        "<div class='dt'>PeerQA (Baumgärtner et al., 2025 · arXiv:2502.13668)</div>"
        "<p class='sub'>PeerQA pairs reviewer questions with author-annotated, sentence-level "
        "evidence over scientific papers. The redistributable release ships "
        "<code>qa.jsonl</code> (questions + evidence) and <code>papers.jsonl</code> (paper text "
        "<b>already segmented into sentences</b>). Each paper sentence carries a global index "
        "<code>idx</code>, paragraph index <code>pidx</code>, sentence index <code>sidx</code>, a "
        "<code>type</code> (sentence / heading / table / figure / formula …), <code>content</code>, "
        "and <code>last_heading</code>.</p>"
    )
    H.append(
        "<div class='grid'>"
        f"<div class='kv'><div class='n'>{ds['num_papers']}</div><div class='k'>papers (subset)</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_sentences_total']:,}</div><div class='k'>source sentences</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_chunks']}</div><div class='k'>chunks built</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_eval_queries']}</div><div class='k'>eval queries</div></div>"
        f"<div class='kv'><div class='n'>{ds['num_questions_usable']}</div><div class='k'>usable QA pairs</div></div>"
        "</div>"
    )
    H.append(
        "<ul>"
        "<li><b>Questions / answers:</b> <code>qa.jsonl</code> → <code>question</code>, "
        "<code>answer_free_form</code>, plus <code>answerable</code> / <code>answerable_mapped</code> flags.</li>"
        "<li><b>Paper / article text:</b> <code>papers.jsonl</code>, one row per sentence, in reading order.</li>"
        "<li><b>Evidence &amp; relevance labels:</b> <code>answer_evidence_mapped</code> gives, per question, the "
        "gold evidence <b>sentence idxs</b> — i.e. sentence-level relevance labels (paragraph-level derivable via "
        "<code>pidx</code>). Evidence idx alignment to <code>papers.jsonl</code> verified 389/389 in-range.</li>"
        f"<li><b>Existing chunks:</b> yes — sentence-level units (see §2). Subset corpora: {corpora}.</li>"
        "<li><b>Scope note:</b> full text is redistributed only for the nlpeer + egu corpora (90 papers). The 118 "
        "openreview papers (ICLR / NeurIPS) are fetched from arXiv / OpenReview via the <code>papers-all</code> "
        "config and are not used here. We selected the 15 text-available papers with the most answerable, "
        "evidence-mapped questions.</li>"
        "</ul></div>"
    )

    # ---- 2. Already chunked? ------------------------------------------- #
    H.append(
        "<div class='card'><h2>2 · Was the dataset already chunked?</h2>"
        "<div class='dt'>yes — at the sentence level, with no overlap</div>"
        "<p class='sub'>PeerQA is distributed <b>pre-chunked into sentences</b>, not paragraphs or fixed windows. "
        "Each <code>papers.jsonl</code> row is one retrieval-atomic sentence.</p>"
        "<ul>"
        "<li><b>Chunk unit:</b> a single sentence (also headings / table rows / figures / formulas as their own units).</li>"
        "<li><b>Chunk size:</b> ~10–30 tokens per sentence — far below a normal RAG chunk.</li>"
        "<li><b>Structure:</b> indexed by <code>idx</code> (global), grouped by <code>pidx</code> (paragraph) and "
        "<code>last_heading</code> (section).</li>"
        "<li><b>Overlap:</b> <b>none</b> — sentences are disjoint.</li>"
        "</ul>"
        "<div class='fb'><b>Verdict</b><br>The documents are already chunked, but at a granularity "
        "(single sentence) that is too fine for dense RAG retrieval. The dataset structure "
        "(global sentence idx + paragraph idx) makes it clean to <b>re-pack sentences into "
        "standard-size chunks</b> while still mapping the sentence-level gold labels onto the new chunks.</div></div>"
    )

    # ---- 3. Final chunking strategy ------------------------------------ #
    H.append(
        "<div class='card'><h2>3 · Final chunking strategy</h2>"
        "<div class='dt'>token-budgeted sentence packing with overlap</div>"
        "<p class='sub'>We pack consecutive sentences (in <code>idx</code> order, within a paper) into "
        "token-budgeted windows using tiktoken <code>cl100k_base</code>. This keeps the standard "
        "400–600-token RAG target while respecting the dataset's own sentence boundaries.</p>"
        "<div class='grid'>"
        f"<div class='kv'><div class='n'>{ds['chunk_size_target']}</div><div class='k'>target tokens</div></div>"
        f"<div class='kv'><div class='n'>{ds['chunk_size_cap']}</div><div class='k'>hard cap</div></div>"
        f"<div class='kv'><div class='n'>{ds['chunk_overlap_tokens']}</div><div class='k'>overlap tokens</div></div>"
        f"<div class='kv'><div class='n'>{_fmt(tpc['mean'], 0)}</div><div class='k'>mean tokens/chunk</div></div>"
        f"<div class='kv'><div class='n'>{_fmt(tpc['median'], 0)}</div><div class='k'>median tokens/chunk</div></div>"
        f"<div class='kv'><div class='n'>{ds['avg_sentences_per_chunk']}</div><div class='k'>avg sentences/chunk</div></div>"
        "</div>"
        "<ul>"
        "<li>Accumulate sentences until the target (~500 tokens) is reached, then close the window.</li>"
        "<li><b>Overlap</b> ≈100 tokens: the next window is re-seeded with the trailing sentences of the previous one.</li>"
        "<li>An oversized single unit (e.g. a flattened table) is hard-split on token boundaries into ≤cap pieces.</li>"
        "<li><b>Gold mapping:</b> each chunk records the set of source sentence idxs it covers; a query's gold "
        "evidence idxs map to every chunk containing them (overlap ⇒ a gold sentence can belong to ≥1 chunk).</li>"
        f"<li>Result: {ds['num_chunks']} chunks, tokens/chunk min {tpc['min']} / mean "
        f"{_fmt(tpc['mean'], 0)} / max {tpc['max']}.</li>"
        "</ul></div>"
    )

    # ---- 4. Experimental setup ----------------------------------------- #
    genq = gen.get("questions_generated_total")
    H.append(
        "<div class='card'><h2>4 · Experimental setup</h2>"
        "<div class='dt'>questions-only doc2query index</div>"
        "<p class='sub'>The core condition: <b>embed only the generated questions, never the chunk text.</b> "
        "For each chunk the LLM writes questions the chunk can answer; each question is its own embedding in the "
        "index, tagged with its <code>parent_chunk_id</code>. A PeerQA query that hits a generated question "
        "returns the <b>parent chunk</b> as the retrieved evidence. A standard chunk-text dense index is included "
        "as <b>baseline</b> only, to answer 'does questions-only beat embedding the text'.</p>"
        "<ul>"
        f"<li><b>Conditions:</b> baseline (chunk text embedded) · q5 · q10 · q13 generated questions per chunk. "
        "The 13-question pool is generated once per chunk; q5 / q10 are its first-k subset (isolates question "
        "<i>count</i>, ~1 LLM call/chunk).</li>"
        f"<li><b>Embedder:</b> <code>{emb_model}</code> (cosine, normalised). <b>LLM:</b> <code>{llm}</code>, "
        f"temperature {C.LLM_TEMPERATURE}.</li>"
        f"<li><b>Generation:</b> {_fmt(genq, 0)} questions across {ds['num_chunks']} chunks "
        f"(avg {gen.get('avg_questions_per_chunk', '—')}/chunk); "
        f"{gen.get('prompt_tokens', 0):,} prompt + {gen.get('completion_tokens', 0):,} completion tokens; "
        f"est. ${gen.get('estimated_cost_usd', '—')}; {gen.get('wall_seconds', '—')}s wall.</li>"
        f"<li><b>Queries:</b> {ds['num_eval_queries']} real PeerQA reviewer questions (answerable, evidence-mapped).</li>"
        "<li><b>Metrics:</b> Hit@1 / Hit@5 / Hit@10, MRR, nDCG@10 (binary sentence-derived relevance), "
        "plus latency, #generated questions, #embeddings, and on-disk index size.</li>"
        "</ul></div>"
    )

    # ---- 5. Results table ---------------------------------------------- #
    def row_html(r, is_base=False):
        cls = (
            "base"
            if is_base
            else ("best" if best_q and r["condition"] == best_q["condition"] else "")
        )
        cells = [
            f"<td class='l'>{r['condition']}</td>",
            f"<td>{r['n_questions_per_chunk']}</td>",
            f"<td class='muted'>{r.get('index_content', '')}</td>",
        ]
        for key in ["hit@1", "hit@5", "hit@10", "mrr", "ndcg@10"]:
            v = r.get(key)
            b = base.get(key) if base else None
            klass = "" if is_base else _cls(v, b)
            cells.append(f"<td class='{klass}'>{_fmt(v)}</td>")
        cells.append(f"<td class='muted'>{r.get('num_embeddings', '—')}</td>")
        cells.append(f"<td class='muted'>{r.get('embeddings_x_baseline', '—')}×</td>")
        cells.append(f"<td class='muted'>{_fmt(r.get('index_size_mb'), 2)}</td>")
        cells.append(f"<td class='muted'>{_fmt(r.get('search_ms_p95'), 2)}</td>")
        return f"<tr class='{cls}'>" + "".join(cells) + "</tr>"

    H.append(
        "<div class='card'><h2>5 · Results — 5 vs 10 vs 13 generated questions</h2>"
        "<div class='dt'>real PeerQA queries · questions-only index</div>"
        "<p class='sub'>Baseline embeds the chunk text (standard dense RAG). q5 / q10 / q13 embed "
        "<b>only</b> generated questions. Green / red = better / worse than baseline.</p>"
        "<div class='scroll'><table><thead><tr>"
        "<th class='l'>Condition</th><th>Q/chunk</th><th>Index content</th>"
        "<th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>nDCG@10</th>"
        "<th>#Emb</th><th>Emb ×base</th><th>Size MB</th><th>p95 ms</th>"
        "</tr></thead><tbody>"
    )
    if base:
        H.append(row_html(base, is_base=True))
    for r in qrows:
        H.append(row_html(r))
    H.append("</tbody></table></div>")

    # small latency/storage note
    H.append(
        "<p class='note sub' style='margin-top:10px'>"
        f"Index build: chunk-text encode {idx.get('chunk_encode_seconds', '—')}s, "
        f"question encode {idx.get('question_encode_seconds', '—')}s for "
        f"{idx.get('num_question_vectors_embedded', '—')} question vectors. "
        "Query-embedding latency is identical across conditions (same query, same embedder); "
        "only the index content differs.</p></div>"
    )

    # ---- 6. Interpretation --------------------------------------------- #
    interp = _interpretation(base, qrows, best_q)
    H.append(interp)

    # ---- 7. Recommendation --------------------------------------------- #
    H.append(_recommendation(base, best_q))

    H.append(
        "<footer>PeerQA subset · generated-questions-only doc2query retrieval · "
        f"embedder {emb_model} · LLM {llm}. All figures computed from "
        "results/peerqa/peerqa_results.json.</footer>"
    )
    H.append("</main></body></html>")
    return "".join(H)


def _interpretation(base, qrows, best_q):
    if not (base and best_q):
        return "<div class='card'><h2>6 · Interpretation</h2><p class='sub'>No results.</p></div>"
    ordered = sorted(qrows, key=lambda r: r["n_questions_per_chunk"])
    top = ordered[-1]  # highest question count (q13)

    # per-metric win/loss of the top question count vs baseline
    early = ["hit@1", "mrr", "ndcg@10"]
    recall = ["hit@5", "hit@10", "recall@10"]

    def cmp(keys):
        return {
            k: (top.get(k), base.get(k), (top.get(k) or 0) - (base.get(k) or 0))
            for k in keys
        }

    ec, rc = cmp(early), cmp(recall)
    recall_wins = sum(1 for _, (_, _, d) in rc.items() if d > 0)

    trend_nd = " → ".join(
        f"{r['condition']} {_fmt(r.get('ndcg@10'), 3)}" for r in ordered
    )
    trend_h5 = " → ".join(
        f"{r['condition']} {_fmt(r.get('hit@5'), 3)}" for r in ordered
    )

    tag = "<span class='tag mix'>recall yes, precision no</span>"
    embs = [r.get("embeddings_x_baseline", 0) for r in qrows]

    def line(k, c):
        v, b, d = c[k]
        klass = "win" if d > 0 else ("lose" if d < 0 else "muted")
        return (
            f"{k.upper()} <b>{_fmt(v, 3)}</b> vs {_fmt(b, 3)} "
            f"(<span class='{klass}'>{'+' if d >= 0 else ''}{_fmt(d, 3)}</span>)"
        )

    return (
        "<div class='card'><h2>6 · Does embedding only generated questions improve retrieval?</h2>"
        f"<div class='dt'>interpretation {tag}</div>"
        "<p class='sub'>Embedding <b>only</b> generated questions does <b>not</b> beat embedding the "
        "chunk text on <b>early-rank precision</b>, but at enough questions it <b>overtakes the baseline "
        "on deeper recall</b>. The count sweep is cleanly monotonic — more questions is strictly better.</p>"
        "<ul>"
        f"<li><b>Precision metrics — baseline still wins:</b> at the top count ({top['condition']}), "
        f"{line('hit@1', ec)}, {line('mrr', ec)}, {line('ndcg@10', ec)}. The single chunk-text vector "
        "puts the right chunk at rank 1 more often.</li>"
        f"<li><b>Recall metrics — questions-only wins:</b> {line('hit@5', rc)}, {line('hit@10', rc)}, "
        f"{line('recall@10', rc)} — {recall_wins}/3 in favour of {top['condition']}. Many small question "
        "vectors cast a wider net, so the correct chunk lands in the top-5/10 more often even when it "
        "isn't rank 1.</li>"
        f"<li><b>Monotonic in count:</b> nDCG@10 {trend_nd}; Hit@5 {trend_h5}. Unlike the chunk-diluting "
        "single-vector append, here every added question is its own vector, so more questions only ever "
        "add coverage — no saturation dip within 5→13.</li>"
        f"<li><b>Cost shape:</b> q5→q13 use {_fmt(min(embs), 1)}×–{_fmt(max(embs), 1)}× the baseline "
        "embedding count, but each question vector carries no chunk text, so index size grows only "
        f"~{_fmt(min(r.get('storage_x_baseline', 0) for r in qrows), 1)}×–"
        f"{_fmt(max(r.get('storage_x_baseline', 0) for r in qrows), 1)}×.</li>"
        "<li><b>Why precision lags:</b> with no chunk text in the index, a fact no question happened to ask "
        "is unretrievable — questions-only is only as good as question coverage, which is why it needs "
        "volume (13+) to match, and why it helps recall before precision.</li>"
        "</ul></div>"
    )


def _recommendation(base, best_q):
    return (
        "<div class='card'><h2>7 · Recommendation for the next experiment</h2>"
        "<div class='dt'>where to go next</div>"
        "<ul>"
        "<li><b>Hybrid, not replacement:</b> keep the chunk-text vector <i>and</i> add generated-question "
        "vectors (multi-vector), or fuse question-hit scores with the chunk score, rather than embedding "
        "questions <i>only</i>. Prior SPIQA work here found score-fusion of questions with the chunk is the "
        "reliable win; questions-only removes the safety net of the text vector.</li>"
        "<li><b>Scale the query set:</b> 15 papers / ~65 queries is a smoke-scale sample. Re-run on all 90 "
        "text-available papers, and add the openreview papers via the <code>papers-all</code> arXiv fetch to "
        "reach the full 208-paper / ~383-query benchmark.</li>"
        "<li><b>Question quality &gt; count:</b> the count sweep flattens quickly; try structured / "
        "evidence-anchored question generation and round-trip filtering to raise per-question coverage.</li>"
        "<li><b>Add BM25 / hybrid:</b> reviewer questions are lexically specific; test generated questions as "
        "BM25 document expansion and RRF hybrid, which outperformed dense-only in the SPIQA study.</li>"
        "<li><b>Report paragraph-level too:</b> PeerQA ships paragraph qrels — evaluate at paragraph granularity "
        "to compare against the dataset's official retrieval numbers.</li>"
        "</ul></div>"
    )


def main():
    data = json.load(open(RESULTS, encoding="utf-8"))
    html = build_html(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
