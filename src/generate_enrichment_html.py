"""
Generate report/enrichment_results.html — a focused HTML summary of the THREE
enrichment methods (+ combined) tested per dataset, one table per dataset, with
baseline vs enrichment results and a short feedback note explaining why each
dataset's enrichment is (or isn't) beneficial.

Reads the by-method CSVs in results/enrichment_method_tests/. Data-driven, so
re-running the experiment then this script keeps the HTML in sync.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

import config

DIR = config.RESULTS_DIR / "enrichment_method_tests"
OUT = config.PROJECT_ROOT / "report" / "enrichment_results.html"

DATA_TYPE = {
    "scifact": "Structured scientific text",
    "nfcorpus": "Unstructured biomedical text",
    "wikitablequestions": "Tables",
    "chartqa": "Charts / graphs",
    "formulareasoning": "Mathematical formulas",
}
# human-readable method labels
METHOD_LABEL = {
    "baseline": "Baseline (raw content)",
    "title_abstract_context": "M1 · Title + abstract context",
    "neighboring_context": "M2 · Neighboring sentence summary",
    "llm_generated_chunk_context": "M3 · LLM-generated chunk context",
    "generated_questions": "M1 · Generated questions",
    "keywords_entities": "M2 · Keywords / entities",
    "plain_summary": "M3 · Plain-language summary",
    "column_headers_per_row": "M1 · Column headers per row",
    "table_page_title": "M2 · Table / page title",
    "natural_language_row_summary": "M3 · Natural-language row summary",
    "chart_to_table_data": "M1 · Chart-to-table data*",
    "axis_legend_title_metadata": "M2 · Axis/legend/title metadata*",
    "chart_summary": "M3 · Chart summary",
    "surrounding_text": "M1 · Surrounding text*",
    "variable_definitions": "M2 · Variable definitions",
    "latex_structure": "M3 · LaTeX + structure",
    "combined_best": "Combined best",
}
# why each data type's enrichment tends to help or not (domain rationale)
RATIONALE = {
    "scifact": ("Scientific sentences are already self-contained, so the baseline is "
                "strong; light document context helps ranking a little, but heavier "
                "enrichment mostly adds noise that lowers answer faithfulness."),
    "nfcorpus": ("Long biomedical passages are vague on their own; expansion methods "
                 "(generated questions, plain summaries) tend to help answer grounding "
                 "more than raw retrieval, since they restate what the passage answers."),
    "wikitablequestions": ("Table rows match queries on exact cell values; adding "
                           "headers/titles/NL summaries can raise retrieval but the "
                           "extra prose often dilutes the precise values an answer needs."),
    "chartqa": ("ChartQA ships images with almost no text, so the baseline is near-random; "
                "any added metadata/summary gives the retriever something to match, which is "
                "why enrichment helps most here (chart-to-table/axis are OCR placeholders)."),
    "formulareasoning": ("Terse formulas are hard to match to long word-problems; variable "
                         "definitions add the quantity names a question mentions, which lifts "
                         "retrieval from near-zero, though answer accuracy stays low."),
}
CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1e293b;background:#f1f5f9;line-height:1.55}
header{background:linear-gradient(135deg,#1e293b,#7c3aed);color:#fff;padding:38px 24px}
header .wrap{max-width:1080px;margin:0 auto}header h1{margin:0 0 8px;font-size:26px}header p{margin:0;opacity:.92;font-size:14.5px}
main{max-width:1080px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 22px;margin:18px 0;box-shadow:0 1px 3px rgba(15,23,42,.05)}
h2{font-size:20px;margin:2px 0 2px;border-left:4px solid #7c3aed;padding-left:10px}
.dt{color:#7c3aed;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:2px 0 12px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0}
th,td{padding:7px 9px;text-align:center;border-bottom:1px solid #eef2f7}
th{background:#f8fafc;color:#64748b;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}tr.base td{background:#faf5ff;font-weight:600}
.win{color:#16a34a;font-weight:700}.lose{color:#dc2626}
.fb{background:#f5f3ff;border:1px solid #ddd6fe;border-radius:9px;padding:13px 15px;margin-top:14px;font-size:14px}
.fb b{color:#6d28d9}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;margin-left:6px}
.tag.up{background:#dcfce7;color:#166534}.tag.down{background:#fee2e2;color:#991b1b}.tag.mix{background:#fef9c3;color:#854d0e}
footer{max-width:1080px;margin:0 auto;padding:10px 24px 44px;color:#64748b;font-size:12px}
.note{color:#64748b;font-size:12px;margin-top:6px}
"""


def _load(name) -> Dict:
    p = DIR / name
    out = {}
    if p.exists():
        for r in csv.DictReader(p.open()):
            out[(r["dataset"], r["method"])] = r
    return out


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cell(cur, base, higher_better=True) -> str:
    f = _f(cur)
    if f is None:
        return "<td>n/a</td>"
    b = _f(base)
    cls = ""
    if b is not None and cur is not base:
        if (f > b and higher_better) or (f < b and not higher_better):
            cls = "win"
        elif (f < b and higher_better) or (f > b and not higher_better):
            cls = "lose"
    return f'<td class="{cls}">{f:.3f}</td>'


def main() -> None:
    ret = _load("retrieval_metrics_by_method.csv")
    ans = _load("answer_quality_by_method.csv")
    cost = _load("token_cost_by_method.csv")
    if not ret:
        print("No results yet. Run run_enrichment_experiments.py first.")
        return

    datasets, methods = [], {}
    for (ds, m) in ret:
        if ds not in datasets:
            datasets.append(ds)
        methods.setdefault(ds, []).append(m)
    order = ["baseline", "combined_best"]
    for ds in methods:
        methods[ds].sort(key=lambda m: (m == "combined_best", m != "baseline", m))

    H: List[str] = []
    H.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    H.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    H.append("<title>Context Enrichment Results by Dataset</title>")
    H.append(f"<style>{CSS}</style></head><body>")
    H.append("<header><div class='wrap'><h1>Context Enrichment Results — by Dataset</h1>"
             "<p>Three enrichment methods (+ combined) vs. baseline, for each of the five "
             "document types. Green = better than baseline, red = worse.</p></div></header><main>")

    for ds in datasets:
        base_r = ret.get((ds, "baseline"), {})
        base_a = ans.get((ds, "baseline"), {})
        b_nd, b_fa = _f(base_r.get("nDCG@10")), _f(base_a.get("faithfulness"))
        H.append("<div class='card'>")
        H.append(f"<h2>{ds}</h2><div class='dt'>{DATA_TYPE.get(ds,'')}</div>")
        H.append("<table><thead><tr>"
                 "<th class='l'>Enrichment method</th><th>Recall@5</th><th>nDCG@10</th>"
                 "<th>MRR</th><th>Hit@5</th><th>Faithfulness</th><th>Answer rel.</th>"
                 "<th>Token cost ($)</th></tr></thead><tbody>")
        for m in methods[ds]:
            rr, aa, cc = ret[(ds, m)], ans.get((ds, m), {}), cost.get((ds, m), {})
            cls = "base" if m == "baseline" else ""
            label = METHOD_LABEL.get(m, m)
            base_flag = (m == "baseline")
            H.append(f"<tr class='{cls}'><td class='l'>{label}</td>"
                     + _cell(rr.get("Recall@5"), None if base_flag else base_r.get("Recall@5"))
                     + _cell(rr.get("nDCG@10"), None if base_flag else base_r.get("nDCG@10"))
                     + _cell(rr.get("MRR"), None if base_flag else base_r.get("MRR"))
                     + _cell(rr.get("Hit@5"), None if base_flag else base_r.get("Hit@5"))
                     + _cell(aa.get("faithfulness"), None if base_flag else base_a.get("faithfulness"))
                     + _cell(aa.get("answer_relevance"), None if base_flag else base_a.get("answer_relevance"))
                     + f"<td>{cc.get('estimated_cost_usd','n/a')}</td></tr>")
        H.append("</tbody></table>")

        # ---- data-driven feedback ----
        nb = [m for m in methods[ds] if m != "baseline"]
        ret_cand = [(m, _f(ret[(ds, m)].get("nDCG@10"))) for m in nb if _f(ret[(ds, m)].get("nDCG@10")) is not None]
        ans_cand = [(m, _f(ans.get((ds, m), {}).get("faithfulness"))) for m in nb if _f(ans.get((ds, m), {}).get("faithfulness")) is not None]
        best_ret = max(ret_cand, key=lambda x: x[1]) if ret_cand else None
        best_ans = max(ans_cand, key=lambda x: x[1]) if ans_cand else None
        ret_up = best_ret and b_nd is not None and best_ret[1] > b_nd
        ans_up = best_ans and b_fa is not None and best_ans[1] > b_fa
        if ret_up and ans_up:
            tag = "<span class='tag up'>beneficial</span>"
        elif ret_up or ans_up:
            tag = "<span class='tag mix'>mixed</span>"
        else:
            tag = "<span class='tag down'>not beneficial</span>"
        parts = []
        if best_ret:
            d = best_ret[1] - (b_nd or 0)
            parts.append(f"best for retrieval was <b>{METHOD_LABEL.get(best_ret[0],best_ret[0])}</b> "
                         f"(nDCG@10 {best_ret[1]:.3f}, {d:+.3f} vs baseline)")
        if best_ans and b_fa is not None:
            d = best_ans[1] - b_fa
            parts.append(f"best for answers was <b>{METHOD_LABEL.get(best_ans[0],best_ans[0])}</b> "
                         f"(faithfulness {best_ans[1]:.3f}, {d:+.3f})")
        verdict = "Here, " + "; ".join(parts) + "." if parts else ""
        H.append(f"<div class='fb'><b>Feedback {tag}</b><br>{RATIONALE.get(ds,'')} {verdict}</div>")
        H.append("</div>")

    H.append("<div class='card note'>* chart-to-table, chart axis metadata, and formula "
             "surrounding-text are documented placeholders (require OCR/vision or are not in "
             "the source data); their effect is structural, not from real extracted values. "
             "Answer grades come from an LLM judge and are noisy on small samples.</div>")
    H.append("</main><footer>Generated from results/enrichment_method_tests/ • "
             "Octen embeddings • Chroma Cloud • gpt-4o-mini (LangChain→LangSmith)</footer>")
    H.append("</body></html>")

    OUT.write_text("\n".join(H), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
