"""
Generate results/enrichment_method_tests/context_enrichment_summary.md from the
by-method CSVs produced by run_enrichment_experiments.py.

Data-driven: every number and ranking is computed from the CSVs, so re-running
the experiment and re-running this script keeps the summary in sync.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import config

DIR = config.RESULTS_DIR / "enrichment_method_tests"
DATA_TYPE = {
    "scifact": "Structured scientific text", "nfcorpus": "Unstructured biomedical text",
    "wikitablequestions": "Tables", "chartqa": "Charts / graphs",
    "formulareasoning": "Mathematical formulas",
}
BASELINE_REPR = {
    "scifact": "Raw abstract sentence", "nfcorpus": "Raw biomedical passage chunk",
    "wikitablequestions": "Linearized table row", "chartqa": "Chart OCR/caption text",
    "formulareasoning": "Raw formula text",
}


def _load(name) -> Dict:
    p = DIR / name
    out = {}
    if not p.exists():
        return out
    for r in csv.DictReader(p.open()):
        out[(r["dataset"], r["method"])] = r
    return out


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v) -> str:
    f = _f(v)
    return "n/a" if f is None else f"{f:.3f}"


def _delta(cur, base) -> str:
    a, b = _f(cur), _f(base)
    if a is None or b is None:
        return "n/a"
    d = a - b
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "■")
    return f"{d:+.3f} {arrow}"


def main() -> None:
    ret = _load("retrieval_metrics_by_method.csv")
    ans = _load("answer_quality_by_method.csv")
    lat = _load("latency_by_method.csv")
    cost = _load("token_cost_by_method.csv")
    if not ret:
        print("No results found. Run run_enrichment_experiments.py first.")
        return

    datasets, methods = [], {}
    for (ds, m) in ret:
        if ds not in datasets:
            datasets.append(ds)
        methods.setdefault(ds, []).append(m)

    L: List[str] = []
    L.append("# Context-Enrichment Method Comparison — Summary\n")
    L.append("## 1. Executive summary\n")
    L.append("We tested whether different context-enrichment methods improve RAG "
             "retrieval and answer quality across structured scientific text, "
             "unstructured biomedical text, tables, charts, and formulas. For each "
             "document type we compared a **baseline** (raw content) against three "
             "targeted enrichment methods and a **combined_best** condition, "
             "measuring retrieval quality, answer quality, latency, token usage, "
             "and cost.\n")

    # ---- 2. dataset overview ----
    L.append("## 2. Dataset overview\n")
    L.append("| Dataset | Data type | Baseline representation | Enrichment methods tested | Docs (chunks) | Queries |")
    L.append("|---|---|---|---|---|---|")
    for ds in datasets:
        ms = methods[ds]
        non_base = [m for m in ms if m != "baseline"]
        nchunks = lat.get((ds, "baseline"), {}).get("num_chunks", "?")
        nq = ret.get((ds, "baseline"), {}).get("num_queries", "?")
        L.append(f"| {ds} | {DATA_TYPE.get(ds,'')} | {BASELINE_REPR.get(ds,'')} | "
                 f"{', '.join(non_base)} | {nchunks} | {nq} |")
    L.append("")

    # ---- 3. results by dataset ----
    L.append("## 3. Results by dataset\n")
    for ds in datasets:
        base_r = ret.get((ds, "baseline"), {})
        base_a = ans.get((ds, "baseline"), {})
        L.append(f"### {ds} — {DATA_TYPE.get(ds,'')}\n")
        # best/worst method by nDCG@10
        cand = [(m, _f(ret[(ds, m)]["nDCG@10"])) for m in methods[ds]
                if m != "baseline" and _f(ret[(ds, m)]["nDCG@10"]) is not None]
        if cand:
            best = max(cand, key=lambda x: x[1]); worst = min(cand, key=lambda x: x[1])
            L.append(f"- **Best retrieval method:** `{best[0]}` "
                     f"(nDCG@10 {best[1]:.3f}, baseline {_fmt(base_r.get('nDCG@10'))})")
            L.append(f"- **Worst retrieval method:** `{worst[0]}` (nDCG@10 {worst[1]:.3f})")
        # answer-quality best by faithfulness
        acand = [(m, _f(ans[(ds, m)]["faithfulness"])) for m in methods[ds]
                 if m != "baseline" and _f(ans[(ds, m)]["faithfulness"]) is not None]
        if acand:
            abest = max(acand, key=lambda x: x[1])
            L.append(f"- **Best answer-quality method:** `{abest[0]}` "
                     f"(faithfulness {abest[1]:.3f}, baseline {_fmt(base_a.get('faithfulness'))})")
        L.append(f"- Baseline → retrieval nDCG@10 {_fmt(base_r.get('nDCG@10'))}, "
                 f"MRR {_fmt(base_r.get('MRR'))}; answer faithfulness "
                 f"{_fmt(base_a.get('faithfulness'))}, relevance {_fmt(base_a.get('answer_relevance'))}.")
        L.append("")

    # ---- 4. method comparison table ----
    L.append("## 4. Method comparison table\n")
    L.append("| Dataset | Method | Recall@5 | MRR | nDCG@10 | Hit@5 | Faithfulness | Answer rel. | Retr. latency (ms) | p95 (ms) | Token cost ($) | Verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        base_nd = _f(ret.get((ds, "baseline"), {}).get("nDCG@10"))
        base_fa = _f(ans.get((ds, "baseline"), {}).get("faithfulness"))
        for m in methods[ds]:
            rr, aa, ll, cc = ret[(ds, m)], ans.get((ds, m), {}), lat.get((ds, m), {}), cost.get((ds, m), {})
            verdict = "—"
            if m != "baseline":
                nd, fa = _f(rr["nDCG@10"]), _f(aa.get("faithfulness"))
                ret_up = nd is not None and base_nd is not None and nd > base_nd
                ans_up = fa is not None and base_fa is not None and fa > base_fa
                verdict = ("retrieval+answer ↑" if ret_up and ans_up else
                           "retrieval ↑ / answer ↓" if ret_up and not ans_up else
                           "retrieval ↓ / answer ↑" if not ret_up and ans_up else
                           "both ↓")
            L.append(f"| {ds} | {m} | {_fmt(rr.get('Recall@5'))} | {_fmt(rr.get('MRR'))} | "
                     f"{_fmt(rr.get('nDCG@10'))} | {_fmt(rr.get('Hit@5'))} | "
                     f"{_fmt(aa.get('faithfulness'))} | {_fmt(aa.get('answer_relevance'))} | "
                     f"{_fmt(ll.get('total_retrieval_latency_ms'))} | {_fmt(ll.get('p95_latency_ms'))} | "
                     f"{cc.get('estimated_cost_usd','n/a')} | {verdict} |")
    L.append("")

    # ---- 5. overall ranking (per-dataset best non-baseline method) ----
    L.append("## 5. Overall ranking (best enrichment method per dataset)\n")
    L.append("| Dataset | Best retrieval Δ | Best answer Δ | Lowest added latency | Best overall trade-off |")
    L.append("|---|---|---|---|---|")
    for ds in datasets:
        base_nd = _f(ret.get((ds, "baseline"), {}).get("nDCG@10"))
        base_fa = _f(ans.get((ds, "baseline"), {}).get("faithfulness"))
        base_lat = _f(lat.get((ds, "baseline"), {}).get("total_retrieval_latency_ms"))
        nb = [m for m in methods[ds] if m != "baseline"]
        def best_by(metric_map, base, key, maximize=True):
            scored = [(m, _f(metric_map[(ds, m)].get(key))) for m in nb
                      if _f(metric_map[(ds, m)].get(key)) is not None]
            if not scored:
                return "n/a"
            pick = (max if maximize else min)(scored, key=lambda x: x[1])
            return f"{pick[0]} ({pick[1]-base:+.3f})" if base is not None else f"{pick[0]} ({pick[1]:.3f})"
        ret_best = best_by(ret, base_nd, "nDCG@10", True)
        ans_best = best_by(ans, base_fa, "faithfulness", True)
        lat_best = best_by(lat, base_lat, "total_retrieval_latency_ms", False)
        # overall trade-off: method maximizing (nDCG delta + faithfulness delta)
        combo = []
        for m in nb:
            nd, fa = _f(ret[(ds, m)].get("nDCG@10")), _f(ans.get((ds, m), {}).get("faithfulness"))
            if nd is None:
                continue
            score = (nd - (base_nd or 0)) + ((fa - base_fa) if (fa is not None and base_fa is not None) else 0)
            combo.append((m, score))
        overall = max(combo, key=lambda x: x[1])[0] if combo else "n/a"
        L.append(f"| {ds} | {ret_best} | {ans_best} | {lat_best} | {overall} |")
    L.append("")

    # ---- 6. main findings ----
    L.append("## 6. Main findings\n")
    def best_ret(ds):
        nb = [(m, _f(ret[(ds, m)]["nDCG@10"])) for m in methods[ds]
              if m != "baseline" and _f(ret[(ds, m)]["nDCG@10"]) is not None]
        return max(nb, key=lambda x: x[1])[0] if nb else "n/a"
    qa = {
        "Structured scientific text (SciFact)": best_ret("scifact") if "scifact" in datasets else "n/a",
        "Unstructured biomedical text (NFCorpus)": best_ret("nfcorpus") if "nfcorpus" in datasets else "n/a",
        "Tables (WikiTableQuestions)": best_ret("wikitablequestions") if "wikitablequestions" in datasets else "n/a",
        "Charts (ChartQA)": best_ret("chartqa") if "chartqa" in datasets else "n/a",
        "Formulas (FormulaReasoning)": best_ret("formulareasoning") if "formulareasoning" in datasets else "n/a",
    }
    for k, v in qa.items():
        L.append(f"- **Best method for {k}:** `{v}`")
    # methods that hurt
    hurt = []
    for ds in datasets:
        base_nd = _f(ret.get((ds, "baseline"), {}).get("nDCG@10"))
        for m in methods[ds]:
            if m == "baseline":
                continue
            nd = _f(ret[(ds, m)]["nDCG@10"])
            if nd is not None and base_nd is not None and nd < base_nd:
                hurt.append(f"{ds}/{m}")
    L.append(f"- **Methods that hurt retrieval (nDCG@10 below baseline):** "
             f"{', '.join(hurt) if hurt else 'none'}")
    # retrieval-up-answer-down cases
    decoupled = []
    for ds in datasets:
        bnd = _f(ret.get((ds, "baseline"), {}).get("nDCG@10"))
        bfa = _f(ans.get((ds, "baseline"), {}).get("faithfulness"))
        for m in methods[ds]:
            if m == "baseline":
                continue
            nd, fa = _f(ret[(ds, m)]["nDCG@10"]), _f(ans.get((ds, m), {}).get("faithfulness"))
            if None not in (nd, bnd, fa, bfa) and nd > bnd and fa < bfa:
                decoupled.append(f"{ds}/{m}")
    L.append(f"- **Did better retrieval always mean better answers?** No. Cases where "
             f"retrieval improved but answer quality dropped: "
             f"{', '.join(decoupled) if decoupled else 'none observed'}.")
    # latency-heaviest method
    lat_heavy = max(lat.items(), key=lambda kv: _f(kv[1].get("total_index_time_seconds")) or 0)
    L.append(f"- **Highest indexing-latency condition:** {lat_heavy[0][0]}/{lat_heavy[0][1]} "
             f"({lat_heavy[1].get('total_index_time_seconds')}s).")
    L.append("")

    # ---- 7. final recommendation ----
    L.append("## 7. Final recommendation table\n")
    L.append("| Data type | Recommended method | Why | Trade-off | Use or avoid? |")
    L.append("|---|---|---|---|---|")
    for ds in datasets:
        base_nd = _f(ret.get((ds, "baseline"), {}).get("nDCG@10"))
        base_fa = _f(ans.get((ds, "baseline"), {}).get("faithfulness"))
        nb = [m for m in methods[ds] if m != "baseline"]
        combo = []
        for m in nb:
            nd, fa = _f(ret[(ds, m)].get("nDCG@10")), _f(ans.get((ds, m), {}).get("faithfulness"))
            if nd is None:
                continue
            combo.append((m, (nd-(base_nd or 0)) + ((fa-base_fa) if None not in (fa, base_fa) else 0)))
        if not combo:
            continue
        rec, score = max(combo, key=lambda x: x[1])
        use = "Use" if score > 0 else "Avoid (enrichment did not help here)"
        why = "improves retrieval and/or answer grounding" if score > 0 else \
              "no net gain over baseline in this run"
        tradeoff = "adds encoding latency + token cost"
        L.append(f"| {DATA_TYPE.get(ds,'')} | {rec} | {why} | {tradeoff} | {use} |")
    L.append("")

    L.append("## Notes & caveats\n")
    L.append("- Small debug sample; treat magnitudes as indicative, not final.\n"
             "- ChartQA chart-to-table and axis metadata are documented placeholders "
             "(require OCR/vision). FormulaReasoning 'surrounding text' is a placeholder "
             "(formula DB stores standalone formulas).\n"
             "- Answer grades come from an LLM judge (gpt-4o-mini) and are noisy.\n")

    out = DIR / "context_enrichment_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out.relative_to(config.PROJECT_ROOT)} ({len(L)} lines)")


if __name__ == "__main__":
    main()
