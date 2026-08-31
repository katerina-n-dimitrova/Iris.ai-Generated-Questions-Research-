"""
Reporting: rebuildable results.md + a single self-contained results.html.

Both are regenerated purely from the saved per-arm metrics JSON and the saved
rankings (for paired significance), so the report rebuilds without re-running
retrieval:  ``python mhrag_report.py --arms B0 B1``.

Content (per the deliverable):
  * run config + dataset summary
  * one table per experiment group (arms x metrics, with 95% CIs), best arm per
    metric highlighted, arms that significantly beat B1 marked (paired bootstrap)
  * the dense / BM25 / hybrid ablation
  * the per-MultiHop-RAG-query-type breakdown (inference / comparison / temporal)
  * HTML bar charts comparing each arm against B0 and B1 per metric
Experiment-specific extras (e.g. the Exp-1 type cross-tab) are appended by the
experiment modules through ``EXTRA_SECTIONS``.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from typing import Dict, List, Optional

import re

import mhrag_config as C
import mhrag_eval as E

MET = E.METRIC_KEYS
PRIMARY = E.PRIMARY_METRIC
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to in on at for and or but with is are was were be been "
    "this that these those it its as by from into their his her they he she "
    "which who what when where why how did does do has have had will would".split()
)

# Experiment modules may register extra HTML/MD blocks: name -> (md_str, html_str)
EXTRA_SECTIONS: "OrderedDict[str, tuple]" = OrderedDict()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_metrics(arm: str) -> Optional[dict]:
    p = C.metrics_path(arm)
    if not p.exists():
        return None
    return json.load(p.open())


def _dataset_summary() -> dict:
    p = C.PROCESSED_DIR / "dataset_summary.json"
    if p.exists():
        s = json.load(p.open())
        return {
            k: v
            for k, v in s.items()
            if k not in ("unmatched_evidence", "multi_chunk_evidence")
        }
    return {}


def _group_by_experiment(arms: List[str]) -> "OrderedDict[str, List[str]]":
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for a in arms:
        exp = C.ARMS[a].experiment
        groups.setdefault(exp, []).append(a)
    return groups


def _sig_vs(arm: str, ref: str, mode: str, metric: str) -> Optional[dict]:
    """Paired delta arm-vs-ref for a metric/mode (None if either missing)."""
    if arm == ref:
        return None
    try:
        sa = E.load_scored(arm, mode)
        sr = E.load_scored(ref, mode)
    except FileNotFoundError:
        return None
    if not sa or not sr:
        return None
    return E.paired_delta(sa, sr, metric)


def _best_arm(arms: List[str], metricvals: Dict[str, Optional[float]]) -> Optional[str]:
    cand = [(a, v) for a, v in metricvals.items() if v is not None]
    if not cand:
        return None
    return max(cand, key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_metric_table(
    arms: List[str],
    mode: str,
    metrics: Dict[str, dict],
    ref_for_sig: Optional[str] = "B1",
) -> str:
    lines = ["| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for m in MET:
        means = {
            a: (
                metrics[a]["modes"].get(mode, {}).get("overall", {}).get(m, {}) or {}
            ).get("mean")
            for a in arms
        }
        best = _best_arm(arms, means)
        cells = []
        for a in arms:
            ci = metrics[a]["modes"].get(mode, {}).get("overall", {}).get(m)
            if not ci:
                cells.append("-")
                continue
            txt = f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            if a == best:
                txt = f"**{txt}**"
            if ref_for_sig and a != ref_for_sig:
                d = _sig_vs(a, ref_for_sig, mode, m)
                if d and d["significant"]:
                    txt += " †" if d["delta"] > 0 else " ‡"
            cells.append(txt)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_type_tables(
    arms: List[str], metrics: Dict[str, dict], mode: str = "hybrid"
) -> str:
    out = []
    for qtype in C.QUERY_TYPES:
        out.append(
            f"\n**{qtype.capitalize()} queries** (hybrid, {PRIMARY} bolded best):\n"
        )
        lines = ["| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for m in MET:
            means = {
                a: (
                    metrics[a]["modes"]
                    .get(mode, {})
                    .get("by_query_type", {})
                    .get(qtype, {})
                    .get(m, {})
                    or {}
                ).get("mean")
                for a in arms
            }
            best = _best_arm(arms, means)
            cells = []
            for a in arms:
                ci = (
                    metrics[a]["modes"]
                    .get(mode, {})
                    .get("by_query_type", {})
                    .get(qtype, {})
                    .get(m)
                )
                if not ci:
                    cells.append("-")
                    continue
                txt = f"{ci['mean']:.3f}"
                if a == best:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("\n".join(lines))
    return "\n".join(out)


def _md_mode_ablation(arms: List[str], metrics: Dict[str, dict]) -> str:
    out = [
        "\n| arm | mode | " + " | ".join(MET) + " |",
        "|---|---|" + "---|" * len(MET),
    ]
    for a in arms:
        for mode in C.RETRIEVAL_MODES:
            mo = metrics[a]["modes"].get(mode)
            if not mo:
                continue
            vals = [
                f"{mo['overall'].get(m, {}).get('mean', float('nan')):.3f}" for m in MET
            ]
            out.append(f"| {a} | {mode} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def build_md(arms: List[str], metrics: Dict[str, dict]) -> str:
    ds = _dataset_summary()
    cfg = C.run_config_signature()
    parts = [
        "# MultiHop-RAG — question-generation strategy study\n",
        "## Run configuration\n",
        "```json",
        json.dumps(cfg, indent=2),
        "```\n",
        "## Dataset (pilot)\n",
        "```json",
        json.dumps(ds, indent=2),
        "```\n",
        "> Legend: **bold** = best arm for that metric; † = significantly "
        "better than B1 (paired bootstrap, 95%); ‡ = significantly worse "
        "than B1. Pilot scale → wide CIs; treat as directional.\n",
    ]
    for exp, exp_arms in _group_by_experiment(arms).items():
        show = (
            list(dict.fromkeys(["B0", "B1"] + exp_arms))
            if exp != "baselines"
            else exp_arms
        )
        show = [a for a in show if a in metrics]
        parts.append(f"\n## {exp}\n")
        parts.append("### Overall — hybrid (dense + BM25 RRF)\n")
        parts.append(_md_metric_table(show, "hybrid", metrics))
        parts.append("\n### Dense / BM25 / hybrid ablation\n")
        parts.append(_md_mode_ablation(show, metrics))
        parts.append("\n### Per MultiHop-RAG query type (hybrid)\n")
        parts.append(_md_type_tables(show, metrics))
        if exp == "exp1_semantic_type":
            parts.append(_exp1_md())
        elif exp == "exp5_style_match":
            parts.append(_exp5_descriptive_md(show))
        elif exp == "exp6_atomic_units":
            parts.append(_exp6_grid_md(metrics))
    for name, (md, _htm) in EXTRA_SECTIONS.items():
        parts.append(f"\n## {name}\n")
        parts.append(md)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--fg:#1a1d24;--muted:#5b6472;--line:#e2e6ec;--bg:#ffffff;--accent:#2563eb;
--best:#dcfce7;--bestln:#16a34a;--bar:#93b4f5;--barb:#2563eb;--good:#16a34a;--bad:#dc2626;}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
color:var(--fg);background:var(--bg);margin:0;padding:0 24px 80px;line-height:1.5;}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:26px;margin:32px 0 4px} h2{font-size:21px;margin:40px 0 10px;padding-top:12px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);margin:2px 0 18px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0 6px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f6f8fb;font-weight:600}
td.best{background:var(--best);font-weight:700;box-shadow:inset 3px 0 0 var(--bestln)}
.ci{color:var(--muted);font-size:11.5px}
.sig{color:var(--good);font-weight:700} .neg{color:var(--bad);font-weight:700}
.legend{font-size:12.5px;color:var(--muted);background:#f6f8fb;border:1px solid var(--line);
border-radius:8px;padding:10px 14px;margin:10px 0 4px}
.chartgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin:12px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h4{margin:0 0 8px;font-size:13px;color:var(--muted);font-weight:600}
.barrow{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px}
.barlabel{width:34px;color:var(--muted);flex:none}
.bartrack{flex:1;background:#f0f3f8;border-radius:4px;height:16px;position:relative}
.barfill{height:100%;border-radius:4px;background:var(--bar)}
.barfill.ref{background:#c7d2e8} .barfill.win{background:var(--barb)}
.barval{width:70px;flex:none;text-align:right;font-variant-numeric:tabular-nums}
pre{background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px}
.scroll{overflow-x:auto}
code{background:#f0f3f8;padding:1px 5px;border-radius:4px;font-size:12.5px}
.pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;background:#eef2fb;color:var(--accent);margin-left:6px}
"""


def _esc(s):
    return html.escape(str(s))


def _html_metric_table(arms, mode, metrics, ref="B1"):
    head = (
        "<tr><th>metric</th>" + "".join(f"<th>{_esc(a)}</th>" for a in arms) + "</tr>"
    )
    rows = []
    for m in MET:
        means = {
            a: (
                metrics[a]["modes"].get(mode, {}).get("overall", {}).get(m, {}) or {}
            ).get("mean")
            for a in arms
        }
        best = _best_arm(arms, means)
        tds = [f"<td>{_esc(m)}</td>"]
        for a in arms:
            ci = metrics[a]["modes"].get(mode, {}).get("overall", {}).get(m)
            if not ci:
                tds.append("<td>-</td>")
                continue
            mark = ""
            if ref and a != ref:
                d = _sig_vs(a, ref, mode, m)
                if d and d["significant"]:
                    mark = (
                        f" <span class='sig'>▲</span>"
                        if d["delta"] > 0
                        else f" <span class='neg'>▼</span>"
                    )
            cls = " class='best'" if a == best else ""
            tds.append(
                f"<td{cls}>{ci['mean']:.3f}{mark}"
                f"<div class='ci'>[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]</div></td>"
            )
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def _html_bars(arms, mode, metrics, chart_metrics):
    """Grouped bar cards: one card per metric, one bar per arm (vs B0/B1)."""
    cards = []
    for m in chart_metrics:
        means = {
            a: (
                metrics[a]["modes"].get(mode, {}).get("overall", {}).get(m, {}) or {}
            ).get("mean")
            for a in arms
        }
        vals = [means[a] for a in arms if means[a] is not None]
        vmax = max(vals) if vals else 1.0
        vmax = vmax if vmax > 0 else 1.0
        best = _best_arm(arms, means)
        b1 = means.get("B1")
        bars = []
        for a in arms:
            v = means[a]
            if v is None:
                continue
            w = 100 * v / vmax
            cls = (
                "ref"
                if a in ("B0", "B1")
                else ("win" if (b1 is not None and v > b1) else "")
            )
            if a == best:
                cls = "win"
            bars.append(
                f"<div class='barrow'><span class='barlabel'>{_esc(a)}</span>"
                f"<span class='bartrack'><span class='barfill {cls}' style='width:{w:.1f}%'></span></span>"
                f"<span class='barval'>{v:.3f}</span></div>"
            )
        cards.append(
            f"<div class='card'><h4>{_esc(m)} — {mode}</h4>{''.join(bars)}</div>"
        )
    return f"<div class='chartgrid'>{''.join(cards)}</div>"


def _html_type_tables(arms, metrics, mode="hybrid"):
    blocks = []
    for qtype in C.QUERY_TYPES:
        head = (
            "<tr><th>metric</th>"
            + "".join(f"<th>{_esc(a)}</th>" for a in arms)
            + "</tr>"
        )
        rows = []
        for m in MET:
            means = {
                a: (
                    metrics[a]["modes"]
                    .get(mode, {})
                    .get("by_query_type", {})
                    .get(qtype, {})
                    .get(m, {})
                    or {}
                ).get("mean")
                for a in arms
            }
            best = _best_arm(arms, means)
            tds = [f"<td>{_esc(m)}</td>"]
            for a in arms:
                ci = (
                    metrics[a]["modes"]
                    .get(mode, {})
                    .get("by_query_type", {})
                    .get(qtype, {})
                    .get(m)
                )
                if not ci:
                    tds.append("<td>-</td>")
                    continue
                cls = " class='best'" if a == best else ""
                tds.append(f"<td{cls}>{ci['mean']:.3f}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
        n = ""
        # attach n for this type from any arm's overall record
        for a in arms:
            rec = (
                metrics[a]["modes"]
                .get(mode, {})
                .get("by_query_type", {})
                .get(qtype, {})
                .get(PRIMARY)
            )
            if rec:
                n = f" <span class='pill'>n={rec.get('n', '?')}</span>"
                break
        blocks.append(
            f"<h4 style='margin:16px 0 4px'>{qtype.capitalize()} queries{n}</h4>"
            f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    return "".join(blocks)


def _html_mode_ablation(arms, metrics):
    head = (
        "<tr><th>arm</th><th>mode</th>"
        + "".join(f"<th>{_esc(m)}</th>" for m in MET)
        + "</tr>"
    )
    rows = []
    for a in arms:
        for mode in C.RETRIEVAL_MODES:
            mo = metrics[a]["modes"].get(mode)
            if not mo:
                continue
            tds = [f"<td>{_esc(a)}</td>", f"<td>{_esc(mode)}</td>"]
            for m in MET:
                v = mo["overall"].get(m, {}).get("mean")
                tds.append(f"<td>{v:.3f}</td>" if v is not None else "<td>-</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


CHART_METRICS = [f"evidence_recall@{k}" for k in C.K_VALUES] + [
    f"mrr@{C.MRR_K}",
    f"ndcg@{C.NDCG_K}",
]


def build_html(arms: List[str], metrics: Dict[str, dict]) -> str:
    ds = _dataset_summary()
    cfg = C.run_config_signature()
    body = [
        f"<div class='wrap'><h1>MultiHop-RAG — question-generation strategy study</h1>",
        f"<p class='sub'>Which <em>kind</em> of LLM-generated question best enriches a "
        f"chunk for retrieval? Hybrid dense + BM25 + RRF, questions-only dense index. "
        f"Pilot: {ds.get('num_articles', '?')} articles, {ds.get('num_chunks', '?')} chunks, "
        f"{ds.get('num_queries', '?')} queries.</p>",
        "<div class='legend'><b>Legend:</b> green-shaded cell = best arm for that metric; "
        "<span class='sig'>▲</span> = significantly better than B1, "
        "<span class='neg'>▼</span> = significantly worse than B1 (paired bootstrap, 95% CI "
        "excludes 0). Pilot scale → wide CIs; treat as directional signals.</div>",
        "<h3>Run config</h3><pre>" + _esc(json.dumps(cfg, indent=2)) + "</pre>",
        "<h3>Dataset</h3><pre>" + _esc(json.dumps(ds, indent=2)) + "</pre>",
    ]

    for exp, exp_arms in _group_by_experiment(arms).items():
        show = (
            list(dict.fromkeys(["B0", "B1"] + exp_arms))
            if exp != "baselines"
            else exp_arms
        )
        show = [a for a in show if a in metrics]
        title = {"baselines": "Baselines (B0, B1)"}.get(exp, exp)
        body.append(f"<h2>{_esc(title)}</h2>")
        descs = " · ".join(f"<b>{a}</b>: {_esc(C.ARMS[a].description)}" for a in show)
        body.append(f"<p class='sub' style='font-size:12.5px'>{descs}</p>")
        body.append("<h3>Overall — hybrid (dense + BM25 RRF)</h3>")
        body.append(_html_metric_table(show, "hybrid", metrics))
        body.append(_html_bars(show, "hybrid", metrics, CHART_METRICS))
        body.append("<h3>Dense / BM25 / hybrid ablation</h3>")
        body.append(_html_mode_ablation(show, metrics))
        body.append("<h3>Per MultiHop-RAG query type (hybrid)</h3>")
        body.append(_html_type_tables(show, metrics))
        if exp == "exp1_semantic_type":
            body.append("<h3>Type cross-tab &amp; coverage</h3>")
            body.append(_exp1_html())
        elif exp == "exp5_style_match":
            body.append(_exp5_descriptive_html(show))
        elif exp == "exp6_atomic_units":
            body.append("<h3>Atomic-units 2×2 (chunk/atom × statement/question)</h3>")
            body.append(_exp6_grid_html(metrics))

    for name, (_md, htm) in EXTRA_SECTIONS.items():
        body.append(f"<h2>{_esc(name)}</h2>")
        body.append(htm)

    body.append("</div>")
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>MultiHop-RAG question-generation study</title>"
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )


# --------------------------------------------------------------------------- #
# Experiment-specific sections
# --------------------------------------------------------------------------- #
def _descriptive_stats(arm: str):
    """(avg #questions, avg question length in words, lexical overlap with chunk)."""
    import mhrag_data as D
    import mhrag_generate as G

    chunks = {c["chunk_id"]: c["text"] for c in D.load_chunks()}
    qs = G.load_questions(arm)
    if not qs:
        return None
    lens, overlaps, counts = [], [], []
    for cid, qlist in qs.items():
        counts.append(len(qlist))
        chunk_words = set(
            w for w in _WORD.findall(chunks.get(cid, "").lower()) if w not in _STOP
        )
        for q in qlist:
            words = [w for w in _WORD.findall(q.lower()) if w not in _STOP]
            lens.append(len(_WORD.findall(q)))
            if words:
                overlaps.append(sum(1 for w in words if w in chunk_words) / len(words))
    n = max(len(lens), 1)
    return {
        "avg_questions": round(sum(counts) / max(len(counts), 1), 2),
        "avg_len_words": round(sum(lens) / n, 2),
        "lexical_overlap_with_chunk": round(sum(overlaps) / max(len(overlaps), 1), 3),
    }


def _exp5_descriptive_md(arms):
    rows = [
        "\n### Descriptive stats (generated questions)\n",
        "| arm | avg #q | avg length (words) | lexical overlap w/ chunk |",
        "|---|---|---|---|",
    ]
    for a in arms:
        s = _descriptive_stats(a)
        if s:
            rows.append(
                f"| {a} | {s['avg_questions']} | {s['avg_len_words']} | "
                f"{s['lexical_overlap_with_chunk']} |"
            )
    return "\n".join(rows)


def _exp5_descriptive_html(arms):
    head = "<tr><th>arm</th><th>avg #q</th><th>avg length (words)</th><th>lexical overlap w/ chunk</th></tr>"
    rows = []
    for a in arms:
        s = _descriptive_stats(a)
        if s:
            rows.append(
                f"<tr><td>{_esc(a)}</td><td>{s['avg_questions']}</td>"
                f"<td>{s['avg_len_words']}</td><td>{s['lexical_overlap_with_chunk']}</td></tr>"
            )
    return (
        "<h3>Descriptive stats (generated questions)</h3>"
        f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _exp1_crosstab():
    p = C.RESULTS_DIR / "exp1_crosstab.json"
    if not p.exists():
        return None
    return json.load(p.open())


def _exp1_md():
    o = _exp1_crosstab()
    if not o:
        return ""
    r = o["crosstab"]
    cov = o["coverage"]
    out = [
        "\n### Experiment 1 — type cross-tab & coverage\n",
        f"Same-type-on-gold coverage: **B1 {cov['B1']['coverage'] * 100:.0f}%** vs "
        f"**E1 {cov['E1']['coverage'] * 100:.0f}%** "
        f"(fraction of queries whose gold chunk carries a generated question of "
        f"the query's Cao&Wang type).\n",
        f"\nCross-tab on B1 — {r['success_metric']} by MultiHop query type × "
        "whether a same-type generated question sat on a gold chunk:\n",
        "| query type | n | success (same-type present) | success (absent) |",
        "|---|---|---|---|",
    ]
    for t, row in r["by_query_type"].items():
        out.append(
            f"| {t} | {row['n']} | {row['success_present']} (n={row['n_present']}) | "
            f"{row['success_absent']} (n={row['n_absent']}) |"
        )
    ov = r["overall"]
    out.append(
        f"| **overall** | {ov['n_present'] + ov['n_absent']} | "
        f"**{ov['success_present']}** (n={ov['n_present']}) | "
        f"**{ov['success_absent']}** (n={ov['n_absent']}) |"
    )
    return "\n".join(out)


def _heat(v):
    """Background colour for a 0..1 success rate (light heatmap)."""
    if v is None:
        return "#f3f4f6"
    v = max(0.0, min(1.0, v))
    r = int(255 - v * (255 - 22))
    g = int(255 - v * (255 - 163))
    b = int(255 - v * (255 - 74))
    return f"rgb({r},{g},{b})"


def _exp1_html():
    o = _exp1_crosstab()
    if not o:
        return ""
    r = o["crosstab"]
    cov = o["coverage"]
    dist = o.get("type_distribution", {})
    head = (
        "<tr><th>MultiHop query type</th><th>n</th>"
        "<th>success — same-type present</th><th>success — absent</th></tr>"
    )
    rows = []
    for t, row in r["by_query_type"].items():
        sp, sa = row["success_present"], row["success_absent"]
        rows.append(
            f"<tr><td>{_esc(t)}</td><td>{row['n']}</td>"
            f"<td style='background:{_heat(sp)}'>{'-' if sp is None else f'{sp:.3f}'} "
            f"<span class='ci'>n={row['n_present']}</span></td>"
            f"<td style='background:{_heat(sa)}'>{'-' if sa is None else f'{sa:.3f}'} "
            f"<span class='ci'>n={row['n_absent']}</span></td></tr>"
        )
    ov = r["overall"]
    rows.append(
        f"<tr><td><b>overall</b></td><td>{ov['n_present'] + ov['n_absent']}</td>"
        f"<td style='background:{_heat(ov['success_present'])}'><b>{ov['success_present']}</b> "
        f"<span class='ci'>n={ov['n_present']}</span></td>"
        f"<td style='background:{_heat(ov['success_absent'])}'><b>{ov['success_absent']}</b> "
        f"<span class='ci'>n={ov['n_absent']}</span></td></tr>"
    )
    # type distribution table
    dhead = (
        "<tr><th>Cao&amp;Wang type</th>"
        + "".join(f"<th>{_esc(a)}</th>" for a in dist)
        + "</tr>"
    )
    types = sorted({t for a in dist for t in dist[a]})
    drows = []
    for t in types:
        drows.append(
            "<tr><td>"
            + _esc(t)
            + "</td>"
            + "".join(f"<td>{dist[a].get(t, 0)}</td>" for a in dist)
            + "</tr>"
        )
    return (
        "<p class='sub'>Same-type-on-gold coverage: "
        f"<b>B1 {cov['B1']['coverage'] * 100:.0f}%</b> vs "
        f"<b>E1 {cov['E1']['coverage'] * 100:.0f}%</b>. Cross-tab cell shading = "
        f"{r['success_metric']} success rate (darker = higher).</p>"
        f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<h3>Generated-question type distribution (LLM classifier)</h3>"
        f"<div class='scroll'><table><thead>{dhead}</thead><tbody>{''.join(drows)}</tbody></table></div>"
    )


def _exp6_grid_html(metrics, mode="hybrid"):
    """2x2: chunk/atom × statement/question, cells = B0/B1/E6as/E6aq."""
    grid = {
        ("chunk", "statement"): "B0",
        ("chunk", "question"): "B1",
        ("atom", "statement"): "E6as",
        ("atom", "question"): "E6aq",
    }
    show_metrics = [f"evidence_recall@{k}" for k in C.K_VALUES] + [f"mrr@{C.MRR_K}"]
    blocks = []
    for m in show_metrics:
        head = "<tr><th></th><th>statement rep</th><th>question rep</th></tr>"
        body = []
        for level in ("chunk", "atom"):
            cells = [f"<td><b>{level}-level</b></td>"]
            for rep in ("statement", "question"):
                arm = grid[(level, rep)]
                ci = (
                    (
                        metrics.get(arm, {})
                        .get("modes", {})
                        .get(mode, {})
                        .get("overall", {})
                        .get(m)
                    )
                    if arm in metrics
                    else None
                )
                if ci:
                    cells.append(
                        f"<td>{arm}: <b>{ci['mean']:.3f}</b>"
                        f"<div class='ci'>[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]</div></td>"
                    )
                else:
                    cells.append(f"<td>{arm}: -</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        blocks.append(
            f"<h4 style='margin:14px 0 4px'>{_esc(m)} ({mode})</h4>"
            f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"
        )
    return (
        "<p class='sub'>Raina &amp; Gales (2024) 2×2: does decomposing to atoms "
        "help, and do questions beat statements? Chunk-level cells are B0/B1.</p>"
        + "".join(blocks)
    )


def _exp6_grid_md(metrics, mode="hybrid"):
    grid = {
        ("chunk", "statement"): "B0",
        ("chunk", "question"): "B1",
        ("atom", "statement"): "E6as",
        ("atom", "question"): "E6aq",
    }
    show_metrics = [f"evidence_recall@{k}" for k in C.K_VALUES] + [f"mrr@{C.MRR_K}"]
    out = ["\n### Atomic-units 2×2 (chunk/atom × statement/question)\n"]
    for m in show_metrics:
        out.append(f"\n**{m}** ({mode})\n")
        out.append("| level | statement rep | question rep |")
        out.append("|---|---|---|")
        for level in ("chunk", "atom"):
            cells = [f"{level}-level"]
            for rep in ("statement", "question"):
                arm = grid[(level, rep)]
                ci = (
                    (
                        metrics.get(arm, {})
                        .get("modes", {})
                        .get(mode, {})
                        .get("overall", {})
                        .get(m)
                    )
                    if arm in metrics
                    else None
                )
                cells.append(f"{arm}: {ci['mean']:.3f}" if ci else f"{arm}: -")
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate(arms: List[str]) -> Dict[str, str]:
    metrics = {a: load_metrics(a) for a in arms}
    metrics = {a: m for a, m in metrics.items() if m}
    arms = [a for a in arms if a in metrics]
    md = build_md(arms, metrics)
    htm = build_html(arms, metrics)
    md_path = C.REPORT_DIR / "results.md"
    html_path = C.REPORT_DIR / "results.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(htm, encoding="utf-8")
    return {"md": str(md_path), "html": str(html_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B0", "B1"])
    args = ap.parse_args()
    paths = generate(args.arms)
    print("[report]", paths)
