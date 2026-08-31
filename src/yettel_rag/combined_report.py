#!/usr/bin/env python3
"""Render the combined MultiHop-RAG + Yettel report, including standalone-vs-combined deltas."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "results/combined_mhrag_yettel_experiments/metrics.json"
MHRAG = ROOT / "results/mhrag_adaptive_questions_full/metrics.json"
YETTEL = ROOT / "results/yettel_bg_experiments/metrics.json"
REPORT = ROOT / "report/combined_multihop_yettel_four_experiments.html"

FULL_KEYS = (
    "evidence_recall@1",
    "evidence_recall@5",
    "evidence_recall@10",
    "all_evidence_hit@5",
    "mrr@10",
    "ndcg@10",
)
SHARED_KEYS = FULL_KEYS[:-1]
LABELS = {
    "evidence_recall@1": "Recall@1",
    "evidence_recall@5": "Recall@5",
    "evidence_recall@10": "Recall@10",
    "all_evidence_hit@5": "Full-evidence@5",
    "mrr@10": "MRR@10",
    "ndcg@10": "nDCG@10",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(text: str) -> str:
    return html.escape(str(text))


def headline(rows: list[dict]) -> str:
    """Baseline vs best condition, as four large stat cards."""
    base, best = rows[0]["metrics"], rows[3]["metrics"]
    cards = []
    for key in ("evidence_recall@5", "evidence_recall@10", "mrr@10", "ndcg@10"):
        delta = best[key] - base[key]
        cards.append(
            f"<div class='card'><div class='k'>{LABELS[key]}</div>"
            f"<div class='v'>{best[key]:.3f}</div>"
            f"<div class='d'>{base[key]:.3f} baseline &rarr; <b class='up'>+{delta:.3f}</b>"
            f" (+{delta / base[key] * 100:.1f}%)</div></div>"
        )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def metrics_table(rows: list[dict], selector=None, keys=FULL_KEYS) -> str:
    head = "".join(f"<th>{LABELS[k]}</th>" for k in keys)
    baseline = selector(rows[0]) if selector else rows[0]["metrics"]
    body = []
    for index, row in enumerate(rows):
        metrics = selector(row) if selector else row["metrics"]
        cells = []
        for key in keys:
            value = metrics[key]
            if index == 0:
                cells.append(f"<td><b>{value:.3f}</b></td>")
                continue
            delta = value - baseline[key]
            cls = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            cells.append(
                f"<td><b>{value:.3f}</b><span class='delta {cls}'>"
                f"{delta:+.3f}</span></td>"
            )
        name = esc(row["condition"])
        tag = " <span class='pill'>baseline</span>" if index == 0 else ""
        body.append(
            f"<tr><td class='left'><b>{name}</b>{tag}</td>"
            f"<td>{row['generated_questions']:,}</td>"
            f"<td>{row['stored_vectors']:,}</td>{''.join(cells)}</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th class='left'>Condition</th>"
        "<th>Generated questions</th><th>Stored vectors</th>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def interference_table(standalone: dict, combined: list[dict], dataset: str) -> str:
    """Same queries, own corpus only vs the 949-document shared index."""
    head = "".join(f"<th>{LABELS[k]}</th>" for k in SHARED_KEYS)
    body = []
    for index, row in enumerate(combined):
        alone = standalone["conditions"][index]["metrics"]
        joint = row["by_dataset"][dataset]
        cells = []
        for key in SHARED_KEYS:
            delta = joint[key] - alone[key]
            cls = "up" if delta > 0.0005 else ("down" if delta < -0.0005 else "flat")
            cells.append(
                f"<td>{alone[key]:.3f} &rarr; <b>{joint[key]:.3f}</b>"
                f"<span class='delta {cls}'>{delta:+.3f}</span></td>"
            )
        body.append(
            f"<tr><td class='left'><b>{esc(row['condition'])}</b></td>"
            f"{''.join(cells)}</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th class='left'>Condition</th>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def leakage_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        leak = row["wrong_corpus_top1"]
        body.append(
            f"<tr><td class='left'><b>{esc(row['condition'])}</b></td>"
            f"<td>{leak['mhrag'] * 100:.2f}%</td>"
            f"<td>{leak['yettel'] * 100:.2f}%</td></tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th class='left'>Condition</th>"
        "<th>MultiHop-RAG query &rarr; Yettel chunk at rank 1</th>"
        "<th>Yettel query &rarr; MultiHop-RAG chunk at rank 1</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


CSS = """
:root{--bg:#0f1319;--panel:#161c25;--line:#29313d;--fg:#e6eaf0;--mut:#9aa5b3;
--up:#4ade80;--down:#f87171}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,-apple-system,sans-serif;
margin:0;padding:28px 22px 64px}
main{max-width:1500px;margin:auto}
h1{font-size:30px;line-height:1.25;margin:0 0 8px}
h2{font-size:19px;margin:44px 0 6px;padding-top:18px;border-top:1px solid var(--line)}
p.note{color:var(--mut);margin:6px 0 16px;max-width:95ch}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;min-width:820px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:9px 11px;text-align:center;vertical-align:middle}
th{background:var(--panel);font-weight:600;color:var(--mut);font-size:12px;
letter-spacing:.03em;text-transform:uppercase}
td.left,th.left{text-align:left}
tbody tr:nth-child(odd){background:rgba(255,255,255,.018)}
.delta{display:block;font-size:11px;font-weight:600;margin-top:2px}
.up{color:var(--up)}.down{color:var(--down)}.flat{color:var(--mut)}
.pill{display:inline-block;margin-left:8px;padding:1px 7px;border:1px solid var(--line);
border-radius:9px;font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:22px 0 8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:30px;font-weight:700;line-height:1.2;margin:2px 0}
.card .d{color:var(--mut);font-size:12px}
.meta{color:var(--mut);font-size:12px;margin-top:40px;padding-top:14px;border-top:1px solid var(--line)}
ul{color:var(--mut);max-width:95ch}li{margin:5px 0}
"""


def render() -> Path:
    combined, mhrag, yettel = load(COMBINED), load(MHRAG), load(YETTEL)
    rows = combined["conditions"]
    p = combined["protocol"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Combined MultiHop-RAG + Yettel: four retrieval experiments</title>
<style>{CSS}</style></head><body><main>

<h1>949-document combined benchmark: MultiHop-RAG + Yettel Bulgaria</h1>
<p class="note">609 English MultiHop-RAG news articles and 340 Bulgarian Yettel corporate
documents merged into one shared index of {p["chunks"]:,} chunks, evaluated with
{p["evaluation_queries"]:,} evidence-bearing queries (2,255 from each dataset). All four
conditions use identical 1,024-token chunks with 128-token overlap, Iris 384-dimensional
embeddings, Unicode BM25, and RRF k={p["rrf_k"]}. The {p["yettel_null_excluded"]} Yettel null
queries stay in the dataset but are excluded from evidence-retrieval metrics.</p>

{headline(rows)}

<h2>1. Combined results &mdash; all {p["evaluation_queries"]:,} queries</h2>
<p class="note">Every query is scored against the full 949-document index, so each dataset
acts as a distractor corpus for the other. Deltas are against the no-question baseline.</p>
{metrics_table(rows)}

<h2>2. MultiHop-RAG queries against the combined corpus</h2>
{metrics_table(rows, lambda x: x["by_dataset"]["mhrag"])}

<h2>3. Yettel queries against the combined corpus</h2>
{metrics_table(rows, lambda x: x["by_dataset"]["yettel"])}

<h2>4. Interference: standalone corpus vs the shared 949-document index</h2>
<p class="note">The same queries and the same enrichment, retrieved first against the
dataset's own corpus and then against the merged index. This isolates the cost of the
extra 340 (or 609) distractor documents. nDCG@10 is omitted because the standalone runs
did not record it.</p>
<h3 style="color:var(--mut);font-size:13px;margin:18px 0 6px;text-transform:uppercase;letter-spacing:.05em">MultiHop-RAG: 609 docs &rarr; 949 docs</h3>
{interference_table(mhrag, rows, "mhrag")}
<h3 style="color:var(--mut);font-size:13px;margin:22px 0 6px;text-transform:uppercase;letter-spacing:.05em">Yettel: 340 docs &rarr; 949 docs</h3>
{interference_table(yettel, rows, "yettel")}

<h2>5. Cross-corpus leakage at rank 1</h2>
<p class="note">Share of queries whose top-ranked chunk comes from the other dataset.</p>
{leakage_table(rows)}

<h2>Reading the numbers</h2>
<ul>
<li>Adaptive question enrichment holds up on the merged corpus. The strongest condition
(chunk + whole-document questions) lifts Recall@5 from
{rows[0]["metrics"]["evidence_recall@5"]:.3f} to {rows[3]["metrics"]["evidence_recall@5"]:.3f}
and MRR@10 from {rows[0]["metrics"]["mrr@10"]:.3f} to {rows[3]["metrics"]["mrr@10"]:.3f}.</li>
<li>Bounded (5&ndash;20) and unbounded question counts are statistically indistinguishable,
matching the single-dataset runs; the bound costs nothing and caps generation spend.</li>
<li>Whole-document questions are what separate condition 4 from condition 2, and the gain is
larger on Yettel than on MultiHop-RAG.</li>
<li>Cross-corpus leakage is near zero in every condition. Bulgarian and English chunks barely
compete, so the combined figure is close to the mean of the two per-dataset figures rather
than a genuinely harder retrieval task. Treat this as a robustness check, not a harder
benchmark.</li>
<li>Because the corpora do not interfere, the combined score is dominated by the weaker
dataset. Yettel trails MultiHop-RAG at every condition, which is where the headroom is.</li>
</ul>

<h2>Protocol notes</h2>
<ul>
<li>Chunk and document IDs are namespaced <code>mhrag::</code> and <code>yettel::</code>, so no
collisions are possible across the merged index.</li>
<li>Existing question generations and embeddings were reused byte-for-byte; the runner
validates cached text and ordering against the source JSONL before scoring.</li>
<li>No evaluation query was ever used as enrichment text.</li>
<li>These four runs measure retrieval only. Answer accuracy is not reported.</li>
<li>Yettel evaluation and enrichment questions were generated independently but by the same
model (gpt-5.4-mini), so same-model stylistic coupling remains a limitation of that half of
the benchmark.</li>
</ul>

<p class="meta">Generated {stamp} from
<code>results/combined_mhrag_yettel_experiments/metrics.json</code> ·
runner <code>src/yettel_rag/run_combined_experiments.py</code> ·
report <code>src/yettel_rag/combined_report.py</code></p>
</main></body></html>""",
        encoding="utf-8",
    )
    return REPORT


if __name__ == "__main__":
    print(f"report: {render()}")
