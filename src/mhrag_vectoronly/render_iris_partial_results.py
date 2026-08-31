"""Render the current Iris generation checkpoint as a searchable HTML report."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "mhrag_iris_qwen_5_20_full"
REPORT = ROOT / "report" / "mhrag_iris_qwen_5_20_partial_results.html"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def failure_category(message: str) -> str:
    if message.startswith("Generated "):
        return "Incomplete question count"
    if "failed after three attempts" in message:
        return "LLM request/JSON failure"
    if "has no attribute" in message:
        return "Unexpected response shape"
    return "Other"


def build_payload() -> tuple[list[dict], dict]:
    chunks = read_jsonl(DATA / "chunks.jsonl")
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    generations = read_jsonl(DATA / "adaptive_generations.jsonl")
    failures_path = DATA / "generation_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []

    rows = []
    for generation in generations:
        chunk = chunks_by_id[generation["chunk_id"]]
        rows.append(
            {
                "status": "complete",
                "chunk_id": generation["chunk_id"],
                "title": chunk["document_title"],
                "position": chunk["chunk_position"],
                "tokens": chunk["n_tokens"],
                "facts": generation["facts"],
                "questions": generation["bounded_questions"],
                "budget": generation["bounded_budget"],
            }
        )

    for failure in failures:
        chunk = chunks_by_id.get(failure["chunk_id"], {})
        rows.append(
            {
                "status": "failed",
                "chunk_id": failure["chunk_id"],
                "title": chunk.get("document_title", "Unknown document"),
                "position": chunk.get("chunk_position"),
                "tokens": chunk.get("n_tokens"),
                "error": failure["error"],
                "category": failure_category(failure["error"]),
            }
        )

    completed = len(generations)
    total = len(chunks)
    fact_counts = [len(row["facts"]) for row in generations]
    question_counts = [len(row["bounded_questions"]) for row in generations]
    categories = Counter(
        failure_category(failure["error"]) for failure in failures
    )
    summary = {
        "completed": completed,
        "failed": len(failures),
        "total": total,
        "completion_pct": 100 * completed / total if total else 0,
        "facts": sum(fact_counts),
        "questions": sum(question_counts),
        "facts_mean": sum(fact_counts) / completed if completed else 0,
        "questions_mean": sum(question_counts) / completed if completed else 0,
        "failure_categories": dict(categories),
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
    }
    return rows, summary


def render() -> Path:
    rows, summary = build_payload()
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    status = "Complete" if not summary["failed"] else "Incomplete"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris Qwen generation results</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0f1115; --surface: #171a21; --surface2: #1e222b;
  --text: #e8eaf0; --muted: #9da6b5; --line: #303744;
  --accent: #7aa2f7; --good: #68c98d; --bad: #ef7d7d;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }}
main {{ max-width: 1500px; margin: auto; padding: 28px; }}
h1 {{ margin: 0; font-size: 24px; }}
h2 {{ margin: 0 0 12px; font-size: 17px; }}
.subtitle {{ color: var(--muted); margin: 6px 0 22px; }}
.badge {{ display: inline-block; margin-left: 8px; padding: 2px 8px;
  border: 1px solid var(--bad); color: var(--bad); border-radius: 999px;
  font-size: 12px; vertical-align: 3px; }}
.summary {{ display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr));
  gap: 10px; margin-bottom: 18px; }}
.metric {{ background: var(--surface); border: 1px solid var(--line);
  padding: 14px; border-radius: 8px; }}
.metric strong {{ display: block; font-size: 20px; }}
.metric span {{ color: var(--muted); font-size: 12px; }}
.progress {{ height: 8px; background: var(--surface2); border-radius: 999px;
  overflow: hidden; margin: 10px 0 24px; }}
.progress div {{ height: 100%; background: var(--good);
  width: {summary["completion_pct"]:.3f}%; }}
.panel {{ background: var(--surface); border: 1px solid var(--line);
  padding: 16px; border-radius: 8px; margin-bottom: 18px; }}
.failure-grid {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.failure-chip {{ padding: 5px 9px; background: var(--surface2);
  border-radius: 6px; color: var(--muted); }}
.toolbar {{ display: grid; grid-template-columns: 1fr 170px 110px;
  gap: 10px; margin-bottom: 12px; }}
input, select, button {{ background: var(--surface); color: var(--text);
  border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px;
  font: inherit; }}
button {{ cursor: pointer; }}
button:disabled {{ opacity: .4; cursor: default; }}
.result {{ border: 1px solid var(--line); background: var(--surface);
  border-radius: 8px; margin-bottom: 9px; overflow: hidden; }}
.result summary {{ cursor: pointer; padding: 12px 14px; list-style: none; }}
.result summary::-webkit-details-marker {{ display: none; }}
.result summary::before {{ content: "›"; color: var(--accent); margin-right: 9px; }}
.result[open] summary::before {{ content: "⌄"; }}
.result-body {{ border-top: 1px solid var(--line); padding: 14px; }}
.meta {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
.complete {{ color: var(--good); }} .failed {{ color: var(--bad); }}
.columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
ol {{ margin: 0; padding-left: 22px; }}
li {{ margin-bottom: 9px; }}
.evidence {{ display: block; color: var(--muted); font-size: 12px;
  margin-top: 2px; }}
.ids {{ color: var(--accent); font-size: 12px; }}
.pager {{ display: flex; align-items: center; justify-content: space-between;
  margin-top: 14px; }}
.pager div {{ color: var(--muted); }}
@media (max-width: 850px) {{
  main {{ padding: 16px; }} .summary {{ grid-template-columns: 1fr 1fr; }}
  .toolbar {{ grid-template-columns: 1fr; }} .columns {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <h1>Iris Qwen generation results <span class="badge">{status}</span></h1>
  <p class="subtitle">Qwen/Qwen3.5-4B · generated {summary["generated_at"]} ·
    retrieval evaluation has not run yet</p>

  <section class="summary">
    <div class="metric"><strong>{summary["completed"]:,}</strong><span>completed chunks</span></div>
    <div class="metric"><strong>{summary["failed"]:,}</strong><span>failed chunks</span></div>
    <div class="metric"><strong>{summary["completion_pct"]:.1f}%</strong><span>generation complete</span></div>
    <div class="metric"><strong>{summary["facts"]:,}</strong><span>extracted facts · {summary["facts_mean"]:.1f}/chunk</span></div>
    <div class="metric"><strong>{summary["questions"]:,}</strong><span>questions · {summary["questions_mean"]:.1f}/chunk</span></div>
  </section>
  <div class="progress"><div></div></div>

  <section class="panel">
    <h2>Failure breakdown</h2>
    <div class="failure-grid" id="failure-breakdown"></div>
  </section>

  <div class="toolbar">
    <input id="search" type="search" placeholder="Search chunk ID, title, facts, or questions">
    <select id="status">
      <option value="all">All statuses</option>
      <option value="complete">Completed only</option>
      <option value="failed">Failed only</option>
    </select>
    <select id="page-size">
      <option value="25">25 / page</option>
      <option value="50" selected>50 / page</option>
      <option value="100">100 / page</option>
    </select>
  </div>
  <div id="results"></div>
  <div class="pager">
    <button id="previous">Previous</button>
    <div id="page-label"></div>
    <button id="next">Next</button>
  </div>
</main>

<script>
const DATA = {data_json};
const SUMMARY = {summary_json};
let page = 1;

const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({{
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}}[char]));

function searchable(row) {{
  return [
    row.chunk_id, row.title, row.error,
    ...(row.facts || []).flatMap(f => [f.fact, f.evidence]),
    ...(row.questions || []).map(q => q.question)
  ].join(" ").toLowerCase();
}}
DATA.forEach(row => row.search = searchable(row));

function filteredRows() {{
  const query = document.querySelector("#search").value.trim().toLowerCase();
  const status = document.querySelector("#status").value;
  return DATA.filter(row =>
    (status === "all" || row.status === status) &&
    (!query || row.search.includes(query))
  );
}}

function completeBody(row) {{
  const facts = row.facts.map((fact, index) => `<li>
    <strong>[${{index}}] ${{esc(fact.fact)}}</strong>
    <span class="evidence">Evidence: “${{esc(fact.evidence)}}” ·
      importance ${{esc(fact.importance)}} · distinctiveness ${{esc(fact.distinctiveness)}}</span>
  </li>`).join("");
  const questions = row.questions.map(question => `<li>
    <strong>${{esc(question.question)}}</strong>
    <span class="ids">source facts: ${{esc((question.source_fact_ids || []).join(", "))}}</span>
  </li>`).join("");
  return `<div class="columns">
    <section><h2>Facts (${{row.facts.length}})</h2><ol>${{facts}}</ol></section>
    <section><h2>Questions (${{row.questions.length}} / budget ${{row.budget}})</h2>
      <ol>${{questions}}</ol></section>
  </div>`;
}}

function resultHtml(row) {{
  const status = row.status === "complete"
    ? `<span class="complete">completed</span>`
    : `<span class="failed">failed</span>`;
  const body = row.status === "complete"
    ? completeBody(row)
    : `<h2>${{esc(row.category)}}</h2><p>${{esc(row.error)}}</p>`;
  return `<details class="result">
    <summary><strong>${{esc(row.chunk_id)}}</strong> · ${{esc(row.title)}} · ${{status}}
      <div class="meta">position ${{esc(row.position)}} · ${{esc(row.tokens)}} tokens</div>
    </summary>
    <div class="result-body">${{body}}</div>
  </details>`;
}}

function render() {{
  const rows = filteredRows();
  const size = Number(document.querySelector("#page-size").value);
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages);
  const visible = rows.slice((page - 1) * size, page * size);
  document.querySelector("#results").innerHTML = visible.map(resultHtml).join("");
  document.querySelector("#page-label").textContent =
    `${{rows.length.toLocaleString()}} results · page ${{page}} of ${{pages}}`;
  document.querySelector("#previous").disabled = page <= 1;
  document.querySelector("#next").disabled = page >= pages;
}}

document.querySelector("#failure-breakdown").innerHTML =
  Object.entries(SUMMARY.failure_categories)
    .map(([name, count]) => `<span class="failure-chip">${{esc(name)}}: <strong>${{count}}</strong></span>`)
    .join("");
document.querySelector("#search").addEventListener("input", () => {{ page = 1; render(); }});
document.querySelector("#status").addEventListener("change", () => {{ page = 1; render(); }});
document.querySelector("#page-size").addEventListener("change", () => {{ page = 1; render(); }});
document.querySelector("#previous").addEventListener("click", () => {{ page--; render(); scrollTo(0, 0); }});
document.querySelector("#next").addEventListener("click", () => {{ page++; render(); scrollTo(0, 0); }});
render();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return REPORT


if __name__ == "__main__":
    print(render())
