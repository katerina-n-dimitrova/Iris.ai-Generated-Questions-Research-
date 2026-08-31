"""Render the new three-table controlled retrieval report."""

from __future__ import annotations

import json
from html import escape

import controlled_suite_generate as G
import controlled_suite_optimize as O
import controlled_suite_remaining as X
import controlled_suite_retrieval as R
import controlled_article_first as A
import vo_config as C

REPORT = C.PROJECT_ROOT / "report" / "mhrag_controlled_three_experiments.html"


def _delta(value: float, base: float) -> str:
    difference = value - base
    cls = "good" if difference > 0 else ("bad" if difference < 0 else "flat")
    return (
        f'<div class="v">{value:.3f}</div>'
        f'<div class="d {cls}">({difference:+.3f})</div>'
    )


def _question_description(condition: str, config: str) -> tuple[str, str]:
    if "GEPA" in condition:
        return (
            "10 general questions per chunk"
            if "E3-" not in condition
            else "3 closed-answer questions per atomic fact",
            "gepa_optimized_v1",
        )
    if "MIPRO" in condition:
        return (
            "10 general questions per chunk"
            if "E3-" not in condition
            else "3 closed-answer questions per atomic fact",
            "miprov2_optimized_v1",
        )
    if "atomic" in config:
        if "adaptive" in config:
            return (
                "Atomic facts; 1/2/3 questions by fact complexity",
                "manual_atomic_adaptive_v1",
            )
        return ("3 closed-answer questions per atomic fact", "manual_atomic_v1")
    if "adaptive" in config:
        return ("5/10/15 by distinct-information level", "manual_adaptive_v1")
    return ("10 general questions per chunk", "manual_general_v1")


def _metadata(row: dict, manifests: dict) -> dict:
    config = row["config"]
    question_method, prompt = _question_description(row["condition"], config)
    chunk_size = 1024 if "1024" in config else 512
    indexes = manifests[config]
    retrieval = {
        "dual": "Separate chunk + question vectors; 0.5/0.5 score fusion",
        "dual_bm25": "Dual-index vector fusion + BM25 chunk RRF",
        "fallback": "Question vectors first; chunk-vector fallback",
        "fallback_bm25": "Question vectors first; chunk vector + BM25 fallback",
        "question": "Atomic-question vectors only; parent-chunk max score",
        "question_bm25": "Atomic-question vectors + BM25 parent-chunk RRF",
        "quester_bm25": "Qwen keyword rewrite sent directly to BM25",
        "atomic_splade_rrf": "Atomic Iris vectors + SPLADE sparse RRF",
        "atomic_bge_m3_sparse_rrf": "Atomic Iris vectors + BGE-M3 sparse RRF",
    }[row["retrieval_method"]]
    if row["condition"] == "E1-QueStER":
        question_method = "None; query-side keyword rewriting"
        prompt = "quester_keyword_v1"
        index_label = "BM25 over original chunks"
    elif row["condition"] == "E3-SPLADE":
        index_label = f"{escape(indexes['question_collection'])} + SPLADE sparse chunks"
    elif row["condition"] == "E3-BGE-M3-Sparse":
        index_label = f"{escape(indexes['question_collection'])} + BGE-M3 sparse chunks"
    else:
        index_label = (
            f"{escape(indexes['chunk_collection'])} + "
            f"{escape(indexes['question_collection'])}"
            if row["experiment"] != "Experiment 3"
            else escape(indexes["question_collection"])
        )
    return {
        "question_method": question_method,
        "question_count": (
            0 if row["condition"] == "E1-QueStER" else indexes["questions"]
        ),
        "chunking": f"{chunk_size} / 128",
        "indexes": index_label,
        "retrieval": retrieval,
        "prompt": prompt,
    }


def _conclusion(experiment: str, rows: list[dict]) -> str:
    best = max(rows, key=lambda row: row["metrics"]["evidence_recall@5"])
    base = next(row for row in rows if row["condition"].endswith("ManualPrompt"))
    delta = best["metrics"]["evidence_recall@5"] - base["metrics"]["evidence_recall@5"]
    if experiment == "Experiment 1":
        relation = "Generated questions complement original chunk vectors."
    elif experiment == "Experiment 2":
        relation = (
            "Generated questions are the first layer; original chunks remain "
            "a fallback rather than being replaced."
        )
    else:
        relation = "Atomic questions replace original vectors in dense retrieval."
    variable = (
        "the 1024-token chunk size"
        if "LargeChunk" in best["condition"]
        else "the retrieval/prompt variation"
    )
    if best["stored_vectors"] <= base["stored_vectors"] and delta > 0:
        cost = (
            f"The improvement came from {variable} while reducing stored "
            f"vectors from {base['stored_vectors']:,} to "
            f"{best['stored_vectors']:,}, so the extra-cost tradeoff is "
            "justified on this test."
        )
    else:
        cost = (
            f"The improvement came from {variable} and uses "
            f"{best['stored_vectors']:,} stored vectors; its added cost should "
            "be judged against the reported gain."
        )
    return (
        f"<b>{escape(best['condition'])}</b> has the best test "
        f"Evidence Recall@5 ({best['metrics']['evidence_recall@5']:.3f}, "
        f"{delta:+.3f} versus the experiment base). {relation} {cost}"
    )


def _article_first_section(payload: dict, experiment: str, metrics: tuple) -> str:
    rows = [row for row in payload["conditions"] if row["experiment"] == experiment]
    base = next(row for row in rows if row["condition"].endswith("ManualPrompt"))
    body = ""
    for row in rows:
        cells = "".join(
            f"<td>{_delta(row['metrics'][key], base['metrics'][key])}</td>"
            for key, _ in metrics
        )
        retrieval = {
            "article_dual": (
                "Separate chunk + mapped article-question vectors; 0.5/0.5 score fusion"
            ),
            "article_dual_bm25": (
                "Article-question/chunk vector fusion + BM25 chunk RRF"
            ),
            "article_fallback": (
                "Mapped article-question vectors first; chunk-vector fallback"
            ),
            "article_fallback_bm25": (
                "Mapped article-question vectors first; chunk vector + BM25 fallback"
            ),
            "article_question": (
                "Mapped article-question vectors only; supporting-chunk max score"
            ),
            "article_question_bm25": (
                "Mapped article-question vectors + BM25 supporting-chunk RRF"
            ),
        }[row["retrieval_method"]]
        indexes_used = (
            row["indexes"][1:] if experiment == "Experiment 3B" else row["indexes"]
        )
        indexes = " + ".join(escape(index) for index in indexes_used)
        route = ""
        if row.get("route_analysis"):
            analysis = row["route_analysis"]
            route = (
                f'<div class="small">question route '
                f"{analysis['question_route_queries']}/17; fallback "
                f"{analysis['fallback_route_queries']}/17; weak-match "
                f"fallback ΔR@5 "
                f"{analysis['fallback_delta_evidence_recall@5']:+.3f}</div>"
            )
        body += f"""<tr>
<td class="left"><b>{escape(row["condition"])}</b>{route}</td>
<td class="left">{escape(row["question_generation"])}; evidence mapped to supporting chunk ID(s)</td>
<td>{row["generated_questions"]:,}</td>
<td>{row["chunk_size"]} / {row["chunk_overlap"]}</td>
<td class="left small">{indexes}</td>
<td class="left">{escape(retrieval)}</td>
<td>{escape(row["generation_model"])}<br><span class="small">T=0.3 · seed 42 · JSON</span></td>
{cells}</tr>"""
    heads = "".join(f"<th>{label}</th>" for _, label in metrics)
    best = max(rows, key=lambda row: row["metrics"]["evidence_recall@5"])
    gain = best["metrics"]["evidence_recall@5"] - base["metrics"]["evidence_recall@5"]
    relation = {
        "Experiment 1B": (
            "Article-level generated questions complement the original chunk vectors."
        ),
        "Experiment 2B": (
            "Article-level generated questions are searched first; original "
            "chunks remain a gated fallback."
        ),
        "Experiment 3B": (
            "Article-level generated-question vectors replace original chunk "
            "vectors in dense retrieval."
        ),
    }[experiment]
    number = experiment.removeprefix("Experiment ")
    detail = {
        "Experiment 1B": (
            "question and original-chunk indexes are searched in parallel"
        ),
        "Experiment 2B": (
            "the question index is searched first and weak matches fall back "
            "to the original chunks"
        ),
        "Experiment 3B": ("only the mapped question index is used for dense retrieval"),
    }[experiment]
    return f"""<h3>Experiment {number} — Article-first question generation</h3>
<p class="cap">For each complete article, 10 questions are generated before
chunking. The article is then split, each question is linked to its supporting
chunk ID(s), and {detail}. This changes the question-generation unit and
evidence-mapping order while preserving Experiment {number[0]}'s retrieval
policy.</p>
<table><thead><tr><th>Condition</th><th>Question generation</th>
<th>Generated questions</th><th>Chunk / overlap</th><th>Indexes used</th><th>Retrieval</th>
<th>Generator</th>
{heads}</tr></thead><tbody>{body}</tbody></table>
<p class="conclusion"><b>{escape(best["condition"])}</b> has the best
Evidence Recall@5 ({best["metrics"]["evidence_recall@5"]:.3f},
{gain:+.3f} versus the article-first base), using
{best["stored_vectors"]:,} stored vectors. {relation}</p>"""


def render() -> None:
    payload = json.loads(R.METRICS.read_text())
    remaining = json.loads(X.REMAINING_METRICS.read_text())
    article_first = json.loads(A.METRICS.read_text())
    known = {row["condition"] for row in payload["conditions"]}
    payload["conditions"].extend(
        row for row in remaining["conditions"] if row["condition"] not in known
    )
    manifests = json.loads(R.MANIFEST.read_text())
    metrics = (
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    )
    sections = []
    for experiment in ("Experiment 1", "Experiment 2", "Experiment 3"):
        rows = [row for row in payload["conditions"] if row["experiment"] == experiment]
        base = next(row for row in rows if row["condition"].endswith("ManualPrompt"))
        body = ""
        for row in rows:
            metadata = _metadata(row, manifests)
            cells = "".join(
                f"<td>{_delta(row['metrics'][key], base['metrics'][key])}</td>"
                for key, _ in metrics
            )
            route = ""
            if row["route_analysis"]:
                analysis = row["route_analysis"]
                route = (
                    f'<div class="small">question route '
                    f"{analysis['question_route_queries']}/17; fallback "
                    f"{analysis['fallback_route_queries']}/17; weak-match "
                    f"fallback ΔR@5 "
                    f"{analysis['fallback_delta_evidence_recall@5']:+.3f}</div>"
                )
            generator = row.get("generation_model", "gpt-5.4-mini")
            generator_meta = (
                "T=0 · deterministic JSON"
                if row["condition"] == "E1-QueStER"
                else "T=0.3 · seed 42 · JSON"
            )
            body += f"""<tr>
<td class="left"><b>{escape(row["condition"])}</b>{route}</td>
<td class="left">{escape(metadata["question_method"])}</td>
<td>{metadata["question_count"]:,}</td>
<td>{metadata["chunking"]}</td>
<td class="left small">{metadata["indexes"]}</td>
<td class="left">{escape(metadata["retrieval"])}</td>
<td>{escape(generator)}<br><span class="small">{generator_meta}</span></td>
{cells}</tr>"""
        heads = "".join(f"<th>{label}</th>" for _, label in metrics)
        sections.append(f"""<h2>{experiment}</h2>
<table><thead><tr><th>Condition</th><th>Question generation</th>
<th>Generated questions</th><th>Chunk / overlap</th><th>Indexes used</th><th>Retrieval</th>
<th>Generator</th>
{heads}</tr></thead><tbody>{body}</tbody></table>
<p class="conclusion">{_conclusion(experiment, rows)}</p>""")
        article_experiment = f"{experiment}B"
        if any(
            row["experiment"] == article_experiment
            for row in article_first["conditions"]
        ):
            sections.append(
                _article_first_section(article_first, article_experiment, metrics)
            )

    protocol = payload["protocol"]
    REPORT.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Controlled MultiHop-RAG retrieval experiments</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#dce2ea;--card:#f6f8fb;--good:#08783d;--bad:#bf3b30;--flat:#7c8795}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#29313d;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1800px;margin:auto;padding:30px 18px 70px}}h1{{font-size:25px;margin:0}}h2{{font-size:19px;margin:34px 0 8px}}h3{{font-size:17px;margin:28px 0 8px}}.cap,.small{{color:var(--muted);font-size:11px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border:1px solid var(--line);padding:7px;text-align:center;vertical-align:top}}th{{background:var(--card)}}.left{{text-align:left}}.v{{font-weight:700;font-size:13px}}.d{{font-size:10px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}.conclusion{{background:var(--card);padding:10px 12px;margin:0;border:1px solid var(--line);border-top:0}}
</style></head><body><div class="wrap">
<h1>MultiHop-RAG — three controlled retrieval experiments</h1>
<p class="cap">Final metrics are on the untouched 17-query test split.
Train/development/test = 50/17/17, seed 42, stratified by inference,
temporal, and comparison. Iris dim-384 embeds all chunks, questions, and user
queries. Rank depth={protocol["rank_depth"]}; RRF k={protocol["rrf_k"]}.
Experiment 2 thresholds {protocol["e2_threshold"]:.6f} (vector fallback) and
{protocol["e2_bm25_threshold"]:.6f} (hybrid fallback) were selected only on
development Evidence Recall@5. Every metric cell shows value and change from
that experiment’s base.</p>
{"".join(sections)}
<p class="cap">Question caches and manifests:
<code>{escape(str(G.DATA))}</code>. Machine-readable metrics:
<code>{escape(str(R.METRICS))}</code>. This report is new and does not
overwrite earlier reports.</p>
</div></body></html>""")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    render()
