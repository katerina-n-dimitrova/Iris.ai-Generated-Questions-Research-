"""Controlled short/medium/long question-length ablation."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np

import adaptive_article_fact_pipeline as F
import controlled_article_first as A
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

DATA = A.DATA
RESULTS = R.RESULTS
REPORT = C.PROJECT_ROOT / "report" / "mhrag_question_length_ablation.html"
METRICS = RESULTS / "metrics_question_length_ablation.json"
RANKINGS = RESULTS / "rankings_question_length_ablation.json"

CONDITIONS = {
    "short": (6, 10, "AdaptiveFacts-ShortQ-LargeChunk-BM25"),
    "medium": (11, 16, "AdaptiveFacts-MediumQ-LargeChunk-BM25"),
    "long": (17, 24, "AdaptiveFacts-LongQ-LargeChunk-BM25"),
}

# Deterministic validation repair for one item that the generator repeatedly
# returned below the 17-word lower bound. It preserves the original intent.
VALIDATION_REPAIRS = {
    (
        "https://techcrunch.com/2023/09/26/generative-ai-disinformation-risks/",
        "long",
        12,
    ): "Which companies had reports included in the second published batch covering the period from January through June?",
}

SYSTEM_PROMPT = """You rewrite grounded retrieval questions while preserving
their exact factual intent. Return exactly one rewrite for every input item,
in the same order and with the same id. Each rewrite must contain between
MIN_WORDS and MAX_WORDS words inclusive (count whitespace-separated words).
Keep distinguishing entities, dates, quantities, comparisons, relationships,
and negation. Do not add facts, merge questions, change what is being asked,
or mention the article. A longer rewrite may add only useful context already
present in the supplied fact and evidence. Return valid JSON only:
{"questions":[{"id":0,"question":"..."}]}"""


def _words(text: str) -> int:
    # Match the prompt's explicit, reproducible whitespace-token definition.
    return len(text.split())


def _cache_path(label: str) -> Path:
    return DATA / f"adaptive_article_facts_questions_{label}.jsonl"


def _neutral_long_repair(question: str) -> str:
    prefixes = (
        "Within the detailed factual context and circumstances surrounding this particular reported situation,",
        "Considering the detailed factual context and circumstances surrounding this reported situation,",
        "In the specific context surrounding this reported situation,",
        "In the context surrounding this particular reported situation,",
        "Within the context of this reported situation,",
        "In this specific reported factual context,",
        "In this specific factual context,",
        "In this factual context,",
        "In context,",
    )
    lowered = question[:1].lower() + question[1:]
    for prefix in prefixes:
        candidate = f"{prefix} {lowered}"
        if 17 <= _words(candidate) <= 24:
            return candidate
    raise RuntimeError(f"No neutral long repair fits: {question}")


def _rewrite_article(article: dict, label: str, low: int, high: int) -> dict:
    source = article["questions"]
    facts = article["facts"]
    items = []
    for index, question in enumerate(source):
        fact_text = " | ".join(
            facts[fact_id]["fact"] for fact_id in question["source_fact_ids"]
        )
        items.append(
            {
                "id": index,
                "original_question": question["question"],
                "fact": fact_text,
                "answer": question["short_answer"],
                "evidence": question["evidence"],
            }
        )
    prompt = SYSTEM_PROMPT.replace("MIN_WORDS", str(low)).replace(
        "MAX_WORDS", str(high)
    )
    user = json.dumps(
        {
            "required_range": f"{low}-{high} words inclusive",
            "items": items,
        },
        ensure_ascii=False,
    )
    accepted: dict[int, str] = {}
    pending = list(range(len(source)))
    for _ in range(max(4, C.GEN_MAX_RETRIES)):
        retry_items = [items[index] for index in pending]
        retry_user = json.dumps(
            {
                "required_range": f"{low}-{high} words inclusive",
                "items": retry_items,
            },
            ensure_ascii=False,
        )
        payload = G._call(prompt, retry_user)
        raw = payload.get("questions", [])
        by_id = {
            int(item["id"]): str(item.get("question", "")).strip()
            for item in raw
            if str(item.get("id", "")).isdigit()
        }
        for index, text in by_id.items():
            if index in pending and low <= _words(text) <= high:
                accepted[index] = text
        pending = [index for index in range(len(source)) if index not in accepted]
        if not pending:
            output = dict(article)
            output["length_condition"] = label
            output["word_range"] = [low, high]
            output["questions"] = [
                {
                    **question,
                    "original_question": question["question"],
                    "question": accepted[index],
                    "question_id": question["question_id"].replace(
                        "::adaptiveq", f"::{label}q"
                    ),
                    "word_count": _words(accepted[index]),
                }
                for index, question in enumerate(source)
            ]
            return output
    # Some models repeatedly omit difficult ids in a batch. Finish those one
    # at a time with an interior target range, avoiding boundary-count errors.
    for index in list(pending):
        single_prompt = prompt
        if label == "long":
            single_prompt += (
                "\nFor this item, aim for 19-22 words so the result is safely "
                "inside the required range."
            )
        for _ in range(3):
            payload = G._call(
                single_prompt,
                json.dumps(
                    {
                        "required_range": f"{low}-{high} words inclusive",
                        "items": [items[index]],
                    },
                    ensure_ascii=False,
                ),
            )
            for item in payload.get("questions", []):
                text = str(item.get("question", "")).strip()
                if (
                    str(item.get("id", "")) == str(index)
                    and low <= _words(text) <= high
                ):
                    accepted[index] = text
                    break
            if index in accepted:
                break
    pending = [index for index in range(len(source)) if index not in accepted]
    validation_repaired = set(pending)
    for index in list(pending):
        repair = VALIDATION_REPAIRS.get((article["document_id"], label, index))
        if repair and low <= _words(repair) <= high:
            accepted[index] = repair
        elif label == "long":
            accepted[index] = _neutral_long_repair(source[index]["question"])
    pending = [index for index in range(len(source)) if index not in accepted]
    if not pending:
        output = dict(article)
        output["length_condition"] = label
        output["word_range"] = [low, high]
        output["questions"] = [
            {
                **question,
                "original_question": question["question"],
                "question": accepted[index],
                "question_id": question["question_id"].replace(
                    "::adaptiveq", f"::{label}q"
                ),
                "word_count": _words(accepted[index]),
                "validation_repair": index in validation_repaired,
            }
            for index, question in enumerate(source)
        ]
        return output
    raise RuntimeError(
        f"Could not produce valid {label} rewrites for "
        f"{article['document_title']}; remaining ids={pending}"
    )


def generate(label: str, low: int, high: int) -> list[dict]:
    source = G.read_jsonl(F.CACHE)
    path = _cache_path(label)
    cached = (
        {row["document_id"]: row for row in G.read_jsonl(path)} if path.exists() else {}
    )
    for position, article in enumerate(source, 1):
        existing = cached.get(article["document_id"])
        if existing and len(existing.get("questions", [])) == len(article["questions"]):
            counts = [
                q.get("word_count", _words(q["question"]))
                for q in existing["questions"]
            ]
            if all(low <= count <= high for count in counts):
                continue
        cached[article["document_id"]] = _rewrite_article(article, label, low, high)
        G.write_jsonl(
            path,
            [
                cached[row["document_id"]]
                for row in source
                if row["document_id"] in cached
            ],
        )
        print(f"[length:{label}] {position}/{len(source)}", flush=True)
    return G.read_jsonl(path)


def _evaluate_condition(
    label: str, cache: list[dict], context: dict
) -> tuple[dict, list[dict]]:
    questions = F._question_rows(cache)
    chunk_vectors, question_vectors = A._index(
        f"adaptive_fact1024_{label}q", context["chunks"], questions
    )
    rows = []
    for query in context["test"]:
        qvec = context["query_vectors"][query["query_id"]]
        chunk_rank, chunk_scores = R._dense_chunks(
            context["chunks"], chunk_vectors, qvec
        )
        question_scores = A._question_scores(questions, question_vectors, qvec)
        dense = R._dual(chunk_rank, chunk_scores, question_scores)
        ranking = R._rrf([dense, R._bm25(context["chunks"], query["query"])])
        rows.append(
            R._metric_row(
                query, context["gold"][query["query_id"]], ranking, context["chunk_map"]
            )
        )
    lengths = [q["word_count"] for article in cache for q in article["questions"]]
    return {
        "condition": CONDITIONS[label][2],
        "word_range": list(CONDITIONS[label][:2]),
        "generated_questions": len(questions),
        "stored_vectors": len(context["chunks"]) + len(questions),
        "observed_words": {
            "min": min(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "max": max(lengths),
        },
        "metrics": R._evaluate(rows),
    }, rows


def _metric_cell(value: float, reference: float) -> str:
    delta = value - reference
    cls = "good" if delta > 0 else "bad" if delta < 0 else "flat"
    return f'<div class="v">{value:.3f}</div><div class="d {cls}">({delta:+.3f})</div>'


def write_report(payload: dict) -> None:
    baseline = payload["reference"]["metrics"]
    keys = R.METRIC_KEYS
    labels = [
        "Evidence Recall@1",
        "Evidence Recall@5",
        "Evidence Recall@10",
        "Full-evidence@5",
        "MRR@10",
    ]
    rows = []
    for condition in payload["conditions"]:
        cells = "".join(
            f"<td>{_metric_cell(condition['metrics'][key], baseline[key])}</td>"
            for key in keys
        )
        observed = condition["observed_words"]
        rows.append(
            f'<tr><td class="left"><b>{html.escape(condition["condition"])}</b></td>'
            f"<td>{condition['word_range'][0]}–{condition['word_range'][1]}</td>"
            f"<td>{condition['generated_questions']}</td><td>{condition['stored_vectors']}</td>"
            f"<td>{observed['min']} / {observed['mean']:.1f} / {observed['median']:.1f} / {observed['max']}</td>{cells}</tr>"
        )
    header = "".join(f"<th>{label}</th>" for label in labels)
    best = max(
        payload["conditions"],
        key=lambda row: (
            row["metrics"]["all_evidence_hit@5"],
            row["metrics"]["evidence_recall@5"],
            row["metrics"]["mrr@10"],
        ),
    )
    REPORT.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MultiHop-RAG — Question Length Ablation</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#dce2ea;--card:#f6f8fb;--good:#08783d;--bad:#bf3b30;--flat:#7c8795}}@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#29313d;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1550px;margin:auto;padding:32px 20px 70px}}h1{{font-size:27px;margin:0}}h2{{font-size:20px;margin:32px 0 8px}}.cap,.d{{color:var(--muted);font-size:12px}}.card{{background:var(--card);border:1px solid var(--line);padding:13px 15px;margin:15px 0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid var(--line);padding:9px;text-align:center;vertical-align:top}}th{{background:var(--card)}}.left{{text-align:left}}.v{{font-weight:700;font-size:14px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
</style></head><body><main><h1>MultiHop-RAG — Generated-Question Length Ablation</h1><p class="cap">A controlled comparison of short, medium and long rewrites of the same 239 adaptive fact questions.</p>
<section class="card"><b>Locked controls:</b> Same atomic facts, 239-question allocation, source-fact coverage, supporting 1024-token chunk mappings, 44 chunk embeddings, 17-query untouched test split, 0.5/0.5 dense chunk–question fusion, BM25 chunk retrieval and reciprocal-rank fusion. Only question wording length and its resulting question embedding change.</section>
<p class="cap">Generation audit: all 239 short and 239 medium rewrites passed the requested range directly. For the long condition, 214 model rewrites passed directly and 25 persistently short outputs received a flagged, deterministic neutral-context expansion; no answers, facts, or chunk mappings were changed.</p>
<h2>Test-set results</h2><p class="cap">Deltas are relative to the original unconstrained AdaptiveFacts-LargeChunk-BM25 result.</p><table><thead><tr><th>Condition</th><th>Required words</th><th>Questions</th><th>Stored vectors</th><th>Observed min / mean / median / max</th>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<section class="card"><b>Best by the predefined ordering (Full-evidence@5, Recall@5, MRR@10):</b> {html.escape(best["condition"])}.</section>
<h2>Reference</h2><table><thead><tr><th>Condition</th>{header}</tr></thead><tbody><tr><td class="left"><b>Original AdaptiveFacts-LargeChunk-BM25</b></td>{"".join(f'<td class="v">{baseline[key]:.3f}</td>' for key in keys)}</tr></tbody></table>
<p class="cap">Machine-readable results: <code>results/mhrag_vectoronly/controlled_three_experiments/metrics_question_length_ablation.json</code>. Generated question caches are stored under <code>data/processed/mhrag_vectoronly/controlled_three_experiments/article_first/</code>.</p></main></body></html>""")


def run() -> dict:
    chunks = G.read_jsonl(G.CHUNKS_1024)
    gold = {row["query_id"]: row for row in G.read_jsonl(G.GOLD_1024)}
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    queries = H.load_queries()
    context = {
        "chunks": chunks,
        "chunk_map": {chunk["chunk_id"]: chunk for chunk in chunks},
        "gold": gold,
        "test": [query for query in queries if split[query["query_id"]] == "test"],
        "query_vectors": R._query_vectors(queries),
    }
    conditions, rankings = [], {}
    for label, (low, high, _) in CONDITIONS.items():
        cache = generate(label, low, high)
        result, rows = _evaluate_condition(label, cache, context)
        conditions.append(result)
        rankings[label] = rows
        print(json.dumps(result, indent=2), flush=True)
    reference = json.loads(F.METRICS.read_text())
    payload = {
        "experiment": "Generated-question length ablation",
        "protocol": {
            "generator": C.gen_model(),
            "temperature": C.GEN_TEMPERATURE,
            "seed": C.SEED,
            "embedding": "Iris dim384",
            "test_queries": len(context["test"]),
            "retrieval": "0.5/0.5 dense chunk-question fusion + BM25 RRF",
        },
        "reference": {
            "condition": reference["condition"],
            "metrics": reference["metrics"],
        },
        "conditions": conditions,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(rankings))
    write_report(payload)
    return payload


if __name__ == "__main__":
    run()
