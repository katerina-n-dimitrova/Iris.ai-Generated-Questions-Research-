"""Article-only Baseline C and Adaptive facts C retrieval experiments."""

from __future__ import annotations

import html
import json

import numpy as np
from rank_bm25 import BM25Okapi

import adaptive_article_fact_pipeline as F
import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

METRICS = R.RESULTS / "metrics_article_only_variants.json"
RANKINGS = R.RESULTS / "rankings_article_only_variants.json"
REPORT = C.PROJECT_ROOT / "report" / "mhrag_article_only_variants.html"


def _fixed_questions() -> list[dict]:
    return [
        {
            "id": question["question_id"],
            "text": question["question"],
            "document_id": row["document_id"],
        }
        for row in G.read_jsonl(H.DOCQ_PATH)
        for question in row["questions"]
    ]


def _adaptive_questions() -> list[dict]:
    return [
        {
            "id": question["question_id"],
            "text": question["question"],
            "document_id": row["document_id"],
        }
        for row in G.read_jsonl(F.CACHE)
        for question in row["questions"]
    ]


def _question_scores(
    questions: list[dict],
    vectors: np.ndarray,
    query_vector: np.ndarray,
) -> dict[str, float]:
    scores = H._cosine_matrix(vectors, query_vector)
    best: dict[str, float] = {}
    for question, score in zip(questions, scores):
        document_id = question["document_id"]
        best[document_id] = max(best.get(document_id, -1.0), float(score))
    return best


def _dense(
    articles: list[dict],
    article_vectors: np.ndarray,
    questions: list[dict],
    question_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> list[str]:
    article_scores_raw = H._cosine_matrix(article_vectors, query_vector)
    article_scores = {
        article["article_id"]: float(score)
        for article, score in zip(articles, article_scores_raw)
    }
    question_scores = _question_scores(questions, question_vectors, query_vector)
    fused = {
        article["article_id"]: (
            0.5 * article_scores[article["article_id"]]
            + 0.5 * question_scores.get(article["article_id"], -1.0)
        )
        for article in articles
    }
    return [
        key for key, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]


def _bm25_articles(
    articles: list[dict],
    model: BM25Okapi,
    query: str,
) -> list[str]:
    scores = model.get_scores(H.tokens(query))
    return [articles[index]["article_id"] for index in H._rank_indices(scores)]


def _per_query(query: dict, ranking: list[str]) -> dict:
    required = set(query["required_article_ids"])
    row = {
        "query_id": query["query_id"],
        "question_type": query["question_type"],
        "required_article_ids": sorted(required),
        "ranked_article_ids": ranking,
    }
    for k in (1, 5, 10):
        hit = required & set(ranking[:k])
        row[f"article_recall@{k}"] = len(hit) / len(required) if required else 0.0
    row["all_required_articles@5"] = float(required <= set(ranking[:5]))
    first = next(
        (
            rank
            for rank, article_id in enumerate(ranking[:10], 1)
            if article_id in required
        ),
        None,
    )
    row["article_mrr@10"] = 1.0 / first if first else 0.0
    return row


def _evaluate(rows: list[dict]) -> dict:
    keys = (
        "article_recall@1",
        "article_recall@5",
        "article_recall@10",
        "all_required_articles@5",
        "article_mrr@10",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _write_report(payload: dict) -> None:
    keys = (
        "article_recall@1",
        "article_recall@5",
        "article_recall@10",
        "all_required_articles@5",
        "article_mrr@10",
    )
    labels = (
        "Article Recall@1",
        "Article Recall@5",
        "Article Recall@10",
        "All required articles@5",
        "Article MRR@10",
    )
    reference = payload["conditions"][0]["metrics"]
    body = []
    for condition in payload["conditions"]:
        cells = []
        for key in keys:
            value = condition["metrics"][key]
            delta = value - reference[key]
            cls = "good" if delta > 0 else "bad" if delta < 0 else "flat"
            cells.append(
                f'<td><div class="v">{value:.3f}</div><div class="d {cls}">({delta:+.3f})</div></td>'
            )
        body.append(
            f'<tr><td class="left"><b>{html.escape(condition["condition"])}</b></td>'
            f'<td class="left">{html.escape(condition["question_generation"])}</td>'
            f"<td>{condition['generated_questions']}</td><td>{condition['articles_embedded']}</td>"
            f"<td>{condition['stored_vectors']}</td>{''.join(cells)}</tr>"
        )
    header = "".join(f"<th>{label}</th>" for label in labels)
    winner = max(
        payload["conditions"],
        key=lambda row: (
            row["metrics"]["all_required_articles@5"],
            row["metrics"]["article_recall@5"],
            row["metrics"]["article_mrr@10"],
        ),
    )
    REPORT.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MultiHop-RAG — Article-only C Experiments</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#dce2ea;--card:#f6f8fb;--good:#08783d;--bad:#bf3b30;--flat:#7c8795}}@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#29313d;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1550px;margin:auto;padding:32px 20px 70px}}h1{{font-size:27px;margin:0}}h2{{font-size:20px;margin:32px 0 8px}}.cap,.d{{color:var(--muted);font-size:12px}}.card{{background:var(--card);border:1px solid var(--line);padding:13px 15px;margin:15px 0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid var(--line);padding:9px;text-align:center;vertical-align:top}}th{{background:var(--card)}}.left{{text-align:left}}.v{{font-weight:700;font-size:14px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}code{{overflow-wrap:anywhere}}
</style></head><body><main><h1>MultiHop-RAG — Baseline C and Adaptive Facts C</h1><p class="cap">True article-level retrieval: complete articles are not chunked or embedded as chunks. Deltas are relative to Baseline C.</p>
<section class="card"><b>Retrieval pipeline:</b> 0.5 complete-article cosine + 0.5 best parent-question cosine, combined through RRF with BM25 over complete article text. The returned units are complete articles.</section>
<section class="card"><b>Important metric boundary:</b> Chunk Evidence Recall is not defined in this experiment because no chunks are retrieved. The analogous article-level metrics measure coverage and rank of the gold source articles. These values must not be directly compared numerically with chunk Recall@k from Baseline B or Adaptive Facts B.</section>
<h2>Article-level test results</h2><table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Articles embedded</th><th>Stored vectors</th>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>
<section class="card"><b>Best condition:</b> {html.escape(winner["condition"])}, selected by all-required-articles@5, then Article Recall@5 and Article MRR@10.</section>
<h2>What is indexed</h2><table><tbody><tr><th class="left">Dense article index</th><td class="left">15 complete-article embeddings</td></tr><tr><th class="left">Dense question index</th><td class="left">150 fixed questions for Baseline C or 239 adaptive-fact questions for Adaptive Facts C</td></tr><tr><th class="left">Sparse index</th><td class="left">BM25 over 15 complete articles</td></tr><tr><th class="left">Excluded</th><td class="left">Chunk creation, chunk embeddings, chunk-vector index, chunk BM25 and question-to-chunk score mapping</td></tr></tbody></table>
<p class="cap">Machine-readable results: <code>results/mhrag_vectoronly/controlled_three_experiments/metrics_article_only_variants.json</code>.</p></main></body></html>""")


def run() -> dict:
    articles = H.read_jsonl(H.ARTICLES_PATH)
    queries = H.load_queries()
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    test = [query for query in queries if split[query["query_id"]] == "test"]
    query_vectors = R._query_vectors(queries)
    article_vectors = R._embed_cached(
        "whole_articles", [article["cleaned_body"] for article in articles]
    )
    bm25 = BM25Okapi([H.tokens(article["cleaned_body"]) for article in articles])
    configs = (
        (
            "Baseline C",
            "Fixed 10 questions per complete article",
            _fixed_questions(),
            "article_article_manual1024_questions",
        ),
        (
            "Adaptive facts C",
            "Adaptive 5–20 questions from distinct-fact analysis",
            _adaptive_questions(),
            "article_adaptive_fact1024_questions",
        ),
    )
    conditions, rankings = [], {}
    for condition, generation, questions, vector_cache in configs:
        question_vectors = R._embed_cached(
            vector_cache, [question["text"] for question in questions]
        )
        rows = []
        for query in test:
            dense = _dense(
                articles,
                article_vectors,
                questions,
                question_vectors,
                query_vectors[query["query_id"]],
            )
            ranking = R._rrf([dense, _bm25_articles(articles, bm25, query["query"])])
            rows.append(_per_query(query, ranking))
        rankings[condition] = rows
        conditions.append(
            {
                "condition": condition,
                "question_generation": generation,
                "generated_questions": len(questions),
                "articles_embedded": len(articles),
                "chunks_created": 0,
                "chunks_embedded": 0,
                "stored_vectors": len(articles) + len(questions),
                "embedding": "Iris dim384",
                "dense_fusion": "0.5 complete article + 0.5 best parent question cosine",
                "sparse_retrieval": "BM25 over complete articles, fused with dense ranking by RRF k=60",
                "metrics": _evaluate(rows),
            }
        )
    payload = {
        "experiment": "Article-only C variants",
        "protocol": {
            "test_queries": len(test),
            "retrieval_unit": "complete article",
            "chunking": "disabled",
            "chunk_embeddings": 0,
            "metric_unit": "required source articles",
        },
        "conditions": conditions,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    RANKINGS.write_text(json.dumps(rankings))
    _write_report(payload)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
