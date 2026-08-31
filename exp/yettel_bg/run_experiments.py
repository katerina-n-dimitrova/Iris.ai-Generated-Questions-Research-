#!/usr/bin/env python3
"""Run the four requested Yettel retrieval experiments and render one report."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "multihoprag"))
import adaptive_lib as A  # noqa: E402
import chunk_article_questions as C  # noqa: E402
from ragkit.embeddings import embedding_signature  # noqa: E402
from ragkit.fusion import rrf_merge  # noqa: E402
from ragkit.text import read_jsonl as _read_jsonl  # noqa: E402
from ragkit.text import tokenize_unicode as tokenize  # noqa: E402
from ragkit.vectors import normalized_dot  # noqa: E402

SOURCE = ROOT / "data" / "processed" / "yettel_bg"
DATA = ROOT / "data" / "processed" / "yettel_bg_experiments"
RESULTS = ROOT / "results" / "yettel_bg_experiments"
REPORT = ROOT / "report" / "yettel_bg_adaptive_questions.html"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

A.DATA = DATA
A.RESULTS = RESULTS
A.REPORT = REPORT
A.CHUNKS_PATH = DATA / "chunks.jsonl"
A.QUERIES_PATH = DATA / "queries.jsonl"
A.SUMMARY_PATH = DATA / "summary.json"
A.GEN_PATH = DATA / "adaptive_generations.jsonl"
A.CHUNK_VECTORS = RESULTS / "chunk_vectors.json"
A.QUERY_VECTORS = RESULTS / "query_vectors.json"
A.BOUNDED_VECTORS = RESULTS / "bounded_question_vectors.json"
A.UNBOUNDED_VECTORS = RESULTS / "unbounded_question_vectors.json"
A.METRICS = RESULTS / "metrics.json"
A.RANKINGS = RESULTS / "rankings.json"
A.MAX_WORKERS = 24
A.MAX_RETRIES = 6
A.FACT_PROMPT = """Анализирай предоставения откъс от Yettel. Извлечи всички
значими атомарни факти, които са изрично подкрепени от текста, и премахни
повторенията. Игнорирай заглавия, общи рекламни фрази и неподкрепени изводи.
За всеки факт върни кратко твърдение, кратък дословен цитат, importance 1-5 и
distinctiveness 1-5. Върни само валиден JSON:
{"facts":[{"fact":"...","evidence":"...","importance":1,"distinctiveness":1}]}"""
A.QUESTION_PROMPT = """Създай точно заявения брой разнообразни и обосновани
въпроси за търсене на български език от откъса и дедупликираните атомарни
факти. Всеки въпрос трябва да може да се отговори само от откъса. Избягвай
почти еднакви въпроси. Върни само валиден JSON:
{"questions":[{"question":"...","source_fact_ids":[0]}]}"""

C.DATA = DATA
C.RESULTS = RESULTS
C.REPORT = REPORT
C.ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
C.ARTICLE_VECTORS = RESULTS / "article_question_vectors.json"
C.ARTICLE_RANKINGS = RESULTS / "article_chunk_question_rankings.jsonl"
C.MAX_WORKERS = 8

UNBOUNDED_COMBINED = "Adaptive chunk + whole-article questions unbounded"
UNBOUNDED_ARTICLE_GENERATIONS = DATA / "article_question_generations_unbounded.jsonl"
UNBOUNDED_ARTICLE_VECTORS = RESULTS / "article_question_vectors_unbounded.json"
UNBOUNDED_ARTICLE_RANKINGS = RESULTS / "article_chunk_question_unbounded_rankings.jsonl"

A.tokenize = tokenize
A.cosine_scores = C.cosine


def read_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(path, encoding="utf-8")


def prepare() -> tuple[list[dict], list[dict], dict]:
    if A.CHUNKS_PATH.exists() and A.QUERIES_PATH.exists() and A.SUMMARY_PATH.exists():
        return (
            A.read_jsonl(A.CHUNKS_PATH),
            A.read_jsonl(A.QUERIES_PATH),
            json.loads(A.SUMMARY_PATH.read_text()),
        )
    source_chunks = read_jsonl(SOURCE / "chunks_1024.jsonl")
    chunks = [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "document_title": row["title"],
            "chunk_position": int(row["chunk_id"].rsplit("c", 1)[-1]),
            "n_tokens": row["token_count"],
            "content": row["text"],
        }
        for row in source_chunks
    ]
    raw_queries = read_jsonl(SOURCE / "questions.jsonl")
    queries = []
    for row in raw_queries:
        if row["question_type"] == "null_query":
            continue
        units = [[chunk_id] for chunk_id in row["gold_chunk_ids"]]
        queries.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "question_type": row["question_type"].replace("_query", ""),
                "required_article_ids": row["gold_document_ids"],
                "n_required_documents": len(row["gold_document_ids"]),
                "n_required_evidence_facts": len(units),
                "evidence_units": units,
                "gold_chunk_ids": row["gold_chunk_ids"],
            }
        )
    summary = {
        "articles": 340,
        "chunks": len(chunks),
        "eligible_queries": len(queries),
        "unresolved_queries": 0,
        "alignment_methods": {
            "canonical_chunk_id": sum(len(q["evidence_units"]) for q in queries)
        },
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "selection_seed": 20260817,
        "null_queries_excluded": len(raw_queries) - len(queries),
    }
    A.write_jsonl(A.CHUNKS_PATH, chunks)
    A.write_jsonl(A.QUERIES_PATH, queries)
    A.SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return chunks, queries, summary


def generate_until_complete(chunks: list[dict]) -> list[dict]:
    while True:
        try:
            return A.generate(chunks)
        except RuntimeError as error:
            print(f"[generation-retry] {error}", flush=True)
            time.sleep(2)


def baseline(chunks: list[dict], queries: list[dict]) -> dict:
    chunk_vectors = A.embed_resumable(
        A.CHUNK_VECTORS, [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        A.QUERY_VECTORS, [row["query"] for row in queries]
    )
    chunk_ids = [row["chunk_id"] for row in chunks]
    bm25 = BM25Okapi([tokenize(row["content"]) for row in chunks])
    rows = []
    rankings = []
    for qi, query in enumerate(queries):
        dense = [
            chunk_ids[i]
            for i in np.argsort(-C.cosine(chunk_vectors, query_vectors[qi]))
        ]
        sparse = [
            chunk_ids[i] for i in np.argsort(-bm25.get_scores(tokenize(query["query"])))
        ]
        ranking = rrf_merge((dense, sparse), A.RRF_K)
        metric = A.metric_row(query, ranking)
        rows.append(metric)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": metric,
            }
        )
        if (qi + 1) % 250 == 0:
            print(f"[retrieve:Baseline] {qi + 1}/{len(queries)}", flush=True)
    keys = list(rows[0])
    A.write_jsonl(RESULTS / "baseline_rankings.jsonl", rankings)
    return {
        "condition": "Baseline",
        "question_rule": "No generated questions",
        "quality": {
            "generated_questions": 0,
            "questions_per_chunk_min": 0,
            "questions_per_chunk_mean": 0,
            "questions_per_chunk_max": 0,
        },
        "stored_vectors": len(chunks),
        "metrics": {key: float(np.mean([row[key] for row in rows])) for key in keys},
    }


def article_inputs(chunks: list[dict], generated: list[dict]) -> list[dict]:
    documents = {
        row["document_id"]: row for row in read_jsonl(SOURCE / "documents.jsonl")
    }
    generated_by_chunk = {row["chunk_id"]: row for row in generated}
    chunks_by_document = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk)
    articles = []
    for document_id in sorted(chunks_by_document):
        document_chunks = sorted(
            chunks_by_document[document_id], key=lambda row: row["chunk_position"]
        )
        facts, seen = [], set()
        for chunk in document_chunks:
            for fact in generated_by_chunk[chunk["chunk_id"]]["facts"]:
                key = A.normalize(fact["fact"])
                if key and key not in seen:
                    seen.add(key)
                    facts.append(fact)
        articles.append(
            {
                "article_id": document_id,
                "title": documents[document_id]["title"],
                "content": documents[document_id]["body"],
                "chunk_ids": [row["chunk_id"] for row in document_chunks],
                "facts": facts,
                "question_budget": min(20, max(5, round(len(facts) * 0.5))),
            }
        )
    return articles


def adaptive_fast(
    chunks: list[dict], queries: list[dict], generated: list[dict]
) -> list[dict]:
    """Recompute the two chunk-question arms with Unicode BM25 and cached vectors."""
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vectors = A.embed_resumable(
        A.CHUNK_VECTORS, [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        A.QUERY_VECTORS, [row["query"] for row in queries]
    )
    chunk_score = normalized_dot(query_vectors, chunk_vectors)
    bm25 = BM25Okapi([tokenize(row["content"]) for row in chunks])
    ranking_output = {}
    conditions = []
    for name, field, vector_path, rule in (
        (
            "Adaptive generated questions 5–20",
            "bounded_questions",
            A.BOUNDED_VECTORS,
            "clamp(5, 20, round(facts * 0.5))",
        ),
        (
            "Adaptive generated questions unbounded",
            "unbounded_questions",
            A.UNBOUNDED_VECTORS,
            "round(facts * 0.5), no bounds",
        ),
    ):
        questions = [
            (question["question"], row["chunk_id"])
            for row in generated
            for question in row[field]
        ]
        question_vectors = A.embed_resumable(vector_path, [row[0] for row in questions])
        question_score = normalized_dot(query_vectors, question_vectors)
        q_indices = defaultdict(list)
        for index, (_, chunk_id) in enumerate(questions):
            q_indices[chunk_id].append(index)
        chunk_q_max = np.column_stack(
            [
                question_score[:, q_indices[cid]].max(axis=1)
                if q_indices[cid]
                else chunk_score[:, i]
                for i, cid in enumerate(chunk_ids)
            ]
        )
        fused = 0.5 * chunk_score + 0.5 * chunk_q_max
        metrics, rankings = [], []
        for qi, query in enumerate(queries):
            dense = [chunk_ids[i] for i in np.argsort(-fused[qi])]
            sparse = [
                chunk_ids[i]
                for i in np.argsort(-bm25.get_scores(tokenize(query["query"])))
            ]
            ranking = rrf_merge((dense, sparse), A.RRF_K)
            metric = A.metric_row(query, ranking)
            metrics.append(metric)
            rankings.append(
                {
                    "query_id": query["query_id"],
                    "ranked_chunk_ids": ranking[:10],
                    "metrics": metric,
                }
            )
            if (qi + 1) % 250 == 0:
                print(f"[retrieve:{name}] {qi + 1}/{len(queries)}", flush=True)
        counts = [len(row[field]) for row in generated]
        keys = list(metrics[0])
        ranking_output[name] = rankings
        conditions.append(
            {
                "condition": name,
                "question_rule": rule,
                "quality": {
                    "generated_questions": sum(counts),
                    "questions_per_chunk_min": min(counts),
                    "questions_per_chunk_mean": float(np.mean(counts)),
                    "questions_per_chunk_max": max(counts),
                },
                "stored_vectors": len(chunks) + sum(counts),
                "metrics": {
                    key: float(np.mean([row[key] for row in metrics])) for key in keys
                },
            }
        )
        print(f"[done] {name}", flush=True)
    A.RANKINGS.write_text(json.dumps(ranking_output), encoding="utf-8")
    return conditions


def combined_fast(
    chunks: list[dict],
    queries: list[dict],
    generated: list[dict],
    article_generated: list[dict],
) -> dict:
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vectors = A.embed_resumable(
        A.CHUNK_VECTORS, [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        A.QUERY_VECTORS, [row["query"] for row in queries]
    )
    chunk_questions = [
        (question["question"], row["chunk_id"])
        for row in generated
        for question in row["bounded_questions"]
    ]
    chunk_q_vectors = A.embed_resumable(
        A.BOUNDED_VECTORS, [row[0] for row in chunk_questions]
    )
    article_questions = [
        (question["question"], row["article_id"])
        for row in article_generated
        for question in row["questions"]
    ]
    article_q_vectors = A.embed_resumable(
        C.ARTICLE_VECTORS, [row[0] for row in article_questions]
    )

    chunk_score = normalized_dot(query_vectors, chunk_vectors)
    question_score = normalized_dot(query_vectors, chunk_q_vectors)
    article_score = normalized_dot(query_vectors, article_q_vectors)
    q_indices = defaultdict(list)
    for index, (_, chunk_id) in enumerate(chunk_questions):
        q_indices[chunk_id].append(index)
    a_indices = defaultdict(list)
    for index, (_, document_id) in enumerate(article_questions):
        a_indices[document_id].append(index)
    chunk_q_max = np.column_stack(
        [question_score[:, q_indices[cid]].max(axis=1) for cid in chunk_ids]
    )
    article_q_max = np.column_stack(
        [article_score[:, a_indices[row["document_id"]]].max(axis=1) for row in chunks]
    )
    dense_score = (chunk_score + chunk_q_max + article_q_max) / 3
    bm25 = BM25Okapi([tokenize(row["content"]) for row in chunks])
    metrics, rankings = [], []
    for qi, query in enumerate(queries):
        dense = [chunk_ids[i] for i in np.argsort(-dense_score[qi])]
        sparse = [
            chunk_ids[i] for i in np.argsort(-bm25.get_scores(tokenize(query["query"])))
        ]
        ranking = rrf_merge((dense, sparse), A.RRF_K)
        row = A.metric_row(query, ranking)
        metrics.append(row)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": row,
            }
        )
        if (qi + 1) % 250 == 0:
            print(f"[retrieve:{C.CONDITION}] {qi + 1}/{len(queries)}", flush=True)
    A.write_jsonl(C.ARTICLE_RANKINGS, rankings)
    chunk_counts = [len(row["bounded_questions"]) for row in generated]
    article_counts = [len(row["questions"]) for row in article_generated]
    keys = list(metrics[0])
    return {
        "condition": C.CONDITION,
        "question_rule": "chunk and document: clamp(5, 20, round(facts * 0.5))",
        "quality": {
            "generated_questions": sum(chunk_counts) + sum(article_counts),
            "chunk_questions": sum(chunk_counts),
            "article_questions": sum(article_counts),
            "questions_per_chunk_min": min(chunk_counts),
            "questions_per_chunk_max": max(chunk_counts),
            "questions_per_article_min": min(article_counts),
            "questions_per_article_max": max(article_counts),
        },
        "stored_vectors": len(chunks) + sum(chunk_counts) + sum(article_counts),
        "metrics": {key: float(np.mean([row[key] for row in metrics])) for key in keys},
    }


def generate_unbounded_articles(articles: list[dict]) -> list[dict]:
    """Generate round(article facts * 0.5) questions without a 5--20 clamp."""
    cached = (
        {row["article_id"]: row for row in read_jsonl(UNBOUNDED_ARTICLE_GENERATIONS)}
        if UNBOUNDED_ARTICLE_GENERATIONS.exists()
        else {}
    )
    todo = [article for article in articles if article["article_id"] not in cached]
    print(
        f"[unbounded-article-generation] cached={len(cached)} todo={len(todo)}",
        flush=True,
    )

    def generate_one(article: dict) -> dict:
        facts = article["facts"]
        budget = round(len(facts) * 0.5)
        proxy = {"chunk_id": article["article_id"], "content": article["content"]}
        questions = A.generate_questions(proxy, facts, budget)
        for question in questions:
            question["supporting_article_id"] = article["article_id"]
            question["supporting_chunk_ids"] = article["chunk_ids"]
        return {
            "article_id": article["article_id"],
            "title": article["title"],
            "chunk_ids": article["chunk_ids"],
            "n_distinct_article_facts": len(facts),
            "question_budget": budget,
            "questions": questions,
        }

    failures = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with UNBOUNDED_ARTICLE_GENERATIONS.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=C.MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_one, article): article for article in todo
            }
            for position, future in enumerate(as_completed(futures), 1):
                article = futures[future]
                try:
                    row = future.result()
                    cached[article["article_id"]] = row
                    checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    print(
                        f"[unbounded-article-generation] {position}/{len(todo)} "
                        f"{article['article_id']} questions={len(row['questions'])}",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(
                        {"article_id": article["article_id"], "error": str(error)}
                    )
                    print(
                        f"[unbounded-article-generation:error] {article['article_id']}: {error}",
                        flush=True,
                    )
    if failures:
        (DATA / "article_question_unbounded_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} unbounded articles need a cached retry")
    ordered = [cached[article["article_id"]] for article in articles]
    A.write_jsonl(UNBOUNDED_ARTICLE_GENERATIONS, ordered)
    return ordered


def combined_unbounded_fast(
    chunks: list[dict],
    queries: list[dict],
    generated: list[dict],
    article_generated: list[dict],
) -> dict:
    """Evaluate equal-third fusion with unbounded chunk and article questions."""
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vectors = A.embed_resumable(
        A.CHUNK_VECTORS, [row["content"] for row in chunks]
    )
    query_vectors = A.embed_resumable(
        A.QUERY_VECTORS, [row["query"] for row in queries]
    )
    chunk_questions = [
        (q["question"], row["chunk_id"])
        for row in generated
        for q in row["unbounded_questions"]
    ]
    article_questions = [
        (q["question"], row["article_id"])
        for row in article_generated
        for q in row["questions"]
    ]
    chunk_q_vectors = A.embed_resumable(
        A.UNBOUNDED_VECTORS, [x[0] for x in chunk_questions]
    )
    article_q_vectors = A.embed_resumable(
        UNBOUNDED_ARTICLE_VECTORS, [x[0] for x in article_questions]
    )
    chunk_score = normalized_dot(query_vectors, chunk_vectors)
    question_score = normalized_dot(query_vectors, chunk_q_vectors)
    article_score = normalized_dot(query_vectors, article_q_vectors)
    q_indices, a_indices = defaultdict(list), defaultdict(list)
    for index, (_, owner) in enumerate(chunk_questions):
        q_indices[owner].append(index)
    for index, (_, owner) in enumerate(article_questions):
        a_indices[owner].append(index)
    chunk_q_max = np.column_stack(
        [
            question_score[:, q_indices[cid]].max(axis=1)
            if q_indices[cid]
            else chunk_score[:, i]
            for i, cid in enumerate(chunk_ids)
        ]
    )
    article_q_max = np.column_stack(
        [article_score[:, a_indices[row["document_id"]]].max(axis=1) for row in chunks]
    )
    dense_score = (chunk_score + chunk_q_max + article_q_max) / 3
    bm25 = BM25Okapi([tokenize(row["content"]) for row in chunks])
    metrics, rankings = [], []
    for qi, query in enumerate(queries):
        dense = np.argsort(-dense_score[qi])
        sparse = np.argsort(-bm25.get_scores(tokenize(query["query"])))
        ranking = rrf_merge(
            ([chunk_ids[i] for i in dense], [chunk_ids[i] for i in sparse]),
            A.RRF_K,
        )
        metric = A.metric_row(query, ranking)
        metrics.append(metric)
        rankings.append(
            {
                "query_id": query["query_id"],
                "ranked_chunk_ids": ranking[:10],
                "metrics": metric,
            }
        )
        if (qi + 1) % 250 == 0:
            print(
                f"[retrieve:{UNBOUNDED_COMBINED}] {qi + 1}/{len(queries)}", flush=True
            )
    A.write_jsonl(UNBOUNDED_ARTICLE_RANKINGS, rankings)
    chunk_counts = [len(row["unbounded_questions"]) for row in generated]
    article_counts = [len(row["questions"]) for row in article_generated]
    keys = list(metrics[0])
    return {
        "condition": UNBOUNDED_COMBINED,
        "question_rule": "chunk and document: round(facts * 0.5), no bounds",
        "quality": {
            "generated_questions": sum(chunk_counts) + sum(article_counts),
            "chunk_questions": sum(chunk_counts),
            "article_questions": sum(article_counts),
            "questions_per_chunk_min": min(chunk_counts),
            "questions_per_chunk_max": max(chunk_counts),
            "questions_per_article_min": min(article_counts),
            "questions_per_article_max": max(article_counts),
        },
        "stored_vectors": len(chunks) + sum(chunk_counts) + sum(article_counts),
        "metrics": {key: float(np.mean([row[key] for row in metrics])) for key in keys},
    }


def render(payload: dict) -> None:
    metric_keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    )
    rows = []
    for condition in payload["conditions"]:
        q = condition["quality"]
        name = condition["condition"]
        if name == "Baseline":
            generation = "No generated questions"
            count = "0<br><span>0–0/chunk</span>"
            retrieval = "Dense chunk-vector retrieval + BM25 chunk RRF"
            generator = "None"
        elif name in (C.CONDITION, UNBOUNDED_COMBINED):
            generation = (
                "5–20 per chunk + 5–20 per whole document"
                if name == C.CONDITION
                else "Unbounded per chunk + unbounded per whole document"
            ) + ", based on deduplicated atomic facts"
            count = f"{q['generated_questions']}<br><span>{q['chunk_questions']} chunk + {q['article_questions']} document questions</span>"
            retrieval = "Equal 1/3 chunk, chunk-question, document-question vector fusion + BM25 chunk RRF"
            generator = A.MODEL
        else:
            generation = f"<code>{condition['question_rule']}</code>, based on deduplicated atomic facts per chunk"
            count = f"{q['generated_questions']}<br><span>{q['questions_per_chunk_min']}–{q['questions_per_chunk_max']}/chunk</span>"
            retrieval = "0.5/0.5 chunk-question vector fusion + BM25 chunk RRF"
            generator = A.MODEL
        metrics = "".join(
            f"<td><b>{condition['metrics'][key]:.3f}</b></td>" for key in metric_keys
        )
        rows.append(
            f'<tr><td class="left"><b>{name}</b></td><td class="left">{generation}</td><td>{count}</td><td>1024 / 128</td><td>{condition["stored_vectors"]}</td><td class="left">{retrieval}</td><td>{generator}</td>{metrics}</tr>'
        )
    protocol = payload["protocol"]
    REPORT.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Yettel Bulgaria RAG experiments</title><style>body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:20px}}main{{max-width:2100px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:9px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}span,.note{{color:#9aa5b3;font-size:11px}}code{{overflow-wrap:anywhere}}</style></head><body><main><h1>340-document Yettel Bulgaria RAG — baseline and adaptive generated questions</h1><p class="note">Same 340 documents, {protocol["eligible_queries"]} evidence-bearing evaluation queries, {protocol["chunks"]} chunks, canonical gold mapping, {protocol["embedding"]}, Unicode Bulgarian BM25, and RRF k=60 for all {len(payload["conditions"])} conditions. The 301 null queries have no gold evidence and are excluded from evidence-retrieval metrics.</p><table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Chunk / overlap</th><th>Stored vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead><tbody>{"".join(rows)}</tbody></table><p class="note">Dataset evaluation questions and adaptive enrichment questions use separate prompts, seeds, and artifacts, so there is no direct question leakage. Both were generated by gpt-5.4-mini, however, which can introduce same-model stylistic coupling; these results should therefore be described as a controlled same-model synthetic benchmark.</p></main></body></html>""",
        encoding="utf-8",
    )


def run() -> dict:
    chunks, queries, summary = prepare()
    generated = generate_until_complete(chunks)
    print(
        f"[run] chunks={len(chunks)} queries={len(queries)} generated={len(generated)}",
        flush=True,
    )
    base = baseline(chunks, queries)
    print("[done] Baseline", flush=True)
    adaptive_conditions = adaptive_fast(chunks, queries, generated)
    articles = article_inputs(chunks, generated)
    article_generated = C.generate_articles(articles)
    combined = combined_fast(chunks, queries, generated, article_generated)
    print("[done] Adaptive chunk + whole-article questions 5–20", flush=True)
    unbounded_article_generated = generate_unbounded_articles(articles)
    combined_unbounded = combined_unbounded_fast(
        chunks, queries, generated, unbounded_article_generated
    )
    print(f"[done] {UNBOUNDED_COMBINED}", flush=True)
    from ragkit.embeddings import get_embedder

    embedder = get_embedder()
    if not embedder.dim:
        sample = json.loads(A.CHUNK_VECTORS.read_text())
        embedder.dim = len(sample["vectors"][0])
    payload = {
        "protocol": {
            **summary,
            "generation_model": A.MODEL,
            "embedding": embedding_signature(),
            "retrieval": "dense + Unicode BM25 RRF",
            "rrf_k": A.RRF_K,
        },
        "conditions": [base] + adaptive_conditions + [combined, combined_unbounded],
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run()
