"""Run the four adaptive-question retrieval arms on 600 LongRAG NQ documents.

The LongRAG ``nq`` configuration stores retrieved Wikipedia documents inside
each query context, but does not expose corpus ids or qrels.  This experiment
therefore builds a deterministic closed collection from the first 600 unique
context documents in ``subset_1000`` and labels every chunk containing a gold
short answer as relevant.  Queries without an answer-bearing chunk are
excluded.  This protocol is intentionally reported as an answer-containment
evaluation, not as the official LongRAG full-corpus benchmark.
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mhrag_vectoronly"))

import config
from embeddings import get_embedder
import adaptive_questions_100 as A
from full_corpus_no_question_baseline import chunk_article, tokenize

DATA = ROOT / "data" / "processed" / "longrag_nq_600"
RESULTS = ROOT / "results" / "longrag_nq_600"
REPORT = ROOT / "report" / "longrag_nq_600_four_experiments.html"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

DOCUMENTS = DATA / "documents.jsonl"
CHUNKS = DATA / "chunks.jsonl"
QUERIES = DATA / "queries.jsonl"
SUMMARY = DATA / "summary.json"
GENERATIONS = DATA / "adaptive_generations.jsonl"
ARTICLE_GENERATIONS = DATA / "article_question_generations.jsonl"
METRICS = RESULTS / "metrics.json"
RANKINGS = RESULTS / "rankings.json"
RRF_K = 60
DOCUMENT_COUNT = 600
MAX_WORKERS = 24

TITLE_RE = re.compile(r"(?:^|\n)Title:\s*(.*?)\nText:\s*", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.open() if line.strip()]
        if path.exists()
        else []
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def split_context(context: str) -> list[tuple[str, str]]:
    matches = list(TITLE_RE.finditer(context or ""))
    output = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        text = context[match.end() : end].strip()
        if title and text:
            output.append((title, text))
    return output


def answer_pattern(answer: str) -> re.Pattern | None:
    words = re.findall(r"\w+", (answer or "").casefold())
    if not words:
        return None
    return re.compile(r"(?<!\w)" + r"\W+".join(map(re.escape, words)) + r"(?!\w)", re.I)


def prepare() -> tuple[list[dict], list[dict], list[dict], dict]:
    if all(path.exists() for path in (DOCUMENTS, CHUNKS, QUERIES, SUMMARY)):
        return (
            read_jsonl(DOCUMENTS),
            read_jsonl(CHUNKS),
            read_jsonl(QUERIES),
            json.loads(SUMMARY.read_text()),
        )

    dataset = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_100")
    documents_by_title: dict[str, dict] = {}
    raw_queries = []
    for row in dataset:
        raw_queries.append(dict(row))
        for title, text in split_context(row["context"]):
            key = title.casefold()
            if (
                key not in documents_by_title
                and len(documents_by_title) < DOCUMENT_COUNT
            ):
                documents_by_title[key] = {
                    "document_id": f"d{len(documents_by_title):04d}",
                    "title": title,
                    "body": text,
                }
        if len(documents_by_title) >= DOCUMENT_COUNT:
            # Continue no further: selection is the deterministic first 600.
            break
    documents = list(documents_by_title.values())
    chunks = []
    for index, document in enumerate(documents):
        article = {
            "url": document["document_id"],
            "title": document["title"],
            "body": document["body"],
        }
        chunks.extend(chunk_article(article, index))

    eligible = []
    for row in raw_queries:
        patterns = [p for p in (answer_pattern(a) for a in row.get("answer", [])) if p]
        gold = [
            chunk["chunk_id"]
            for chunk in chunks
            if any(pattern.search(chunk["content"]) for pattern in patterns)
        ]
        if gold:
            eligible.append(
                {
                    "query_id": str(row["query_id"]),
                    "query": row["query"].strip(),
                    "answers": row.get("answer", []),
                    "gold_chunk_ids": gold,
                    "evidence_units": [gold],
                }
            )
    summary = {
        "dataset": "TIGER-Lab/LongRAG",
        "configuration": "nq",
        "source_split": "subset_100",
        "selection": "first 600 unique context titles",
        "documents": len(documents),
        "chunks": len(chunks),
        "source_queries_scanned": len(raw_queries),
        "eligible_queries": len(eligible),
        "gold_definition": "chunk contains at least one normalized gold short answer",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "seed": 42,
    }
    write_jsonl(DOCUMENTS, documents)
    write_jsonl(CHUNKS, chunks)
    write_jsonl(QUERIES, eligible)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return documents, chunks, eligible, summary


def configure_generation() -> None:
    A.GEN_PATH = GENERATIONS
    A.MAX_WORKERS = MAX_WORKERS
    A.MAX_RETRIES = 8


def generate_chunks(chunks: list[dict]) -> list[dict]:
    configure_generation()
    return A.generate(chunks)


def generate_articles(
    documents: list[dict], chunks: list[dict], generated: list[dict]
) -> list[dict]:
    cache = {row["article_id"]: row for row in read_jsonl(ARTICLE_GENERATIONS)}
    facts_by_chunk = {row["chunk_id"]: row.get("facts", []) for row in generated}
    chunks_by_document = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk)

    def generate_one(document: dict) -> dict:
        article_id = document["document_id"]
        facts, seen = [], set()
        for chunk in chunks_by_document[article_id]:
            for fact in facts_by_chunk.get(chunk["chunk_id"], []):
                key = A.normalize(fact.get("fact", ""))
                if key and key not in seen:
                    seen.add(key)
                    facts.append(fact)
        if not facts:
            user = f'Source article:\n"""\n{document["body"]}\n"""'
            facts = A.dedup_facts(A.call_json(A.FACT_PROMPT, user).get("facts", []))
        budget = min(20, max(5, round(len(facts) * 0.5)))
        proxy = {"chunk_id": article_id, "content": document["body"]}
        questions = A.generate_questions(proxy, facts, budget)
        return {
            "article_id": article_id,
            "title": document["title"],
            "question_budget": budget,
            "questions": questions,
        }

    todo = [document for document in documents if document["document_id"] not in cache]
    print(f"[article-generation] cached={len(cache)} todo={len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_one, document): document for document in todo
        }
        for position, future in enumerate(as_completed(futures), 1):
            document = futures[future]
            row = future.result()
            cache[row["article_id"]] = row
            write_jsonl(
                ARTICLE_GENERATIONS,
                [
                    cache[d["document_id"]]
                    for d in documents
                    if d["document_id"] in cache
                ],
            )
            print(
                f"[article-generation] {position}/{len(todo)} {document['title']}",
                flush=True,
            )
    return [cache[document["document_id"]] for document in documents]


def embed(path: Path, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("texts") == texts:
            return np.asarray(payload["vectors"], dtype=float)
        if payload.get("texts") == texts[: len(payload.get("texts", []))]:
            vectors = payload["vectors"]
    model = get_embedder()
    for start in range(len(vectors), len(texts), batch_size):
        vectors.extend(model.embed_documents(texts[start : start + batch_size]))
        path.write_text(
            json.dumps({"texts": texts[: len(vectors)], "vectors": vectors})
        )
        print(f"[embed] {path.name} {len(vectors)}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=float)


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity using an explicit dot product for BLAS portability."""
    dots = np.einsum("ij,j->i", matrix, vector, optimize=False)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return dots / np.maximum(norms, 1e-12)


def metric_row(query: dict, ranking: list[str]) -> dict:
    gold = set(query["gold_chunk_ids"])
    recall = lambda k: float(bool(gold & set(ranking[:k])))
    first = next((i for i, cid in enumerate(ranking[:10], 1) if cid in gold), None)
    return {
        "evidence_recall@1": recall(1),
        "evidence_recall@5": recall(5),
        "evidence_recall@10": recall(10),
        "all_evidence_hit@5": recall(5),
        "mrr@10": 0.0 if first is None else 1.0 / first,
    }


def rrf(
    dense_order: np.ndarray, sparse_order: np.ndarray, chunk_ids: list[str]
) -> list[str]:
    score = defaultdict(float)
    for order in (dense_order, sparse_order):
        for rank, item in enumerate(order, 1):
            score[chunk_ids[item]] += 1 / (RRF_K + rank)
    return [cid for cid, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


def evaluate(
    documents: list[dict],
    chunks: list[dict],
    queries: list[dict],
    generated: list[dict],
    article_generated: list[dict],
    summary: dict,
) -> dict:
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_vec = embed(
        RESULTS / "chunk_vectors_iris.json", [c["content"] for c in chunks]
    )
    query_vec = embed(
        RESULTS / "query_vectors_iris.json", [q["query"] for q in queries]
    )
    generated_by_chunk = {row["chunk_id"]: row for row in generated}
    question_sets = {}
    for field in ("bounded_questions", "unbounded_questions"):
        pairs = [
            (q["question"], row["chunk_id"]) for row in generated for q in row[field]
        ]
        vec = embed(RESULTS / f"{field}_vectors_iris.json", [x[0] for x in pairs])
        indices = defaultdict(list)
        for index, (_, cid) in enumerate(pairs):
            indices[cid].append(index)
        question_sets[field] = (pairs, vec, indices)
    article_pairs = [
        (q["question"], row["article_id"])
        for row in article_generated
        for q in row["questions"]
    ]
    article_vec = embed(
        RESULTS / "article_question_vectors_iris.json", [x[0] for x in article_pairs]
    )
    article_indices = defaultdict(list)
    for index, (_, did) in enumerate(article_pairs):
        article_indices[did].append(index)
    bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])

    names = [
        "Baseline",
        "Adaptive generated questions 5–20",
        "Adaptive generated questions unbounded",
        "Adaptive chunk + whole-article questions 5–20",
    ]
    rows = {name: [] for name in names}
    ranking_output = {name: [] for name in names}
    for qi, query in enumerate(queries):
        cscore = cosine_scores(chunk_vec, query_vec[qi])
        sparse = np.argsort(-bm25.get_scores(tokenize(query["query"])))
        dense_scores = {"Baseline": cscore}
        for name, field in zip(
            names[1:3], ("bounded_questions", "unbounded_questions")
        ):
            _, qvec, indices = question_sets[field]
            qscore = cosine_scores(qvec, query_vec[qi])
            dense_scores[name] = np.asarray(
                [
                    0.5 * cscore[i] + 0.5 * max(qscore[indices[cid]])
                    for i, cid in enumerate(chunk_ids)
                ]
            )
        _, bounded_vec, bounded_indices = question_sets["bounded_questions"]
        bscore = cosine_scores(bounded_vec, query_vec[qi])
        ascore = cosine_scores(article_vec, query_vec[qi])
        dense_scores[names[3]] = np.asarray(
            [
                (
                    cscore[i]
                    + max(bscore[bounded_indices[cid]])
                    + max(ascore[article_indices[chunks[i]["document_id"]]])
                )
                / 3
                for i, cid in enumerate(chunk_ids)
            ]
        )
        for name in names:
            ranking = rrf(np.argsort(-dense_scores[name]), sparse, chunk_ids)
            metrics = metric_row(query, ranking)
            rows[name].append(metrics)
            ranking_output[name].append(
                {
                    "query_id": query["query_id"],
                    "ranked_chunk_ids": ranking[:10],
                    "metrics": metrics,
                }
            )
        if (qi + 1) % 25 == 0:
            print(f"[retrieve] {qi + 1}/{len(queries)}", flush=True)
    keys = list(rows[names[0]][0])
    bounded_n = sum(len(row["bounded_questions"]) for row in generated)
    unbounded_n = sum(len(row["unbounded_questions"]) for row in generated)
    article_n = sum(len(row["questions"]) for row in article_generated)
    condition_meta = [
        (names[0], 0, "No generated questions"),
        (names[1], bounded_n, "clamp(5, 20, round(facts × 0.5))"),
        (names[2], unbounded_n, "round(facts × 0.5), no bounds"),
        (names[3], bounded_n + article_n, "5–20 per chunk + 5–20 per whole document"),
    ]
    payload = {
        "protocol": {
            **summary,
            "generation_model": config.OPENAI_CHAT_MODEL,
            "embedding": "Iris dim-384",
            "rrf_k": RRF_K,
        },
        "conditions": [
            {
                "condition": name,
                "generated_questions": count,
                "question_rule": rule,
                "stored_vectors": len(chunks) + count,
                "metrics": {
                    k: float(np.mean([r[k] for r in rows[name]])) for k in keys
                },
            }
            for name, count, rule in condition_meta
        ],
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    RANKINGS.write_text(json.dumps(ranking_output))
    render(payload)
    return payload


def render(payload: dict) -> None:
    metric_keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    )
    body = []
    for index, row in enumerate(payload["conditions"]):
        metrics = "".join(
            f"<td><b>{row['metrics'][key]:.3f}</b></td>" for key in metric_keys
        )
        retrieval = (
            "Dense chunk-vector + BM25 chunk RRF"
            if index == 0
            else "0.5/0.5 chunk-question vector fusion + BM25 chunk RRF"
            if index < 3
            else "Equal 1/3 chunk, chunk-question, document-question fusion + BM25 chunk RRF"
        )
        body.append(
            f"<tr><td class='left'><b>{html.escape(row['condition'])}</b></td>"
            f"<td class='left'><code>{html.escape(row['question_rule'])}</code></td>"
            f"<td>{row['generated_questions']}</td><td>1024 / 128</td>"
            f"<td>{row['stored_vectors']}</td><td class='left'>{retrieval}</td>"
            f"<td>{'None' if index == 0 else html.escape(config.OPENAI_CHAT_MODEL)}</td>{metrics}</tr>"
        )
    p = payload["protocol"]
    REPORT.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LongRAG NQ 600-document experiments</title><style>
body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:20px}}main{{max-width:2100px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:9px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}.note{{color:#9aa5b3;font-size:12px}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<h1>600-document LongRAG–NQ — baseline and adaptive generated questions</h1>
<p class='note'>TIGER-Lab/LongRAG, <code>nq</code> configuration, <code>{p["source_split"]}</code>: first 600 unique retrieved context documents; {p["eligible_queries"]} eligible queries from {p["source_queries_scanned"]} scanned; {p["chunks"]} chunks. Gold relevance is answer containment, not official LongRAG qrels.</p>
<p class='note'>All four conditions use the identical document collection, query set, chunks, Iris embeddings, BM25, and RRF k=60.</p>
<table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Chunk / overlap</th><th>Stored vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead><tbody>{"".join(body)}</tbody></table>
<p class='note'>Machine-readable metrics: <code>results/longrag_nq_600/metrics.json</code></p>
</main></body></html>""")


def run() -> dict:
    documents, chunks, queries, summary = prepare()
    if not queries:
        raise RuntimeError("No answer-containing queries in the selected collection")
    generated = generate_chunks(chunks)
    article_generated = generate_articles(documents, chunks, generated)
    payload = evaluate(
        documents, chunks, queries, generated, article_generated, summary
    )
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
