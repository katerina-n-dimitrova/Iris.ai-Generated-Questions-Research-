#!/usr/bin/env python3
"""Evaluate the four retrieval conditions on the union of MultiHop-RAG and Yettel."""

from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ragkit.text import read_jsonl as _read_jsonl  # noqa: E402
from ragkit.text import tokenize_unicode as tokenize  # noqa: E402
from ragkit.text import write_jsonl_utf8 as write_jsonl  # noqa: E402
from ragkit.vectors import unit  # noqa: E402

M_DATA = ROOT / "data/processed/mhrag_adaptive_questions_full"
Y_DATA = ROOT / "data/processed/yettel_bg_experiments"
M_RESULTS = ROOT / "results/mhrag_adaptive_questions_full"
Y_RESULTS = ROOT / "results/yettel_bg_experiments"
OUT_DATA = ROOT / "data/processed/combined_mhrag_yettel_experiments"
OUT_RESULTS = ROOT / "results/combined_mhrag_yettel_experiments"
REPORT = ROOT / "report/combined_multihop_yettel_adaptive_questions.html"
for directory in (OUT_DATA, OUT_RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

RRF_K = 60
CONDITIONS = (
    "Baseline",
    "Adaptive generated questions 5–20",
    "Adaptive generated questions unbounded",
    "Adaptive chunk + whole-document questions 5–20",
)


def read_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(path, encoding="utf-8")


def cache(path: Path) -> tuple[list[str], np.ndarray]:
    obj = json.loads(path.read_text())
    return obj["texts"], np.asarray(obj["vectors"], dtype=np.float32)


def prefix_query(row: dict, prefix: str) -> dict:
    result = dict(row)
    result["query_id"] = f"{prefix}::{row['query_id']}"
    result["dataset"] = prefix
    result["required_article_ids"] = [
        f"{prefix}::{x}" for x in row["required_article_ids"]
    ]
    result["evidence_units"] = [
        [f"{prefix}::{x}" for x in unit_] for unit_ in row["evidence_units"]
    ]
    result["gold_chunk_ids"] = [f"{prefix}::{x}" for x in row["gold_chunk_ids"]]
    return result


def prefix_chunk(row: dict, prefix: str) -> dict:
    result = dict(row)
    result["chunk_id"] = f"{prefix}::{row['chunk_id']}"
    result["document_id"] = f"{prefix}::{row['document_id']}"
    result["dataset"] = prefix
    return result


def flatten_questions(
    generated: list[dict], field: str, prefix: str
) -> tuple[list[str], list[str]]:
    texts, owners = [], []
    for row in generated:
        for question in row[field]:
            texts.append(question["question"])
            owners.append(f"{prefix}::{row['chunk_id']}")
    return texts, owners


def flatten_articles(generated: list[dict], prefix: str) -> tuple[list[str], list[str]]:
    texts, owners = [], []
    for row in generated:
        for question in row["questions"]:
            texts.append(question["question"])
            owners.append(f"{prefix}::{row['article_id']}")
    return texts, owners


def validate_texts(label: str, expected: list[str], actual: list[str]) -> None:
    if expected != actual:
        raise RuntimeError(
            f"{label} cache text/order mismatch ({len(expected)} vs {len(actual)})"
        )


def metric_row(query: dict, ranking: list[str]) -> dict[str, float]:
    units = [set(x) for x in query["evidence_units"]]
    gold = set(query["gold_chunk_ids"])
    result = {}
    for k in (1, 5, 10):
        top = set(ranking[:k])
        result[f"evidence_recall@{k}"] = sum(bool(x & top) for x in units) / len(units)
    result["all_evidence_hit@5"] = float(all(x & set(ranking[:5]) for x in units))
    result["mrr@10"] = next(
        (1 / i for i, cid in enumerate(ranking[:10], 1) if cid in gold), 0.0
    )
    dcg = sum(
        1 / np.log2(i + 1) for i, cid in enumerate(ranking[:10], 1) if cid in gold
    )
    ideal = sum(1 / np.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    result["ndcg@10"] = float(dcg / ideal) if ideal else 0.0
    return result


def summarize(rows: list[dict]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def rrf_ranking(
    dense_scores: np.ndarray, sparse_scores: np.ndarray, chunk_ids: list[str]
) -> list[str]:
    dense_order = np.argsort(-dense_scores)
    sparse_order = np.argsort(-sparse_scores)
    score = np.zeros(len(chunk_ids), dtype=np.float32)
    score[dense_order] += 1 / (RRF_K + np.arange(1, len(chunk_ids) + 1))
    score[sparse_order] += 1 / (RRF_K + np.arange(1, len(chunk_ids) + 1))
    order = sorted(
        range(len(chunk_ids)), key=lambda i: (-float(score[i]), chunk_ids[i])
    )
    return [chunk_ids[i] for i in order[:10]]


def owner_max(
    similarity: np.ndarray,
    owners: list[str],
    target_ids: list[str],
    fallback: np.ndarray,
) -> np.ndarray:
    indexes = defaultdict(list)
    for index, owner in enumerate(owners):
        indexes[owner].append(index)
    result = np.empty((similarity.shape[0], len(target_ids)), dtype=np.float32)
    for col, target in enumerate(target_ids):
        positions = indexes.get(target)
        result[:, col] = (
            similarity[:, positions].max(axis=1) if positions else fallback[:, col]
        )
    return result


def build() -> tuple[list[dict], list[dict], dict]:
    m_chunks_raw, y_chunks_raw = (
        read_jsonl(M_DATA / "chunks.jsonl"),
        read_jsonl(Y_DATA / "chunks.jsonl"),
    )
    m_queries_raw, y_queries_raw = (
        read_jsonl(M_DATA / "queries.jsonl"),
        read_jsonl(Y_DATA / "queries.jsonl"),
    )
    chunks = [prefix_chunk(x, "mhrag") for x in m_chunks_raw] + [
        prefix_chunk(x, "yettel") for x in y_chunks_raw
    ]
    queries = [prefix_query(x, "mhrag") for x in m_queries_raw] + [
        prefix_query(x, "yettel") for x in y_queries_raw
    ]
    write_jsonl(OUT_DATA / "chunks.jsonl", chunks)
    write_jsonl(OUT_DATA / "queries.jsonl", queries)

    mg, yg = (
        read_jsonl(M_DATA / "adaptive_generations.jsonl"),
        read_jsonl(Y_DATA / "adaptive_generations.jsonl"),
    )
    ma, ya = (
        read_jsonl(M_DATA / "article_question_generations.jsonl"),
        read_jsonl(Y_DATA / "article_question_generations.jsonl"),
    )
    chunk_ids = [x["chunk_id"] for x in chunks]
    document_ids = [x["document_id"] for x in chunks]

    mct, mcv = cache(M_RESULTS / "chunk_vectors_iris.json")
    yct, ycv = cache(Y_RESULTS / "chunk_vectors.json")
    mqt, mqv = cache(M_RESULTS / "query_vectors_iris.json")
    yqt, yqv = cache(Y_RESULTS / "query_vectors.json")
    validate_texts("MultiHop chunks", [x["content"] for x in m_chunks_raw], mct)
    validate_texts("Yettel chunks", [x["content"] for x in y_chunks_raw], yct)
    validate_texts("MultiHop queries", [x["query"] for x in m_queries_raw], mqt)
    validate_texts("Yettel queries", [x["query"] for x in y_queries_raw], yqt)

    vectors = {
        "chunk": unit(np.vstack((mcv, ycv))),
        "query": unit(np.vstack((mqv, yqv))),
    }
    owners = {}
    for field, key, mf, yf in (
        (
            "bounded_questions",
            "bounded",
            "bounded_question_vectors_iris.json",
            "bounded_question_vectors.json",
        ),
        (
            "unbounded_questions",
            "unbounded",
            "unbounded_question_vectors_iris.json",
            "unbounded_question_vectors.json",
        ),
    ):
        mt, mo = flatten_questions(mg, field, "mhrag")
        yt, yo = flatten_questions(yg, field, "yettel")
        mcache_t, mvec = cache(M_RESULTS / mf)
        ycache_t, yvec = cache(Y_RESULTS / yf)
        validate_texts(f"MultiHop {key}", mt, mcache_t)
        validate_texts(f"Yettel {key}", yt, ycache_t)
        vectors[key] = unit(np.vstack((mvec, yvec)))
        owners[key] = mo + yo
    mt, mo = flatten_articles(ma, "mhrag")
    yt, yo = flatten_articles(ya, "yettel")
    mcache_t, mvec = cache(M_RESULTS / "article_question_vectors_iris.json")
    ycache_t, yvec = cache(Y_RESULTS / "article_question_vectors.json")
    validate_texts("MultiHop article", mt, mcache_t)
    validate_texts("Yettel article", yt, ycache_t)
    vectors["article"] = unit(np.vstack((mvec, yvec)))
    owners["article"] = mo + yo
    payload = {
        "vectors": vectors,
        "owners": owners,
        "chunk_ids": chunk_ids,
        "document_ids": document_ids,
        "counts": {
            "bounded": len(owners["bounded"]),
            "unbounded": len(owners["unbounded"]),
            "article": len(owners["article"]),
        },
    }
    return chunks, queries, payload


def evaluate(chunks: list[dict], queries: list[dict], data: dict) -> dict:
    vectors, owners = data["vectors"], data["owners"]
    chunk_ids, document_ids = data["chunk_ids"], data["document_ids"]
    bm25 = BM25Okapi([tokenize(x["content"]) for x in chunks])
    by_condition = {c: [] for c in CONDITIONS}
    by_source = {c: defaultdict(list) for c in CONDITIONS}
    wrong_source = {c: defaultdict(int) for c in CONDITIONS}
    rankings = []
    batch_size = 64
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        qv = vectors["query"][start : start + batch_size]
        chunk_score = np.einsum("ik,jk->ij", qv, vectors["chunk"], optimize=True)
        bounded = owner_max(
            np.einsum("ik,jk->ij", qv, vectors["bounded"], optimize=True),
            owners["bounded"],
            chunk_ids,
            chunk_score,
        )
        unbounded = owner_max(
            np.einsum("ik,jk->ij", qv, vectors["unbounded"], optimize=True),
            owners["unbounded"],
            chunk_ids,
            chunk_score,
        )
        article_by_doc = owner_max(
            np.einsum("ik,jk->ij", qv, vectors["article"], optimize=True),
            owners["article"],
            sorted(set(document_ids)),
            np.zeros((len(batch), len(set(document_ids))), dtype=np.float32),
        )
        doc_col = {doc: i for i, doc in enumerate(sorted(set(document_ids)))}
        article = article_by_doc[:, [doc_col[x] for x in document_ids]]
        scores = {
            CONDITIONS[0]: chunk_score,
            CONDITIONS[1]: (chunk_score + bounded) / 2,
            CONDITIONS[2]: (chunk_score + unbounded) / 2,
            CONDITIONS[3]: (chunk_score + bounded + article) / 3,
        }
        for local, query in enumerate(batch):
            sparse = bm25.get_scores(tokenize(query["query"]))
            record = {
                "query_id": query["query_id"],
                "dataset": query["dataset"],
                "conditions": {},
            }
            for condition in CONDITIONS:
                ranking = rrf_ranking(scores[condition][local], sparse, chunk_ids)
                metric = metric_row(query, ranking)
                by_condition[condition].append(metric)
                by_source[condition][query["dataset"]].append(metric)
                wrong_source[condition][query["dataset"]] += int(
                    not ranking[0].startswith(query["dataset"] + "::")
                )
                record["conditions"][condition] = {
                    "ranked_chunk_ids": ranking,
                    "metrics": metric,
                }
            rankings.append(record)
        print(
            f"evaluated {min(start + batch_size, len(queries))}/{len(queries)}",
            flush=True,
        )
    write_jsonl(OUT_RESULTS / "rankings.jsonl", rankings)
    counts = data["counts"]
    result = {
        "protocol": {
            "documents": 949,
            "chunks": len(chunks),
            "evaluation_queries": len(queries),
            "total_questions_including_yettel_null": len(queries) + 301,
            "yettel_null_excluded": 301,
            "embedding": "Iris dim-384",
            "chunk_size": 1024,
            "overlap": 128,
            "rrf_k": RRF_K,
        },
        "conditions": [],
    }
    stored = (
        len(chunks),
        len(chunks) + counts["bounded"],
        len(chunks) + counts["unbounded"],
        len(chunks) + counts["bounded"] + counts["article"],
    )
    generated = (
        0,
        counts["bounded"],
        counts["unbounded"],
        counts["bounded"] + counts["article"],
    )
    for i, condition in enumerate(CONDITIONS):
        result["conditions"].append(
            {
                "condition": condition,
                "generated_questions": generated[i],
                "stored_vectors": stored[i],
                "metrics": summarize(by_condition[condition]),
                "by_dataset": {
                    source: summarize(rows)
                    for source, rows in by_source[condition].items()
                },
                "wrong_corpus_top1": {
                    source: wrong_source[condition][source]
                    / len(by_source[condition][source])
                    for source in ("mhrag", "yettel")
                },
            }
        )
    (OUT_RESULTS / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def table(conditions: list[dict], selector=None) -> str:
    keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
        "ndcg@10",
    )
    rows = []
    for row in conditions:
        metrics = selector(row) if selector else row["metrics"]
        values = "".join(f"<td><b>{metrics[k]:.3f}</b></td>" for k in keys)
        rows.append(
            f"<tr><td class='left'><b>{html.escape(row['condition'])}</b></td><td>{row['generated_questions']}</td><td>{row['stored_vectors']}</td>{values}</tr>"
        )
    return (
        "<table><thead><tr><th>Condition</th><th>Generated questions</th><th>Stored vectors</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th><th>nDCG@10</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render(result: dict) -> None:
    p = result["protocol"]
    overall = table(result["conditions"])
    mhrag = table(result["conditions"], lambda x: x["by_dataset"]["mhrag"])
    yettel = table(result["conditions"], lambda x: x["by_dataset"]["yettel"])
    REPORT.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>Combined MultiHop-RAG + Yettel experiments</title><style>
body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:22px}}main{{max-width:1800px;margin:auto}}h1{{font-size:30px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:10px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}.note{{color:#9aa5b3}}code{{color:#bdc7d6}}</style></head><body><main>
<h1>949-document combined MultiHop-RAG + Yettel Bulgaria benchmark</h1>
<p class='note'>609 MultiHop-RAG articles + 340 Yettel documents; {p["chunks"]} shared-corpus chunks and {p["evaluation_queries"]} evidence-bearing queries. All four conditions use the same 1024-token / 128-token-overlap chunks, Iris 384-dimensional embeddings, Unicode BM25, and RRF k=60. The 301 Yettel null queries are retained in the dataset but excluded from evidence-retrieval metrics.</p>
<h2>Combined results</h2>{overall}
<h2>MultiHop-RAG queries against the combined corpus</h2>{mhrag}
<h2>Yettel queries against the combined corpus</h2>{yettel}
<h2>Protocol notes</h2><p class='note'>IDs are namespaced with <code>mhrag::</code> and <code>yettel::</code>, preventing collisions. Existing question generations and embeddings were reused exactly; no combined evaluation query was used as enrichment text. Metrics are computed against one common 2,793-chunk index, so distractors from both domains are present. Answer accuracy is not reported here because these four runs evaluate retrieval, not answer generation. The Yettel evaluation and enrichment questions were generated independently but by the same model, so same-model stylistic coupling remains a benchmark limitation.</p>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    chunks, queries, data = build()
    result = evaluate(chunks, queries, data)
    render(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
