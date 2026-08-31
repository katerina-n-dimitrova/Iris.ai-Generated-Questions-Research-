"""Full MultiHop-RAG no-question baseline: 1024/128 dense + BM25 RRF."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embeddings import get_embedder


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "multihoprag"
DATA = ROOT / "data" / "processed" / "mhrag_full_baseline_1024_128"
RESULTS = ROOT / "results" / "mhrag_full_baseline_1024_128"
REPORT = ROOT / "report" / "mhrag_full_no_question_baseline.html"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = DATA / "chunks.jsonl"
QUERIES_PATH = DATA / "queries.jsonl"
ALIGNMENT_PATH = DATA / "alignment_summary.json"
CHUNK_VECTORS = RESULTS / "chunk_vectors_iris.json"
QUERY_VECTORS = RESULTS / "query_vectors_iris.json"
RANKINGS = RESULTS / "rankings.jsonl"
METRICS = RESULTS / "metrics.json"

ENC = tiktoken.get_encoding("cl100k_base")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")
RRF_K = 60


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def norm(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip().casefold()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall((text or "").casefold())


def chunk_article(article: dict, article_index: int) -> list[dict]:
    units = []
    paragraphs = re.split(r"\n\s*\n", article.get("body", ""))
    for paragraph_id, paragraph in enumerate(paragraphs):
        for sentence in SENT_RE.split(WS_RE.sub(" ", paragraph).strip()):
            if not sentence:
                continue
            encoded = ENC.encode(sentence)
            if len(encoded) <= 1024:
                units.append((paragraph_id, sentence, len(encoded)))
            else:
                for start in range(0, len(encoded), 1024):
                    text = ENC.decode(encoded[start : start + 1024]).strip()
                    units.append((paragraph_id, text, len(ENC.encode(text))))
    windows, current, current_n, position = [], [], 0, 0
    while position < len(units):
        unit = units[position]
        if current and current_n + unit[2] > 1024:
            windows.append(current)
            overlap, overlap_n = [], 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_n += previous[2]
                if overlap_n >= 128:
                    break
            current = overlap if len(overlap) < len(current) else []
            current_n = sum(item[2] for item in current)
            continue
        current.append(unit)
        current_n += unit[2]
        position += 1
    if current and (not windows or current != windows[-1]):
        windows.append(current)
    return [
        {
            "chunk_id": f"a{article_index:03d}::l{index}",
            "document_id": article["url"],
            "document_title": article.get("title", ""),
            "chunk_position": index,
            "n_tokens": len(ENC.encode(" ".join(item[1] for item in window))),
            "content": " ".join(item[1] for item in window).strip(),
        }
        for index, window in enumerate(windows)
    ]


def locate_fact(fact: str, candidates: list[dict]) -> tuple[list[str], str]:
    target = norm(fact)
    hits = [row["chunk_id"] for row in candidates if target in norm(row["content"])]
    if hits:
        return hits, "exact"
    best_id, best = None, 0.0
    for row in candidates:
        for sentence in SENT_RE.split(row["content"]):
            score = SequenceMatcher(None, target, norm(sentence)).ratio()
            if score > best:
                best_id, best = row["chunk_id"], score
    return ([best_id], "fuzzy") if best_id and best >= 0.90 else ([], "unresolved")


def prepare() -> tuple[list[dict], list[dict], dict]:
    if CHUNKS_PATH.exists() and QUERIES_PATH.exists() and ALIGNMENT_PATH.exists():
        return (
            read_jsonl(CHUNKS_PATH),
            read_jsonl(QUERIES_PATH),
            json.loads(ALIGNMENT_PATH.read_text()),
        )
    corpus = json.loads((RAW / "corpus.json").read_text())
    raw_queries = json.loads((RAW / "MultiHopRAG.json").read_text())
    chunks = [
        chunk
        for index, article in enumerate(corpus)
        for chunk in chunk_article(article, index)
    ]
    by_document = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)
    queries, unresolved = [], []
    method_counts = defaultdict(int)
    null_count = 0
    for index, query in enumerate(raw_queries):
        query_id = f"q{index:05d}"
        if query.get("question_type") == "null_query" or not query.get("evidence_list"):
            null_count += 1
            continue
        units, failed = [], False
        required_documents = set()
        for evidence_index, evidence in enumerate(query["evidence_list"]):
            document_id = evidence["url"]
            required_documents.add(document_id)
            hits, method = locate_fact(
                evidence.get("fact", ""), by_document.get(document_id, [])
            )
            method_counts[method] += 1
            if not hits:
                failed = True
                unresolved.append(
                    {
                        "query_id": query_id,
                        "evidence_index": evidence_index,
                        "document_id": document_id,
                        "fact": evidence.get("fact", ""),
                    }
                )
            units.append(sorted(set(hits)))
        if failed:
            continue
        queries.append(
            {
                "query_id": query_id,
                "query": query["query"].strip(),
                "question_type": query["question_type"].replace("_query", ""),
                "required_article_ids": sorted(required_documents),
                "n_required_documents": len(required_documents),
                "n_required_evidence_facts": len(units),
                "evidence_units": units,
                "gold_chunk_ids": sorted({cid for unit in units for cid in unit}),
            }
        )
    summary = {
        "articles": len(corpus),
        "chunks": len(chunks),
        "raw_queries": len(raw_queries),
        "null_queries_excluded": null_count,
        "evidence_queries": len(raw_queries) - null_count,
        "fully_aligned_queries": len(queries),
        "queries_excluded_for_unresolved_evidence": (
            len(raw_queries) - null_count - len(queries)
        ),
        "unresolved_evidence_facts": len(unresolved),
        "alignment_methods": dict(method_counts),
        "chunk_size": 1024,
        "chunk_overlap": 128,
    }
    write_jsonl(CHUNKS_PATH, chunks)
    write_jsonl(QUERIES_PATH, queries)
    (DATA / "unresolved_evidence.json").write_text(json.dumps(unresolved, indent=2))
    ALIGNMENT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return chunks, queries, summary


def embed_resumable(path: Path, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("texts") == texts:
            return np.asarray(payload["vectors"], dtype=float)
        if payload.get("texts") == texts[: len(payload.get("texts", []))]:
            vectors = payload.get("vectors", [])
    embedder = get_embedder()
    for start in range(len(vectors), len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embedder.embed_documents(batch))
        path.write_text(
            json.dumps({"texts": texts[: len(vectors)], "vectors": vectors})
        )
        print(f"[embed] {path.name}: {len(vectors)}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=float)


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    return matrix @ vector / np.maximum(matrix_norm * vector_norm, 1e-12)


def rrf(dense: list[str], sparse: list[str]) -> list[str]:
    scores = defaultdict(float)
    for ranking in (dense, sparse):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1.0 / (RRF_K + rank)
    return [item[0] for item in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def per_query(query: dict, ranking: list[str]) -> dict:
    units = [set(unit) for unit in query["evidence_units"]]

    def recall(k: int) -> float:
        found = set(ranking[:k])
        return sum(bool(unit & found) for unit in units) / len(units)

    first = next(
        (
            rank
            for rank, cid in enumerate(ranking[:10], 1)
            if any(cid in unit for unit in units)
        ),
        None,
    )
    return {
        "evidence_recall@1": recall(1),
        "evidence_recall@5": recall(5),
        "evidence_recall@10": recall(10),
        "all_evidence_hit@5": float(recall(5) == 1.0),
        "mrr@10": 0.0 if first is None else 1.0 / first,
    }


def render(payload: dict) -> None:
    metrics, protocol = payload["metrics"], payload["protocol"]
    cells = "".join(
        f"<td><b>{metrics[key]:.3f}</b></td>"
        for key in (
            "evidence_recall@1",
            "evidence_recall@5",
            "evidence_recall@10",
            "all_evidence_hit@5",
            "mrr@10",
        )
    )
    REPORT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Full MultiHop-RAG no-question baseline</title><style>
body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;margin:0;padding:24px}}main{{max-width:1800px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:10px;text-align:center;vertical-align:top}}th{{background:#161c25}}.base{{background:#403a18}}.left{{text-align:left}}.note{{color:#9aa5b3}}code{{overflow-wrap:anywhere}}
</style></head><body><main><h1>Full MultiHop-RAG — no-question baseline</h1>
<p class="note">Complete 609-document corpus; {protocol["evaluated_queries"]} fully aligned evidence-bearing queries; 1024-token chunks with 128-token overlap. Null queries have no gold evidence and are excluded from evidence-retrieval metrics.</p>
<table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Documents</th><th>Chunk / overlap</th><th>Chunks / vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead>
<tbody><tr class="base"><td><b>Baseline</b></td><td class="left">None</td><td>0</td><td>609</td><td>1024 / 128</td><td>{protocol["chunks"]}</td><td class="left">Dense chunk-vector retrieval + BM25 keyword retrieval; RRF k=60</td><td>None</td>{cells}</tr></tbody></table>
<p class="note">Source: <code>results/mhrag_full_baseline_1024_128/metrics.json</code></p></main></body></html>""")


def run() -> dict:
    chunks, queries, alignment = prepare()
    chunk_vectors = embed_resumable(CHUNK_VECTORS, [c["content"] for c in chunks])
    query_vectors = embed_resumable(QUERY_VECTORS, [q["query"] for q in queries])
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    bm25 = BM25Okapi([tokenize(chunk["content"]) for chunk in chunks])
    rows, scores = [], []
    with RANKINGS.open("w") as output:
        for index, query in enumerate(queries):
            dense_order = np.argsort(
                -cosine_scores(chunk_vectors, query_vectors[index])
            )
            dense = [chunk_ids[item] for item in dense_order]
            sparse_order = np.argsort(-bm25.get_scores(tokenize(query["query"])))
            sparse = [chunk_ids[item] for item in sparse_order]
            ranking = rrf(dense, sparse)
            values = per_query(query, ranking)
            scores.append(values)
            row = {**query, "ranked_chunk_ids": ranking[:10], "metrics": values}
            output.write(json.dumps(row) + "\n")
            if (index + 1) % 100 == 0:
                print(f"[retrieve] {index + 1}/{len(queries)}", flush=True)
    metric_keys = list(scores[0])
    payload = {
        "condition": "Baseline",
        "protocol": {
            "dataset": "yixuantt/MultiHopRAG full corpus",
            "documents": 609,
            "chunks": len(chunks),
            "raw_queries": alignment["raw_queries"],
            "null_queries_excluded": alignment["null_queries_excluded"],
            "evaluated_queries": len(queries),
            "queries_excluded_for_unresolved_evidence": alignment[
                "queries_excluded_for_unresolved_evidence"
            ],
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "generated_questions": 0,
            "stored_vectors": len(chunks),
            "embedding": "Iris dim-384",
            "retrieval": "dense chunks + BM25 RRF",
            "rrf_k": RRF_K,
        },
        "metrics": {
            key: float(np.mean([row[key] for row in scores])) for key in metric_keys
        },
        "alignment": alignment,
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    render(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
