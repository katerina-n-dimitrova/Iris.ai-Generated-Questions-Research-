"""Two chunk-level adaptive-question experiments on 100 MultiHop-RAG articles."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ragkit import config
from ragkit.embeddings import get_embedder
from ragkit.fusion import rrf_merge
from ragkit.metrics import metric_row
from ragkit.text import (
    read_jsonl as _read_jsonl,
    tokenize_ascii as tokenize,
    write_jsonl,
)
from ragkit.vectors import cosine_scores

from baseline import chunk_article, locate_fact


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "multihoprag"
DATA = ROOT / "data" / "processed" / "mhrag_adaptive_questions_100"
RESULTS = ROOT / "results" / "mhrag_adaptive_questions_100"
REPORT = ROOT / "report" / "mhrag_adaptive_questions_100.html"
for directory in (DATA, RESULTS, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = DATA / "chunks.jsonl"
QUERIES_PATH = DATA / "queries.jsonl"
SUMMARY_PATH = DATA / "summary.json"
GEN_PATH = DATA / "adaptive_generations.jsonl"
CHUNK_VECTORS = RESULTS / "chunk_vectors_iris.json"
QUERY_VECTORS = RESULTS / "query_vectors_iris.json"
BOUNDED_VECTORS = RESULTS / "bounded_question_vectors_iris.json"
UNBOUNDED_VECTORS = RESULTS / "unbounded_question_vectors_iris.json"
METRICS = RESULTS / "metrics.json"
RANKINGS = RESULTS / "rankings.json"

SEED = 42
ARTICLE_COUNT = 100
RRF_K = 60
MODEL = config.OPENAI_CHAT_MODEL
MAX_RETRIES = 3
MAX_WORKERS = int(os.getenv("MHRAG_ADAPTIVE_WORKERS", "8"))

FACT_PROMPT = """Analyze the supplied source chunk. Extract every meaningful
atomic fact explicitly supported by it, then deduplicate closely related
restatements. Ignore headings, boilerplate, vague opinion, and unsupported
inference. For each distinct fact return a concise factual statement, a short
verbatim evidence quote, importance 1-5, and distinctiveness 1-5. Return valid
JSON only: {"facts":[{"fact":"...","evidence":"...","importance":1,
"distinctiveness":1}]}"""

QUESTION_PROMPT = """Generate exactly the requested number of diverse,
grounded retrieval questions from the source chunk and its deduplicated atomic
facts. Prioritize important and distinctive facts while maximizing coverage.
Avoid near-duplicates. Each question must be answerable using only the chunk.
Return valid JSON only: {"questions":[{"question":"...",
"source_fact_ids":[0]}]}"""


def read_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(path, missing_ok=True)


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").casefold()).strip()


def prepare() -> tuple[list[dict], list[dict], dict]:
    if CHUNKS_PATH.exists() and QUERIES_PATH.exists() and SUMMARY_PATH.exists():
        return (
            read_jsonl(CHUNKS_PATH),
            read_jsonl(QUERIES_PATH),
            json.loads(SUMMARY_PATH.read_text()),
        )
    corpus = json.loads((RAW / "corpus.json").read_text())
    raw_queries = json.loads((RAW / "MultiHopRAG.json").read_text())
    for index, query in enumerate(raw_queries):
        query["query_id"] = f"q{index:05d}"
    corpus_ids = {row["url"] for row in corpus}
    nonnull = [
        q
        for q in raw_queries
        if q.get("question_type") != "null_query"
        and q.get("evidence_list")
        and {e["url"] for e in q["evidence_list"]} <= corpus_ids
    ]
    shuffled = nonnull[:]
    random.Random(SEED).shuffle(shuffled)
    selected = set()
    for query in shuffled:
        required = {e["url"] for e in query["evidence_list"]}
        if len(selected | required) <= ARTICLE_COUNT:
            selected |= required
        if len(selected) >= ARTICLE_COUNT:
            break
    ordered_articles = [
        row for row in sorted(corpus, key=lambda x: x["url"]) if row["url"] in selected
    ]
    chunks = [
        chunk
        for index, article in enumerate(ordered_articles)
        for chunk in chunk_article(article, index)
    ]
    by_document = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)
    queries, methods = [], defaultdict(int)
    unresolved_queries = 0
    for query in nonnull:
        required = {e["url"] for e in query["evidence_list"]}
        if not required <= selected:
            continue
        units, failed = [], False
        for evidence in query["evidence_list"]:
            hits, method = locate_fact(
                evidence.get("fact", ""), by_document[evidence["url"]]
            )
            methods[method] += 1
            failed |= not bool(hits)
            units.append(sorted(set(hits)))
        if failed:
            unresolved_queries += 1
            continue
        queries.append(
            {
                "query_id": query["query_id"],
                "query": query["query"].strip(),
                "question_type": query["question_type"].replace("_query", ""),
                "required_article_ids": sorted(required),
                "n_required_documents": len(required),
                "n_required_evidence_facts": len(units),
                "evidence_units": units,
                "gold_chunk_ids": sorted({cid for unit in units for cid in unit}),
            }
        )
    summary = {
        "articles": len(selected),
        "chunks": len(chunks),
        "eligible_queries": len(queries),
        "unresolved_queries": unresolved_queries,
        "alignment_methods": dict(methods),
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "selection_seed": SEED,
    }
    write_jsonl(CHUNKS_PATH, chunks)
    write_jsonl(QUERIES_PATH, queries)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print("[data]", summary, flush=True)
    return chunks, queries, summary


def call_json(system: str, user: str, seed: int = SEED) -> dict:
    client = config.get_openai_client()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def dedup_facts(items: list[dict]) -> list[dict]:
    output, keys = [], []
    for item in items:
        fact, evidence = (
            str(item.get("fact", "")).strip(),
            str(item.get("evidence", "")).strip(),
        )
        key = normalize(fact)
        if (
            not key
            or not evidence
            or any(SequenceMatcher(None, key, old).ratio() >= 0.90 for old in keys)
        ):
            continue
        keys.append(key)
        output.append(
            {
                "fact": fact,
                "evidence": evidence,
                "importance": min(5, max(1, int(item.get("importance", 1)))),
                "distinctiveness": min(5, max(1, int(item.get("distinctiveness", 1)))),
            }
        )
    return output


def generate_questions(chunk: dict, facts: list[dict], count: int) -> list[dict]:
    if count == 0:
        return []
    facts_text = "\n".join(
        f"[{i}] importance={f['importance']} distinctiveness={f['distinctiveness']} | {f['fact']}"
        for i, f in enumerate(facts)
    )
    user = f'''Source chunk:\n"""\n{chunk["content"]}\n"""\n\nFacts:\n{facts_text}\n\nGenerate exactly {count} questions.'''
    output, keys = [], []
    for attempt in range(MAX_RETRIES):
        active_user = user
        if output:
            accepted = "\n".join(f"- {item['question']}" for item in output)
            covered = {
                fact_id for item in output for fact_id in item["source_fact_ids"]
            }
            uncovered = [index for index in range(len(facts)) if index not in covered]
            active_user += f"""\n\nAlready accepted questions (do not repeat or paraphrase these):\n{accepted}\n\nPrefer these not-yet-covered source fact IDs: {uncovered}.\nGenerate exactly {count - len(output)} additional distinct questions."""
        items = call_json(QUESTION_PROMPT, active_user, SEED + attempt).get(
            "questions", []
        )
        duplicate_threshold = 0.94 if attempt == 0 else 0.98
        for item in items:
            question, key = (
                str(item.get("question", "")).strip(),
                normalize(str(item.get("question", ""))),
            )
            ids = sorted(
                {
                    int(x)
                    for x in item.get("source_fact_ids", [])
                    if str(x).lstrip("-").isdigit() and 0 <= int(x) < len(facts)
                }
            )
            if (
                question
                and key
                and ids
                and not any(
                    SequenceMatcher(None, key, old).ratio() >= duplicate_threshold
                    for old in keys
                )
            ):
                keys.append(key)
                output.append(
                    {
                        "question": question,
                        "source_fact_ids": ids,
                        "supporting_chunk_ids": [chunk["chunk_id"]],
                    }
                )
        if len(output) >= count:
            return output[:count]
    raise RuntimeError(
        f"Generated {len(output)}/{count} questions for {chunk['chunk_id']}"
    )


def generate_one(chunk: dict) -> dict:
    user = f'''Source chunk:\n"""\n{chunk["content"]}\n"""'''
    facts = []
    for _ in range(MAX_RETRIES):
        facts = dedup_facts(call_json(FACT_PROMPT, user).get("facts", []))
        if facts:
            break
    if not facts:
        raise RuntimeError(f"No facts for {chunk['chunk_id']}")
    raw_budget = round(len(facts) * 0.5)
    bounded_budget = min(20, max(5, raw_budget))
    unbounded = generate_questions(chunk, facts, raw_budget)
    bounded = (
        unbounded
        if bounded_budget == raw_budget
        else generate_questions(chunk, facts, bounded_budget)
    )
    return {
        "chunk_id": chunk["chunk_id"],
        "n_distinct_facts": len(facts),
        "unbounded_budget": raw_budget,
        "bounded_budget": bounded_budget,
        "facts": facts,
        "bounded_questions": bounded,
        "unbounded_questions": unbounded,
    }


def generate(chunks: list[dict]) -> list[dict]:
    cache = {row["chunk_id"]: row for row in read_jsonl(GEN_PATH)}
    todo = [chunk for chunk in chunks if chunk["chunk_id"] not in cache]
    failures = []
    with tqdm(
        total=len(chunks),
        initial=len(cache),
        desc="Chunk generation",
        unit="chunk",
        dynamic_ncols=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_one, chunk): chunk for chunk in todo}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    cache[chunk["chunk_id"]] = future.result()
                    write_jsonl(
                        GEN_PATH,
                        [
                            cache[c["chunk_id"]]
                            for c in chunks
                            if c["chunk_id"] in cache
                        ],
                    )
                    row = cache[chunk["chunk_id"]]
                    progress.set_postfix(
                        facts=row["n_distinct_facts"],
                        questions=row["bounded_budget"],
                        errors=len(failures),
                    )
                except Exception as error:
                    failures.append(
                        {"chunk_id": chunk["chunk_id"], "error": str(error)}
                    )
                    progress.write(f"[generation:error] {chunk['chunk_id']}: {error}")
                finally:
                    progress.update()
    if failures:
        (DATA / "generation_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError(f"{len(failures)} chunks need another cached retry")
    failure_path = DATA / "generation_failures.json"
    if failure_path.exists():
        failure_path.unlink()
    return [cache[chunk["chunk_id"]] for chunk in chunks]


def embed_resumable(path: Path, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    if path.exists():
        payload = json.loads(path.read_text())
        saved_texts = payload.get("texts", [])
        if saved_texts == texts:
            return np.asarray(payload["vectors"], dtype=float)
        if saved_texts == texts[: len(saved_texts)]:
            vectors = payload.get("vectors", [])
    embedder = get_embedder()
    for start in range(len(vectors), len(texts), batch_size):
        vectors.extend(embedder.embed_documents(texts[start : start + batch_size]))
        path.write_text(
            json.dumps({"texts": texts[: len(vectors)], "vectors": vectors})
        )
        print(f"[embed] {path.name} {len(vectors)}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=float)


def evaluate(
    chunks: list[dict], queries: list[dict], generated: list[dict], summary: dict
) -> dict:
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_vectors = embed_resumable(CHUNK_VECTORS, [c["content"] for c in chunks])
    query_vectors = embed_resumable(QUERY_VECTORS, [q["query"] for q in queries])
    conditions = {}
    for name, field, vector_path in (
        ("Adaptive generated questions 5–20", "bounded_questions", BOUNDED_VECTORS),
        (
            "Adaptive generated questions unbounded",
            "unbounded_questions",
            UNBOUNDED_VECTORS,
        ),
    ):
        questions = [
            (question["question"], row["chunk_id"])
            for row in generated
            for question in row[field]
        ]
        question_vectors = embed_resumable(vector_path, [q[0] for q in questions])
        by_chunk = defaultdict(list)
        for index, (_, chunk_id) in enumerate(questions):
            by_chunk[chunk_id].append(index)
        conditions[name] = {
            "questions": questions,
            "vectors": question_vectors,
            "by_chunk": by_chunk,
            "rows": [],
        }
    bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])
    ranking_output = {name: [] for name in conditions}
    for qi, query in enumerate(queries):
        chunk_score = cosine_scores(chunk_vectors, query_vectors[qi])
        sparse_order = np.argsort(-bm25.get_scores(tokenize(query["query"])))
        sparse_rank = [chunk_ids[i] for i in sparse_order]
        for name, condition in conditions.items():
            qscore = cosine_scores(condition["vectors"], query_vectors[qi])
            # Chunks without generated questions fall back to their chunk-vector
            # score. This preserves the complete retrieval corpus when a
            # generation service cannot enrich a small number of chunks.
            fused = np.asarray(
                [
                    0.5 * chunk_score[i]
                    + 0.5
                    * (
                        max(qscore[condition["by_chunk"][cid]])
                        if condition["by_chunk"][cid]
                        else chunk_score[i]
                    )
                    for i, cid in enumerate(chunk_ids)
                ]
            )
            dense_rank = [chunk_ids[i] for i in np.argsort(-fused)]
            ranking = rrf_merge((dense_rank, sparse_rank), RRF_K)
            values = metric_row(query, ranking)
            condition["rows"].append(values)
            ranking_output[name].append(
                {
                    "query_id": query["query_id"],
                    "ranked_chunk_ids": ranking[:10],
                    "metrics": values,
                }
            )
        if (qi + 1) % 50 == 0:
            print(f"[retrieve] {qi + 1}/{len(queries)}", flush=True)
    keys = list(next(iter(conditions.values()))["rows"][0])
    quality = {}
    for name, field, _ in (
        ("Adaptive generated questions 5–20", "bounded_questions", None),
        ("Adaptive generated questions unbounded", "unbounded_questions", None),
    ):
        counts = [len(row[field]) for row in generated]
        quality[name] = {
            "generated_questions": sum(counts),
            "questions_per_chunk_min": min(counts),
            "questions_per_chunk_mean": float(np.mean(counts)),
            "questions_per_chunk_max": max(counts),
        }
    payload = {
        "protocol": {
            **summary,
            "generation_model": MODEL,
            "embedding": "Iris dim-384",
            "retrieval": "0.5 chunk/question vector fusion + BM25 RRF",
            "rrf_k": RRF_K,
        },
        "conditions": [
            {
                "condition": name,
                "question_rule": (
                    "clamp(5, 20, round(facts * 0.5))"
                    if "5–20" in name
                    else "round(facts * 0.5), no bounds"
                ),
                "quality": quality[name],
                "stored_vectors": len(chunks) + quality[name]["generated_questions"],
                "metrics": {
                    key: float(np.mean([r[key] for r in condition["rows"]]))
                    for key in keys
                },
            }
            for name, condition in conditions.items()
        ],
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    RANKINGS.write_text(json.dumps(ranking_output))
    render(payload)
    return payload


def render(payload: dict) -> None:
    keys = (
        "evidence_recall@1",
        "evidence_recall@5",
        "evidence_recall@10",
        "all_evidence_hit@5",
        "mrr@10",
    )
    rows = ""
    for condition in payload["conditions"]:
        q, m = condition["quality"], condition["metrics"]
        cells = "".join(f"<td><b>{m[key]:.3f}</b></td>" for key in keys)
        baseline = condition["condition"] == "Baseline"
        generation = (
            "No generated questions"
            if baseline
            else f"<code>{condition['question_rule']}</code>, based on deduplicated atomic facts per chunk"
        )
        retrieval = (
            "Dense chunk-vector retrieval + BM25 chunk RRF"
            if baseline
            else "0.5/0.5 chunk-question vector fusion + BM25 chunk RRF"
        )
        generator = "None" if baseline else MODEL
        rows += f"""<tr><td class="left"><b>{condition["condition"]}</b></td><td class="left">{generation}</td><td>{q["generated_questions"]}<br><span>{q["questions_per_chunk_min"]}–{q["questions_per_chunk_max"]}/chunk</span></td><td>1024 / 128</td><td>{condition["stored_vectors"]}</td><td class="left">{retrieval}</td><td>{generator}</td>{cells}</tr>"""
    p = payload["protocol"]
    has_baseline = any(row["condition"] == "Baseline" for row in payload["conditions"])
    study = (
        "baseline and adaptive generated questions"
        if has_baseline
        else "adaptive generated questions"
    )
    condition_count = len(payload["conditions"])
    skip_note = (
        f'<p class="note">Adaptive generation succeeded for '
        f"{p['chunks'] - p['skipped_question_generation_chunks']} of "
        f"{p['chunks']} chunks; the remaining "
        f"{p['skipped_question_generation_chunks']} chunks use chunk-vector "
        f"fallback and remain in BM25 retrieval.</p>"
        if p.get("skipped_question_generation_chunks")
        else ""
    )
    REPORT.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>{p["articles"]}-article {study}</title><style>body{{background:#0f1319;color:#e6eaf0;font:14px/1.5 system-ui;padding:20px}}main{{max-width:1900px;margin:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #29313d;padding:9px;text-align:center;vertical-align:top}}th{{background:#161c25}}.left{{text-align:left}}span,.note{{color:#9aa5b3;font-size:11px}}code{{overflow-wrap:anywhere}}</style></head><body><main><h1>{p["articles"]}-article MultiHop-RAG — {study}</h1><p class="note">Same query-first {p["articles"]} articles, {p["eligible_queries"]} eligible queries, {p["chunks"]} chunks, gold mapping, Iris embeddings, BM25, and RRF settings for all {condition_count} conditions.</p>{skip_note}<table><thead><tr><th>Condition</th><th>Question generation</th><th>Questions</th><th>Chunk / overlap</th><th>Stored vectors</th><th>Retrieval</th><th>Generator</th><th>Evidence Recall@1</th><th>Evidence Recall@5</th><th>Evidence Recall@10</th><th>Full-evidence@5</th><th>MRR@10</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>"""
    )


def run() -> dict:
    chunks, queries, summary = prepare()
    generated = generate(chunks)
    payload = evaluate(chunks, queries, generated, summary)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
