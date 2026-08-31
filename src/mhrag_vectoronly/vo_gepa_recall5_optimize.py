"""Optimize the routing-question prompt directly for Evidence Recall@5."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)
from rank_bm25 import BM25Okapi

import vo_config as C
import vo_hierarchical_hybrid as H
from embeddings import get_embedder

BASE = H.RESULTS / "gepa_recall5"
BASE.mkdir(parents=True, exist_ok=True)
RUN_DIR = BASE / "run"
RESULT_PATH = BASE / "optimization_result.json"
CANDIDATES_PATH = BASE / "candidate_prompts.json"
BEST_PATH = BASE / "best_prompt.txt"
SEED_PATH = H.RESULTS / "gepa" / "best_prompt.txt"

ARTICLES = H.read_jsonl(H.ARTICLES_PATH)
CHUNKS = H.load_chunks()
QUERIES = H.load_queries()
GOLD = H.load_gold()
ANALYSES = {x["query_id"]: x for x in H.query_understanding(False)}
BY_DOC = defaultdict(list)
for chunk in CHUNKS:
    BY_DOC[chunk["document_id"]].append(chunk)

rng = np.random.default_rng(C.SEED)
order = rng.permutation(len(QUERIES)).tolist()
cut = round(len(order) * 0.72)
DEV = [QUERIES[i] for i in order[:cut]]
TEST = [QUERIES[i] for i in order[cut:]]

CLIENT = C.openai_client()
EMBEDDER = get_embedder()
CHUNK_VECTORS = np.asarray(
    EMBEDDER.embed_documents([x["content"] for x in CHUNKS]),
    dtype=np.float64,
)
CHUNK_BM25 = BM25Okapi([H.tokens(x["content"]) for x in CHUNKS])
QUERY_VECTORS = {
    q["query_id"]: np.asarray(EMBEDDER.embed_query(q["query"]), dtype=np.float64)
    for q in QUERIES
}
MEMO: dict[str, dict] = {}


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _generate(prompt: str, article: dict) -> list[dict]:
    chunk_block = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in BY_DOC[article["article_id"]]
    )
    user = f"""Document ID: {article["article_id"]}
Title: {article["title"]}

Complete document with retrievable chunk IDs:
{chunk_block}

Generate exactly 10 document-routing questions. Return JSON:
{{"questions":[{{"question":"...","short_answer":"...",
"evidence":"verbatim quote","primary_chunk_id":"...",
"supporting_chunk_ids":["..."],"question_type":"...",
"evidence_role":"...","important_entities":["..."]}}]}}"""
    response = CLIENT.chat.completions.create(
        model=C.gen_model(),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
    )
    raw = json.loads(response.choices[0].message.content)
    items = raw.get("questions", raw if isinstance(raw, list) else [])
    output, seen = [], set()
    valid_ids = {x["chunk_id"] for x in BY_DOC[article["article_id"]]}
    for item in items:
        question = str(item.get("question", "")).strip()
        key = _norm(question)
        if not question or key in seen:
            continue
        seen.add(key)
        cited = [
            str(x) for x in item.get("supporting_chunk_ids", []) if str(x) in valid_ids
        ]
        primary = str(item.get("primary_chunk_id", ""))
        if primary in valid_ids and primary not in cited:
            cited.insert(0, primary)
        if not cited:
            cited = H._map_evidence(
                str(item.get("evidence", "")),
                BY_DOC[article["article_id"]],
            )
        output.append(
            {
                "question_id": f"{article['article_key']}::r5q{len(output)}",
                "question": question,
                "document_id": article["article_id"],
                "supporting_chunk_ids": cited,
                "short_answer": str(item.get("short_answer", item.get("answer", ""))),
                "question_type": str(item.get("question_type", "factual")),
                "important_entities": [
                    str(x)
                    for x in item.get("important_entities", item.get("entities", []))
                ],
            }
        )
    return output[:10]


def _evaluate(prompt: str, queries: list[dict]) -> dict:
    cache_key = _norm(prompt) + f"::{len(queries)}"
    if cache_key in MEMO:
        return MEMO[cache_key]
    questions, failures = [], []
    counts = {}
    for article in ARTICLES:
        try:
            rows = _generate(prompt, article)
        except Exception as exc:
            rows = []
            failures.append(f"{article['title']}: {type(exc).__name__}: {exc}")
        counts[article["article_id"]] = len(rows)
        questions.extend(rows)

    if not questions:
        result = {
            "score": 0.0,
            "evidence_recall@5": 0.0,
            "question_count": 0,
            "exact_10_documents": 0,
            "misses": [],
            "failures": failures,
        }
        MEMO[cache_key] = result
        return result

    qvectors = np.asarray(
        EMBEDDER.embed_documents([x["question"] for x in questions]),
        dtype=np.float64,
    )
    index = H.Indexes(
        chunks=CHUNKS,
        chunk_map={x["chunk_id"]: x for x in CHUNKS},
        chunk_vectors=CHUNK_VECTORS,
        chunk_bm25=CHUNK_BM25,
        doc_questions=questions,
        docq_vectors=qvectors,
        docq_bm25=BM25Okapi([H.tokens(x["question"]) for x in questions]),
    )

    values, misses = [], []
    for query in queries:
        qvec = QUERY_VECTORS[query["query_id"]]
        analysis = ANALYSES[query["query_id"]]
        routed, _ = H.route_documents(index, qvec, analysis)
        ranked, _ = H._chunk_hybrid(index, qvec, analysis, set(routed))
        row = H._as_metric_row(
            query,
            GOLD[query["query_id"]],
            ranked[: C.RANK_DEPTH],
            index.chunk_map,
        )
        metric = H.VM.per_query(row)["evidence_recall@5"]
        values.append(metric)
        if metric < 1:
            gold_ids = set(GOLD[query["query_id"]]["gold_chunk_ids"])
            misses.append(
                {
                    "query": query["query"],
                    "recall_at_5": metric,
                    "missing_gold_chunks": sorted(gold_ids - set(ranked[:5])),
                    "retrieved_top5": ranked[:5],
                    "routed_documents": routed,
                }
            )
    result = {
        "score": float(np.mean(values)),
        "evidence_recall@5": float(np.mean(values)),
        "question_count": len(questions),
        "exact_10_documents": sum(x == 10 for x in counts.values()),
        "misses": misses[:10],
        "failures": failures,
    }
    MEMO[cache_key] = result
    return result


def evaluator(candidate: str):
    result = _evaluate(candidate, DEV)
    feedback = {
        "evidence_recall_at_5": result["evidence_recall@5"],
        "question_count": result["question_count"],
        "documents_with_exactly_10_questions": result["exact_10_documents"],
        "representative_evidence_failures": result["misses"],
        "generation_failures": result["failures"],
        "instruction": (
            "Revise the generation prompt to make the top five restricted "
            "hybrid chunk results cover more missing gold evidence units. "
            "Increase distinct evidence-chunk and multi-hop bridge coverage "
            "without sacrificing exact entities, dates, numbers, and grounding."
        ),
    }
    print(f"[gepa-r5] ER@5={result['score']:.4f} questions={result['question_count']}")
    return result["score"], feedback


def main():
    seed = SEED_PATH.read_text()
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(RUN_DIR),
            seed=C.SEED,
            display_progress_bar=True,
            max_candidate_proposals=4,
            max_metric_calls=12,
            parallel=False,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=f"openai/{C.gen_model()}",
            reflection_minibatch_size=1,
        ),
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=(
            "Maximize end-to-end Evidence Recall@5 after generated-question "
            "document routing and restricted BM25+Iris-vector chunk retrieval. "
            "This is the only optimization metric."
        ),
        background=(
            "MultiHop-RAG uses 15 documents, 512-token chunks with 128 overlap, "
            "RRF document routing, and RRF chunk retrieval. Feedback lists "
            "missing gold chunks and the retrieved top five."
        ),
        config=config,
    )
    candidates = list(result.candidates)
    best = result.best_candidate
    best_prompt = (
        best.get("candidate", next(iter(best.values())))
        if isinstance(best, dict)
        else str(best)
    )
    seed_dev, best_dev = _evaluate(seed, DEV), _evaluate(best_prompt, DEV)
    seed_test, best_test = _evaluate(seed, TEST), _evaluate(best_prompt, TEST)
    payload = {
        "protocol": {
            "documents": len(ARTICLES),
            "development_queries": len(DEV),
            "held_out_queries": len(TEST),
            "metric": "evidence_recall@5",
            "candidate_proposal_budget": 4,
        },
        "best_idx": int(result.best_idx),
        "total_metric_calls": int(result.total_metric_calls),
        "candidate_count": len(candidates),
        "seed_development": seed_dev["evidence_recall@5"],
        "best_development": best_dev["evidence_recall@5"],
        "seed_held_out": seed_test["evidence_recall@5"],
        "best_held_out": best_test["evidence_recall@5"],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2))
    BEST_PATH.write_text(best_prompt)
    print(json.dumps(payload, indent=2))
    print(f"[gepa-r5] best prompt: {BEST_PATH}")


if __name__ == "__main__":
    main()
