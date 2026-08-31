"""GEPA optimization of the hierarchical document-routing question prompt."""

from __future__ import annotations

import json
import re
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vo_config as C
import vo_hierarchical_hybrid as H
from embeddings import get_embedder

OUT = H.RESULTS / "gepa"
OUT.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUT / "run"
RESULT_PATH = OUT / "optimization_result.json"
CANDIDATES_PATH = OUT / "candidate_prompts.json"
BEST_PROMPT_PATH = OUT / "best_prompt.txt"

SEED_PROMPT = """You generate document-level retrieval questions for routing.
Given one complete document, produce exactly 10 diverse, concise questions that
represent the document as a whole. Questions must be explicitly answerable,
specific enough to distinguish this document, and cover different factual,
comparison, process, numerical/result, limitation, and application intents when
supported. Preserve distinguishing people, organizations, products, datasets,
methods, dates, locations, and numbers. Do not create vague questions,
paraphrases, unsupported facts, or combinations of unrelated sections.

For every question provide a concise supported answer, one short verbatim
evidence quote, question type, and important entities. Return valid JSON only."""

ARTICLES = H.read_jsonl(H.ARTICLES_PATH)
QUERIES = H.load_queries()
ARTICLE_BY_ID = {a["article_id"]: a for a in ARTICLES}

# Query-held-out protocol: all source documents are available to construct the
# index, but no held-out evaluation query influences GEPA candidate selection.
rng = np.random.default_rng(C.SEED)
order = rng.permutation(len(QUERIES)).tolist()
cut = round(len(order) * 0.72)
DEV_QUERIES = [QUERIES[i] for i in order[:cut]]
TEST_QUERIES = [QUERIES[i] for i in order[cut:]]

CLIENT = None
EMBEDDER = None
QUERY_VECTORS: dict[str, np.ndarray] = {}
EVAL_CACHE: dict[str, dict] = {}


def _client():
    global CLIENT
    if CLIENT is None:
        CLIENT = C.openai_client()
    return CLIENT


def _embedder():
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = get_embedder()
    return EMBEDDER


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _generate(candidate: str, article: dict) -> list[dict]:
    prompt = f'''Document ID: {article["article_id"]}
Document title: {article["title"]}
Complete document:
"""
{article["cleaned_body"]}
"""

Generate exactly 10 document-routing questions. Return:
{{"questions":[{{"question":"...","short_answer":"...",
"evidence":"verbatim quote","question_type":"...",
"important_entities":["..."]}}]}}'''
    response = _client().chat.completions.create(
        model=C.gen_model(),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": candidate},
            {"role": "user", "content": prompt},
        ],
    )
    raw = json.loads(response.choices[0].message.content)
    rows = []
    seen = set()
    body_norm = _normalize(article["cleaned_body"])
    for item in raw.get("questions", []):
        question = str(item.get("question", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        key = _normalize(question)
        evidence_supported = bool(evidence and _normalize(evidence) in body_norm)
        if not question or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "question": question,
                "short_answer": str(item.get("short_answer", "")).strip(),
                "evidence": evidence,
                "evidence_supported": evidence_supported,
                "question_type": str(item.get("question_type", "factual")),
                "important_entities": [
                    str(x) for x in item.get("important_entities", [])
                ],
                "document_id": article["article_id"],
            }
        )
    return rows


def _rrf_docs(
    question_rows: list[dict], question_vectors: np.ndarray, query: dict
) -> list[str]:
    texts = [x["question"] for x in question_rows]
    bm25 = BM25Okapi([H.tokens(x) for x in texts])
    query_terms = H.tokens(query["query"])
    sparse_scores = bm25.get_scores(query_terms)
    if query["query_id"] not in QUERY_VECTORS:
        QUERY_VECTORS[query["query_id"]] = np.asarray(
            _embedder().embed_query(query["query"]), dtype=np.float64
        )
    dense_scores = H._cosine_matrix(question_vectors, QUERY_VECTORS[query["query_id"]])

    def doc_rank(scores):
        best = defaultdict(lambda: -float("inf"))
        for row, score in zip(question_rows, scores):
            best[row["document_id"]] = max(best[row["document_id"]], float(score))
        return [
            doc for doc, _ in sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
        ]

    fused = H._rrf([doc_rank(sparse_scores), doc_rank(dense_scores)])
    return [
        doc for doc, _ in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _score(candidate: str, queries: list[dict]) -> dict:
    cache_key = _normalize(candidate) + "::" + str(len(queries))
    if cache_key in EVAL_CACHE:
        return EVAL_CACHE[cache_key]

    question_rows = []
    generation_failures = []
    for article in ARTICLES:
        try:
            rows = _generate(candidate, article)
        except Exception as exc:
            rows = []
            generation_failures.append(
                f"{article['title']}: {type(exc).__name__}: {exc}"
            )
        question_rows.extend(rows)

    if question_rows:
        vectors = np.asarray(
            _embedder().embed_documents([x["question"] for x in question_rows]),
            dtype=np.float64,
        )
    else:
        vectors = np.zeros((0, 384), dtype=np.float64)

    recalls, reciprocal_ranks, misses = [], [], []
    for query in queries:
        ranking = _rrf_docs(question_rows, vectors, query)
        required = set(query["required_article_ids"])
        top5 = set(ranking[:5])
        recall = len(required & top5) / len(required)
        first = min(
            (ranking.index(doc) + 1 for doc in required if doc in ranking),
            default=len(ARTICLES) + 1,
        )
        recalls.append(recall)
        reciprocal_ranks.append(1.0 / first)
        if recall < 1:
            misses.append(
                {
                    "query": query["query"],
                    "missed_documents": sorted(required - top5),
                    "top_documents": ranking[:5],
                }
            )

    counts = defaultdict(int)
    supported = 0
    typed = set()
    for row in question_rows:
        counts[row["document_id"]] += 1
        supported += int(row["evidence_supported"])
        typed.add(row["question_type"])
    exact_count_rate = sum(counts[a["article_id"]] == 10 for a in ARTICLES) / len(
        ARTICLES
    )
    support_rate = supported / max(1, len(question_rows))
    routing_recall = float(np.mean(recalls))
    routing_mrr = float(np.mean(reciprocal_ranks))
    quality = 0.5 * exact_count_rate + 0.5 * support_rate
    score = 0.55 * routing_recall + 0.30 * routing_mrr + 0.15 * quality
    result = {
        "score": score,
        "routing_recall@5": routing_recall,
        "routing_mrr": routing_mrr,
        "exact_10_rate": exact_count_rate,
        "evidence_support_rate": support_rate,
        "question_count": len(question_rows),
        "question_types": sorted(typed),
        "misses": misses[:8],
        "generation_failures": generation_failures,
        "questions": question_rows,
    }
    EVAL_CACHE[cache_key] = result
    return result


def evaluator(candidate: str):
    result = _score(candidate, DEV_QUERIES)
    feedback = {
        "routing_recall_at_5": result["routing_recall@5"],
        "routing_mrr": result["routing_mrr"],
        "exactly_ten_questions_rate": result["exact_10_rate"],
        "verbatim_evidence_support_rate": result["evidence_support_rate"],
        "observed_question_types": result["question_types"],
        "representative_routing_failures": result["misses"],
        "generation_failures": result["generation_failures"],
        "guidance": (
            "Improve document discrimination and coverage of entities, dates, "
            "numbers, comparisons, methods, and limitations implicated by the "
            "misses. Preserve strict grounding, diversity, and exactly 10."
        ),
    }
    print(
        f"[gepa] score={result['score']:.4f} "
        f"R@5={result['routing_recall@5']:.4f} "
        f"MRR={result['routing_mrr']:.4f} "
        f"supported={result['evidence_support_rate']:.3f}"
    )
    return result["score"], feedback


def main():
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(RUN_DIR),
            seed=C.SEED,
            display_progress_bar=True,
            max_candidate_proposals=6,
            max_metric_calls=7,
            parallel=False,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=f"openai/{C.gen_model()}",
            reflection_minibatch_size=1,
        ),
    )
    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=evaluator,
        objective=(
            "Optimize a drop-in document-routing question-generation system "
            "prompt. Maximize hybrid BM25+dense document routing Recall@5 and "
            "MRR while producing exactly 10 diverse, discriminative, explicitly "
            "supported questions per document with verbatim evidence."
        ),
        background=(
            "The downstream system embeds each generated question with Iris, "
            "also indexes it in BM25, fuses document ranks with RRF, and routes "
            "MultiHop-RAG queries before chunk retrieval. False-positive "
            "documents are the central failure mode."
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
    seed_dev = _score(SEED_PROMPT, DEV_QUERIES)
    best_dev = _score(best_prompt, DEV_QUERIES)
    seed_test = _score(SEED_PROMPT, TEST_QUERIES)
    best_test = _score(best_prompt, TEST_QUERIES)
    payload = {
        "protocol": {
            "documents": len(ARTICLES),
            "development_queries": len(DEV_QUERIES),
            "held_out_queries": len(TEST_QUERIES),
            "candidate_proposal_budget": 6,
            "generation_model": C.gen_model(),
            "embedding_backend": "iris",
        },
        "best_idx": int(result.best_idx),
        "total_metric_calls": int(result.total_metric_calls),
        "seed_development": {
            k: v for k, v in seed_dev.items() if k not in ("questions", "misses")
        },
        "best_development": {
            k: v for k, v in best_dev.items() if k not in ("questions", "misses")
        },
        "seed_held_out": {
            k: v for k, v in seed_test.items() if k not in ("questions", "misses")
        },
        "best_held_out": {
            k: v for k, v in best_test.items() if k not in ("questions", "misses")
        },
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2))
    BEST_PROMPT_PATH.write_text(best_prompt)
    print(json.dumps(payload, indent=2))
    print(f"[gepa] best prompt: {BEST_PROMPT_PATH}")


if __name__ == "__main__":
    main()
