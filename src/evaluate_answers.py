"""
Evaluate generated answers from answers_<dataset>_<condition>.jsonl.

First-version metrics (cheap, no extra services):
    - exact_match            normalized string equality vs gold
    - token_f1               token overlap F1 vs gold
    - llm_correctness        optional 0/1 LLM judge (--use-llm-judge)

Placeholders are emitted for the richer RAG metrics so the schema is stable and
you can wire them in later:
    - faithfulness
    - citation_accuracy
    - answer_relevance
These default to None unless --use-ragas is passed (requires the `ragas`
package and an OpenAI key); the hook is in compute_ragas_metrics().

Writes results/answer_metrics/answer_scores.csv.

Usage:
    python src/evaluate_answers.py
    python src/evaluate_answers.py --use-llm-judge
"""

from __future__ import annotations

import argparse
import csv
import re
import string
from typing import Dict, List, Optional

import common
import config

ANSWER_SCORES = config.ANSWER_METRICS_DIR / "answer_scores.csv"
SCORE_FIELDS = [
    "dataset",
    "condition",
    "num_answers",
    "exact_match",
    "token_f1",
    "llm_correctness",
    "answer_correctness",
    "faithfulness",
    "citation_accuracy",
    "answer_relevance",
]


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return 0.0
    common_tokens = set(p) & set(g)
    if not common_tokens:
        return 0.0
    # multiset overlap
    overlap = sum(min(p.count(t), g.count(t)) for t in common_tokens)
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def llm_judge(client, question: str, gold: str, pred: str) -> float:
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict grader. Reply with only '1' if the "
                    "predicted answer is correct given the gold answer, else '0'.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nGold: {gold}\nPredicted: {pred}\n"
                    "Correct (1/0)?",
                },
            ],
            temperature=0.0,
            max_tokens=2,
        )
        return 1.0 if resp.choices[0].message.content.strip().startswith("1") else 0.0
    except Exception:
        return 0.0


_LC_NULL = {
    "faithfulness": None,
    "citation_accuracy": None,
    "answer_relevance": None,
    "answer_correctness": None,
}

_GRADE_SYSTEM = (
    "You are a strict RAG answer evaluator. Given a question, the retrieved "
    "context, the generated answer, and optionally a gold answer, score each "
    "metric from 0.0 to 1.0:\n"
    "- faithfulness: every claim in the answer is supported by the context.\n"
    "- answer_relevance: the answer directly addresses the question.\n"
    "- citation_accuracy: the answer's facts actually appear in the context "
    "(not hallucinated).\n"
    "- answer_correctness: the answer matches the gold answer (use null if no "
    "gold answer is given).\n"
    "Reply with ONLY compact JSON, no prose: "
    '{"faithfulness":0.0,"answer_relevance":0.0,"citation_accuracy":0.0,'
    '"answer_correctness":0.0}'
)


def _get_langchain_llm():
    """ChatOpenAI grader. With LANGCHAIN_TRACING_V2=true these calls are traced
    to LangSmith automatically (no extra code needed)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=0,
        max_tokens=120,
        api_key=config.OPENAI_API_KEY,
    )


def _grade_one(
    llm, question: str, context: str, answer: str, gold: str
) -> Dict[str, Optional[float]]:
    import json as _json

    user = (
        f"Question: {question}\n\nContext:\n{context[:4000]}\n\n"
        f"Answer: {answer}\n\nGold answer: {gold or 'N/A'}"
    )
    try:
        out = llm.invoke([("system", _GRADE_SYSTEM), ("human", user)]).content
        start, end = out.find("{"), out.rfind("}")
        data = _json.loads(out[start : end + 1])
        res = {}
        for k in (
            "faithfulness",
            "answer_relevance",
            "citation_accuracy",
            "answer_correctness",
        ):
            v = data.get(k)
            res[k] = float(v) if isinstance(v, (int, float)) else None
        if not gold:
            res["answer_correctness"] = None
        return res
    except Exception:
        return dict(_LC_NULL)


def compute_langchain_metrics(
    rows: List[Dict], concurrency: int = 8
) -> Dict[str, Optional[float]]:
    """LLM-graded RAG metrics via langchain_openai (faithfulness, answer
    relevance, citation accuracy, answer correctness). Averaged over answers.
    Calls run in a thread pool (latency-bound) and are traced to LangSmith when
    LANGCHAIN_TRACING_V2=true."""
    from concurrent.futures import ThreadPoolExecutor

    llm = _get_langchain_llm()

    def grade(r):
        return _grade_one(
            llm,
            r.get("query_text", ""),
            r.get("context", ""),
            r.get("generated_answer", ""),
            r.get("gold_answer", ""),
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        graded = list(ex.map(grade, rows))

    acc = {
        k: []
        for k in (
            "faithfulness",
            "answer_relevance",
            "citation_accuracy",
            "answer_correctness",
        )
    }
    for g in graded:
        for k, v in g.items():
            if v is not None:
                acc[k].append(v)
    return {k: (round(sum(v) / len(v), 4) if v else None) for k, v in acc.items()}


def evaluate_file(path, client, use_langchain: bool) -> Dict | None:
    rows = list(common.read_jsonl(path))
    if not rows:
        return None

    em = f1 = judged = 0.0
    n_judged = 0
    for r in rows:
        pred, gold = r.get("generated_answer", ""), r.get("gold_answer", "")
        em += exact_match(pred, gold)
        f1 += token_f1(pred, gold)
        if client is not None and gold:
            judged += llm_judge(client, r.get("query_text", ""), gold, pred)
            n_judged += 1

    n = len(rows)
    out = {
        "dataset": rows[0]["dataset"],
        "condition": rows[0]["condition"],
        "num_answers": n,
        "exact_match": round(em / n, 4),
        "token_f1": round(f1 / n, 4),
        "llm_correctness": round(judged / n_judged, 4) if n_judged else None,
    }
    out.update(compute_langchain_metrics(rows) if use_langchain else dict(_LC_NULL))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=config.ALL_DATASETS,
        choices=config.ALL_DATASETS,
    )
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=list(config.CONDITIONS),
        choices=config.CONDITIONS,
    )
    ap.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use the chat model as a 0/1 correctness judge.",
    )
    ap.add_argument(
        "--use-langchain",
        action="store_true",
        help="Compute LLM-graded RAG metrics via langchain_openai "
        "(faithfulness, answer relevance, citation accuracy, "
        "answer correctness). Traced to LangSmith if "
        "LANGCHAIN_TRACING_V2=true.",
    )
    args = ap.parse_args()

    client = config.get_openai_client() if args.use_llm_judge else None

    results: List[Dict] = []
    for dataset in args.datasets:
        for condition in args.conditions:
            path = config.ANSWER_METRICS_DIR / f"answers_{dataset}_{condition}.jsonl"
            if not path.exists():
                continue
            r = evaluate_file(path, client, args.use_langchain)
            if r:
                results.append(r)

    if not results:
        print("No answer files found. Run run_answer_generation.py first.")
        return

    groups = {(r["dataset"], r["condition"]) for r in results}
    common.upsert_csv(
        ANSWER_SCORES, SCORE_FIELDS, results, ("dataset", "condition"), groups
    )

    print(f"\n{'dataset':<20}{'cond':<10}{'EM':>8}{'F1':>8}{'LLM':>8}")
    print("-" * 54)
    for r in results:
        print(
            f"{r['dataset']:<20}{r['condition']:<10}"
            f"{r['exact_match']:>8}{r['token_f1']:>8}"
            f"{str(r['llm_correctness']):>8}"
        )
    print(f"\nSaved -> {ANSWER_SCORES.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
