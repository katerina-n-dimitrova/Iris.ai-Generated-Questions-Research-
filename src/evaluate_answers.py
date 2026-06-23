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
    "dataset", "condition", "num_answers",
    "exact_match", "token_f1", "llm_correctness",
    "faithfulness", "citation_accuracy", "answer_relevance",
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
                {"role": "system", "content":
                    "You are a strict grader. Reply with only '1' if the "
                    "predicted answer is correct given the gold answer, else '0'."},
                {"role": "user", "content":
                    f"Question: {question}\nGold: {gold}\nPredicted: {pred}\n"
                    "Correct (1/0)?"},
            ],
            temperature=0.0,
            max_tokens=2,
        )
        return 1.0 if resp.choices[0].message.content.strip().startswith("1") else 0.0
    except Exception:
        return 0.0


def compute_ragas_metrics(rows: List[Dict]) -> Dict[str, Optional[float]]:
    """
    Hook for ragas-based faithfulness / answer_relevance (+ a citation proxy).
    Left as a stub returning None so the first version runs without ragas/network.
    Implement by building a ragas Dataset from rows (question, answer, contexts,
    ground_truth) and calling ragas.evaluate(...).
    """
    return {"faithfulness": None, "citation_accuracy": None,
            "answer_relevance": None}


def evaluate_file(path, client, use_ragas: bool) -> Dict | None:
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
    out.update(compute_ragas_metrics(rows) if use_ragas
               else {"faithfulness": None, "citation_accuracy": None,
                     "answer_relevance": None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=config.ALL_DATASETS,
                    choices=config.ALL_DATASETS)
    ap.add_argument("--conditions", nargs="*", default=list(config.CONDITIONS),
                    choices=config.CONDITIONS)
    ap.add_argument("--use-llm-judge", action="store_true",
                    help="Use the chat model as a 0/1 correctness judge.")
    ap.add_argument("--use-ragas", action="store_true",
                    help="Compute ragas faithfulness/relevance (requires ragas).")
    args = ap.parse_args()

    client = config.get_openai_client() if args.use_llm_judge else None

    results: List[Dict] = []
    for dataset in args.datasets:
        for condition in args.conditions:
            path = config.ANSWER_METRICS_DIR / f"answers_{dataset}_{condition}.jsonl"
            if not path.exists():
                continue
            r = evaluate_file(path, client, args.use_ragas)
            if r:
                results.append(r)

    if not results:
        print("No answer files found. Run run_answer_generation.py first.")
        return

    with ANSWER_SCORES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'dataset':<20}{'cond':<10}{'EM':>8}{'F1':>8}{'LLM':>8}")
    print("-" * 54)
    for r in results:
        print(f"{r['dataset']:<20}{r['condition']:<10}"
              f"{r['exact_match']:>8}{r['token_f1']:>8}"
              f"{str(r['llm_correctness']):>8}")
    print(f"\nSaved -> {ANSWER_SCORES.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
