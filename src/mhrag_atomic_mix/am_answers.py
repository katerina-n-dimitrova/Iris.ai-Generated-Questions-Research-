"""
Stage: end-to-end RAG answer generation, Condition A vs E (§23).

Reuses the 15-article harness answer prompt + scorers (`vo_answers`): same answer
LLM, prompt, decoding, and EM/F1/yes-no metrics for both conditions. Both receive
the top-5 ORIGINAL parent chunk texts (never the generated questions). Results are
split by how much gold evidence the retriever delivered (all / partial / none).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import am_config as C
import am_data as D
import am_retrieval as R
import vo_answers as VA


def run_answers() -> dict:
    if not C.ANSWER_ENABLED:
        return {"enabled": False}
    k = C.ANSWER_TOP_K
    text_of = {c["chunk_id"]: c["text"] for c in D.load_chunks()}
    client = C.openai_client()
    rank = {
        C.COND_A: {r["query_id"]: r for r in R.load_rankings(C.COND_A)},
        C.COND_E: {r["query_id"]: r for r in R.load_rankings(C.COND_E)},
    }
    queries = D.load_eligible_queries()

    tasks = []
    for cond in C.CONDITIONS:
        for q in queries:
            row = rank[cond][q["query_id"]]
            ctx = [text_of[r["chunk_id"]] for r in row["ranked"][:k]]
            tasks.append((cond, q, row, ctx))

    out_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(VA._answer_one, client, t[1]["query"], t[3]): t for t in tasks
        }
        for fut in as_completed(futs):
            cond, q, row, ctx = futs[fut]
            try:
                ans = fut.result()
            except Exception as e:  # noqa: BLE001
                ans = f"[error:{e.__class__.__name__}]"
            gold = q["gold_answer"]
            out_rows.append(
                {
                    "query_id": q["query_id"],
                    "condition": cond,
                    "top_k": k,
                    "question_type": q["question_type"],
                    "retrieved_chunk_ids": [r["chunk_id"] for r in row["ranked"][:k]],
                    "retrieved_article_ids": sorted(
                        {r["parent_document_id"] for r in row["ranked"][:k]}
                    ),
                    "generated_answer": ans,
                    "gold_answer": gold,
                    "evidence_bucket": VA._evidence_bucket(row, k),
                    "em": VA._em(ans, gold),
                    "f1": round(VA._f1(ans, gold), 4),
                    "is_yesno": VA._is_yesno(gold),
                    "yesno_correct": (
                        int(VA._normalize(ans).startswith(VA._normalize(gold)))
                        if VA._is_yesno(gold)
                        else None
                    ),
                }
            )

    D._write_jsonl(C.ANSWER_RESULTS, out_rows)

    def agg(rows):
        n = len(rows)
        yn = [r for r in rows if r["is_yesno"]]
        return {
            "n": n,
            "exact_match": round(sum(r["em"] for r in rows) / n, 4) if n else 0,
            "token_f1": round(sum(r["f1"] for r in rows) / n, 4) if n else 0,
            "yesno_accuracy": round(sum(r["yesno_correct"] for r in yn) / len(yn), 4)
            if yn
            else None,
        }

    summary = {"top_k": k, "by_condition": {}, "by_condition_bucket": {}}
    for cond in C.CONDITIONS:
        rows = [r for r in out_rows if r["condition"] == cond]
        summary["by_condition"][cond] = agg(rows)
        summary["by_condition_bucket"][cond] = {
            b: agg([r for r in rows if r["evidence_bucket"] == b])
            for b in ("all", "partial", "none")
        }
    json.dump(summary, open(C.ANSWER_METRICS_JSON, "w"), indent=2)
    a, e = summary["by_condition"][C.COND_A], summary["by_condition"][C.COND_E]
    print(
        f"[answers] EM  A={a['exact_match']} E={e['exact_match']} | F1 A={a['token_f1']} E={e['token_f1']}"
    )
    return summary


if __name__ == "__main__":
    run_answers()
