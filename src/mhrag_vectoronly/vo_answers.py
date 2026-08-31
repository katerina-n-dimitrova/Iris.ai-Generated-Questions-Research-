"""
Stage: end-to-end RAG answer generation for both conditions (§16).

For each condition we take the top-k UNIQUE PARENT CHUNK texts (not the generated
questions), give them + the query to the LLM under a strict grounded prompt, and
produce a concise answer. Answers are scored with normalized exact match, token
F1, and yes/no accuracy, then split into buckets by how much gold evidence the
retriever actually delivered (all / partial / none) so retrieval failure is
separated from answer-generation failure.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import vo_config as C
import vo_data as D
import vo_retrieval as R

_ANSWER_SYS = (
    "You answer questions using ONLY the supplied evidence chunks. Combine "
    "information across chunks when needed. Do not use outside knowledge. If the "
    "evidence is insufficient, reply exactly 'Insufficient evidence.'\n"
    "Answer with the SHORTEST possible span — just the entity, name, number, "
    "date, or a single 'Yes'/'No'. No sentence, no explanation, no punctuation "
    "beyond what the answer itself needs."
)


def _chunk_text_lookup() -> Dict[str, str]:
    return {c["chunk_id"]: c["text"] for c in D.load_chunks()}


# --------------------------------------------------------------------------- #
# Answer scoring (SQuAD-style normalization)
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _em(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def _is_yesno(gold: str) -> bool:
    return _normalize(gold) in {"yes", "no"}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _answer_one(client, query: str, chunk_texts: List[str]) -> str:
    context = "\n\n".join(f"[Chunk {i + 1}] {t}" for i, t in enumerate(chunk_texts))
    resp = client.chat.completions.create(
        model=C.gen_model(),
        temperature=0.0,
        messages=[
            {"role": "system", "content": _ANSWER_SYS},
            {
                "role": "user",
                "content": f"Question: {query}\n\nEvidence:\n{context}\n\nShortest-span answer:",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def _evidence_bucket(row: dict, k: int) -> str:
    units = [set(u) for u in row["evidence_units"]]
    topk = {r["chunk_id"] for r in row["ranked"][:k]}
    covered = sum(1 for u in units if u & topk)
    if not units:
        return "none"
    if covered == len(units):
        return "all"
    return "partial" if covered > 0 else "none"


def run_answers() -> dict:
    if not C.ANSWER_ENABLED:
        return {"enabled": False}
    k = C.ANSWER_TOP_K
    text_of = _chunk_text_lookup()
    client = C.openai_client()

    rank = {
        C.COND_A: {r["query_id"]: r for r in R.load_rankings(C.COND_A)},
        C.COND_B: {r["query_id"]: r for r in R.load_rankings(C.COND_B)},
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
        futs = {ex.submit(_answer_one, client, t[1]["query"], t[3]): t for t in tasks}
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
                    "evidence_bucket": _evidence_bucket(row, k),
                    "em": _em(ans, gold),
                    "f1": round(_f1(ans, gold), 4),
                    "is_yesno": _is_yesno(gold),
                    "yesno_correct": (
                        int(_normalize(ans).startswith(_normalize(gold)))
                        if _is_yesno(gold)
                        else None
                    ),
                }
            )

    D._write_jsonl(C.RESULTS_DIR / "generation_results.jsonl", out_rows)

    # ---- aggregate ------------------------------------------------------- #
    def agg(rows):
        n = len(rows)
        yn = [r for r in rows if r["is_yesno"]]
        return {
            "n": n,
            "exact_match": round(sum(r["em"] for r in rows) / n, 4) if n else 0,
            "token_f1": round(sum(r["f1"] for r in rows) / n, 4) if n else 0,
            "yesno_accuracy": (
                round(sum(r["yesno_correct"] for r in yn) / len(yn), 4) if yn else None
            ),
        }

    summary = {"top_k": k, "by_condition": {}, "by_condition_bucket": {}}
    for cond in C.CONDITIONS:
        rows = [r for r in out_rows if r["condition"] == cond]
        summary["by_condition"][cond] = agg(rows)
        summary["by_condition_bucket"][cond] = {
            b: agg([r for r in rows if r["evidence_bucket"] == b])
            for b in ("all", "partial", "none")
        }
    with C.ANSWER_METRICS_JSON.open("w") as fh:
        json.dump(summary, fh, indent=2)
    a, b = summary["by_condition"][C.COND_A], summary["by_condition"][C.COND_B]
    print(
        f"[answers] EM  A={a['exact_match']} B={b['exact_match']} | "
        f"F1 A={a['token_f1']} B={b['token_f1']}"
    )
    return summary


if __name__ == "__main__":
    run_answers()
