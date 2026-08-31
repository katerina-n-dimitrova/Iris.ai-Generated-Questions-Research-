"""
Condition B — "10 general questions per chunk" (Experiment-1 style) run on the
SAME 10-article collection as the atomic+chunk-level experiment, so all three are
directly comparable on one closed set:

    A = original chunk vectors            (identical baseline)
    B = 10 unrestricted questions/chunk   (this module)
    E = atomic + chunk-level questions     (am_* pipeline)

Dense cosine only, local Chroma, same embedding model, same 69 eligible queries,
same gpt-5.4-mini generator. Reuses am_data (chunks/gold/queries), am_retrieval
(retrieval math), and vo_metrics (metric suite). Writes its own collection + files.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List

import am_config as C
import am_data as D
import am_index as IDX
import am_retrieval as AR
from embeddings import get_embedder
import vo_metrics as VM

VM.KS = C.TOP_K_VALUES
KS = C.TOP_K_VALUES

GENERAL_COLLECTION = "mhrag_acm10_general_questions"
GENERAL_Q = C.DATA_DIR / "general_questions.jsonl"
GENERAL_RANKINGS = C.RESULTS_DIR / "general_retrieval_results.jsonl"
GENERAL_METRICS = C.RESULTS_DIR / "general_metrics.json"
QUESTIONS_PER_CHUNK = 10

_SYS = (
    "You write search questions for a news-article passage. Output ONLY a JSON "
    "object. Every question must be answerable using only this passage."
)
_USER = """Passage:
\"\"\"
{chunk}
\"\"\"

Write EXACTLY {n} natural-language questions that this passage can answer. Cover
different facts (don't paraphrase one question 10 times). Name the central entity in
each; never say "the passage/article/text"; avoid vague questions like "what happened".
Return JSON: {{"questions": ["...", "..."]}}"""

_WS = re.compile(r"\s+")


def _norm(t):
    return _WS.sub(" ", (t or "")).strip().lower()


def _gen_one(client, chunk: dict) -> List[str]:
    prompt = _USER.format(chunk=chunk["text"], n=QUESTIONS_PER_CHUNK)
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": prompt},
                ],
            )
            qs = json.loads(r.choices[0].message.content).get("questions", [])
            out, seen = [], set()
            for q in qs:
                q = (q or "").strip()
                k = _norm(q)
                if (
                    q
                    and k not in seen
                    and not any(
                        SequenceMatcher(None, k, s).ratio() >= 0.92 for s in seen
                    )
                ):
                    seen.add(k)
                    out.append(q)
            if out:
                return out[:QUESTIONS_PER_CHUNK]
        except Exception:  # noqa: BLE001
            continue
    return []


def generate(force: bool = False) -> dict:
    chunks = D.load_chunks()
    cache = {}
    if not force and GENERAL_Q.exists():
        cache = {r["chunk_id"]: r for r in D.read_jsonl(GENERAL_Q)}
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"[general] {len(chunks)} chunks, {len(cache)} cached, {len(todo)} to do")
    t0 = time.perf_counter()
    if todo:
        client = C.openai_client()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_gen_one, client, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                cache[c["chunk_id"]] = {
                    "chunk_id": c["chunk_id"],
                    "parent_document_id": c["parent_document_id"],
                    "questions": fut.result(),
                }
                if i % 25 == 0:
                    print(f"  [general] {i}/{len(todo)}", flush=True)
    rows = [cache[c["chunk_id"]] for c in chunks if c["chunk_id"] in cache]
    D._write_jsonl(GENERAL_Q, rows)
    n_q = sum(len(r["questions"]) for r in rows)
    secs = round(time.perf_counter() - t0, 2)
    print(f"[general] {n_q} questions ({n_q / len(chunks):.1f}/chunk) in {secs}s")
    return {"chunks": len(chunks), "questions": n_q, "seconds": secs}


def build_index() -> int:
    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    rows = D.read_jsonl(GENERAL_Q)
    embedder = get_embedder()
    flat, owner = [], []
    for r in rows:
        for j, q in enumerate(r["questions"]):
            flat.append(q)
            owner.append((r, j))
    vecs = embedder.embed_documents(flat)
    coll = C.reset_collection(GENERAL_COLLECTION)
    ids, docs, metas = [], [], []
    for (r, j), q in zip(owner, flat):
        cid = r["chunk_id"]
        ids.append(f"{cid}::g{j}")
        docs.append(q)
        metas.append(
            {
                "record_type": "generated_question",
                "question_type": "general",
                "parent_chunk_id": cid,
                "parent_document_id": r["parent_document_id"],
                "title": chunks[cid]["title"],
                "source": chunks[cid]["source"],
            }
        )
    IDX._add_batches(coll, ids, vecs, docs, metas)
    print(f"[general] indexed {len(ids)} question vectors")
    return len(ids)


def evaluate() -> dict:
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    coll_a = C.get_collection(C.BASELINE_COLLECTION)
    coll_b = C.get_collection(GENERAL_COLLECTION)
    base_rows, gen_rows = [], []
    for q in queries:
        g = gold[q["query_id"]]
        qv = embedder.embed_query(q["query"])
        a = AR.retrieve_baseline(coll_a, qv, C.RANK_DEPTH)
        b = AR.retrieve_mixed(coll_b, qv, C.RANK_DEPTH, C.CANDIDATE_MULTIPLIER)
        common = {
            "query_id": q["query_id"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        base_rows.append({**common, "ranked": a})
        gen_rows.append({**common, "ranked": b})
    D._write_jsonl(GENERAL_RANKINGS, gen_rows)

    per_a = [VM.per_query(r) for r in sorted(base_rows, key=lambda r: r["query_id"])]
    per_b = [VM.per_query(r) for r in sorted(gen_rows, key=lambda r: r["query_id"])]
    overall = VM.aggregate({"baseline": per_a, "generated": per_b})
    pair = VM.paired(per_a, per_b)
    rollup = {
        "k_values": KS,
        "n_queries": len(per_b),
        "overall": overall,
        "paired": pair,
    }
    json.dump(rollup, open(GENERAL_METRICS, "w"), indent=2)
    a5 = overall["baseline"]["evidence_recall@5"]["mean"]
    b5 = overall["generated"]["evidence_recall@5"]["mean"]
    print(
        f"[general] EvidenceRecall@5  A={a5:.3f}  B(general)={b5:.3f}  "
        f"Δ={b5 - a5:+.3f} (p={pair['evidence_recall@5']['paired_p']})"
    )
    return rollup


def run():
    generate()
    build_index()
    return evaluate()


if __name__ == "__main__":
    run()
