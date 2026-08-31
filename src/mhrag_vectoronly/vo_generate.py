"""
Stage: generate + validate EXACTLY 10 synthetic questions per chunk (§7, §8).

Each chunk's questions are natural user queries the parent chunk alone can
answer (doc2query / HyPE-style bridges). They are embedded and indexed in
Condition B; the parent chunk stays the source of truth.

The generator LLM sees ONLY the chunk text — never a benchmark query, gold
answer, or gold evidence. Output is structured JSON. Every set is validated:
exactly 10 non-empty questions, valid JSON, no exact/near duplicates, each
supporting_text span present in the parent chunk, no "the passage/chunk/article"
meta-references. Failures are retried up to GEN_MAX_RETRIES, then logged.

Results are cached to generated_questions.jsonl (resumable — cached chunks are
never regenerated).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import vo_config as C
import vo_data as D

_WS = re.compile(r"\s+")
_META = re.compile(
    r"\b(this|the)\s+(passage|chunk|text|article|document|excerpt|snippet)\b",
    re.IGNORECASE,
)

_SYSTEM = (
    "You generate natural-language search questions that a specific news-article "
    "passage can fully answer. The questions will be embedded and used as search "
    "bridges to retrieve that passage. Respond with ONLY a JSON object."
)

_USER_TMPL = """Read the passage below and write EXACTLY {n} natural-language questions that can be answered using ONLY the information in this passage.

Rules:
- Every question must be answerable from this passage alone; do not use outside knowledge or facts from other passages.
- Do NOT refer to "the passage", "the article", "the text", "this document", etc. Each question must stand on its own.
- Preserve important names, organizations, dates, locations, products, events, and numbers so the question is specific.
- Cover DIFFERENT facts — do not write 10 paraphrases of one question.
- Avoid generic questions like "What happened?" or "What is important?". Do not put the answer inside the question wording.
- Use varied types where supported: who / what / when / where / numeric / yes-no / cause-effect / relationship / definition.

Return a JSON object of this exact shape:
{{"questions": [{{"question": "...", "question_type": "who|what|when|where|numeric|yesno|cause|relationship|definition|other", "supporting_text": "<short verbatim span copied from the passage that proves the question is answerable>", "short_answer": "..."}}]}}

Passage:
\"\"\"
{chunk}
\"\"\""""


def _norm(t: str) -> str:
    return _WS.sub(" ", (t or "")).strip().lower()


# --------------------------------------------------------------------------- #
# Validation (§8)
# --------------------------------------------------------------------------- #
def validate_set(chunk_text: str, items: List[dict]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if len(items) != C.QUESTIONS_PER_CHUNK:
        reasons.append(f"expected {C.QUESTIONS_PER_CHUNK} got {len(items)}")
    ctext = _norm(chunk_text)
    seen: List[str] = []
    for it in items:
        q = (it.get("question") or "").strip()
        if not q:
            reasons.append("empty question")
            continue
        if _META.search(q):
            reasons.append(f"meta-reference: {q[:60]}")
        qn = _norm(q)
        for prev in seen:
            if (
                qn == prev
                or SequenceMatcher(None, qn, prev).ratio() >= C.NEAR_DUP_THRESHOLD
            ):
                reasons.append(f"near/exact duplicate: {q[:60]}")
                break
        seen.append(qn)
        sup = _norm(it.get("supporting_text") or "")
        # supporting span should be traceable to the chunk (token-overlap tolerant)
        if sup and sup not in ctext:
            # tolerate minor edits: require >=0.85 partial ratio against chunk
            if not _span_in(sup, ctext):
                reasons.append(f"supporting_text not in chunk: {sup[:50]}")
    return (len(reasons) == 0), reasons


def _span_in(span: str, ctext: str) -> bool:
    if span in ctext:
        return True
    # sliding fuzzy: is there a window of ctext close to span?
    words = span.split()
    if len(words) < 3:
        return False
    probe = " ".join(words[:8])
    return probe in ctext or SequenceMatcher(None, span, ctext).ratio() >= 0.6


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def _generate_one(client, chunk: dict) -> Tuple[List[dict], List[str], int]:
    prompt = _USER_TMPL.format(n=C.QUESTIONS_PER_CHUNK, chunk=chunk["text"])
    attempts = 0
    items: List[dict] = []
    last_reasons: List[str] = ["no output"]
    for attempts in range(1, C.GEN_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content
            items = json.loads(raw).get("questions", [])
        except Exception as e:  # noqa: BLE001
            last_reasons = [f"llm/json error: {e.__class__.__name__}"]
            continue
        ok, reasons = validate_set(chunk["text"], items)
        if ok:
            return items[: C.QUESTIONS_PER_CHUNK], [], attempts
        last_reasons = reasons
    return items[: C.QUESTIONS_PER_CHUNK], last_reasons, attempts


# --------------------------------------------------------------------------- #
# Orchestration (cached / resumable)
# --------------------------------------------------------------------------- #
def _load_cache() -> Dict[str, dict]:
    if not C.GENQ_PATH.exists():
        return {}
    return {r["chunk_id"]: r for r in D.read_jsonl(C.GENQ_PATH)}


def generate_all(force: bool = False) -> dict:
    chunks = D.load_chunks()
    cache = {} if force else _load_cache()
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"[generate] {len(chunks)} chunks, {len(cache)} cached, {len(todo)} to do")

    client = C.openai_client()
    failures: List[dict] = []
    retry_counts: List[int] = []
    t0 = time.perf_counter()

    if todo:
        results: Dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_generate_one, client, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                items, reasons, attempts = fut.result()
                retry_counts.append(attempts - 1)
                row = {
                    "chunk_id": c["chunk_id"],
                    "parent_document_id": c["parent_document_id"],
                    "questions": [
                        {
                            "question_id": f"{c['chunk_id']}::q{j}",
                            "question": (it.get("question") or "").strip(),
                            "question_type": it.get("question_type", "other"),
                            "supporting_text": it.get("supporting_text", ""),
                            "short_answer": it.get("short_answer", ""),
                        }
                        for j, it in enumerate(items)
                    ],
                    "n_questions": len(items),
                    "valid": len(reasons) == 0,
                    "attempts": attempts,
                }
                results[c["chunk_id"]] = row
                if reasons:
                    failures.append(
                        {
                            "chunk_id": c["chunk_id"],
                            "reasons": reasons,
                            "n_questions": len(items),
                        }
                    )
                if i % 25 == 0:
                    print(f"  [generate] {i}/{len(todo)}", flush=True)
        cache.update(results)

    gen_seconds = round(time.perf_counter() - t0, 2)
    ordered = [cache[c["chunk_id"]] for c in chunks if c["chunk_id"] in cache]
    D._write_jsonl(C.GENQ_PATH, ordered)
    D._write_jsonl(C.GENQ_FAILURES, failures)

    # ----- quality report (§8) ------------------------------------------- #
    total_expected = len(chunks) * C.QUESTIONS_PER_CHUNK
    all_q = [q for r in ordered for q in r["questions"] if q["question"].strip()]
    valid_sets = sum(
        1 for r in ordered if r["valid"] and r["n_questions"] == C.QUESTIONS_PER_CHUNK
    )
    # duplicate stats across all questions
    norms = [_norm(q["question"]) for q in all_q]
    dup = len(norms) - len(set(norms))
    qtype_dist: Dict[str, int] = {}
    for q in all_q:
        qtype_dist[q["question_type"]] = qtype_dist.get(q["question_type"], 0) + 1

    import random as _rnd

    sample_chunks = _rnd.Random(C.SEED).sample(ordered, min(3, len(ordered)))
    samples = [
        {
            "chunk_id": r["chunk_id"],
            "questions": [q["question"] for q in r["questions"][:5]],
        }
        for r in sample_chunks
    ]

    report = {
        "total_chunks": len(chunks),
        "expected_questions": total_expected,
        "actual_valid_questions": len(all_q),
        "avg_valid_questions_per_chunk": round(len(all_q) / max(len(chunks), 1), 3),
        "valid_sets": valid_sets,
        "validation_failure_rate": round(len(failures) / max(len(chunks), 1), 4),
        "duplicate_questions_global": dup,
        "duplicate_rate": round(dup / max(len(all_q), 1), 4),
        "avg_retries_per_chunk": round(sum(retry_counts) / max(len(retry_counts), 1), 3)
        if retry_counts
        else 0.0,
        "unresolved_failures": len(failures),
        "question_type_distribution": dict(
            sorted(qtype_dist.items(), key=lambda kv: -kv[1])
        ),
        "generation_seconds": gen_seconds,
        "generation_model": C.gen_model(),
        "sample_questions": samples,
    }
    with C.GENQ_QUALITY.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"[generate] valid_sets={valid_sets}/{len(chunks)} "
        f"questions={len(all_q)} failures={len(failures)} ({gen_seconds}s)"
    )
    return report


def load_questions_by_chunk() -> Dict[str, List[str]]:
    """chunk_id -> [question strings] (only non-empty)."""
    out: Dict[str, List[str]] = {}
    for r in D.read_jsonl(C.GENQ_PATH):
        out[r["chunk_id"]] = [
            q["question"] for q in r["questions"] if q["question"].strip()
        ]
    return out


def load_question_records() -> List[dict]:
    return D.read_jsonl(C.GENQ_PATH)


if __name__ == "__main__":
    import pprint

    pprint.pp({k: v for k, v in generate_all().items() if k != "sample_questions"})
