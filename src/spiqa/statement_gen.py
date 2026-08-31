"""
Statement-level generated-question generation (structured, anchor-heavy).

For each chunk the LLM:
  1. splits the chunk into ATOMIC factual statements (one clear fact each,
     grounded only in the chunk — no invented facts), then
  2. for those statements writes up to 10 SPECIFIC questions, each answerable
     from a single statement and carrying >=1 concrete anchor (dataset/method/
     model/metric name, number, table/figure name, scientific entity, ...).

Structured output per question (cached to JSONL, one row per chunk, resumable):
    {chunk_id, paper_id, questions: [ {
        statement, question, answer,
        question_type,          # metric|method|dataset|result|comparison|
                                # table|figure|definition|limitation|numeric_detail
        required_anchors: [...],
        source_text_span        # exact substring of the chunk supporting it
    }, ... ]}

Same LLM as every other condition; generation is offline + cached so it never
re-runs. Generic questions ("what did the paper find?") are explicitly banned in
the prompt and pruned in post-processing.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import spiqa_config as C
from llm_adapter import get_llm, Usage
from question_gen import estimate_cost

CACHE = C.PROCESSED_DIR / "test-B_statement_questions.jsonl"

QUESTION_TYPES = {
    "metric",
    "method",
    "dataset",
    "result",
    "comparison",
    "table",
    "figure",
    "definition",
    "limitation",
    "numeric_detail",
}

_GENERIC = [
    "what did the paper find",
    "what method was used",
    "what is the main idea",
    "what were the results",
    "what did the authors conclude",
    "what is the paper about",
    "what did the study find",
    "what was the conclusion",
]


def _prompt(chunk_text: str):
    system = (
        "You build statement-level retrieval questions for a scientific-paper RAG "
        "system. First mentally split the passage into ATOMIC factual statements "
        "(one clear fact each, grounded ONLY in the passage — never invent facts). "
        "Then write up to 10 SPECIFIC questions, each answerable from ONE statement "
        "and each containing at least one CONCRETE ANCHOR taken verbatim from the "
        "passage: a dataset/method/model/metric name, a number, a table/figure name, "
        "a scientific entity, an experimental setting, or a result value.\n"
        "BAN generic questions that could apply to many papers, e.g. 'What did the "
        "paper find?', 'What method was used?', 'What were the results?'.\n"
        'Return STRICT JSON: {"questions": [{"statement": str, "question": str, '
        '"answer": str, "question_type": one of '
        "[metric,method,dataset,result,comparison,table,figure,definition,limitation,"
        'numeric_detail], "required_anchors": [str, ...], "source_text_span": '
        "str (exact substring of the passage)}]}. Output at most 10 questions; fewer "
        "if the passage cannot support 10 specific, anchored ones."
    )
    user = f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\nReturn the JSON now.'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _is_generic(q: str) -> bool:
    ql = re.sub(r"[^a-z ]", "", q.lower()).strip()
    return any(g in ql for g in _GENERIC)


def _clean(objs: List[dict], chunk_text: str) -> List[dict]:
    out, seen = [], set()
    low = chunk_text.lower()
    for o in objs:
        if not isinstance(o, dict):
            continue
        q = str(o.get("question", "")).strip()
        if "?" not in q or _is_generic(q):
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        qt = str(o.get("question_type", "")).strip().lower()
        if qt not in QUESTION_TYPES:
            qt = "other"
        anchors = o.get("required_anchors", []) or []
        anchors = [str(a).strip() for a in anchors if str(a).strip()]
        # keep only anchors that actually appear in the chunk (grounding check)
        anchors = [a for a in anchors if a.lower() in low]
        out.append(
            {
                "statement": str(o.get("statement", "")).strip(),
                "question": q,
                "answer": str(o.get("answer", "")).strip(),
                "question_type": qt,
                "required_anchors": anchors,
                "source_text_span": str(o.get("source_text_span", "")).strip(),
            }
        )
        if len(out) >= 10:
            break
    return out


def _gen_one(chunk_text: str):
    llm = get_llm()
    t0 = time.perf_counter()
    text, usage = llm.chat(_prompt(chunk_text), temperature=0.4, max_tokens=1600)
    secs = time.perf_counter() - t0
    # parse JSON (tolerate code fences / stray text)
    objs = []
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        objs = data.get("questions", []) if isinstance(data, dict) else data
    except Exception:
        objs = []
    return _clean(objs, chunk_text), usage, secs


def _load_cache() -> Dict[str, dict]:
    cache = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["chunk_id"]] = r
    return cache


def generate(
    chunks: List[Dict],
    *,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    cache = _load_cache()
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]
    total = Usage()
    t_wall = time.perf_counter()
    fh = CACHE.open("a", encoding="utf-8")

    def work(c):
        objs, usage, secs = _gen_one(c["text"])
        return c, objs, usage, secs

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futs):
                c, objs, usage, secs = fut.result()
                total += usage
                row = {
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "n_generated": len(objs),
                    "questions": objs,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "gen_seconds": round(secs, 3),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[c["chunk_id"]] = row
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"  [stmt-gen] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()
    return {
        "chunks_newly_generated": len(todo),
        "chunks_cached_now": len(cache),
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "estimated_cost_usd": estimate_cost(
            total.prompt_tokens, total.completion_tokens
        ),
        "wall_seconds": round(time.perf_counter() - t_wall, 1),
    }


def load() -> Dict[str, List[dict]]:
    """Return {chunk_id: [question_obj, ...]} from the cache."""
    out = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["chunk_id"]] = r["questions"]
    return out


if __name__ == "__main__":
    import argparse
    from spiqa_chunker import load_chunks

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test-B")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    chunks = load_chunks(args.split)
    print(json.dumps(generate(chunks, limit=args.limit), indent=2))
