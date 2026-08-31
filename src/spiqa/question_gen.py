"""
Generated-question enrichment (doc2query-style).

For each chunk we ask the LLM for up to MAX_QUESTIONS (default 100) distinct,
specific questions that are *directly answerable from the chunk* and introduce
no facts not supported by it. These questions become alternative retrieval
representations of the chunk (each embedded separately); the chunk itself stays
the source of evidence.

Efficiency / experimental design
--------------------------------
We generate the MAX once per chunk and derive the smaller conditions (Q1, Q10,
Q50) as the first-k subset of that pool. This:
  * isolates the variable of interest (the *number* of questions) instead of
    confounding it with different prompts, and
  * costs ~1 generation call per chunk instead of one per condition.
Set SPIQA_INDEPENDENT_GEN=1 to instead generate each condition independently.

Results are cached to disk (JSONL, one row per chunk) and generation is
resumable — chunks already present in the cache are skipped, so an interrupted
or repeated run never re-pays for questions it already has.

Latency/cost accounting (per chunk): wall-clock generation time and prompt/
completion token usage are recorded so Stage 7/8 can report generation cost.
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

_NUM_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+?)\s*$")


def questions_path(split: str) -> Path:
    return C.PROCESSED_DIR / f"{split}_questions.jsonl"


def _build_prompt(chunk_text: str, n: int) -> List[Dict[str, str]]:
    system = (
        "You generate retrieval questions for a scientific-paper RAG system. "
        "Given a passage, you output questions that are SPECIFIC and FULLY "
        "answerable using ONLY the information in the passage. Never introduce "
        "facts, numbers, or entities that are not present in the passage. Prefer "
        "diverse questions covering different facts, relations, methods, results, "
        "and definitions in the passage. Output ONLY a numbered list, one "
        "question per line, no preamble or commentary."
    )
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Write up to {n} distinct questions answerable solely from this passage. "
        f"If the passage cannot support {n} distinct questions, output as many "
        f"high-quality ones as it genuinely supports."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_questions(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in text.splitlines():
        m = _NUM_LINE.match(line)
        cand = (m.group(1) if m else line).strip()
        if not cand or "?" not in cand:
            continue
        # keep up to the question mark; drop trailing junk
        cand = cand[: cand.rindex("?") + 1].strip()
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _max_output_tokens(n: int) -> int:
    # ~20 tokens/question + headroom, capped so we don't over-request.
    return min(4000, max(256, n * 22))


def generate_for_chunk(chunk_text: str, n: int = C.MAX_QUESTIONS):
    """Generate up to n questions for one chunk. Returns (questions, usage, secs)."""
    llm = get_llm()
    t0 = time.perf_counter()
    text, usage = llm.chat(
        _build_prompt(chunk_text, n),
        temperature=C.LLM_TEMPERATURE,
        max_tokens=_max_output_tokens(n),
    )
    secs = time.perf_counter() - t0
    return _parse_questions(text)[:n], usage, secs


# --------------------------------------------------------------------------- #
# Batch generation with resumable disk cache
# --------------------------------------------------------------------------- #
def _load_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[row["chunk_id"]] = row
    return cache


def generate_questions(
    chunks: List[Dict],
    split: str,
    *,
    n: int = C.MAX_QUESTIONS,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    """
    Generate (and cache) up to n questions for each chunk of a split.

    Returns a summary dict with totals (chunks processed, questions produced,
    token usage, wall-clock time). Safe to re-run: cached chunks are skipped.
    """
    path = questions_path(split)
    cache = _load_cache(path)

    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    total_usage = Usage()
    total_gen_secs = 0.0
    t_wall = time.perf_counter()

    # Append as we go so progress survives interruption.
    fh = path.open("a", encoding="utf-8")
    try:

        def work(c):
            qs, usage, secs = generate_for_chunk(c["text"], n)
            return c["chunk_id"], qs, usage, secs

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futures):
                chunk_id, qs, usage, secs = fut.result()
                total_usage += usage
                total_gen_secs += secs
                row = {
                    "chunk_id": chunk_id,
                    "n_requested": n,
                    "n_generated": len(qs),
                    "questions": qs,
                    "gen_seconds": round(secs, 3),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[chunk_id] = row
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(
                        f"  [{split}] generated {done}/{len(todo)} "
                        f"(cache total {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()

    wall = time.perf_counter() - t_wall
    per_chunk = [
        cache[c["chunk_id"]]["n_generated"] for c in chunks if c["chunk_id"] in cache
    ]
    summary = {
        "split": split,
        "requested_per_chunk": n,
        "chunks_total": len(chunks),
        "chunks_newly_generated": len(todo),
        "chunks_cached_now": len(cache),
        "questions_generated_this_run": sum(
            cache[c["chunk_id"]]["n_generated"] for c in todo if c["chunk_id"] in cache
        ),
        "avg_questions_per_chunk": round(sum(per_chunk) / max(len(per_chunk), 1), 1),
        "prompt_tokens_this_run": total_usage.prompt_tokens,
        "completion_tokens_this_run": total_usage.completion_tokens,
        "sum_gen_seconds_this_run": round(total_gen_secs, 1),
        "avg_gen_seconds_per_chunk": round(total_gen_secs / max(len(todo), 1), 3),
        "wall_seconds_this_run": round(wall, 1),
    }
    return summary


def load_questions(split: str) -> Dict[str, List[str]]:
    """Return {chunk_id: [questions...]} from the cache."""
    path = questions_path(split)
    out: Dict[str, List[str]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = row["questions"]
    return out


def questions_for_condition(
    all_qs: Dict[str, List[str]], n: int
) -> Dict[str, List[str]]:
    """First-n subset per chunk (n==0 -> baseline, no questions)."""
    if n == 0:
        return {}
    return {cid: qs[:n] for cid, qs in all_qs.items()}


def estimate_cost(
    prompt_tokens: int, completion_tokens: int, model: str = C.LLM_MODEL
) -> float:
    price = C.PRICE_PER_1M.get(model)
    if not price:
        return -1.0
    return round(
        prompt_tokens / 1e6 * price["input"]
        + completion_tokens / 1e6 * price["output"],
        4,
    )


if __name__ == "__main__":
    import argparse
    from spiqa_chunker import load_chunks

    ap = argparse.ArgumentParser(description="Generate questions per chunk.")
    ap.add_argument("--split", default=C.DEFAULT_SPLIT)
    ap.add_argument("-n", "--num", type=int, default=C.MAX_QUESTIONS)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate for the first N uncached chunks (smoke test).",
    )
    args = ap.parse_args()

    chunks = load_chunks(args.split)
    summary = generate_questions(chunks, args.split, n=args.num, limit=args.limit)
    summary["estimated_cost_usd_this_run"] = estimate_cost(
        summary["prompt_tokens_this_run"], summary["completion_tokens_this_run"]
    )
    print(json.dumps(summary, indent=2))
