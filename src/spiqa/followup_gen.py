"""
LLM generation for the follow-up "question quality" experiments (cached, resumable).

Two new generators, both keyed by chunk_id in JSONL caches under
``data/processed/spiqa/`` so an interrupted or repeated run never re-pays:

  * generate_bm25_aware(...)  -> {split}_questions_bm25aware.jsonl
        For each chunk we extract lexical anchors (followup_lib.extract_terms)
        and ask the LLM for questions that naturally embed those exact terms and
        plausible synonyms — doc2query with deliberate lexical grounding so BM25
        can match verbatim.

  * build_structured(...)     -> {split}_structured_figs.jsonl
        For figure/table chunks, turn the caption + cached vision description +
        section heading + nearby paragraph into a STRUCTURED record (Figure ID,
        type, caption, axes, main trend, key values, compared methods, dataset/
        task, metrics, nearby context). We then generate questions FROM that
        structured text (via question_gen) and index the structured text itself.

Same LLM (gpt-4o-mini via llm_adapter), same temperature/worker settings as the
original question generation. No image calls here — the structured builder reuses
the already-cached vision descriptions, so it is text-only and cheap.
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
from question_gen import _parse_questions, _max_output_tokens


# --------------------------------------------------------------------------- #
# small resumable JSONL cache helper
# --------------------------------------------------------------------------- #
def _load_jsonl(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["chunk_id"]] = r
    return cache


# --------------------------------------------------------------------------- #
# 2 · BM25-aware questions
# --------------------------------------------------------------------------- #
def _bm25_prompt(chunk_text: str, terms: List[str], n: int) -> List[Dict[str, str]]:
    system = (
        "You generate retrieval questions for a scientific-paper RAG system that "
        "uses BOTH dense and lexical (BM25) search. Given a passage and a list of "
        "KEY TERMS from it, output questions that are SPECIFIC and FULLY answerable "
        "using ONLY the passage. Each question must read naturally, and across the "
        "set you must reuse the key terms VERBATIM (their exact surface form) while "
        "also weaving in plausible synonyms or expansions of those terms, so a "
        "lexical retriever can match them. Never introduce facts, numbers, or "
        "entities not present in the passage. Output ONLY a numbered list, one "
        "question per line, no preamble."
    )
    user = (
        f'Passage:\n"""\n{chunk_text.strip()}\n"""\n\n'
        f"Key terms (reuse these exact strings, plus natural synonyms): "
        f"{', '.join(terms) if terms else '(none — use the passage’s own salient terms)'}\n\n"
        f"Write up to {n} distinct questions answerable solely from this passage, "
        f"collectively covering these key terms."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_bm25_aware(
    chunks: List[Dict],
    terms_by_chunk: Dict[str, List[str]],
    split: str,
    *,
    n: int = 10,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    path = C.PROCESSED_DIR / f"{split}_questions_bm25aware.jsonl"
    cache = _load_jsonl(path)
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    usage = Usage()
    t0 = time.perf_counter()
    fh = path.open("a", encoding="utf-8")
    llm = get_llm()

    def work(c):
        terms = terms_by_chunk.get(c["chunk_id"], [])
        t1 = time.perf_counter()
        text, u = llm.chat(
            _bm25_prompt(c["text"], terms, n),
            temperature=C.LLM_TEMPERATURE,
            max_tokens=_max_output_tokens(n),
        )
        return (
            c["chunk_id"],
            _parse_questions(text)[:n],
            u,
            time.perf_counter() - t1,
            terms,
        )

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futs):
                cid, qs, u, secs, terms = fut.result()
                usage += u
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": cid,
                            "n_generated": len(qs),
                            "questions": qs,
                            "anchor_terms": terms,
                            "gen_seconds": round(secs, 3),
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()
                cache[cid] = {"questions": qs}
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"  [bm25aware:{split}] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()
    return {
        "split": split,
        "newly_generated": len(todo),
        "cached_now": len(cache),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def load_bm25_aware(split: str) -> Dict[str, List[str]]:
    path = C.PROCESSED_DIR / f"{split}_questions_bm25aware.jsonl"
    out: Dict[str, List[str]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["chunk_id"]] = r["questions"]
    return out


# --------------------------------------------------------------------------- #
# 6 · Structured figure/table representation (+ its questions)
# --------------------------------------------------------------------------- #
_STRUCT_FIELDS = [
    "Figure/table type",
    "Section heading",
    "Caption",
    "X-axis",
    "Y-axis",
    "Main trend",
    "Key values",
    "Compared methods/models",
    "Dataset/task",
    "Important metrics",
    "Nearby paragraph context",
]


def _struct_prompt(
    fig_id: str, fig_type: str, caption: str, section: str, vision: str, nearby: str
) -> List[Dict[str, str]]:
    system = (
        "You convert a scientific figure/table into a STRUCTURED record for "
        "retrieval. Fill each field using ONLY the provided caption, visual "
        "description, and nearby text. If a field is unknown or not applicable "
        "(e.g. a schematic has no axes), write 'n/a'. Never invent numbers, "
        "methods, datasets, or entities. Be concise and factual. Output EXACTLY "
        "the requested fields, one per line, as 'Field: value'."
    )
    fields = "\n".join(f"{f}:" for f in _STRUCT_FIELDS)
    user = (
        f"Figure ID: {fig_id}\n"
        f"Declared type: {fig_type or 'unknown'}\n"
        f"Caption: {caption or 'n/a'}\n"
        f"Section heading: {section or 'n/a'}\n"
        f"Visual description (from a vision model): {vision or 'n/a'}\n"
        f"Nearby text: {nearby[:900] or 'n/a'}\n\n"
        f"Produce these fields:\n{fields}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _struct_qprompt(structured_text: str, n: int) -> List[Dict[str, str]]:
    system = (
        "You generate retrieval questions for a scientific-paper RAG system. "
        "Given a STRUCTURED figure/table record, output SPECIFIC questions that a "
        "reader could answer from this figure/table — about its axes, trends, key "
        "values, compared methods, dataset/task and metrics. Use only information "
        "present in the record; never invent numbers or entities. Output ONLY a "
        "numbered list, one question per line."
    )
    user = (
        f'Structured record:\n"""\n{structured_text.strip()}\n"""\n\n'
        f"Write up to {n} distinct, answerable questions about this figure/table."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_structured(
    fig_chunks: List[Dict],
    vision: Dict[str, str],
    nearby_by_chunk: Dict[str, str],
    split: str,
    *,
    n_questions: int = 10,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    """
    For each figure/table chunk: build a structured record, then generate
    questions from it. Both cached together, keyed by chunk_id.
    """
    path = C.PROCESSED_DIR / f"{split}_structured_figs.jsonl"
    cache = _load_jsonl(path)
    todo = [c for c in fig_chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    usage = Usage()
    t0 = time.perf_counter()
    fh = path.open("a", encoding="utf-8")
    llm = get_llm()

    def work(c):
        md = c["metadata"]
        fid = md.get("fig_id", "")
        struct_text, u1 = llm.chat(
            _struct_prompt(
                fid,
                md.get("fig_type", ""),
                md.get("caption", ""),
                md.get("section_heading", ""),
                vision.get(fid, ""),
                nearby_by_chunk.get(c["chunk_id"], ""),
            ),
            temperature=C.LLM_TEMPERATURE,
            max_tokens=400,
        )
        qtext, u2 = llm.chat(
            _struct_qprompt(struct_text, n_questions),
            temperature=C.LLM_TEMPERATURE,
            max_tokens=_max_output_tokens(n_questions),
        )
        qs = _parse_questions(qtext)[:n_questions]
        return c["chunk_id"], struct_text.strip(), qs, u1, u2

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futs):
                cid, struct_text, qs, u1, u2 = fut.result()
                usage += u1
                usage += u2
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": cid,
                            "structured_text": struct_text,
                            "n_generated": len(qs),
                            "questions": qs,
                            "prompt_tokens": u1.prompt_tokens + u2.prompt_tokens,
                            "completion_tokens": u1.completion_tokens
                            + u2.completion_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()
                cache[cid] = {"questions": qs}
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(
                        f"  [structured:{split}] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()
    return {
        "split": split,
        "newly_generated": len(todo),
        "cached_now": len(cache),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def load_structured(split: str):
    """Return (structured_text_by_chunk, questions_by_chunk)."""
    path = C.PROCESSED_DIR / f"{split}_structured_figs.jsonl"
    text_by: Dict[str, str] = {}
    q_by: Dict[str, List[str]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                text_by[r["chunk_id"]] = r["structured_text"]
                q_by[r["chunk_id"]] = r["questions"]
    return text_by, q_by
