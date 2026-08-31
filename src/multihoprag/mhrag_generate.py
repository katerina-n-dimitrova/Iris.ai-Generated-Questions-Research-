"""
Synthetic question generation (doc2query-style), arm-aware and cached.

One LLM + temperature is fixed for every arm (config); only the PROMPT changes.
Each arm's questions are cached to its own resumable JSONL so re-runs never
regenerate. A generation that parses to the wrong count / non-questions is
retried once, then logged and kept as-is; the per-arm failure rate is reported.

Prompts registry
----------------
``PROMPTS[name] -> (system_str, user_template_fn, parser_fn)``. This deliverable
ships the ``naive`` prompt (baseline B1). Experiments 1-5 add more prompts here
and a matching Arm in mhrag_config -- indexing/retrieval/eval never change.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mhrag_config as C
import config as base_config

_NUM_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+?)\s*$")
# "TYPE: question?" (optionally numbered); label stripped from the embedded text.
_LABELED = re.compile(r"^\s*(?:\d+[\.\)]\s*)?([A-Za-z]+)\s*[:\-]\s*(.+)$")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_questions(text: str, n: int) -> List[str]:
    """Extract up to n distinct question strings from a numbered/bulleted list."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _NUM_LINE.match(line)
        cand = (m.group(1) if m else line).strip()
        if not cand or "?" not in cand:
            continue
        cand = cand[: cand.rindex("?") + 1].strip()
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def parse_labeled_questions(text: str, n: int) -> List[str]:
    """Parse 'TYPE: question?' lines, stripping the type label from the embedded
    question (the authoritative type comes from the classifier pass)."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _LABELED.match(line)
        cand = (m.group(2) if m else line).strip()
        if "?" not in cand:
            continue
        cand = cand[: cand.rindex("?") + 1].strip()
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def parse_keywords(text: str, n: int) -> List[str]:
    """Parse short keyword queries (2-6 words, no question form)."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _NUM_LINE.match(line)
        cand = (m.group(1) if m else line).strip().strip('."')
        cand = cand.rstrip("?").strip()
        if not cand:
            continue
        if not (2 <= len(cand.split()) <= 6):
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def parse_qa(text: str, n: int) -> List[str]:
    """Parse 'Q: ..? A: ..' pairs into single concatenated strings."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        s = _NUM_LINE.match(line)
        cand = (s.group(1) if s else line).strip()
        if "q:" not in cand.lower() or "a:" not in cand.lower():
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #
_NAIVE_SYSTEM = (
    "You write questions for a search system over news articles. "
    "Output ONLY a numbered list of questions, one per line, no preamble, "
    "no answers."
)


def _naive_user(chunk_text: str, n: int) -> str:
    # B1 -- the industry-default strategy: no constraints beyond count.
    return (
        f"Generate {n} questions this passage answers.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 1: semantic type (Cao & Wang ontology) ---- #
_TYPED_SYSTEM = (
    "You generate retrieval questions for a news RAG system, covering specified "
    "SEMANTIC TYPES. Every question must be specific and answerable from the "
    "passage alone. Output ONLY 'TYPE: question?' lines, no preamble, no answers."
)


def _typed_user(chunk_text: str, n: int) -> str:
    import mhrag_ontology as O

    alloc = O.load_allocation()
    slots = [t for t, cnt in alloc.items() for _ in range(cnt)]
    template = "\n".join(
        f"{i + 1}. {t}: a {O.ONTOLOGY[t]['definition']} question "
        f'(like "{O.ONTOLOGY[t]["exemplar"]}")'
        for i, t in enumerate(slots)
    )
    return (
        f"Fill EACH of these {n} numbered slots with a DISTINCT question that this "
        f"passage answers, of the stated semantic type:\n{template}\n\n"
        "Every question must be fully answerable using ONLY the passage; do not "
        "invent facts; make repeated-type slots genuinely different questions. "
        f"Output exactly {n} lines as 'N. TYPE: question?' (keep number + TYPE), "
        "nothing else.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 2: scope (local / summary / mixed) ---- #
def _local_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions about this passage. EACH must be answerable from a "
        "SINGLE sentence of the passage, and together they should cover as many "
        "DIFFERENT sentences as possible — each targets one specific stated fact. "
        "Do not ask questions that need combining sentences.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _summary_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions about this passage. EACH must require reading and "
        "COMBINING information from the whole passage — none answerable from a "
        "single sentence alone. Ask about the passage's overall point, synthesis, "
        "or how its parts relate.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _mixed_user(chunk_text: str, n: int) -> str:
    local = max(1, round(n * 0.8))
    summ = n - local
    return (
        f"Generate {n} questions about this passage: the first {local} each "
        "answerable from a SINGLE sentence (different sentences, one fact each), "
        f"then {summ} that require the WHOLE passage (synthesis across sentences). "
        "Output as one numbered list.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 3: explicitness ---- #
def _explicit_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions this passage answers where the answer is LITERALLY "
        "STATED in the passage (extractive). Use the passage's own wording.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _explicit_implicit_user(chunk_text: str, n: int) -> str:
    half = n // 2
    return (
        f"Generate {n} questions this passage answers, in two groups.\n"
        f"First {half}: EXPLICIT — the answer is literally stated in the passage.\n"
        f"Next {n - half}: IMPLICIT — the answer is inferable but you must NOT reuse "
        "content words from the passage; paraphrase heavily and ask as a person who "
        "has NOT read it would, using different vocabulary from the text.\n"
        "Output all as one numbered list of questions.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 4: surface form (keywords, Q&A pairs) ---- #
_KEYWORD_SYSTEM = (
    "You write short keyword search queries (like web-search queries) for a news "
    "retrieval system. Output ONLY a numbered list, one 2-to-5-word keyword phrase "
    "per line, no question marks, no sentences, no preamble."
)


def _keyword_user(chunk_text: str, n: int) -> str:
    return (
        f"Write {n} short keyword search queries (2-5 words each, no question form) "
        "that should retrieve this passage.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


_QA_SYSTEM = (
    "You write question-answer pairs for a news retrieval system. Output ONLY a "
    "numbered list; each line is exactly 'Q: <question>? A: <short answer>' with the "
    "answer taken from the passage. No preamble."
)


def _qa_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} question-answer pairs this passage answers. Each line: "
        "'Q: <question>? A: <short answer from the passage>'.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 5: style match (few-shot) ---- #
_FEWSHOT_SYSTEM = (
    "You write questions for a search system over news articles, imitating the "
    "STYLE of the example queries provided (their length, phrasing, and "
    "specificity). Output ONLY a numbered list of questions, no preamble."
)


def _fewshot_user(chunk_text: str, n: int) -> str:
    import mhrag_style as S

    examples = S.load_exemplars()
    ex_block = "\n".join(f"- {q}" for q in examples)
    return (
        f"Here are {len(examples)} example queries from the target distribution — "
        f"imitate their style, length, and specificity:\n{ex_block}\n\n"
        f"Now generate {n} questions THIS passage answers, written in that same "
        "style.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# name -> (system, user_template(chunk_text, n) -> str, parser(text, n) -> [str])
PROMPTS: Dict[
    str, Tuple[str, Callable[[str, int], str], Callable[[str, int], List[str]]]
] = {
    "naive": (_NAIVE_SYSTEM, _naive_user, parse_questions),
    "typed": (_TYPED_SYSTEM, _typed_user, parse_labeled_questions),
    "local": (_NAIVE_SYSTEM, _local_user, parse_questions),
    "summary": (_NAIVE_SYSTEM, _summary_user, parse_questions),
    "mixed": (_NAIVE_SYSTEM, _mixed_user, parse_questions),
    "explicit": (_NAIVE_SYSTEM, _explicit_user, parse_questions),
    "explicit_implicit": (_NAIVE_SYSTEM, _explicit_implicit_user, parse_questions),
    "keyword": (_KEYWORD_SYSTEM, _keyword_user, parse_keywords),
    "qa": (_QA_SYSTEM, _qa_user, parse_qa),
    "fewshot": (_FEWSHOT_SYSTEM, _fewshot_user, parse_questions),
}


def build_messages(prompt_name: str, chunk_text: str, n: int) -> List[Dict[str, str]]:
    system, user_fn, _ = PROMPTS[prompt_name]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_fn(chunk_text, n)},
    ]


def get_parser(prompt_name: str) -> Callable[[str, int], List[str]]:
    return PROMPTS[prompt_name][2]


# --------------------------------------------------------------------------- #
# Cache IO
# --------------------------------------------------------------------------- #
def _load_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[row["chunk_id"]] = row
    return cache


def load_questions(arm: str) -> Dict[str, List[str]]:
    """chunk_id -> list of generated questions for an arm (from cache)."""
    path = C.gen_cache_path(arm)
    out: Dict[str, List[str]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = row["questions"]
    return out


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class GenStats:
    arm: str
    prompt_name: str
    chunks_total: int
    chunks_newly_generated: int
    questions_total: int
    avg_questions_per_chunk: float
    parse_failures: int  # chunks that still fell short after 1 retry
    failure_rate: float
    retried: int
    prompt_tokens: int
    completion_tokens: int
    gen_seconds_sum: float
    wall_seconds: float
    estimated_cost_usd: float


def _estimate_cost(pt: int, ct: int) -> float:
    price = C.PRICE_PER_1M.get(C.LLM_MODEL)
    if not price:
        return -1.0
    return round(pt / 1e6 * price["input"] + ct / 1e6 * price["output"], 4)


def generate_for_arm(
    arm_name: str,
    chunks: List[dict],
    *,
    n: int = C.QUESTION_BUDGET,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> GenStats:
    """Generate exactly ``n`` questions per chunk for an arm (cached, resumable).

    A generation that parses to < n questions is retried once; if it still falls
    short it is logged as a parse failure and whatever parsed is kept."""
    arm = C.ARMS[arm_name]
    if arm.kind != "enrichment":
        raise ValueError(
            f"Arm {arm_name} is not an enrichment arm; nothing to generate."
        )
    prompt_name = arm.prompt
    parser = get_parser(prompt_name)
    path = C.gen_cache_path(arm_name)
    client = base_config.get_openai_client()

    cache = _load_cache(path)
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if limit:
        todo = todo[:limit]

    def _call(chunk_text: str) -> Tuple[str, float, int, int]:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=C.LLM_MODEL,
            messages=build_messages(prompt_name, chunk_text, n),
            temperature=C.LLM_TEMPERATURE,
            max_tokens=min(4000, max(256, n * 60)),
        )
        secs = time.perf_counter() - t0
        u = resp.usage
        return (
            resp.choices[0].message.content or "",
            secs,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
        )

    def work(c: dict) -> dict:
        raw, secs, pt, ct = _call(c["text"])
        qs = parser(raw, n)
        retries = 0
        if len(qs) < n:  # retry once
            retries = 1
            raw2, secs2, pt2, ct2 = _call(c["text"])
            secs += secs2
            pt += pt2
            ct += ct2
            qs2 = parser(raw2, n)
            if len(qs2) > len(qs):
                qs, raw = qs2, raw2
        return {
            "chunk_id": c["chunk_id"],
            "arm": arm_name,
            "prompt_name": prompt_name,
            "n_requested": n,
            "n_generated": len(qs),
            "questions": qs,
            "raw": raw,
            "retries": retries,
            "ok": len(qs) >= n,
            "gen_seconds": round(secs, 3),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }

    t_wall = time.perf_counter()
    pt_sum = ct_sum = gen_secs = retried = 0
    fh = path.open("a", encoding="utf-8")
    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(work, c): c for c in todo}
            for fut in as_completed(futures):
                row = fut.result()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[row["chunk_id"]] = row
                pt_sum += row["prompt_tokens"]
                ct_sum += row["completion_tokens"]
                gen_secs += row["gen_seconds"]
                retried += row["retries"]
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"  [gen:{arm_name}] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()

    rows = [cache[c["chunk_id"]] for c in chunks if c["chunk_id"] in cache]
    n_gen = [r["n_generated"] for r in rows]
    failures = sum(1 for r in rows if not r["ok"])
    return GenStats(
        arm=arm_name,
        prompt_name=prompt_name,
        chunks_total=len(chunks),
        chunks_newly_generated=len(todo),
        questions_total=sum(n_gen),
        avg_questions_per_chunk=round(sum(n_gen) / max(len(rows), 1), 2),
        parse_failures=failures,
        failure_rate=round(failures / max(len(rows), 1), 4),
        retried=retried,
        prompt_tokens=pt_sum,
        completion_tokens=ct_sum,
        gen_seconds_sum=round(gen_secs, 1),
        wall_seconds=round(time.perf_counter() - t_wall, 1),
        estimated_cost_usd=_estimate_cost(pt_sum, ct_sum),
    )
