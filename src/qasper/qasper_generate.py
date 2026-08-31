"""
Synthetic question generation (doc2query-style), arm-aware and cached.

One LLM + temperature is fixed for every arm (config); only the PROMPT changes.
Each arm's questions are cached to its own resumable JSONL so re-runs never
regenerate. Parsing that yields the wrong count / non-questions is retried once,
then logged and kept as-is; the per-arm failure rate is reported.

Prompts registry
----------------
``PROMPTS[name] -> (system_str, user_template_fn)``. This deliverable ships only
the ``naive`` prompt (baseline B1). Experiments 1-5 add more prompts here and a
matching Arm in qasper_config -- indexing/retrieval/eval never change.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import qasper_config as C
import config as base_config

_NUM_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+?)\s*$")
# "TYPE: question?" (optionally numbered), label stripped for the embedded text.
_LABELED = re.compile(r"^\s*(?:\d+[\.\)]\s*)?([A-Za-z]+)\s*[:\-]\s*(.+)$")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_questions(text: str, n: int) -> List[str]:
    """Extract up to n distinct question strings from a numbered list."""
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


def parse_keywords(text: str, n: int) -> List[str]:
    """Parse short keyword phrases (2-5 words, no '?' requirement)."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _NUM_LINE.match(line)
        cand = (m.group(1) if m else line).strip().rstrip(".").strip()
        cand = cand.strip('"').strip()
        if not cand:
            continue
        wc = len(cand.split())
        if wc < 1 or wc > 6 or "?" in cand:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= n:
            break
    return out


_QA = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?Q\s*[:\-]\s*(.+?)\s*A\s*[:\-]\s*(.+)$", re.IGNORECASE
)


def parse_qa_pairs(text: str, n: int) -> List[str]:
    """Parse 'Q: ...? A: ...' lines into concatenated 'Q: q A: a' strings."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _QA.match(line)
        if not m:
            continue
        q, a = m.group(1).strip(), m.group(2).strip()
        if not q or not a:
            continue
        combined = f"Q: {q} A: {a}"
        key = combined.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(combined)
        if len(out) >= n:
            break
    return out


def parse_labeled_questions(text: str, n: int) -> List[str]:
    """Parse 'TYPE: question?' lines, stripping the type label from the question
    that gets embedded (the authoritative type comes from the classifier pass)."""
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


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #
def _naive_user(chunk_text: str, n: int) -> str:
    # B1 -- the industry-default strategy: no constraints beyond count.
    return (
        f"Generate {n} questions this passage answers.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


_NAIVE_SYSTEM = (
    "You write questions for a search system over scientific papers. "
    "Output ONLY a numbered list of questions, one per line, no preamble, "
    "no answers."
)


def _typed_user(chunk_text: str, n: int) -> str:
    # E1 -- type-stratified generation over the Cao & Wang ontology.
    # The allocation is expanded into n explicit numbered slots (a doubled type
    # appears as two slots) so the model reliably returns exactly n DISTINCT
    # questions rather than collapsing a doubled type into one.
    import qasper_ontology as O

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
        f"Output exactly {n} lines in the form 'N. TYPE: question?' (keep the "
        "number and the TYPE label), nothing else.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


_TYPED_SYSTEM = (
    "You generate retrieval questions for a scientific-paper RAG system, covering "
    "specified SEMANTIC TYPES. Every question must be specific and answerable from "
    "the passage alone. Output ONLY 'TYPE: question?' lines, no preamble, no answers."
)


# ---- Experiment 2: scope (local / summary / mixed) ---- #
def _local_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions about this passage. EACH question must be "
        "answerable from a SINGLE sentence of the passage, and together they should "
        "cover as many DIFFERENT sentences as possible — each question targets one "
        "specific stated fact. Do not ask questions that need combining sentences.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _summary_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions about this passage. EACH question must require "
        "reading and COMBINING information from the whole passage to answer — none "
        "should be answerable from a single sentence alone. Ask about the passage's "
        "overall point, synthesis, trade-offs, or how its parts relate.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _mixed_user(chunk_text: str, n: int) -> str:
    local = max(1, round(n * 0.8))
    summ = n - local
    return (
        f"Generate {n} questions about this passage: the first {local} each "
        "answerable from a SINGLE sentence (covering different sentences, one fact "
        f"each), then {summ} that require the WHOLE passage to answer (synthesis "
        "across sentences). Output all as one numbered list.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


# ---- Experiment 3: explicitness (explicit vs explicit+implicit) ---- #
def load_style_exemplars(n_exemplars: int = 8) -> List[str]:
    """8 REAL QASPER questions from dev papers OUTSIDE the selected set, seeded and
    cached, for Experiment 5's few-shot style prompt. Asserts no leakage (no
    exemplar comes from a selected paper)."""
    import json as _json
    import random as _random

    path = C.PROCESSED_DIR / "e5_exemplars.json"
    if path.exists():
        return _json.load(path.open())["questions"]
    import qasper_data as D

    dev = D.load_dev()
    selected = set(_json.load(C.SELECTED_PAPERS_PATH.open())["paper_ids"])
    candidates = []
    for pid, p in dev.items():
        if pid in selected:
            continue
        for qa in p.get("qas", []):
            q = (qa.get("question") or "").strip()
            if q:
                candidates.append((pid, q))
    _random.Random(C.SELECTION_SEED).shuffle(candidates)
    chosen = candidates[:n_exemplars]
    for pid, _q in chosen:  # no-leakage assertion (in code)
        assert pid not in selected, f"LEAKAGE: exemplar from selected paper {pid}"
    questions = [q for _pid, q in chosen]
    _json.dump(
        {
            "questions": questions,
            "source_papers": [pid for pid, _ in chosen],
            "seed": C.SELECTION_SEED,
            "selected_excluded": sorted(selected),
        },
        path.open("w"),
        indent=2,
    )
    return questions


def _fewshot_user(chunk_text: str, n: int) -> str:
    exemplars = load_style_exemplars()
    ex_block = "\n".join(f"- {q}" for q in exemplars)
    return (
        "Here are example questions written by readers of NLP papers (the target "
        f"style):\n{ex_block}\n\n"
        f"Now generate {n} questions this passage answers, imitating the STYLE, "
        "length, and specificity of the examples above (they were written by people "
        "who saw only the title and abstract, so keep them high-level and natural).\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _explicit_user(chunk_text: str, n: int) -> str:
    return (
        f"Generate {n} questions this passage answers, where the answer to each is "
        "LITERALLY stated in the passage (you could point to the exact words that "
        "answer it). Stay close to the passage's wording.\n\n"
        f'Passage:\n"""\n{chunk_text.strip()}\n"""'
    )


def _implicit_mix_user(chunk_text: str, n: int) -> str:
    exp = n // 2
    imp = n - exp
    return (
        f"Generate {n} questions this passage answers.\n"
        f"- The first {exp} are EXPLICIT: the answer is literally stated in the passage.\n"
        f"- The last {imp} are IMPLICIT: the answer is inferable from the passage but "
        "you must NOT reuse the passage's content words. Paraphrase heavily, use "
        "synonyms and general phrasing, and write each as a person who has NOT read "
        "the passage would ask it.\n\n"
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
    "implicit_mix": (_NAIVE_SYSTEM, _implicit_mix_user, parse_questions),
    "fewshot": (_NAIVE_SYSTEM, _fewshot_user, parse_questions),
    "keywords": (
        "You produce short keyword search queries for a scientific-paper search "
        "engine. Output ONLY a numbered list, one 2-5 word keyword phrase per line, "
        "no question marks, no sentences.",
        lambda t, n: (
            f"Generate {n} short keyword queries (2-5 words each, no "
            "question form) that a searcher might type to find this "
            f'passage.\n\nPassage:\n"""\n{t.strip()}\n"""'
        ),
        parse_keywords,
    ),
    "qa_pairs": (
        "You generate question-answer pairs for scientific-paper retrieval. Output "
        "ONLY lines of the form 'Q: <question>? A: <answer>', one pair per line.",
        lambda t, n: (
            f"Generate {n} question-answer pairs this passage supports. "
            "Each answer must come directly from the passage.\n\n"
            f'Passage:\n"""\n{t.strip()}\n"""'
        ),
        parse_qa_pairs,
    ),
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
    short it is logged as a parse failure and whatever parsed is kept.
    """
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
            max_tokens=min(4000, max(256, n * 30)),
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
