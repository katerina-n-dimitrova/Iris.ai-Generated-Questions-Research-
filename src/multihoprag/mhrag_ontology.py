"""
Cao & Wang (2021) semantic question-type ontology + an LLM type classifier.

Used by Experiment 1 in two ways:
  1. Generation: the type-stratified arm (E1) must cover distinct types from this
     ontology (JUDGMENTAL skipped -- opinion questions don't suit a factual news
     corpus), with the type name + a one-line definition + one exemplar in the
     prompt. Slot distribution (per the spec): one question per type + one extra
     COMPARISON (MultiHop-RAG is heavy on comparison / temporal reasoning).
  2. Analysis: the SAME ontology is applied by an LLM classifier (temperature 0,
     cached) to label BOTH the real MultiHop-RAG queries and the generated
     questions, to produce the Experiment-1 cross-tab.

Everything the classifier produces is cached to disk so re-runs are free.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import mhrag_config as C
import config as base_config

# type -> (one-line definition, one exemplar) — exemplars written for news.
ONTOLOGY: Dict[str, Dict[str, str]] = {
    "VERIFICATION": {
        "definition": "is X true? (yes/no confirmation of a fact)",
        "exemplar": "Did the company confirm the acquisition?",
    },
    "DISJUNCTIVE": {
        "definition": "which option holds? (choose between alternatives)",
        "exemplar": "Was the decision made by the board or the regulator?",
    },
    "CONCEPT": {
        "definition": "definition or identity of something",
        "exemplar": "What is the new policy called?",
    },
    "EXTENT": {
        "definition": "quantity, size or degree",
        "exemplar": "How much funding was raised?",
    },
    "EXAMPLE": {
        "definition": "instances or members of a set",
        "exemplar": "Which companies were affected by the outage?",
    },
    "COMPARISON": {
        "definition": "compare two or more things",
        "exemplar": "How does this quarter's revenue compare to last year's?",
    },
    "CAUSE": {
        "definition": "the reason for something",
        "exemplar": "Why did the CEO step down?",
    },
    "CONSEQUENCE": {
        "definition": "the result or effect of something",
        "exemplar": "What happened after the ruling was announced?",
    },
    "PROCEDURAL": {
        "definition": "the method or steps for doing something",
        "exemplar": "How was the vulnerability discovered?",
    },
}
TYPES: List[str] = list(ONTOLOGY.keys())
_TYPE_SET = set(TYPES)


def ontology_block() -> str:
    lines = []
    for t in TYPES:
        o = ONTOLOGY[t]
        lines.append(f'- {t}: {o["definition"]}. Example: "{o["exemplar"]}"')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM classifier (temperature 0, cached)
# --------------------------------------------------------------------------- #
_CLASSIFIER_SYSTEM = (
    "You label questions by semantic type. Choose EXACTLY ONE type per question "
    "from this fixed ontology:\n"
    + ontology_block()
    + "\nIf a question fits several, pick the single best. Reply with one type name "
    "per line, in the same order as the numbered questions, and nothing else."
)


def _classify_batch(questions: List[str]) -> List[str]:
    if not questions:
        return []
    client = base_config.get_openai_client()
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    resp = client.chat.completions.create(
        model=C.LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": numbered},
        ],
        max_tokens=len(questions) * 8 + 32,
    )
    text = resp.choices[0].message.content or ""
    labels: List[str] = []
    for line in text.splitlines():
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip().upper()
        s = re.sub(r"[^A-Z]", "", s)
        labels.append(s if s in _TYPE_SET else "")
    labels = (labels + ["CONCEPT"] * len(questions))[: len(questions)]
    return [l if l in _TYPE_SET else "CONCEPT" for l in labels]


def _query_type_path():
    return C.PROCESSED_DIR / "query_cw_types.json"


def classify_queries(queries: List[dict], force: bool = False) -> Dict[str, str]:
    """query_id -> Cao&Wang semantic type (cached)."""
    path = _query_type_path()
    if path.exists() and not force:
        cached = json.load(path.open())
        if all(q["query_id"] in cached for q in queries):
            return cached
    labels = _classify_batch([q["question"] for q in queries])
    out = {q["query_id"]: t for q, t in zip(queries, labels)}
    json.dump(out, path.open("w"), indent=2)
    return out


def _arm_type_path(arm: str):
    return C.GEN_DIR / f"question_types_{arm}.json"


def classify_arm_questions(
    arm: str,
    questions_by_chunk: Dict[str, List[str]],
    *,
    max_workers: int = C.LLM_MAX_WORKERS,
    force: bool = False,
) -> Dict[str, List[str]]:
    """chunk_id -> list of per-question types for an arm (cached, one call/chunk)."""
    path = _arm_type_path(arm)
    cache: Dict[str, List[str]] = {}
    if path.exists() and not force:
        cache = json.load(path.open())
    todo = [cid for cid in questions_by_chunk if cid not in cache]
    if todo:

        def work(cid):
            return cid, _classify_batch(questions_by_chunk[cid])

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, cid): cid for cid in todo}
            for fut in as_completed(futs):
                cid, labels = fut.result()
                cache[cid] = labels
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"  [classify:{arm}] {done}/{len(todo)}", flush=True)
        json.dump(cache, path.open("w"), indent=2)
    return cache


# --------------------------------------------------------------------------- #
# Slot allocation for the type-stratified arm (E1)
# --------------------------------------------------------------------------- #
def allocate_slots(budget: int = C.QUESTION_BUDGET) -> Dict[str, int]:
    """Per the spec: one question per type + one extra COMPARISON. With 9 types
    and a budget of 10 that gives COMPARISON=2, all others=1. Extra budget beyond
    10 adds further COMPARISON/EXTENT slots (temporal reasoning leans on both)."""
    alloc = {t: 1 for t in TYPES}
    remaining = budget - len(TYPES)
    boosters = ["COMPARISON", "EXTENT", "CONSEQUENCE"]
    i = 0
    while remaining > 0:
        alloc[boosters[i % len(boosters)]] += 1
        remaining -= 1
        i += 1
    # if budget < #types, keep the first `budget` types
    if budget < len(TYPES):
        keep = TYPES[:budget]
        alloc = {t: (1 if t in keep else 0) for t in TYPES}
    return {t: alloc[t] for t in TYPES}


def _allocation_path():
    return C.PROCESSED_DIR / "e1_allocation.json"


def save_allocation(alloc: Dict[str, int]) -> None:
    json.dump(alloc, _allocation_path().open("w"), indent=2)


def load_allocation() -> Dict[str, int]:
    path = _allocation_path()
    if path.exists():
        return json.load(path.open())
    return allocate_slots()
