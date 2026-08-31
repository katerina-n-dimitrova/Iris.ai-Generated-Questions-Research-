"""
Cao & Wang (2021) semantic question-type ontology + an LLM type classifier.

Used by Experiment 1 in two ways:
  1. Generation: the type-stratified arm (E1) must cover distinct types from this
     ontology (JUDGMENTAL is skipped -- opinion questions don't suit a factual
     corpus), with the type name + a one-line definition + one exemplar in the prompt.
  2. Analysis: the SAME ontology is applied by an LLM classifier (temperature 0,
     cached) to label BOTH the real QASPER queries and the generated questions, to
     produce the Experiment-1 cross-tab.

Everything the classifier produces is cached to disk so re-runs are free.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import qasper_config as C
import config as base_config

# --------------------------------------------------------------------------- #
# The ontology (order fixed; JUDGMENTAL intentionally excluded)
# --------------------------------------------------------------------------- #
# type -> (one-line definition, one exemplar)
ONTOLOGY: Dict[str, Dict[str, str]] = {
    "VERIFICATION": {
        "definition": "is X true? (yes/no confirmation of a fact)",
        "exemplar": "Does the model use pretraining?",
    },
    "DISJUNCTIVE": {
        "definition": "which option holds? (choose between alternatives)",
        "exemplar": "Is the encoder frozen or fine-tuned?",
    },
    "CONCEPT": {
        "definition": "definition or identity of something",
        "exemplar": "What is the proposed method called?",
    },
    "EXTENT": {
        "definition": "quantity, size or degree",
        "exemplar": "How large is the training set?",
    },
    "EXAMPLE": {
        "definition": "instances or members of a set",
        "exemplar": "Which datasets are used for evaluation?",
    },
    "COMPARISON": {
        "definition": "compare two or more things",
        "exemplar": "How does the method differ from the baseline?",
    },
    "CAUSE": {
        "definition": "the reason for something",
        "exemplar": "Why does performance drop on long inputs?",
    },
    "CONSEQUENCE": {
        "definition": "the result or effect of something",
        "exemplar": "What is the effect of removing the graph component?",
    },
    "PROCEDURAL": {
        "definition": "the method or steps for doing something",
        "exemplar": "How are the embeddings trained?",
    },
}
TYPES: List[str] = list(ONTOLOGY.keys())
_TYPE_SET = set(TYPES)


def ontology_block() -> str:
    """The type list (name + definition + exemplar) for prompts."""
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
    """One LLM call classifying a list of questions; returns a type per question."""
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
    # pad/truncate to len(questions); default unmatched to CONCEPT (never crash)
    labels = (labels + ["CONCEPT"] * len(questions))[: len(questions)]
    return [l if l in _TYPE_SET else "CONCEPT" for l in labels]


# ---- queries -------------------------------------------------------------- #
def _query_type_path():
    return C.PROCESSED_DIR / "query_types.json"


def classify_queries(queries: List[dict], force: bool = False) -> Dict[str, str]:
    """query_id -> semantic type (cached)."""
    path = _query_type_path()
    if path.exists() and not force:
        cached = json.load(path.open())
        if all(q["query_id"] in cached for q in queries):  # cache covers this query set
            return cached
    labels = _classify_batch([q["question"] for q in queries])
    out = {q["query_id"]: t for q, t in zip(queries, labels)}
    json.dump(out, path.open("w"), indent=2)
    return out


# ---- generated questions (per arm) ---------------------------------------- #
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
# Slot allocation for the type-stratified arm
# --------------------------------------------------------------------------- #
def allocate_slots(
    query_type_dist: Dict[str, int], budget: int = C.QUESTION_BUDGET
) -> Dict[str, int]:
    """Allocate ``budget`` question slots across the 9 types.

    Spec: sample proportionally to the real-query type distribution if feasible,
    else one per type + one extra CONCEPT. With 10 slots and 9 types, full type
    coverage leaves ~1 free slot, so we give every type 1 (coverage the cross-tab
    needs) and place the remaining slot(s) on the most common query type(s),
    falling back to CONCEPT. Returns an ordered {type: count} summing to budget.
    """
    alloc = {t: 1 for t in TYPES}
    remaining = budget - len(TYPES)
    if remaining > 0:
        ranked = sorted(
            TYPES, key=lambda t: (-query_type_dist.get(t, 0), TYPES.index(t))
        )
        # tie / empty distribution -> CONCEPT first
        if not query_type_dist:
            ranked = ["CONCEPT"] + [t for t in TYPES if t != "CONCEPT"]
        for i in range(remaining):
            alloc[ranked[i % len(ranked)]] += 1
    elif remaining < 0:  # budget < 9: keep the most common |budget| types
        keep = sorted(
            TYPES, key=lambda t: (-query_type_dist.get(t, 0), TYPES.index(t))
        )[:budget]
        alloc = {t: (1 if t in keep else 0) for t in TYPES}
    return {t: alloc[t] for t in TYPES}


def _allocation_path():
    return C.PROCESSED_DIR / "e1_allocation.json"


def save_allocation(alloc: Dict[str, int]) -> None:
    json.dump(alloc, _allocation_path().open("w"), indent=2)


def load_allocation() -> Dict[str, int]:
    """Load the saved E1 slot allocation; fall back to the spec default
    (1/type + 1 extra CONCEPT) if generation runs before analysis."""
    path = _allocation_path()
    if path.exists():
        return json.load(path.open())
    return allocate_slots({})
