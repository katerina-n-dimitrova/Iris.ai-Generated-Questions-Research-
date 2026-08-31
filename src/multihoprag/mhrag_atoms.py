"""
Experiment 6 — Atomic Units (Raina & Gales 2024, arXiv:2405.12363).

Their pipeline: (1) decompose a chunk into stand-alone atomic facts; (2) generate
closed-answer synthetic questions over each atom, using the whole chunk as
context; (3) dense-retrieve on the atom questions, a hit routing to its parent
chunk (max-over-vectors scoring — already how our dense index works). This gives
two ablation axes for free:
  * chunk-level vs atom-level  (chunk cells are supplied by B0 / B1)
  * statement (atom) vs question representations

Two arms are produced here, both written to the standard generated-questions
cache format so indexing/retrieval/eval treat them like any other arm:
  * E6as : atom STATEMENTS  — the atoms themselves are the vectors.
  * E6aq : atom QUESTIONS   — closed-answer questions generated over the atoms.

Budget note: to stay comparable with the other arms (fixed factor = 10 vectors /
chunk of generated content) the atom-question arm is capped at QUESTION_BUDGET
questions per chunk, distributed across the atoms; the atom-statement arm embeds
all atoms of a chunk (a variable count -- the representation being tested). The
original paper generates up to 15 questions per atom; we hold the budget fixed
for a controlled comparison and note this in the analysis.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import mhrag_config as C
import config as base_config
from mhrag_generate import parse_questions, GenStats, _estimate_cost

MAX_ATOMS = int(__import__("os").getenv("MHRAG_MAX_ATOMS", "12"))

_DECOMP_SYSTEM = (
    "You decompose a news paragraph into stand-alone atomic facts. Each fact is a "
    "single self-contained statement that makes sense without the paragraph. "
    "Output ONLY the facts, one per line, no numbering, no preamble."
)

_ATOMQ_SYSTEM = (
    "You write closed-answer retrieval questions for a news RAG system. Each "
    "question has a specific answer contained in the passage. Output ONLY a "
    "numbered list of questions, one per line, no answers, no preamble."
)


def _bullet_strip(line: str) -> str:
    return re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", line).strip()


def parse_atoms(text: str, n: int) -> List[str]:
    out, seen = [], set()
    for line in (text or "").splitlines():
        cand = _bullet_strip(line).strip()
        if len(cand.split()) < 3:
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
# Shared cached generation loop
# --------------------------------------------------------------------------- #
def _load_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[row["chunk_id"]] = row
    return cache


def _run_cached(
    arm_name: str,
    chunks: List[dict],
    work_fn: Callable[[dict], dict],
    *,
    max_workers: int = C.LLM_MAX_WORKERS,
) -> GenStats:
    path = C.gen_cache_path(arm_name)
    cache = _load_cache(path)
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    t_wall = time.perf_counter()
    pt_sum = ct_sum = gen_secs = retried = 0
    fh = path.open("a", encoding="utf-8")
    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(work_fn, c): c for c in todo}
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
        prompt_name=rows[0]["prompt_name"] if rows else "-",
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


def _call(messages) -> Tuple[str, float, int, int]:
    client = base_config.get_openai_client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=C.LLM_MODEL,
        messages=messages,
        temperature=C.LLM_TEMPERATURE,
        max_tokens=700,
    )
    secs = time.perf_counter() - t0
    u = resp.usage
    return (
        resp.choices[0].message.content or "",
        secs,
        getattr(u, "prompt_tokens", 0) or 0,
        getattr(u, "completion_tokens", 0) or 0,
    )


# --------------------------------------------------------------------------- #
# E6as — atom statements (decompose chunk -> atoms)
# --------------------------------------------------------------------------- #
def decompose_for_arm(arm_name: str = "E6as", chunks: List[dict] = None) -> GenStats:
    def work(c: dict) -> dict:
        msgs = [
            {"role": "system", "content": _DECOMP_SYSTEM},
            {
                "role": "user",
                "content": "Break down the following paragraph into stand-alone atomic facts. "
                f'Return each fact on a new line.\n\n"""\n{c["text"].strip()}\n"""',
            },
        ]
        raw, secs, pt, ct = _call(msgs)
        atoms = parse_atoms(raw, MAX_ATOMS)
        retries = 0
        if not atoms:
            retries = 1
            raw2, s2, p2, ct2 = _call(msgs)
            secs += s2
            pt += p2
            ct += ct2
            atoms = parse_atoms(raw2, MAX_ATOMS) or [c["text"].strip()]
            raw = raw2
        return {
            "chunk_id": c["chunk_id"],
            "arm": arm_name,
            "prompt_name": "atoms",
            "n_requested": MAX_ATOMS,
            "n_generated": len(atoms),
            "questions": atoms,
            "raw": raw,
            "retries": retries,
            "ok": len(atoms) >= 1,
            "gen_seconds": round(secs, 3),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }

    return _run_cached(arm_name, chunks, work)


# --------------------------------------------------------------------------- #
# E6aq — atom questions (questions generated over the atoms, chunk as context)
# --------------------------------------------------------------------------- #
def gen_atom_questions_for_arm(
    arm_name: str = "E6aq",
    chunks: List[dict] = None,
    atoms_arm: str = "E6as",
    n: int = C.QUESTION_BUDGET,
) -> GenStats:
    """Requires the atoms cache (``atoms_arm``) to exist. One call per chunk asks
    for exactly ``n`` closed-answer questions distributed across the chunk's atoms
    (chunk given as context), keeping the fixed 10-question/chunk budget."""
    atoms_cache = _load_cache(C.gen_cache_path(atoms_arm))
    if not atoms_cache:
        raise RuntimeError(f"Atom cache for {atoms_arm} missing; run decompose first.")

    def work(c: dict) -> dict:
        atoms = atoms_cache.get(c["chunk_id"], {}).get("questions", [])
        atom_block = "\n".join(f"- {a}" for a in atoms) or "- (no atoms)"
        user = (
            f'Passage (context):\n"""\n{c["text"].strip()}\n"""\n\n'
            f"Atomic facts from the passage:\n{atom_block}\n\n"
            f"Generate exactly {n} closed-answer questions, each answerable from ONE "
            "of the atomic facts above (the answer must be present in that fact). "
            "Spread the questions across DIFFERENT atoms. Output a numbered list of "
            "questions only."
        )
        msgs = [
            {"role": "system", "content": _ATOMQ_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw, secs, pt, ct = _call(msgs)
        qs = parse_questions(raw, n)
        retries = 0
        if len(qs) < n:
            retries = 1
            raw2, s2, p2, ct2 = _call(msgs)
            secs += s2
            pt += p2
            ct += ct2
            qs2 = parse_questions(raw2, n)
            if len(qs2) > len(qs):
                qs, raw = qs2, raw2
        return {
            "chunk_id": c["chunk_id"],
            "arm": arm_name,
            "prompt_name": "atom_questions",
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

    return _run_cached(arm_name, chunks, work)
