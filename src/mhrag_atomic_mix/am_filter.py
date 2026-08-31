"""
Stage: validate + filter Condition-E questions (§15).

Sequential gates (a question must pass all to be accepted):
  15.1 structural      — non-empty question, parent id, >=1 supporting span, short answer.
  15.2 grounding       — a supporting span appears in the parent chunk (normalized).
  15.3 self-contained  — reject vague references (this company / the device / the report / ...)
                         when no entity is named.
  round-trip embed     — embed survivors; query the baseline chunk index.
  15.5 round-trip      — parent chunk must rank within parent_retrieval_top_k.
  15.6 confusion margin— margin = sim(parent) - max sim(non-parent) must be >= min.
  15.4 near-duplicate  — within a chunk, drop cosine>=threshold duplicates (keep higher margin).
  15.7 coverage-aware  — greedy pick covering distinct atoms; add 2-3 chunk-level; cap 10/chunk.

Needs the baseline chunk index (built here if absent). Never sees benchmark labels.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

import am_config as C
import am_data as D
import am_index as IDX
from embeddings import get_embedder
from vo_data import _norm

_VAGUE = re.compile(
    r"\b(this company|the team|the device|the report|the product|"
    r"the firm|this device|the organization|the group)\b",
    re.IGNORECASE,
)


def _grounded(q: dict, chunk_text: str) -> bool:
    ctext = _norm(chunk_text)
    for span in q.get("supporting_spans", []):
        ns = _norm(span)
        if ns and (ns in ctext or _partial(ns, ctext)):
            return True
    return False


def _partial(span: str, ctext: str) -> bool:
    words = span.split()
    if len(words) < 4:
        return False
    return " ".join(words[:6]) in ctext or " ".join(words[-6:]) in ctext


def _self_contained(q: dict) -> bool:
    if _VAGUE.search(q["question"]) and not q.get("entities"):
        return False
    return True


def run_filter(force: bool = False) -> dict:
    if not force and C.QUESTIONS_FILTERED.exists():
        return json.load(open(C.FILTER_REPORT))

    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    raw = D.read_jsonl(C.QUESTIONS_RAW)
    rejected: List[dict] = []
    stage = defaultdict(int)
    t0 = time.perf_counter()

    def reject(q, reason):
        rejected.append(
            {
                "question_id": q["question_id"],
                "parent_chunk_id": q["parent_chunk_id"],
                "question": q["question"],
                "question_type": q["question_type"],
                "reason": reason,
            }
        )
        stage[reason] += 1

    # 15.1 structural + 15.2 grounding + 15.3 self-contained
    survivors: List[dict] = []
    for q in raw:
        if not q["question"].strip() or not q.get("parent_chunk_id"):
            reject(q, "structural")
            continue
        if not q.get("supporting_spans") or not q.get("short_answer"):
            reject(q, "structural_missing_span_or_answer")
            continue
        ctext = chunks.get(q["parent_chunk_id"], {}).get("text", "")
        if not _grounded(q, ctext):
            reject(q, "grounding_span_absent")
            continue
        if not _self_contained(q):
            reject(q, "not_self_contained")
            continue
        survivors.append(q)
    stage["passed_structural_grounding_selfcontained"] = len(survivors)

    # ensure baseline index exists (round-trip / margin need it)
    try:
        coll = C.get_collection(C.BASELINE_COLLECTION)
        if coll.count() == 0:
            raise ValueError("empty")
    except Exception:
        IDX.build_baseline_index(list(chunks.values()))
        coll = C.get_collection(C.BASELINE_COLLECTION)
    n_chunks = coll.count()

    # embed survivors once (normalized -> dot = cosine)
    embedder = get_embedder()
    embs = (
        embedder.embed_documents([q["question"] for q in survivors])
        if survivors
        else []
    )
    emb_of = {}
    for q, v in zip(survivors, embs):
        emb_of[q["question_id"]] = np.array(v)

    # round-trip + confusion margin (query full ranking so parent sim is exact)
    strict_rank1 = 0
    passed_rt: List[dict] = []
    for q, v in zip(survivors, embs):
        res = coll.query(
            query_embeddings=[v], n_results=n_chunks, include=["metadatas", "distances"]
        )
        ids = [m["parent_chunk_id"] for m in res["metadatas"][0]]
        sims = [1.0 - d for d in res["distances"][0]]
        parent = q["parent_chunk_id"]
        try:
            prank = ids.index(parent) + 1
            psim = sims[ids.index(parent)]
        except ValueError:
            prank, psim = 10**9, 0.0
        nonparent = [s for i, s in zip(ids, sims) if i != parent]
        best_np = max(nonparent) if nonparent else 0.0
        margin = psim - best_np
        q["_parent_rank"] = prank
        q["_parent_sim"] = round(psim, 5)
        q["_margin"] = round(margin, 5)
        if prank == 1:
            strict_rank1 += 1
        if prank > C.PARENT_TOPK:
            reject(q, "roundtrip_parent_below_topk")
            continue
        if C.REQUIRE_POS_MARGIN and margin < C.MIN_MARGIN:
            reject(q, "confusion_margin_too_low")
            continue
        passed_rt.append(q)
    stage["passed_roundtrip_margin"] = len(passed_rt)
    stage["would_pass_strict_rank1"] = strict_rank1

    # group by chunk for near-dup + coverage
    by_chunk: Dict[str, List[dict]] = defaultdict(list)
    for q in passed_rt:
        by_chunk[q["parent_chunk_id"]].append(q)

    accepted: List[dict] = []
    for cid, qs in by_chunk.items():
        # 15.4 near-duplicate removal (keep higher margin)
        qs_sorted = sorted(qs, key=lambda x: x["_margin"], reverse=True)
        kept: List[dict] = []
        for q in qs_sorted:
            dup = False
            for k in kept:
                sim = float(emb_of[q["question_id"]] @ emb_of[k["question_id"]])
                if sim >= C.NEAR_DUP_THRESHOLD:
                    dup = True
                    break
            if dup:
                reject(q, "near_duplicate")
            else:
                kept.append(q)

        # 15.7 coverage-aware greedy: atomic first, one per distinct atom, by margin
        atomic = [q for q in kept if q["question_type"] == "atomic"]
        chunk_lv = [q for q in kept if q["question_type"] == "chunk_level"]
        atomic.sort(key=lambda x: x["_margin"], reverse=True)
        chunk_lv.sort(key=lambda x: x["_margin"], reverse=True)

        chosen, covered_atoms = [], set()
        # pass 1: best question for each uncovered atom
        for q in atomic:
            if q["atom_id"] in covered_atoms:
                continue
            chosen.append(q)
            covered_atoms.add(q["atom_id"])
            if len(chosen) >= C.MAX_TOTAL_Q - C.DEFAULT_CHUNK_LEVEL_Q:
                break
        # pass 2: fill remaining atomic budget with alternate-view questions
        for q in atomic:
            if len(chosen) >= C.MAX_TOTAL_Q - C.DEFAULT_CHUNK_LEVEL_Q:
                break
            if q not in chosen:
                chosen.append(q)
        # add 2-3 chunk-level
        for q in chunk_lv[: C.MAX_CHUNK_LEVEL_Q]:
            if len(chosen) >= C.MAX_TOTAL_Q:
                break
            chosen.append(q)
        # anything kept but not chosen -> coverage-trimmed
        for q in kept:
            if q not in chosen:
                reject(q, "coverage_trimmed")

        for q in chosen:
            q["parent_chunk_text"] = chunks[cid]["text"]
            accepted.append(q)

    filt_seconds = round(time.perf_counter() - t0, 2)
    D._write_jsonl(C.QUESTIONS_FILTERED, accepted)
    D._write_jsonl(C.REJECTED, rejected)

    per_chunk = defaultdict(int)
    a_per_chunk = defaultdict(int)
    cl_per_chunk = defaultdict(int)
    for q in accepted:
        per_chunk[q["parent_chunk_id"]] += 1
        if q["question_type"] == "atomic":
            a_per_chunk[q["parent_chunk_id"]] += 1
        else:
            cl_per_chunk[q["parent_chunk_id"]] += 1
    n_chunks_total = len(chunks)
    n_atomic = sum(1 for q in accepted if q["question_type"] == "atomic")
    report = {
        "raw_questions": len(raw),
        "accepted_questions": len(accepted),
        "rejected_questions": len(rejected),
        "accepted_atomic": n_atomic,
        "accepted_chunk_level": len(accepted) - n_atomic,
        "avg_accepted_per_chunk": round(len(accepted) / n_chunks_total, 2),
        "avg_atomic_per_chunk": round(n_atomic / n_chunks_total, 2),
        "avg_chunk_level_per_chunk": round(
            (len(accepted) - n_atomic) / n_chunks_total, 2
        ),
        "chunks_with_zero_accepted": n_chunks_total - len(per_chunk),
        "rejections_by_reason": {
            k: v
            for k, v in sorted(stage.items())
            if k.startswith(
                (
                    "structural",
                    "grounding",
                    "not_",
                    "roundtrip",
                    "confusion",
                    "near",
                    "coverage",
                )
            )
        },
        "passed_structural_grounding_selfcontained": stage[
            "passed_structural_grounding_selfcontained"
        ],
        "passed_roundtrip_margin": stage["passed_roundtrip_margin"],
        "would_pass_strict_rank1_roundtrip": stage["would_pass_strict_rank1"],
        "filter_seconds": filt_seconds,
        "filter_config": {
            "parent_retrieval_top_k": C.PARENT_TOPK,
            "require_positive_margin": C.REQUIRE_POS_MARGIN,
            "min_confusion_margin": C.MIN_MARGIN,
            "near_duplicate_threshold": C.NEAR_DUP_THRESHOLD,
            "max_total_per_chunk": C.MAX_TOTAL_Q,
        },
    }
    json.dump(report, open(C.FILTER_REPORT, "w"), indent=2)
    print(
        f"[filter] raw {len(raw)} -> accepted {len(accepted)} "
        f"(atomic {n_atomic}/chunk-level {len(accepted) - n_atomic}); "
        f"{report['chunks_with_zero_accepted']} chunks empty; "
        f"strict-rank1 would keep {stage['would_pass_strict_rank1']}"
    )
    return report


if __name__ == "__main__":
    import pprint

    pprint.pp(run_filter(force=True))
