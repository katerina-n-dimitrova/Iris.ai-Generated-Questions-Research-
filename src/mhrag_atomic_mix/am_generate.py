"""
Stage: Condition-E generation (§9–§14).

For every chunk, ONE LLM call:
  1. decomposes the chunk into self-contained atomic facts (pronouns resolved via
     title/source/date + chunk context),
  2. writes one focused retrieval question per important atom (+ an optional 2nd
     question only for a genuinely different retrieval intent),
  3. writes 2–3 broader chunk-level questions (relationships / outcomes / in-chunk
     comparisons — never vague "what does this discuss").

The model sees only title / source / date / chunk text — never a benchmark query,
gold answer, or gold evidence. Structured JSON out. Cached/resumable. This stage
only PRODUCES candidates; acceptance happens in am_filter.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import am_config as C
import am_data as D

_SYS = (
    "You turn a news-article passage into retrieval questions for a search index. "
    "You first break the passage into self-contained atomic facts, then write "
    "questions those facts answer. Every answer and supporting span must come from "
    "the passage itself. Respond with ONLY a JSON object."
)

_USER = """Article title: {title}
Source: {source} | Published: {published}

Passage:
\"\"\"
{chunk}
\"\"\"

Do the following and return JSON only.

1. ATOMS: break the passage into self-contained atomic facts. Each atom states ONE claim, resolves pronouns to explicit named entities, and preserves names/orgs/products/dates/quantities/units/locations/events. Do not merge separate claims; do not add outside knowledge.

2. For each important atom write 1 focused retrieval question ("direct"). Add a 2nd question ("alternate") ONLY if it is a genuinely different retrieval intent, not a paraphrase. Each question names its central entity, avoids "the passage/article/text", avoids vague pronouns, and is answerable from the atom + passage.

3. CHUNK-LEVEL: write {n_cl} broader questions the whole passage answers — a relationship among facts, a broader event/outcome, or an in-passage comparison. They must be specific and self-contained, never "what does this discuss".

Every question needs a short_answer and at least one supporting_span copied VERBATIM from the passage.

JSON shape:
{{"chunk_id":"{chunk_id}",
 "atoms":[{{"atom_id":"a1","atomic_fact":"...","importance":"high|medium|low",
   "questions":[{{"question_view":"direct|alternate","question":"...","short_answer":"...","supporting_spans":["..."],"entities":["..."],"dates":["..."]}}]}}],
 "chunk_level_questions":[{{"question":"...","short_answer":"...","supporting_spans":["...","..."],"entities":["..."],"dates":["..."]}}]}}"""


def _parse(raw: str, chunk: dict) -> Tuple[List[dict], List[dict]]:
    """Return (atom_records, question_candidate_records)."""
    data = json.loads(raw)
    cid = chunk["chunk_id"]
    meta = {
        "parent_chunk_id": cid,
        "parent_document_id": chunk["parent_document_id"],
        "title": chunk["title"],
        "source": chunk["source"],
        "published_at": chunk["published_at"],
        "chunk_position": chunk["chunk_position"],
    }
    atoms, cands = [], []
    qn = 0
    for ai, atom in enumerate(data.get("atoms", [])):
        aid = atom.get("atom_id") or f"a{ai + 1}"
        atoms.append(
            {
                "chunk_id": cid,
                "atom_id": aid,
                "atomic_fact": atom.get("atomic_fact", ""),
                "importance": atom.get("importance", "medium"),
            }
        )
        for q in atom.get("questions", []):
            if q.get("question_view") == "alternate" and not C.ALLOW_SECOND_ATOMIC:
                continue
            qtext = (q.get("question") or "").strip()
            if not qtext:
                continue
            cands.append(
                {
                    "question_id": f"{cid}::q{qn}",
                    "question": qtext,
                    "question_type": "atomic",
                    "question_view": q.get("question_view", "direct"),
                    "atom_id": aid,
                    "importance": atom.get("importance", "medium"),
                    "short_answer": q.get("short_answer", ""),
                    "supporting_spans": q.get("supporting_spans", []),
                    "entities": q.get("entities", []),
                    "dates": q.get("dates", []),
                    **meta,
                }
            )
            qn += 1
    for q in data.get("chunk_level_questions", []):
        qtext = (q.get("question") or "").strip()
        if not qtext:
            continue
        cands.append(
            {
                "question_id": f"{cid}::q{qn}",
                "question": qtext,
                "question_type": "chunk_level",
                "question_view": "chunk",
                "atom_id": None,
                "importance": "chunk",
                "short_answer": q.get("short_answer", ""),
                "supporting_spans": q.get("supporting_spans", []),
                "entities": q.get("entities", []),
                "dates": q.get("dates", []),
                **meta,
            }
        )
        qn += 1
    return atoms, cands


def _generate_one(client, chunk: dict) -> Tuple[List[dict], List[dict], str]:
    n_cl = C.DEFAULT_CHUNK_LEVEL_Q
    prompt = _USER.format(
        title=chunk["title"],
        source=chunk["source"],
        published=chunk["published_at"],
        chunk=chunk["text"],
        chunk_id=chunk["chunk_id"],
        n_cl=f"2 to {C.MAX_CHUNK_LEVEL_Q}",
    )
    err = ""
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": prompt},
                ],
            )
            atoms, cands = _parse(resp.choices[0].message.content, chunk)
            if atoms and cands:
                return atoms, cands, ""
            err = "empty atoms/questions"
        except Exception as e:  # noqa: BLE001
            err = f"{e.__class__.__name__}"
    return [], [], err


def generate_all(force: bool = False) -> dict:
    chunks = D.load_chunks()
    done = set()
    if not force and C.QUESTIONS_RAW.exists():
        done = {r["parent_chunk_id"] for r in D.read_jsonl(C.QUESTIONS_RAW)}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"[gen] {len(chunks)} chunks, {len(done)} cached, {len(todo)} to do")

    all_atoms = (
        [] if force else (D.read_jsonl(C.ATOMS_PATH) if C.ATOMS_PATH.exists() else [])
    )
    all_cands = (
        []
        if force
        else (D.read_jsonl(C.QUESTIONS_RAW) if C.QUESTIONS_RAW.exists() else [])
    )
    fails = []
    t0 = time.perf_counter()
    if todo:
        client = C.openai_client()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_generate_one, client, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                atoms, cands, err = fut.result()
                if err and not cands:
                    fails.append({"chunk_id": c["chunk_id"], "error": err})
                all_atoms.extend(atoms)
                all_cands.extend(cands)
                if i % 20 == 0:
                    print(f"  [gen] {i}/{len(todo)}", flush=True)
    gen_seconds = round(time.perf_counter() - t0, 2)

    D._write_jsonl(C.ATOMS_PATH, all_atoms)
    D._write_jsonl(C.QUESTIONS_RAW, all_cands)

    n_atomic = sum(1 for q in all_cands if q["question_type"] == "atomic")
    n_chunk = sum(1 for q in all_cands if q["question_type"] == "chunk_level")
    per_chunk = {}
    for q in all_cands:
        per_chunk[q["parent_chunk_id"]] = per_chunk.get(q["parent_chunk_id"], 0) + 1
    report = {
        "total_chunks": len(chunks),
        "chunks_with_output": len(per_chunk),
        "total_atoms": len(all_atoms),
        "avg_atoms_per_chunk": round(len(all_atoms) / max(len(chunks), 1), 2),
        "raw_questions_total": len(all_cands),
        "raw_atomic_questions": n_atomic,
        "raw_chunk_level_questions": n_chunk,
        "avg_raw_questions_per_chunk": round(len(all_cands) / max(len(chunks), 1), 2),
        "generation_failures": len(fails),
        "generation_seconds": gen_seconds,
        "generation_model": C.gen_model(),
    }
    json.dump(report, open(C.GEN_QUALITY, "w"), indent=2)
    print(
        f"[gen] atoms={len(all_atoms)} raw_q={len(all_cands)} "
        f"(atomic {n_atomic} / chunk-level {n_chunk}) {gen_seconds}s"
    )
    return report


if __name__ == "__main__":
    generate_all()
