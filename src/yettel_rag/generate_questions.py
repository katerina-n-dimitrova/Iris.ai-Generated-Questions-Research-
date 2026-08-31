#!/usr/bin/env python3
"""Generate an evidence-grounded Yettel benchmark in MultiHop-RAG format.

This is intentionally isolated from any future chunk-enrichment questions. It
uses a fixed seed, its own prompt, and writes only evaluation artifacts.
Generation is resumable at the individual query-id level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

DATA = ROOT / "data" / "processed" / "yettel_bg"
CHECKPOINT = DATA / "question_generation_checkpoint.jsonl"
OUTPUT = DATA / "questions.jsonl"
MHRAG_OUTPUT = DATA / "MultiHopRAG.json"
REPORT = DATA / "question_manifest.json"
SEED = 20260817
TYPE_TARGETS = {
    "inference_query": 816,
    "comparison_query": 856,
    "temporal_query": 583,
    "null_query": 301,
}
HOP_TARGETS = {2: 1169, 3: 774, 4: 312}
BATCH_SIZE = 4

WORD = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9+.-]{2,}")
SPACE = re.compile(r"\s+")
BG_STOP = {
    "като",
    "които",
    "която",
    "което",
    "този",
    "тази",
    "това",
    "тези",
    "може",
    "могат",
    "има",
    "имат",
    "или",
    "при",
    "през",
    "след",
    "преди",
    "дори",
    "само",
    "вече",
    "също",
    "всички",
    "всяка",
    "всеки",
    "между",
    "към",
    "над",
    "под",
    "без",
    "със",
    "във",
    "един",
    "една",
    "едно",
    "тях",
    "него",
    "нея",
    "своя",
    "своите",
    "твоите",
    "теб",
    "yettel",
    "българия",
    "клиенти",
    "клиентите",
    "услуга",
    "услуги",
    "повече",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "you",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def terms(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in WORD.finditer(text)
        if m.group(0).lower() not in BG_STOP
    }


def norm(text: str) -> str:
    return SPACE.sub(" ", text or "").strip().lower()


def balanced_sequence(targets: dict) -> list:
    """Deterministically interleave labels while preserving exact counts."""
    remaining = dict(targets)
    total = sum(remaining.values())
    output = []
    for position in range(total):
        label = max(
            remaining, key=lambda key: (remaining[key] / targets[key], str(key))
        )
        output.append(label)
        remaining[label] -= 1
        if remaining[label] == 0:
            del remaining[label]
    return output


def build_neighbors(documents: list[dict]) -> dict[str, list[str]]:
    doc_terms = {
        d["document_id"]: terms(d["title"] + " " + d["body"]) for d in documents
    }
    frequency = Counter(term for values in doc_terms.values() for term in values)
    weighted = {
        did: {
            term: math.log((len(documents) + 1) / (frequency[term] + 1)) + 1
            for term in values
        }
        for did, values in doc_terms.items()
    }
    inverted = defaultdict(set)
    for did, values in doc_terms.items():
        for term in values:
            inverted[term].add(did)
    result = {}
    for document in documents:
        did = document["document_id"]
        scores = defaultdict(float)
        # Rare shared terms and title terms dominate, which forms coherent
        # product/service/event clusters without using evaluation questions.
        focus = set(sorted(weighted[did], key=weighted[did].get, reverse=True)[:120])
        focus |= terms(document["title"])
        for term in focus:
            for other in inverted[term]:
                if other != did:
                    scores[other] += weighted[did].get(term, 1.0)
        result[did] = [
            other
            for other, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
    return result


def choose_excerpt(
    document_ids: list[str], chunks_by_doc: dict, document_by_id: dict
) -> list[dict]:
    focus = set()
    for did in document_ids:
        focus |= terms(document_by_id[did]["title"])
    excerpts = []
    for did in document_ids:
        candidates = chunks_by_doc[did]
        chunk = max(
            candidates,
            key=lambda row: (
                len(terms(row["text"]) & focus),
                row["token_count"],
                row["chunk_id"],
            ),
        )
        # A complete chunk remains within the model context for four-document
        # bundles and ensures any returned quote can map to a canonical chunk.
        excerpts.append(
            {
                "document_id": did,
                "chunk_id": chunk["chunk_id"],
                "title": document_by_id[did]["title"],
                "url": document_by_id[did]["url"],
                "source": document_by_id[did]["source"],
                "category": document_by_id[did]["category"],
                "date": document_by_id[did]["date"],
                "text": chunk["text"],
            }
        )
    return excerpts


def make_specs(documents: list[dict], chunks: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    document_by_id = {d["document_id"]: d for d in documents}
    chunks_by_doc = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["document_id"]].append(chunk)
    neighbors = build_neighbors(documents)
    answerable_types = balanced_sequence(
        {k: v for k, v in TYPE_TARGETS.items() if k != "null_query"}
    )
    hops = balanced_sequence(HOP_TARGETS)
    anchors = [d["document_id"] for d in documents]
    rng.shuffle(anchors)
    specs = []
    for index, (question_type, hop_count) in enumerate(zip(answerable_types, hops)):
        # Generation retries may safely use the strongest related pair. Existing
        # checkpoint rows retain their validated 2-4-document evidence, while
        # unresolved IDs avoid repeatedly receiving an incoherent long bundle.
        if os.getenv("YETTEL_RETRY_STRONGEST_PAIR") == "1":
            hop_count = 2
        anchor = anchors[index % len(anchors)]
        retry_offset = int(os.getenv("YETTEL_RETRY_NEIGHBOR_OFFSET", "0"))
        rotation = ((index // len(anchors)) + retry_offset) % max(
            1, len(neighbors[anchor])
        )
        ordered = neighbors[anchor][rotation:] + neighbors[anchor][:rotation]
        selected = [anchor]
        for candidate in ordered:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == hop_count:
                break
        if os.getenv("YETTEL_RETRY_DATE_PAIR") == "1" and index == 229:
            selected = ["yettel_bg_0147", "yettel_bg_0235"]
        if len(selected) < hop_count:
            for candidate in anchors:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == hop_count:
                    break
        specs.append(
            {
                "query_id": f"yq{index:05d}",
                "question_type": question_type,
                "documents": choose_excerpt(selected, chunks_by_doc, document_by_id),
            }
        )
    # Null prompts get representative entity context but never gold evidence.
    null_start = len(specs)
    for offset in range(TYPE_TARGETS["null_query"]):
        did = anchors[(offset * 7) % len(anchors)]
        specs.append(
            {
                "query_id": f"yq{null_start + offset:05d}",
                "question_type": "null_query",
                "documents": choose_excerpt([did], chunks_by_doc, document_by_id),
            }
        )
    return specs


SYSTEM = """You create Bulgarian evaluation questions for a telecommunications RAG benchmark.
Return valid JSON only. Never use outside knowledge. Do not mention documents, passages, evidence, or the benchmark in a question.

For answerable tasks:
- Write one natural Bulgarian question of the requested type.
- It must require combining ALL supplied documents; a single document must be insufficient.
- Return a concise Bulgarian answer.
- For every supplied document copy one short, verbatim evidence span from its text. Do not paraphrase evidence.
- comparison_query compares facts across sources; temporal_query combines dates/order/change over time; inference_query identifies or deduces an answer by connecting facts.

For null_query:
- Write a plausible Bulgarian question in the same domain whose answer is not stated in the supplied context.
- Set answer to "Няма достатъчно информация" and evidence to an empty list.

Avoid yes/no questions, vague pronouns, promotional fluff, and questions whose answer appears directly in the wording."""


def render_task(spec: dict) -> dict:
    docs = []
    for item in spec["documents"]:
        docs.append(
            {
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "date": item["date"],
                "text": item["text"],
            }
        )
    return {
        "query_id": spec["query_id"],
        "question_type": spec["question_type"],
        "documents": docs,
    }


def call_batch(client: OpenAI, model: str, specs: list[dict]) -> list[dict]:
    user = {
        "tasks": [render_task(spec) for spec in specs],
        "output_schema": {
            "results": [
                {
                    "query_id": "exact input id",
                    "query": "Bulgarian question",
                    "answer": "concise Bulgarian answer",
                    "question_type": "exact requested type",
                    "evidence": [
                        {
                            "document_id": "...",
                            "chunk_id": "...",
                            "fact": "verbatim span",
                        }
                    ],
                }
            ]
        },
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["results"]


def canonicalize(raw: dict, spec: dict) -> dict | None:
    if (
        raw.get("query_id") != spec["query_id"]
        or raw.get("question_type") != spec["question_type"]
    ):
        return None
    query = SPACE.sub(" ", str(raw.get("query", ""))).strip()
    answer = SPACE.sub(" ", str(raw.get("answer", ""))).strip()
    if not query or not answer:
        return None
    if spec["question_type"] == "null_query":
        return {
            "query_id": spec["query_id"],
            "query": query,
            "answer": "Няма достатъчно информация",
            "question_type": "null_query",
            "gold_document_ids": [],
            "gold_evidence": [],
            "gold_chunk_ids": [],
            "evidence_list": [],
            "generation_split": "evaluation",
            "generation_seed": SEED,
        }
    supplied = {(d["document_id"], d["chunk_id"]): d for d in spec["documents"]}
    evidence = raw.get("evidence") or []
    # A generation bundle may expose up to four related documents. Preserve the
    # subset the model actually used, provided it is genuinely multi-document;
    # this follows MultiHop-RAG's 2-4 evidence design without assigning unused
    # documents as gold relevance labels.
    if not (2 <= len(evidence) <= len(spec["documents"])):
        return None
    canonical = []
    seen_docs = set()
    for item in evidence:
        key = (item.get("document_id"), item.get("chunk_id"))
        source = supplied.get(key)
        fact = SPACE.sub(" ", str(item.get("fact", ""))).strip()
        if (
            not source
            or source["document_id"] in seen_docs
            or not fact
            or norm(fact) not in norm(source["text"])
        ):
            return None
        seen_docs.add(source["document_id"])
        canonical.append(
            {
                "document_id": source["document_id"],
                "chunk_id": source["chunk_id"],
                "title": source["title"],
                "author": None,
                "url": source["url"],
                "source": source["source"],
                "category": source["category"],
                "published_at": None,
                "date": source["date"],
                "fact": fact,
            }
        )
    return {
        "query_id": spec["query_id"],
        "query": query,
        "answer": answer,
        "question_type": spec["question_type"],
        "gold_document_ids": [e["document_id"] for e in canonical],
        "gold_evidence": [e["fact"] for e in canonical],
        "gold_chunk_ids": [e["chunk_id"] for e in canonical],
        "evidence_list": canonical,
        "generation_split": "evaluation",
        "generation_seed": SEED,
    }


def generate_group(
    client: OpenAI, model: str, specs: list[dict], attempts: int = 4
) -> list[dict]:
    pending = {spec["query_id"]: spec for spec in specs}
    valid = []
    for attempt in range(attempts):
        if not pending:
            break
        try:
            raw_rows = call_batch(client, model, list(pending.values()))
            for raw in raw_rows:
                spec = pending.get(raw.get("query_id"))
                if not spec:
                    continue
                row = canonicalize(raw, spec)
                if row:
                    valid.append(row)
                    pending.pop(spec["query_id"], None)
        except Exception:
            if attempt == attempts - 1:
                raise
        if pending:
            time.sleep(1.5 * (attempt + 1))
    return valid


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[str]]:
    seen = set()
    kept, rejected = [], []
    for row in sorted(rows, key=lambda item: item["query_id"]):
        key = norm(row["query"])
        if key in seen:
            rejected.append(row["query_id"])
        else:
            seen.add(key)
            kept.append(row)
    return kept, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.OPENAI_CHAT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--limit", type=int, default=0, help="Development-only number of tasks"
    )
    args = parser.parse_args()
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    documents = read_jsonl(DATA / "documents.jsonl")
    chunks = read_jsonl(DATA / "chunks_1024.jsonl")
    specs = make_specs(documents, chunks)
    if args.limit:
        specs = specs[: args.limit]
    completed = {}
    if CHECKPOINT.exists():
        for row in read_jsonl(CHECKPOINT):
            completed[row["query_id"]] = row
    pending = [spec for spec in specs if spec["query_id"] not in completed]
    batches = [
        pending[index : index + args.batch_size]
        for index in range(0, len(pending), args.batch_size)
    ]
    client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=120, max_retries=2)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_group, client, args.model, batch): batch
            for batch in batches
        }
        for number, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            if rows:
                append_jsonl(CHECKPOINT, rows)
                for row in rows:
                    completed[row["query_id"]] = row
            if number % 25 == 0:
                print(
                    f"batches={number}/{len(batches)} valid={len(completed)}/{len(specs)}",
                    flush=True,
                )

    rows, duplicates = deduplicate(list(completed.values()))
    expected_ids = {spec["query_id"] for spec in specs}
    rows = [row for row in rows if row["query_id"] in expected_ids]
    missing = sorted(expected_ids - {row["query_id"] for row in rows})
    # Do not silently publish an incomplete benchmark. A rerun resumes missing
    # tasks; duplicate question rows are removed from the checkpoint first.
    if missing or duplicates:
        if duplicates:
            retained = [
                row
                for row in completed.values()
                if row["query_id"] not in set(duplicates)
            ]
            CHECKPOINT.unlink(missing_ok=True)
            append_jsonl(CHECKPOINT, retained)
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "valid": len(rows),
                    "missing": len(missing),
                    "duplicates": len(duplicates),
                },
                indent=2,
            )
        )
        raise SystemExit(2)

    rows.sort(key=lambda item: item["query_id"])
    append_target = OUTPUT
    append_target.unlink(missing_ok=True)
    append_jsonl(append_target, rows)
    MHRAG_OUTPUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "question_count": len(rows),
        "model": args.model,
        "seed": SEED,
        "generation_split": "evaluation",
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "evidence_document_counts": dict(
            Counter(len(row["gold_document_ids"]) for row in rows)
        ),
        "leakage_note": "Evaluation prompts and artifacts are isolated from all chunk-enrichment question generation.",
    }
    REPORT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
