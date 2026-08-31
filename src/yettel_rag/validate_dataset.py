#!/usr/bin/env python3
"""Integrity checks for the Yettel corpus, chunks, and optional questions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "yettel_bg"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    documents = read_jsonl(DATA / "documents.jsonl")
    chunks = read_jsonl(DATA / "chunks_1024.jsonl")
    require(
        len(documents) == len({d["document_id"] for d in documents}),
        "duplicate document_id",
    )
    require(len(documents) == len({d["url"] for d in documents}), "duplicate URL")
    require(
        len(documents) == len({d["body_sha256"] for d in documents}),
        "duplicate clean body",
    )
    require(
        all(1_500 <= d["token_count"] <= 5_000 for d in documents),
        "document outside strict range",
    )
    require(all(d["body"].strip() for d in documents), "empty body")
    require(len(chunks) == len({c["chunk_id"] for c in chunks}), "duplicate chunk_id")
    document_ids = {d["document_id"] for d in documents}
    require(all(c["document_id"] in document_ids for c in chunks), "orphan chunk")
    require(all(0 < c["token_count"] <= 1_024 for c in chunks), "invalid chunk size")
    counts = Counter(c["document_id"] for c in chunks)
    require(
        all(counts[d] >= 2 for d in document_ids),
        "strict document did not produce multiple chunks",
    )

    question_path = DATA / "questions.jsonl"
    question_summary = {"present": question_path.exists()}
    if question_path.exists():
        questions = read_jsonl(question_path)
        chunk_ids = {c["chunk_id"] for c in chunks}
        required = {
            "query",
            "answer",
            "question_type",
            "gold_document_ids",
            "gold_evidence",
            "gold_chunk_ids",
            "evidence_list",
        }
        for index, question in enumerate(questions):
            require(required <= question.keys(), f"question {index} missing fields")
            require(
                question["question_type"]
                in {
                    "inference_query",
                    "comparison_query",
                    "temporal_query",
                    "null_query",
                },
                f"bad type {index}",
            )
            if question["question_type"] == "null_query":
                require(
                    not question["gold_document_ids"]
                    and not question["gold_chunk_ids"]
                    and not question["evidence_list"],
                    f"null has evidence {index}",
                )
            else:
                require(
                    2 <= len(set(question["gold_document_ids"])) <= 4,
                    f"not 2-4 documents {index}",
                )
                require(
                    set(question["gold_document_ids"]) <= document_ids,
                    f"unknown document {index}",
                )
                require(
                    set(question["gold_chunk_ids"]) <= chunk_ids,
                    f"unknown chunk {index}",
                )
        question_summary = {
            "present": True,
            "count": len(questions),
            "types": Counter(q["question_type"] for q in questions),
        }

    print(
        json.dumps(
            {
                "status": "valid",
                "documents": len(documents),
                "chunks": len(chunks),
                "chunks_per_document_min": min(counts.values()),
                "chunks_per_document_max": max(counts.values()),
                "questions": question_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
