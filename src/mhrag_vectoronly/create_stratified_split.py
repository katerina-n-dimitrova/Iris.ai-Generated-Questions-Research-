"""Create the permanent seed-42 50/17/17 split for the 84-query pilot."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "mhrag_vectoronly" / "hierarchical_512_128"
QUERIES = DATA / "queries.jsonl"
OUT = DATA / "query_split_seed42.jsonl"
SUMMARY = DATA / "query_split_seed42_summary.json"
TARGETS = {
    "inference": {"train": 18, "development": 6, "test": 6},
    "temporal": {"train": 17, "development": 6, "test": 6},
    "comparison": {"train": 15, "development": 5, "test": 5},
}


def create() -> list[dict]:
    queries = [json.loads(line) for line in QUERIES.open()]
    grouped = defaultdict(list)
    for query in queries:
        grouped[query["question_type"]].append(query)
    if {key: len(value) for key, value in grouped.items()} != {
        "inference": 30,
        "temporal": 29,
        "comparison": 25,
    }:
        raise RuntimeError("Query population changed; refusing to alter split")

    rows = []
    for question_type in ("inference", "temporal", "comparison"):
        group = sorted(grouped[question_type], key=lambda row: row["query_id"])
        random.Random(f"{SEED}:{question_type}").shuffle(group)
        offset = 0
        for split in ("train", "development", "test"):
            count = TARGETS[question_type][split]
            for query in group[offset : offset + count]:
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "question_type": question_type,
                        "split": split,
                        "seed": SEED,
                    }
                )
            offset += count

    rows.sort(key=lambda row: row["query_id"])
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts = Counter((row["split"], row["question_type"]) for row in rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    summary = {
        "seed": SEED,
        "method": "stratified deterministic shuffle within question type",
        "total": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "strata": {
            split: {
                question_type: counts[(split, question_type)]
                for question_type in ("inference", "temporal", "comparison")
            }
            for split in ("train", "development", "test")
        },
        "sha256": digest,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return rows


if __name__ == "__main__":
    result = create()
    print(f"[split] wrote {len(result)} rows to {OUT}")
    print(SUMMARY.read_text())
