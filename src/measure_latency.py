"""
Summarize the latency + storage logs into a single comparison table.

Reads:
    results/latency_logs/offline_indexing.csv     (from build_chroma_indexes.py)
    results/latency_logs/online_query_latency.csv (from run_retrieval_experiments.py)

Writes:
    results/latency_logs/latency_summary.csv

and prints a baseline-vs-enriched comparison covering offline encoding latency,
index size, and online query latency (mean + p95). This is the headline view
for objectives 3-5 (encoding latency, query latency, index size/storage).

Usage:
    python src/measure_latency.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config

OFFLINE = config.LATENCY_LOG_DIR / "offline_indexing.csv"
ONLINE = config.LATENCY_LOG_DIR / "online_query_latency.csv"
SUMMARY = config.LATENCY_LOG_DIR / "latency_summary.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    offline = _read_csv(OFFLINE)
    online = _read_csv(ONLINE)

    if offline.empty and online.empty:
        print("No latency logs found. Build indexes and run retrieval first.")
        return

    rows: List[Dict] = []
    keys = set()
    if not offline.empty:
        keys |= set(map(tuple, offline[["dataset", "condition"]].values))
    if not online.empty:
        keys |= set(map(tuple, online[["dataset", "condition"]].values))

    for dataset, condition in sorted(keys):
        row: Dict = {"dataset": dataset, "condition": condition}

        if not offline.empty:
            o = offline[(offline.dataset == dataset)
                        & (offline.condition == condition)]
            if not o.empty:
                o = o.iloc[0]
                row.update({
                    "num_chunks": int(o["num_chunks"]),
                    "avg_tokens_per_chunk": o["avg_tokens_per_chunk"],
                    "total_embedding_time_s": o["total_embedding_time_seconds"],
                    "avg_embed_ms_per_chunk": o["avg_embedding_time_per_chunk_ms"],
                    "total_indexing_time_s": o["total_indexing_time_seconds"],
                    "index_size_mb": o["index_size_mb"],
                })

        if not online.empty:
            q = online[(online.dataset == dataset)
                       & (online.condition == condition)]
            if not q.empty:
                t = q["total_retrieval_latency_ms"]
                row.update({
                    "num_queries": len(q),
                    "query_embed_ms_mean":
                        round(q["query_embedding_latency_ms"].mean(), 3),
                    "chroma_search_ms_mean":
                        round(q["chroma_search_latency_ms"].mean(), 3),
                    "retrieval_ms_mean": round(t.mean(), 3),
                    "retrieval_ms_p95": round(t.quantile(0.95), 3),
                })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Latency & storage summary (baseline vs enriched) ===\n")
    print(df.to_string(index=False))
    print(f"\nSaved -> {SUMMARY.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
