"""
Preprocess WikiTableQuestions (tables).

Each table row becomes one chunk. Baseline embeds a linearized row
("header: value | header: value"); enriched prepends dataset/type, table/page
title, column headers, and a natural-language row summary, then the original
row.

Tables are de-duplicated across questions. Produces document records
(baseline + enriched) and a queries file from data/raw/wikitablequestions/.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import common
import config
from common import make_record, read_jsonl, render_enriched

SPEC = config.DATASETS["wikitablequestions"]


def _table_id(table: Dict[str, Any], fallback: str) -> str:
    return str(table.get("name") or table.get("id") or fallback)


def _linearize_row(headers: List[str], row: List[Any]) -> str:
    pairs = []
    for h, v in zip(headers, row):
        h = str(h).strip()
        v = str(v).strip()
        pairs.append(f"{h}: {v}" if h else v)
    return " | ".join(p for p in pairs if p)


def _row_summary(client, title: str, headers: List[str], row: List[Any],
                 use_llm: bool) -> str:
    liner = _linearize_row(headers, row)
    if use_llm:
        return common.llm_summary(
            client, f"Table '{title}'. Row: {liner}", kind="table row")
    # Cheap deterministic NL rendering.
    return f"In table '{title}', " + "; ".join(
        f"{str(h).strip()} is {str(v).strip()}"
        for h, v in zip(headers, row) if str(v).strip()
    )


def build_documents(use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
                    ) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    test_path = SPEC.raw_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} missing. Run download_datasets.py first.")

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []
    seen_tables: set[str] = set()

    for idx, row in enumerate(list(read_jsonl(test_path))[:max_samples]):
        table = row.get("table") or {}
        tid = _table_id(table, fallback=f"table_{idx}")
        if tid in seen_tables:
            continue
        seen_tables.add(tid)

        title = str(table.get("name") or row.get("page_title") or tid).strip()
        headers = [str(h) for h in (table.get("header") or [])]
        rows = table.get("rows") or []

        for ri, data_row in enumerate(rows):
            linear = _linearize_row(headers, data_row)
            if not linear:
                continue
            base_id = f"wtq_{tid}_r{ri}".replace("/", "_")

            baseline.append(make_record(
                chunk_id=f"{base_id}_baseline",
                dataset="wikitablequestions",
                input_type=SPEC.input_type,
                condition="baseline",
                text_for_embedding=linear,
                original_text=linear,
                source_id=tid,
                title=title,
                row_id=str(ri),
            ))

            summary = _row_summary(client, title, headers, data_row, use_llm)
            enriched_text = render_enriched({
                "Dataset": "WikiTableQuestions",
                "Document type": "table",
                "Table title/page title": title,
                "Column headers": ", ".join(headers),
                "Row summary in natural language": summary,
                "Original row": linear,
            })
            enriched.append(make_record(
                chunk_id=f"{base_id}_enriched",
                dataset="wikitablequestions",
                input_type=SPEC.input_type,
                condition="enriched",
                text_for_embedding=enriched_text,
                original_text=linear,
                source_id=tid,
                title=title,
                row_id=str(ri),
            ))

    return {"baseline": baseline, "enriched": enriched}


def build_queries(max_samples: int = config.MAX_DATASET_SAMPLES) -> List[Dict[str, Any]]:
    test_path = SPEC.raw_dir / "test.jsonl"
    if not test_path.exists():
        return []
    queries: List[Dict[str, Any]] = []
    for idx, row in enumerate(list(read_jsonl(test_path))[:max_samples]):
        table = row.get("table") or {}
        tid = _table_id(table, fallback=f"table_{idx}")
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        answers = row.get("answers")
        gold_answer = ", ".join(map(str, answers)) if isinstance(answers, list) \
            else str(answers or "")
        queries.append({
            "query_id": str(row.get("id") or f"wtq_q{idx}"),
            "dataset": "wikitablequestions",
            "text": question,
            "gold_source_ids": [tid],
            "gold_answer": gold_answer,
        })
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    docs = build_documents(args.use_llm, args.max_samples)
    for condition in config.CONDITIONS:
        n = common.write_jsonl(SPEC.processed_path(condition), docs[condition])
        print(f"wikitablequestions {condition}: {n} chunks "
              f"-> {SPEC.processed_path(condition).name}")

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "wikitablequestions_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"wikitablequestions queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
