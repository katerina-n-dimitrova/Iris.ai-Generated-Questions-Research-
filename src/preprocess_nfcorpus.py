"""
Preprocess BEIR NFCorpus (unstructured biomedical text).

Long documents are split into overlapping chunks. Baseline embeds the raw
chunk text; enriched prepends dataset/type/title, key entities/keywords, and a
generated chunk summary.

Produces document records (baseline + enriched) and a queries file from
data/raw/nfcorpus/.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import common
import config
from common import make_record, read_jsonl, render_enriched

SPEC = config.DATASETS["nfcorpus"]


def build_documents(use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
                    ) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    corpus_path = SPEC.raw_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} missing. Run download_datasets.py first.")

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for row in list(read_jsonl(corpus_path))[:max_samples]:
        doc_id = str(row.get("_id") or row.get("id") or row.get("doc_id"))
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        for ci, chunk in enumerate(common.chunk_text(text)):
            base_id = f"nfcorpus_{doc_id}_c{ci}"

            baseline.append(make_record(
                chunk_id=f"{base_id}_baseline",
                dataset="nfcorpus",
                input_type=SPEC.input_type,
                condition="baseline",
                text_for_embedding=chunk,
                original_text=chunk,
                source_id=doc_id,
                title=title,
            ))

            keywords = common.cheap_keywords(f"{title}. {chunk}")
            summary = common.llm_summary(client, chunk, kind="biomedical passage") \
                if use_llm else common.cheap_summary(chunk)
            enriched_text = render_enriched({
                "Dataset": "NFCorpus",
                "Document type": "biomedical text",
                "Document title": title,
                "Key entities/keywords": keywords,
                "Generated chunk summary": summary,
                "Original text": chunk,
            })
            enriched.append(make_record(
                chunk_id=f"{base_id}_enriched",
                dataset="nfcorpus",
                input_type=SPEC.input_type,
                condition="enriched",
                text_for_embedding=enriched_text,
                original_text=chunk,
                source_id=doc_id,
                title=title,
            ))

    return {"baseline": baseline, "enriched": enriched}


def build_queries(max_samples: int = config.MAX_DATASET_SAMPLES) -> List[Dict[str, Any]]:
    queries_path = SPEC.raw_dir / "queries.jsonl"
    if not queries_path.exists():
        print(f"  (no queries file for nfcorpus at {queries_path})")
        return []

    gold: Dict[str, List[str]] = {}
    for split in ("test", "validation", "train"):
        qpath = SPEC.raw_dir / f"qrels_{split}.jsonl"
        if not qpath.exists():
            continue
        for r in read_jsonl(qpath):
            qid = str(r.get("query-id") or r.get("query_id"))
            cid = str(r.get("corpus-id") or r.get("corpus_id"))
            if int(r.get("score", 1)) > 0:
                gold.setdefault(qid, []).append(cid)
        break

    queries: List[Dict[str, Any]] = []
    for row in list(read_jsonl(queries_path))[:max_samples]:
        qid = str(row.get("_id") or row.get("id"))
        text = str(row.get("text") or row.get("query") or "").strip()
        if not text:
            continue
        queries.append({
            "query_id": qid,
            "dataset": "nfcorpus",
            "text": text,
            "gold_source_ids": gold.get(qid, []),
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
        print(f"nfcorpus {condition}: {n} chunks -> {SPEC.processed_path(condition).name}")

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "nfcorpus_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"nfcorpus queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
