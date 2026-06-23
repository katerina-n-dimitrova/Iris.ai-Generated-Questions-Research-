"""
Preprocess SciFact (structured scientific text).

Produces, from data/raw/scifact/:
  - document records (baseline + enriched), one per abstract chunk
  - a queries file (claim text + gold doc ids from qrels) for retrieval eval

Each abstract is chunked at the sentence/paragraph level. Baseline embeds the
raw sentence; enriched prepends dataset/type/title/position context plus a
summary or keyword line.

Run directly to write all three files, or import build_documents/build_queries
from the create_*_chunks orchestrators.
"""

from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List

import common
import config
from common import make_record, read_jsonl, render_enriched

SPEC = config.DATASETS["scifact"]


def _abstract_sentences(row: Dict[str, Any]) -> List[str]:
    # allenai/scifact stored 'abstract' as a list of sentences; the BeIR mirror
    # stores the abstract as a single 'text' string. Support both.
    abstract = row.get("abstract")
    if isinstance(abstract, list):
        return [str(s).strip() for s in abstract if str(s).strip()]
    text = abstract if isinstance(abstract, str) else row.get("text")
    if isinstance(text, str) and text.strip():
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sents if s.strip()]
    return []


def build_documents(use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
                    ) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    corpus_path = SPEC.raw_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} missing. Run download_datasets.py first.")

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for row in list(read_jsonl(corpus_path))[:max_samples]:
        doc_id = str(row.get("doc_id") or row.get("_id") or row.get("id"))
        title = str(row.get("title") or "").strip()
        sentences = _abstract_sentences(row)
        n = len(sentences)

        for pos, sent in enumerate(sentences):
            base_id = f"scifact_{doc_id}_s{pos}"

            baseline.append(make_record(
                chunk_id=f"{base_id}_baseline",
                dataset="scifact",
                input_type=SPEC.input_type,
                condition="baseline",
                text_for_embedding=sent,
                original_text=sent,
                source_id=doc_id,
                title=title,
                page=f"{pos + 1}/{n}",
            ))

            summary = common.llm_summary(client, sent, kind="scientific sentence") \
                if use_llm else common.cheap_keywords(f"{title}. {sent}")
            enriched_text = render_enriched({
                "Dataset": "SciFact",
                "Document type": "scientific abstract",
                "Paper title": title,
                "Sentence position": f"{pos + 1} of {n}",
                "Original text": sent,
                "Generated summary or keywords": summary,
            })
            enriched.append(make_record(
                chunk_id=f"{base_id}_enriched",
                dataset="scifact",
                input_type=SPEC.input_type,
                condition="enriched",
                text_for_embedding=enriched_text,
                original_text=sent,
                source_id=doc_id,
                title=title,
                page=f"{pos + 1}/{n}",
            ))

    return {"baseline": baseline, "enriched": enriched}


def build_queries(max_samples: int = config.MAX_DATASET_SAMPLES) -> List[Dict[str, Any]]:
    queries_path = SPEC.raw_dir / "queries.jsonl"
    if not queries_path.exists():
        print(f"  (no queries file for scifact at {queries_path})")
        return []

    # Gold relevance from qrels (prefer test split).
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
        break  # use the first split that exists

    queries: List[Dict[str, Any]] = []
    for row in list(read_jsonl(queries_path))[:max_samples]:
        qid = str(row.get("_id") or row.get("id"))
        text = str(row.get("text") or row.get("query") or "").strip()
        if not text:
            continue
        queries.append({
            "query_id": qid,
            "dataset": "scifact",
            "text": text,
            "gold_source_ids": gold.get(qid, []),
        })
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true",
                    help="Use OpenAI for enrichment summaries (else cheap keywords).")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    docs = build_documents(args.use_llm, args.max_samples)
    for condition in config.CONDITIONS:
        n = common.write_jsonl(SPEC.processed_path(condition), docs[condition])
        print(f"scifact {condition}: {n} chunks -> {SPEC.processed_path(condition).name}")

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "scifact_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"scifact queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
