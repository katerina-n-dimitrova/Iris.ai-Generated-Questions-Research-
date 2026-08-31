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


# NFCorpus is dense (~40 gold docs per query), so we evaluate fewer queries to
# keep the gold-doc union (and thus the index) a manageable size.
N_EVAL_QUERIES = min(config.MAX_DATASET_SAMPLES, 30)
N_DISTRACTORS = config.MAX_DATASET_SAMPLES


def _corpus_index() -> Dict[str, Dict[str, Any]]:
    corpus_path = SPEC.raw_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} missing. Run download_datasets.py first."
        )
    idx = {}
    for row in read_jsonl(corpus_path):
        doc_id = str(row.get("_id") or row.get("id") or row.get("doc_id"))
        idx[doc_id] = row
    return idx


def _select(corpus: Dict[str, Dict[str, Any]]):
    qtext, gold = common.load_beir_queries_gold(SPEC.raw_dir)
    return common.select_eval_subset(
        qtext, gold, corpus.keys(), N_EVAL_QUERIES, N_DISTRACTORS
    )


def build_documents(
    use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    corpus = _corpus_index()
    _, index_ids = _select(corpus)

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for doc_id in sorted(index_ids):
        row = corpus[doc_id]
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        for ci, chunk in enumerate(common.chunk_text(text)):
            base_id = f"nfcorpus_{doc_id}_c{ci}"

            baseline.append(
                make_record(
                    chunk_id=f"{base_id}_baseline",
                    dataset="nfcorpus",
                    input_type=SPEC.input_type,
                    condition="baseline",
                    text_for_embedding=chunk,
                    original_text=chunk,
                    source_id=doc_id,
                    title=title,
                )
            )

            keywords = common.cheap_keywords(f"{title}. {chunk}")
            summary = (
                common.llm_summary(client, chunk, kind="biomedical passage")
                if use_llm
                else common.cheap_summary(chunk)
            )
            enriched_text = render_enriched(
                {
                    "Dataset": "NFCorpus",
                    "Document type": "biomedical text",
                    "Document title": title,
                    "Key entities/keywords": keywords,
                    "Generated chunk summary": summary,
                    "Original text": chunk,
                }
            )
            enriched.append(
                make_record(
                    chunk_id=f"{base_id}_enriched",
                    dataset="nfcorpus",
                    input_type=SPEC.input_type,
                    condition="enriched",
                    text_for_embedding=enriched_text,
                    original_text=chunk,
                    source_id=doc_id,
                    title=title,
                )
            )

    return {"baseline": baseline, "enriched": enriched}


def build_queries(
    max_samples: int = config.MAX_DATASET_SAMPLES,
) -> List[Dict[str, Any]]:
    corpus = _corpus_index()
    selected, _ = _select(corpus)
    return [
        {
            "query_id": qid,
            "dataset": "nfcorpus",
            "text": text,
            "gold_source_ids": gold_ids,
        }
        for qid, text, gold_ids in selected
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    docs = build_documents(args.use_llm, args.max_samples)
    for condition in config.CONDITIONS:
        n = common.write_jsonl(SPEC.processed_path(condition), docs[condition])
        print(
            f"nfcorpus {condition}: {n} chunks -> {SPEC.processed_path(condition).name}"
        )

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "nfcorpus_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"nfcorpus queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
