"""
Dataset prep for the 10-article atomic+chunk-level mixed-question pilot.

Reuses the validated 15-article pipeline (`vo_data`) for the generic pieces —
light cleaning, the cl100k tokenizer, gold fact->chunk alignment — and only
re-implements the two config-dependent bits: query-first selection with the
10-article budget, and token chunking against this experiment's config. Nothing
here touches the earlier experiment's data.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Dict, List

import am_config as C

# reuse the validated helpers from the 15-article harness
from vo_data import (
    _norm,
    _aid,
    _short_type,
    _query_article_ids,
    clean_body,
    read_jsonl,
    _write_jsonl,
    load_corpus,
    load_all_queries,
    _tok_words,
    _ENC,
)


def _locate_fact(fact: str, chunks: List[dict]):
    """Align labels by normalized substring only, as specified.

    A fact may occur in both sides of an overlapping window.  Returning every
    matching chunk lets evaluation represent those hits as one evidence unit.
    """
    needle = _norm(fact)
    if not needle:
        return [], 0.0, "empty"
    hits = [c["chunk_id"] for c in chunks if needle in _norm(c["text"])]
    return (hits, 1.0, "exact") if hits else ([], 0.0, "unresolved")


# --------------------------------------------------------------------------- #
# §3.1 query-first selection with the 10-article budget
# --------------------------------------------------------------------------- #
def select_articles(corpus: List[dict], queries: List[dict]) -> set:
    corpus_ids = {_aid(c) for c in corpus}
    nonnull = [
        q
        for q in queries
        if q["question_type"] != "null_query"
        and q.get("evidence_list")
        and _query_article_ids(q) <= corpus_ids
    ]
    order = nonnull[:]
    random.Random(C.SEED).shuffle(order)
    selected: set = set()
    for q in order:
        arts = _query_article_ids(q)
        if len(selected | arts) <= C.ARTICLE_COUNT:
            selected |= arts
        if len(selected) >= C.ARTICLE_COUNT:
            break
    return selected


# --------------------------------------------------------------------------- #
# §6 token chunking (same 256/50/80 config)
# --------------------------------------------------------------------------- #
def token_chunk(article_key: str, rec: dict, cleaned: str) -> List[dict]:
    toks = _ENC.encode(cleaned)
    size, overlap, mn = C.CHUNK_SIZE, C.CHUNK_OVERLAP, C.MIN_CHUNK
    stride = max(1, size - overlap)
    windows, i, n = [], 0, len(toks)
    while i < n:
        j = min(i + size, n)
        windows.append((i, j))
        if j >= n:
            break
        i += stride
    if len(windows) >= 2 and (windows[-1][1] - windows[-1][0]) < mn:
        s_prev, _ = windows[-2]
        _, e_last = windows[-1]
        windows[-2] = (s_prev, e_last)
        windows.pop()
    out = []
    for ci, (s, e) in enumerate(windows):
        out.append(
            {
                "chunk_id": f"{article_key}::c{ci}",
                "parent_document_id": _aid(rec),
                "article_key": article_key,
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "category": rec.get("category", ""),
                "published_at": rec.get("published_at", ""),
                "url": _aid(rec),
                "chunk_position": ci,
                "token_start": s,
                "token_end": e,
                "n_tokens": e - s,
                "text": _ENC.decode(toks[s:e]).strip(),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Orchestration (mirrors vo_data.build_all, 10-article namespace)
# --------------------------------------------------------------------------- #
def build_all(force: bool = False) -> dict:
    if (
        not force
        and C.CHUNKS_PATH.exists()
        and C.ELIGIBLE.exists()
        and C.GOLD_MAPPING.exists()
    ):
        return json.load(open(C.SUBSET_REPORT))

    corpus = load_corpus()
    queries = load_all_queries()
    by_id = {_aid(c): c for c in corpus}
    corpus_ids = set(by_id)
    selected = select_articles(corpus, queries)
    assert len(selected) == C.ARTICLE_COUNT, f"selected {len(selected)}"

    nonnull = [
        q
        for q in queries
        if q["question_type"] != "null_query"
        and q.get("evidence_list")
        and _query_article_ids(q) <= corpus_ids
    ]

    ordered = sorted(selected)
    key_of = {aid: f"a{ai:02d}" for ai, aid in enumerate(ordered)}
    processed, chunks, boiler_total, lengths = [], [], 0, []
    for aid in ordered:
        rec = by_id[aid]
        cleaned, removed = clean_body(rec.get("body", ""))
        boiler_total += len(removed)
        lengths.append(_tok_words(cleaned))
        art_chunks = token_chunk(key_of[aid], rec, cleaned)
        chunks.extend(art_chunks)
        processed.append(
            {
                "article_id": aid,
                "article_key": key_of[aid],
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "published_at": rec.get("published_at", ""),
                "category": rec.get("category", ""),
                "url": aid,
                "original_body": rec.get("body", ""),
                "cleaned_body": cleaned,
                "removed_boilerplate": removed,
                "n_chunks": len(art_chunks),
            }
        )

    chunks_by_article = defaultdict(list)
    for c in chunks:
        chunks_by_article[c["parent_document_id"]].append(c)

    eligible, excluded, gold_rows, unresolved = [], [], [], []
    align, fuzzy_log, cross_boundary = Counter(), [], 0
    for q in nonnull:
        req = _query_article_ids(q)
        if not (req <= selected):
            if req & selected:
                excluded.append(
                    {
                        "query_id": q["query_id"],
                        "query": q["query"].strip(),
                        "exclusion_reason": "missing_required_article",
                        "missing_required_article_ids": sorted(req - selected),
                    }
                )
            continue
        fact_to_chunks, q_unres = [], False
        for k, e in enumerate(q["evidence_list"]):
            aid = e[C.ARTICLE_ID_FIELD]
            fact = e.get("fact", "")
            hits, score, how = _locate_fact(fact, chunks_by_article.get(aid, []))
            align[how] += 1
            if how == "fuzzy":
                fuzzy_log.append(
                    {
                        "query_id": q["query_id"],
                        "fact": fact[:200],
                        "score": round(score, 3),
                        "chunk": hits[0],
                    }
                )
            if len(hits) > 1:
                cross_boundary += 1
            if not hits:
                q_unres = True
                unresolved.append(
                    {
                        "query_id": q["query_id"],
                        "article_id": aid,
                        "best_score": round(score, 3),
                        "fact": fact[:250],
                    }
                )
            fact_to_chunks.append(
                {
                    "evidence_fact_id": f"{q['query_id']}::e{k}",
                    "article_id": aid,
                    "fact": fact,
                    "chunk_ids": hits,
                    "match": how,
                }
            )
        if q_unres:
            excluded.append(
                {
                    "query_id": q["query_id"],
                    "query": q["query"].strip(),
                    "exclusion_reason": "unresolved_gold_evidence",
                    "missing_required_article_ids": [],
                }
            )
            continue
        gold_chunk_ids = sorted({cid for f in fact_to_chunks for cid in f["chunk_ids"]})
        evidence_units = [sorted(set(f["chunk_ids"])) for f in fact_to_chunks]
        eligible.append(
            {
                "query_id": q["query_id"],
                "query": q["query"].strip(),
                "question_type": _short_type(q["question_type"]),
                "gold_answer": q.get("answer", ""),
                "required_article_ids": sorted(req),
                "required_evidence_fact_ids": [
                    f["evidence_fact_id"] for f in fact_to_chunks
                ],
                "n_required_documents": len(req),
                "n_required_evidence_facts": len(fact_to_chunks),
                "gold_chunk_ids": gold_chunk_ids,
                "evidence_units": evidence_units,
            }
        )
        gold_rows.append(
            {
                "query_id": q["query_id"],
                "facts": fact_to_chunks,
                "gold_chunk_ids": gold_chunk_ids,
                "evidence_units": evidence_units,
            }
        )

    # persist
    eligible_by_article = Counter()
    for q in eligible:
        for aid in q["required_article_ids"]:
            eligible_by_article[aid] += 1
    articles_out = [
        {
            "article_id": aid,
            "article_key": key_of[aid],
            "title": by_id[aid].get("title", ""),
            "source": by_id[aid].get("source", ""),
            "published_at": by_id[aid].get("published_at", ""),
            "category": by_id[aid].get("category", ""),
            "article_length_words": _tok_words(
                [p for p in processed if p["article_id"] == aid][0]["cleaned_body"]
            ),
            "n_chunks": len(chunks_by_article[aid]),
            "n_eligible_queries": eligible_by_article[aid],
        }
        for aid in ordered
    ]

    json.dump(
        {
            "note": "A closed-collection, 10-article MultiHop-RAG pilot — NOT the "
            "official full-corpus benchmark.",
            "random_seed": C.SEED,
            "selection_method": "query-first greedy fit within "
            "10-article budget (shuffle seed 42), then keep every fully-covered query",
            "num_articles": len(articles_out),
            "articles": articles_out,
            "selected_article_ids": ordered,
        },
        open(C.PILOT_ARTICLES, "w"),
        indent=2,
    )
    _write_jsonl(C.ELIGIBLE, eligible)
    _write_jsonl(C.EXCLUDED, excluded)
    _write_jsonl(C.PROCESSED_ARTICLES, processed)
    _write_jsonl(
        C.CHUNKS_PATH,
        [
            {
                k: c[k]
                for k in (
                    "chunk_id",
                    "parent_document_id",
                    "article_key",
                    "title",
                    "source",
                    "category",
                    "published_at",
                    "url",
                    "chunk_position",
                    "token_start",
                    "token_end",
                    "n_tokens",
                    "text",
                )
            }
            for c in chunks
        ],
    )
    _write_jsonl(C.GOLD_MAPPING, gold_rows)
    _write_jsonl(C.UNRESOLVED_GOLD, unresolved)

    by_type = Counter(q["question_type"] for q in eligible)
    by_doc = Counter(q["n_required_documents"] for q in eligible)
    excl_reasons = Counter(x["exclusion_reason"] for x in excluded)
    gold_sizes = [len(q["gold_chunk_ids"]) for q in eligible]
    fact_counts = [q["n_required_evidence_facts"] for q in eligible]
    gold_align = {
        "total_gold_evidence_facts": sum(align.values()),
        "exact_matched": align["exact"],
        "fuzzy_matched": align["fuzzy"],
        "unresolved": align["unresolved"] + align["empty"],
        "facts_crossing_chunk_boundaries": cross_boundary,
        "avg_gold_chunks_per_query": round(
            sum(gold_sizes) / max(len(gold_sizes), 1), 3
        ),
        "avg_evidence_facts_per_query": round(
            sum(fact_counts) / max(len(fact_counts), 1), 3
        ),
        "fuzzy_matches": fuzzy_log,
    }
    json.dump(gold_align, open(C.GOLD_REPORT, "w"), indent=2)

    report = {
        "collection_note": "A closed-collection, 10-article MultiHop-RAG pilot — NOT "
        "the official full-corpus benchmark.",
        "random_seed": C.SEED,
        "article_selection_method": "query-first greedy budget fit (seed 42)",
        "num_selected_articles": len(articles_out),
        "selected_article_ids": ordered,
        "num_fully_eligible_queries": len(eligible),
        "num_excluded_queries": len(excluded),
        "eligible_by_question_type": dict(by_type),
        "eligible_by_required_document_count": {
            str(k): v for k, v in sorted(by_doc.items())
        },
        "exclusion_reason_counts": dict(excl_reasons),
        "num_chunks": len(chunks),
        "chunks_per_article_avg": round(len(chunks) / len(articles_out), 2),
        "avg_chunk_tokens": round(sum(c["n_tokens"] for c in chunks) / len(chunks), 1),
        "chunking": {
            "chunk_size_tokens": C.CHUNK_SIZE,
            "chunk_overlap_tokens": C.CHUNK_OVERLAP,
            "minimum_chunk_tokens": C.MIN_CHUNK,
            "tokenizer": C.TOKENIZER,
        },
        "avg_article_length_words": round(sum(lengths) / len(lengths), 1),
        "removed_boilerplate_segments": boiler_total,
        "gold_alignment": {k: v for k, v in gold_align.items() if k != "fuzzy_matches"},
    }
    json.dump(report, open(C.SUBSET_REPORT, "w"), indent=2)
    return report


def load_chunks() -> List[dict]:
    return read_jsonl(C.CHUNKS_PATH)


def load_eligible_queries() -> List[dict]:
    return read_jsonl(C.ELIGIBLE)


def load_gold() -> Dict[str, dict]:
    return {g["query_id"]: g for g in read_jsonl(C.GOLD_MAPPING)}


if __name__ == "__main__":
    import pprint

    pprint.pp(build_all(force=True))
