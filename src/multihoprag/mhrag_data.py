"""
MultiHop-RAG dataset loading, query-first selection, chunking, gold labelling.

Source (HuggingFace ``yixuantt/MultiHopRAG``)
--------------------------------------------
* corpus.json      : list of 609 news articles, each with title / author / source
                     / published_at / category / url / body.
* MultiHopRAG.json : ~2,556 queries, each with query / answer / question_type
                     (inference_query | comparison_query | temporal_query |
                     null_query) / evidence_list. Every evidence item carries the
                     source article's title+url and a ``fact`` snippet (the gold
                     evidence). Each non-null query cites 2-4 distinct articles.

What this module produces (all reproducible, cached to disk)
------------------------------------------------------------
* selected_articles.json  : the query-first article sample (see below).
* selected_query_ids.json : the kept query ids.
* chunks.jsonl            : paragraph chunks of the selected articles ONLY.
* queries.jsonl           : the kept queries, each with:
    - gold_chunk_ids : the chunks containing its evidence facts (relevance labels;
      queries are multi-evidence -> 2-4 gold chunks, possibly across articles);
    - query_type     : inference / comparison / temporal (for per-type breakdown).

Query-first selection (see mhrag_config)
----------------------------------------
Random articles would leave almost no fully-answerable query (evidence spans 2-4
articles). So: drop null queries, shuffle (seed), greedily accept a query iff its
evidence articles fit within ARTICLE_BUDGET, until the budget is hit; then keep
every remaining query whose evidence articles are all already selected.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mhrag_config as C

_WS = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")
_PARA_SPLIT = re.compile(r"\n\s*\n")
# Sentence-ish splitter, only used for fuzzy gold matching inside a chunk.
_SENT = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Normalise for evidence<->chunk matching (whitespace + case)."""
    return _WS.sub(" ", text or "").strip().lower()


def clean_body(body: str) -> str:
    """Strip residual HTML/markup; keep paragraph structure (blank lines).

    Conservative on purpose: MultiHop-RAG evidence 'fact' snippets are verbatim
    substrings of the body, so we only remove HTML tags and collapse intra-line
    whitespace -- never drop prose (which could orphan a gold snippet)."""
    text = _HTML.sub(" ", body or "")
    # normalise whitespace WITHIN each paragraph but preserve blank-line breaks
    paras = _PARA_SPLIT.split(text)
    cleaned = [_WS.sub(" ", p).strip() for p in paras]
    return "\n\n".join(p for p in cleaned if p)


def _tok_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Load corpus + queries
# --------------------------------------------------------------------------- #
def _load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus() -> List[dict]:
    if not C.CORPUS_JSON.exists():
        raise FileNotFoundError(
            f"{C.CORPUS_JSON} not found. Download corpus.json + MultiHopRAG.json "
            "from HuggingFace yixuantt/MultiHopRAG into data/raw/multihoprag/."
        )
    return _load_json(C.CORPUS_JSON)


def load_all_queries() -> List[dict]:
    """All raw queries, each tagged with a stable ``query_id`` = its file index."""
    rows = _load_json(C.QUERIES_JSON)
    for i, q in enumerate(rows):
        q["query_id"] = f"q{i:05d}"
    return rows


def _article_id(record: dict) -> str:
    return record[C.ARTICLE_ID_FIELD]


def _short_type(question_type: str) -> str:
    """'inference_query' -> 'inference'."""
    return question_type.replace("_query", "")


# --------------------------------------------------------------------------- #
# Query-first article selection (reproducible)
# --------------------------------------------------------------------------- #
def _query_article_ids(q: dict) -> set:
    return {e[C.ARTICLE_ID_FIELD] for e in q.get("evidence_list", [])}


def select_articles_and_queries(corpus: List[dict], queries: List[dict]):
    """Return (selected_article_ids:set, kept_queries:list) per the query-first
    algorithm. Deterministic given ARTICLE_BUDGET + SELECTION_SEED."""
    corpus_ids = {_article_id(c) for c in corpus}
    nonnull = [
        q
        for q in queries
        if q["question_type"] != "null_query"
        and q.get("evidence_list")
        and _query_article_ids(q) <= corpus_ids
    ]

    order = nonnull[:]
    random.Random(C.SELECTION_SEED).shuffle(order)

    selected: set = set()
    for q in order:
        arts = _query_article_ids(q)
        if len(selected | arts) <= C.ARTICLE_BUDGET:
            selected |= arts
        if len(selected) >= C.ARTICLE_BUDGET:
            break

    # Second pass: keep every non-null query fully covered by the selected set.
    kept = [q for q in nonnull if _query_article_ids(q) <= selected]
    return selected, kept


# --------------------------------------------------------------------------- #
# Chunking (paragraph split + short-paragraph merge)
# --------------------------------------------------------------------------- #
def chunk_article(article_key: str, record: dict) -> List[dict]:
    """Split a cleaned body into paragraph chunks, merging paragraphs shorter
    than MERGE_MIN_TOKENS into a neighbour. Returns chunk dicts with metadata."""
    body = clean_body(record.get("body", ""))
    paras = [p for p in _PARA_SPLIT.split(body) if p.strip()]

    merged: List[str] = []
    for p in paras:
        if merged and (
            _tok_count(p) < C.MERGE_MIN_TOKENS
            or _tok_count(merged[-1]) < C.MERGE_MIN_TOKENS
        ):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)

    chunks = []
    for ci, text in enumerate(merged):
        chunks.append(
            {
                "chunk_id": f"{article_key}::c{ci}",
                "article_id": _article_id(record),
                "article_key": article_key,
                "title": record.get("title", ""),
                "source": record.get("source", ""),
                "category": record.get("category", ""),
                "published_at": record.get("published_at", ""),
                "text": text,
            }
        )
    return chunks


def build_corpus(selected_ids: set, corpus: List[dict]) -> List[dict]:
    """Chunk every selected article. article_key = a{n} in selection order (for
    short, stable chunk ids); selection order = sorted article id for determinism."""
    by_id = {_article_id(c): c for c in corpus}
    ordered_ids = sorted(selected_ids)
    chunks: List[dict] = []
    for ai, aid in enumerate(ordered_ids):
        chunks.extend(chunk_article(f"a{ai:02d}", by_id[aid]))
    return chunks


# --------------------------------------------------------------------------- #
# Gold evidence -> chunk labelling
# --------------------------------------------------------------------------- #
def _locate_fact(fact: str, chunks: List[dict]) -> Tuple[List[str], float]:
    """Return (matching_chunk_ids, best_score) for one evidence fact among the
    chunks of ITS article. Substring match (score 1.0) preferred; else fuzzy over
    sentences with best ratio >= threshold."""
    nf = _norm(fact)
    if not nf:
        return [], 0.0
    hits = [c["chunk_id"] for c in chunks if nf in _norm(c["text"])]
    if hits:
        return hits, 1.0
    best_id, best = None, 0.0
    for c in chunks:
        for sent in _SENT.split(c["text"]):
            r = SequenceMatcher(None, nf, _norm(sent)).ratio()
            if r > best:
                best, best_id = r, c["chunk_id"]
    if best >= C.GOLD_FUZZY_THRESHOLD and best_id is not None:
        return [best_id], best
    return [], best


def build_queries(kept: List[dict], chunks: List[dict]):
    """Attach gold_chunk_ids to each kept query. Log any evidence fact that fails
    to land in exactly one chunk (a chunking bug, not to be ignored)."""
    by_article: Dict[str, List[dict]] = {}
    for c in chunks:
        by_article.setdefault(c["article_id"], []).append(c)

    out: List[dict] = []
    diag = Counter()
    unmatched_log: List[dict] = []
    multi_log: List[dict] = []

    for q in kept:
        gold: set = set()
        for e in q["evidence_list"]:
            aid = e[C.ARTICLE_ID_FIELD]
            fact = e.get("fact", "")
            hits, score = _locate_fact(fact, by_article.get(aid, []))
            if not hits:
                diag["unmatched"] += 1
                unmatched_log.append(
                    {
                        "query_id": q["query_id"],
                        "article_id": aid,
                        "best_score": round(score, 3),
                        "fact": fact[:200],
                    }
                )
            else:
                if len(hits) > 1:
                    diag["multi_chunk"] += 1
                    multi_log.append(
                        {
                            "query_id": q["query_id"],
                            "article_id": aid,
                            "n_chunks": len(hits),
                            "fact": fact[:200],
                        }
                    )
                diag["matched"] += 1
                gold.update(hits)
        if not gold:
            diag["queries_without_gold"] += 1
            continue
        out.append(
            {
                "query_id": q["query_id"],
                "question": q["query"].strip(),
                "answer": q.get("answer", ""),
                "query_type": _short_type(q["question_type"]),
                "gold_chunk_ids": sorted(gold),
                "n_gold": len(gold),
                "n_evidence": len(q["evidence_list"]),
            }
        )
    return out, diag, unmatched_log, multi_log


# --------------------------------------------------------------------------- #
# Orchestration + IO
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_chunks() -> List[dict]:
    return _read_jsonl(C.CHUNKS_PATH)


def load_queries() -> List[dict]:
    return _read_jsonl(C.QUERIES_PATH)


def build_all(force: bool = False) -> dict:
    """Build (or load) the corpus + queries + gold and return a summary dict."""
    if not force and C.CHUNKS_PATH.exists() and C.QUERIES_PATH.exists():
        return _summarize(
            load_chunks(), load_queries(), _load_json(C.SELECTED_ARTICLES_PATH)
        )

    corpus = load_corpus()
    queries = load_all_queries()
    selected_ids, kept = select_articles_and_queries(corpus, queries)
    chunks = build_corpus(selected_ids, corpus)
    eval_queries, diag, unmatched_log, multi_log = build_queries(kept, chunks)

    # Persist reproducibility artifacts.
    by_id = {_article_id(c): c for c in corpus}
    articles_meta = {
        "article_budget": C.ARTICLE_BUDGET,
        "selection_seed": C.SELECTION_SEED,
        "article_id_field": C.ARTICLE_ID_FIELD,
        "num_articles": len(selected_ids),
        "article_ids": sorted(selected_ids),
        "articles": [
            {
                "article_id": aid,
                "title": by_id[aid]["title"],
                "source": by_id[aid]["source"],
                "category": by_id[aid]["category"],
            }
            for aid in sorted(selected_ids)
        ],
    }
    json.dump(articles_meta, C.SELECTED_ARTICLES_PATH.open("w"), indent=2)
    json.dump(
        [q["query_id"] for q in eval_queries],
        C.SELECTED_QUERIES_PATH.open("w"),
        indent=2,
    )
    _write_jsonl(C.CHUNKS_PATH, chunks)
    _write_jsonl(C.QUERIES_PATH, eval_queries)

    summary = _summarize(chunks, eval_queries, articles_meta)
    summary["evidence_matching"] = dict(diag)
    summary["unmatched_evidence"] = unmatched_log
    summary["multi_chunk_evidence"] = multi_log
    with (C.PROCESSED_DIR / "dataset_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    if unmatched_log:
        print(
            f"[data] WARNING: {len(unmatched_log)} evidence facts unmatched "
            "(logged in dataset_summary.json -> chunking bug to fix)."
        )
    if multi_log:
        print(
            f"[data] NOTE: {len(multi_log)} evidence facts matched >1 chunk "
            "(logged); gold uses all matches."
        )
    return summary


def _summarize(chunks, eval_queries, articles_meta) -> dict:
    by_type = Counter(q["query_type"] for q in eval_queries)
    gold_sizes = [q["n_gold"] for q in eval_queries]
    chunk_words = [_tok_count(c["text"]) for c in chunks]
    n_articles = articles_meta.get(
        "num_articles", len(articles_meta.get("article_ids", []))
    )
    return {
        "num_articles": n_articles,
        "num_chunks": len(chunks),
        "chunks_per_article_avg": round(len(chunks) / max(n_articles, 1), 1),
        "chunk_words_avg": round(sum(chunk_words) / max(len(chunk_words), 1), 1),
        "num_queries": len(eval_queries),
        "queries_by_type": dict(by_type),
        "avg_gold_per_query": round(sum(gold_sizes) / max(len(gold_sizes), 1), 3),
        "min_gold_per_query": min(gold_sizes, default=0),
        "max_gold_per_query": max(gold_sizes, default=0),
    }


if __name__ == "__main__":
    import pprint

    s = build_all(force=True)
    pprint.pp(
        {
            k: v
            for k, v in s.items()
            if k not in ("unmatched_evidence", "multi_chunk_evidence")
        }
    )
