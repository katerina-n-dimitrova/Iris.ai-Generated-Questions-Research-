"""
Stage: dataset prep for the dense-vector-only 15-article MultiHop-RAG pilot.

Pipeline (all deterministic, seed-fixed, cached to disk):
  1. Load corpus.json (609 articles) + MultiHopRAG.json (2,556 queries).
  2. Query-first selection of EXACTLY 15 articles (§3): greedily accept queries
     whose evidence articles fit the 15-article budget, then keep every query
     fully covered by the resulting set. Random article samples leave almost no
     answerable query because evidence spans 2-4 articles.
  3. Eligible queries (§4) = fully covered; excluded = touch the set but miss an
     article. Gold answers / evidence are used ONLY here for label building.
  4. Light, documented cleaning of the 15 selected articles (§5).
  5. Token-based chunking 256/50/80 with a short-tail merge (§6).
  6. Gold evidence fact -> chunk alignment: normalized substring (primary) then
     high-threshold fuzzy; overlap-aware so a fact in two overlapping chunks is
     ONE evidence unit (§14).

NB: benchmark query text, gold answers, and gold evidence NEVER enter an index —
they are used only to (a) pick the closed 15-article collection and (b) build
retrieval labels after the fact.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken

import vo_config as C

_WS = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")
_SENT = re.compile(r"(?<=[.!?])\s+")
GOLD_FUZZY_THRESHOLD = 0.90

# Light boilerplate patterns (case-insensitive, whole-line-ish) — §5.
_BOILERPLATE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*sign up for (our|the) newsletter.*$",
        r"^\s*subscribe to .{0,60}newsletter.*$",
        r"^\s*table of contents\s*$",
        r"^\s*loading\.{0,3}\s*$",
        r"^\s*advertisement\s*$",
        r"^\s*share this article.*$",
        r"^\s*follow us on .*$",
    ]
]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def _tok_words(text: str) -> int:
    return len((text or "").split())


# --------------------------------------------------------------------------- #
# Load corpus + queries
# --------------------------------------------------------------------------- #
def load_corpus() -> List[dict]:
    if not C.CORPUS_JSON.exists():
        raise FileNotFoundError(
            f"{C.CORPUS_JSON} missing. Download corpus.json + MultiHopRAG.json "
            "from HuggingFace yixuantt/MultiHopRAG into data/raw/multihoprag/."
        )
    return _load_json(C.CORPUS_JSON)


def load_all_queries() -> List[dict]:
    rows = _load_json(C.QUERIES_JSON)
    for i, q in enumerate(rows):
        q["query_id"] = f"q{i:05d}"
    return rows


def _aid(rec: dict) -> str:
    return rec[C.ARTICLE_ID_FIELD]


def _short_type(t: str) -> str:
    return t.replace("_query", "")


def _query_article_ids(q: dict) -> set:
    return {e[C.ARTICLE_ID_FIELD] for e in q.get("evidence_list", [])}


# --------------------------------------------------------------------------- #
# §3 — query-first selection of exactly 15 articles
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
# §5 — light cleaning
# --------------------------------------------------------------------------- #
def clean_body(body: str) -> Tuple[str, List[str]]:
    """Return (cleaned_text, removed_segments). HTML stripped, whitespace
    collapsed within paragraphs, blank-line paragraph structure preserved, and
    a short list of boilerplate lines removed (recorded for the report)."""
    text = _HTML.sub(" ", body or "")
    removed: List[str] = []
    paras = re.split(r"\n\s*\n", text)
    kept_paras = []
    for p in paras:
        lines = []
        for ln in p.splitlines():
            stripped = ln.strip()
            if any(rx.match(stripped) for rx in _BOILERPLATE):
                removed.append(stripped)
                continue
            lines.append(ln)
        para = _WS.sub(" ", " ".join(lines)).strip()
        if para:
            kept_paras.append(para)
    return "\n\n".join(kept_paras), removed


# --------------------------------------------------------------------------- #
# §6 — token-based chunking (tiktoken), overlap + short-tail merge
# --------------------------------------------------------------------------- #
_ENC = tiktoken.get_encoding(C.TOKENIZER)


def token_chunk(article_key: str, rec: dict, cleaned: str) -> List[dict]:
    toks = _ENC.encode(cleaned)
    size, overlap, mn = C.CHUNK_SIZE, C.CHUNK_OVERLAP, C.MIN_CHUNK
    stride = max(1, size - overlap)

    windows: List[Tuple[int, int]] = []
    i = 0
    n = len(toks)
    while i < n:
        j = min(i + size, n)
        windows.append((i, j))
        if j >= n:
            break
        i += stride
    # Merge a short trailing window into the previous one (avoid tiny tail chunks).
    if len(windows) >= 2 and (windows[-1][1] - windows[-1][0]) < mn:
        s_prev, _ = windows[-2]
        _, e_last = windows[-1]
        windows[-2] = (s_prev, e_last)
        windows.pop()

    chunks = []
    for ci, (s, e) in enumerate(windows):
        text = _ENC.decode(toks[s:e]).strip()
        chunks.append(
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
                "text": text,
            }
        )
    return chunks


# --------------------------------------------------------------------------- #
# §14 — gold evidence fact -> chunk alignment
# --------------------------------------------------------------------------- #
def _locate_fact(fact: str, chunks: List[dict]) -> Tuple[List[str], float, str]:
    nf = _norm(fact)
    if not nf:
        return [], 0.0, "empty"
    hits = [c["chunk_id"] for c in chunks if nf in _norm(c["text"])]
    if hits:
        return hits, 1.0, "exact"
    best_id, best = None, 0.0
    for c in chunks:
        for sent in _SENT.split(c["text"]):
            r = SequenceMatcher(None, nf, _norm(sent)).ratio()
            if r > best:
                best, best_id = r, c["chunk_id"]
    if best >= GOLD_FUZZY_THRESHOLD and best_id is not None:
        return [best_id], best, "fuzzy"
    return [], best, "unresolved"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_all(force: bool = False) -> dict:
    if (
        not force
        and C.CHUNKS_PATH.exists()
        and C.PILOT_ELIGIBLE.exists()
        and C.GOLD_MAPPING.exists()
    ):
        return _load_json(C.PILOT_REPORT)

    corpus = load_corpus()
    queries = load_all_queries()
    by_id = {_aid(c): c for c in corpus}
    corpus_ids = set(by_id)

    selected = select_articles(corpus, queries)
    assert len(selected) == C.ARTICLE_COUNT, (
        f"selected {len(selected)} != {C.ARTICLE_COUNT}"
    )

    nonnull = [
        q
        for q in queries
        if q["question_type"] != "null_query"
        and q.get("evidence_list")
        and _query_article_ids(q) <= corpus_ids
    ]

    # Clean + chunk the 15 selected articles (deterministic order = sorted url).
    ordered = sorted(selected)
    key_of = {aid: f"a{ai:02d}" for ai, aid in enumerate(ordered)}
    processed_articles: List[dict] = []
    chunks: List[dict] = []
    boiler_total = 0
    lengths = []
    for aid in ordered:
        rec = by_id[aid]
        cleaned, removed = clean_body(rec.get("body", ""))
        boiler_total += len(removed)
        lengths.append(_tok_words(cleaned))
        art_chunks = token_chunk(key_of[aid], rec, cleaned)
        chunks.extend(art_chunks)
        processed_articles.append(
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
                "cleaning_actions": [
                    "strip_html",
                    "collapse_whitespace",
                    "remove_boilerplate_lines",
                ],
                "n_words_cleaned": _tok_words(cleaned),
                "n_chunks": len(art_chunks),
            }
        )

    chunks_by_article: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_article[c["parent_document_id"]].append(c)

    # §4 + §14 — eligible / excluded queries, gold alignment.
    eligible: List[dict] = []
    excluded: List[dict] = []
    gold_rows: List[dict] = []
    unresolved: List[dict] = []
    align = Counter()
    fuzzy_log: List[dict] = []
    cross_boundary = 0

    for q in nonnull:
        req = _query_article_ids(q)
        if not (req <= selected):
            # only record queries that actually touch the collection
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

        # Build gold chunk labels per evidence fact.
        fact_to_chunks: List[dict] = []
        q_unresolved = False
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
                q_unresolved = True
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

        if q_unresolved:
            excluded.append(
                {
                    "query_id": q["query_id"],
                    "query": q["query"].strip(),
                    "exclusion_reason": "unresolved_gold_evidence",
                    "missing_required_article_ids": [],
                }
            )
            continue

        # Unique gold chunks (overlap-aware: a fact in 2 chunks stays 1 unit).
        gold_chunk_ids = sorted({cid for f in fact_to_chunks for cid in f["chunk_ids"]})
        # evidence-fact-set: each fact is one unit, represented by its chunk set.
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

    # ----- persist artifacts ---------------------------------------------- #
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
                [p for p in processed_articles if p["article_id"] == aid][0][
                    "cleaned_body"
                ]
            ),
            "n_chunks": len(chunks_by_article[aid]),
            "n_eligible_queries": eligible_by_article[aid],
        }
        for aid in ordered
    ]

    with C.PILOT_ARTICLES.open("w") as fh:
        json.dump(
            {
                "note": "SELECTED 15-article MultiHop-RAG PILOT collection — NOT the "
                "official full-corpus benchmark.",
                "random_seed": C.SEED,
                "article_id_field": C.ARTICLE_ID_FIELD,
                "selection_method": "query-first greedy fit within 15-article budget "
                "(shuffle seed 42), then keep every fully-covered query",
                "num_articles": len(articles_out),
                "articles": articles_out,
                "selected_article_ids": ordered,
            },
            fh,
            indent=2,
        )

    _write_jsonl(C.PILOT_ELIGIBLE, eligible)
    _write_jsonl(C.PILOT_EXCLUDED, excluded)
    _write_jsonl(C.PROCESSED_ARTICLES, processed_articles)
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
    by_doccount = Counter(q["n_required_documents"] for q in eligible)
    excl_reasons = Counter(x["exclusion_reason"] for x in excluded)

    preprocess_report = {
        "num_selected_articles": len(articles_out),
        "avg_article_length_words": round(sum(lengths) / len(lengths), 1),
        "min_article_length_words": min(lengths),
        "max_article_length_words": max(lengths),
        "removed_boilerplate_segments": boiler_total,
        "unusually_short_articles": [
            a["article_id"] for a in articles_out if a["article_length_words"] < 120
        ],
    }
    with C.PREPROCESS_REPORT.open("w") as fh:
        json.dump(preprocess_report, fh, indent=2)

    gold_sizes = [len(q["gold_chunk_ids"]) for q in eligible]
    fact_counts = [q["n_required_evidence_facts"] for q in eligible]
    gold_align_report = {
        "total_gold_evidence_facts": sum(align.values()),
        "exact_matched": align["exact"],
        "fuzzy_matched": align["fuzzy"],
        "unresolved": align["unresolved"] + align["empty"],
        "facts_crossing_chunk_boundaries": cross_boundary,
        "queries_removed_unresolved_mapping": excl_reasons.get(
            "unresolved_gold_evidence", 0
        ),
        "avg_gold_chunks_per_query": round(
            sum(gold_sizes) / max(len(gold_sizes), 1), 3
        ),
        "avg_evidence_facts_per_query": round(
            sum(fact_counts) / max(len(fact_counts), 1), 3
        ),
        "fuzzy_matches": fuzzy_log,
    }
    with C.GOLD_REPORT.open("w") as fh:
        json.dump(gold_align_report, fh, indent=2)

    report = {
        "collection_note": "SELECTED 15-article MultiHop-RAG pilot collection — "
        "NOT the official full-corpus MultiHop-RAG benchmark.",
        "random_seed": C.SEED,
        "article_selection_method": "query-first greedy budget fit (seed 42)",
        "num_selected_articles": len(articles_out),
        "selected_article_ids": ordered,
        "num_fully_eligible_queries": len(eligible),
        "num_excluded_queries": len(excluded),
        "eligible_by_question_type": dict(by_type),
        "eligible_by_required_document_count": {
            str(k): v for k, v in sorted(by_doccount.items())
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
        "preprocessing": preprocess_report,
        "gold_alignment": {
            k: v for k, v in gold_align_report.items() if k != "fuzzy_matches"
        },
    }
    with C.PILOT_REPORT.open("w") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_chunks() -> List[dict]:
    return read_jsonl(C.CHUNKS_PATH)


def load_eligible_queries() -> List[dict]:
    return read_jsonl(C.PILOT_ELIGIBLE)


def load_gold() -> Dict[str, dict]:
    return {g["query_id"]: g for g in read_jsonl(C.GOLD_MAPPING)}


if __name__ == "__main__":
    import pprint

    pprint.pp(build_all(force=True))
