"""
Shared helpers used across the pipeline.

Kept deliberately small and dependency-light so the preprocessing scripts can
stay focused on dataset-specific logic. Not part of the originally requested
file list, but factored out to avoid copy-pasting JSONL I/O and the LLM
enrichment helper into four near-identical preprocessors.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import config


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write an iterable of dicts to a JSONL file. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def upsert_csv(path: Path, fieldnames: Sequence[str], new_rows: List[Dict],
               group_fields: Sequence[str], rebuilt_groups: set) -> None:
    """
    Write `new_rows` to a CSV, preserving any existing rows whose group
    (tuple of group_fields, e.g. (dataset, condition)) was NOT rebuilt this run.
    Lets us update a subset of datasets without clobbering the others' logs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: List[Dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                grp = tuple(str(r.get(k)) for k in group_fields)
                if grp not in rebuilt_groups:
                    kept.append(r)
    rows = kept + new_rows
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Chunk record schema
# --------------------------------------------------------------------------- #
def make_record(
    *,
    chunk_id: str,
    dataset: str,
    input_type: str,
    condition: str,
    text_for_embedding: str,
    original_text: str,
    source_id: Optional[str] = None,
    question_id: Optional[str] = None,
    title: Optional[str] = None,
    page: Optional[str] = None,
    row_id: Optional[str] = None,
    chart_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a chunk record in the canonical JSONL schema for this project."""
    return {
        "id": chunk_id,
        "dataset": dataset,
        "input_type": input_type,
        "condition": condition,
        "text_for_embedding": text_for_embedding.strip(),
        "original_text": original_text.strip(),
        "metadata": {
            "source_id": source_id,
            "question_id": question_id,
            "title": title,
            "page": page,
            "row_id": row_id,
            "chart_id": chart_id,
        },
    }


def render_enriched(fields: Dict[str, Optional[str]]) -> str:
    """
    Render an ordered dict of label -> value into the enriched chunk format,
    skipping empty values. Preserves insertion order.
    """
    lines = []
    for label, value in fields.items():
        if value is None:
            continue
        value = str(value).strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Token estimate (cheap, no tokenizer dependency)
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """
    Rough token count (~4 chars/token heuristic). Good enough for the
    avg_tokens_per_chunk reporting field without pulling in tiktoken.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


# --------------------------------------------------------------------------- #
# LLM enrichment (summaries / keywords)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "this", "that",
    "with", "from", "they", "have", "were", "their", "which", "these", "such",
    "also", "into", "than", "then", "them", "been", "more", "most", "other",
    "some", "what", "when", "where", "while", "about", "between",
}


def cheap_keywords(text: str, top_n: int = 8) -> str:
    """
    Frequency-based keyword extraction with no API calls. Used as a fallback
    when LLM enrichment is disabled so the enriched chunks are still richer
    than baseline.
    """
    counts: Dict[str, int] = {}
    for m in _WORD_RE.finditer(text.lower()):
        w = m.group(0)
        if w in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(w for w, _ in ranked[:top_n])


def cheap_summary(text: str, max_chars: int = 240) -> str:
    """First-sentence(s) extractive summary, no API calls."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    # take whole sentences up to the budget
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(sent) > max_chars:
            break
        out = (out + " " + sent).strip()
    return out or text[:max_chars]


def llm_summary(client, text: str, kind: str = "passage", max_words: int = 40) -> str:
    """
    One-line LLM summary. `client` is an OpenAI client (or None to fall back
    to the cheap extractive summary).
    """
    if client is None:
        return cheap_summary(text)
    text = text.strip()
    if not text:
        return ""
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, factual one-line summaries used to "
                        "improve retrieval. No preamble, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize this {kind} in at most {max_words} words:\n\n{text}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Never let an enrichment hiccup kill the whole preprocessing run.
        return cheap_summary(text)


def load_beir_queries_gold(raw_dir):
    """
    Load BEIR-style queries text + gold judgments from a raw dataset dir.
    Returns (qtext: {qid: text}, gold: {qid: [corpus_id, ...]}). Gold is taken
    from the first available qrels split (test > validation > train).
    """
    qtext: Dict[str, str] = {}
    qpath = raw_dir / "queries.jsonl"
    if qpath.exists():
        for row in read_jsonl(qpath):
            qid = str(row.get("_id") or row.get("id"))
            text = str(row.get("text") or row.get("query") or "").strip()
            if text:
                qtext[qid] = text

    gold: Dict[str, List[str]] = {}
    for split in ("test", "validation", "train"):
        gp = raw_dir / f"qrels_{split}.jsonl"
        if not gp.exists():
            continue
        for r in read_jsonl(gp):
            qid = str(r.get("query-id") or r.get("query_id"))
            cid = str(r.get("corpus-id") or r.get("corpus_id"))
            try:
                score = int(r.get("score", 1))
            except (TypeError, ValueError):
                score = 1
            if score > 0:
                gold.setdefault(qid, []).append(cid)
        break
    return qtext, gold


def select_eval_subset(qtext, gold, corpus_ids, n_queries: int, n_distractors: int):
    """
    Build a small but valid cross-document retrieval task:
      - pick up to n_queries queries that have gold docs present in the corpus,
      - the index must contain every selected query's gold docs (so retrieval is
        answerable) plus up to n_distractors non-gold distractor docs.

    Returns (selected, index_ids):
      selected  = [(qid, text, [gold_ids_in_corpus]), ...]
      index_ids = set of corpus ids to index (gold ∪ distractors)
    """
    # Preserve a deterministic order: iterate the corpus in its on-disk order
    # (do NOT iterate a set — set ordering is randomized per process and would
    # make baseline vs enriched index different docs).
    corpus_order = list(corpus_ids)
    corpus_set = set(corpus_order)
    selected = []
    required: set = set()
    for qid, cids in gold.items():
        if qid not in qtext:
            continue
        in_corpus = [c for c in cids if c in corpus_set]
        if not in_corpus:
            continue
        selected.append((qid, qtext[qid], in_corpus))
        required.update(in_corpus)
        if len(selected) >= n_queries:
            break

    distractors = []
    for cid in corpus_order:
        if cid not in required:
            distractors.append(cid)
            if len(distractors) >= n_distractors:
                break

    index_ids = set(required) | set(distractors)
    return selected, index_ids


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Sentence-aware character chunking for long documents (used by NFCorpus).
    Returns a list with at least one element.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    cur = ""
    for sent in sentences:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            # start next chunk with a bit of overlap for context continuity
            tail = cur[-overlap:] if overlap and cur else ""
            cur = (tail + " " + sent).strip()
    if cur:
        chunks.append(cur)
    return chunks
