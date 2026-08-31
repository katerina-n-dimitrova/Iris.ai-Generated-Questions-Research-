"""
Reusable building blocks for the follow-up "question quality" experiments.

Everything here is retrieval/analysis logic with NO new LLM calls, so it can be
re-run cheaply and re-used across conditions. The heavy retrieval primitives
(DenseIndex, BM25Index, _embed, _enriched, _tok, _query_metrics, RRF_K) are
imported from `hybrid_doc2query_experiment` (H) so the follow-up study shares the
exact same embedder, tokeniser, RRF constant and metric definitions as the
report it extends.

Contents
--------
* extract_terms(chunk, idf)      lexical anchors for BM25-aware prompting
* build_idf(texts)               corpus IDF used both for anchors and filtering
* filter_questions(...)          keep answerable / specific / non-duplicate qs
* QuestionIndex                  dense index over generated questions -> parent
* eval_fusion(...)               RRF over an arbitrary list of ranker callables
* CrossEncoderReranker           first-stage top-N -> cross-encoder -> top-k
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore")

import hybrid_doc2query_experiment as H
from spiqa_eval import _query_metrics

RRF_K = H.RRF_K
_tok = H._tok
_embed = H._embed
_ranks_from_scores = H._ranks_from_scores

# very small English stop set — enough to keep lexical anchors content-bearing
_STOP = set(
    """a an the of to in on for and or is are was were be been being this that these those
with without from into as at by it its their our your his her they we you i he she them us
which who whom whose what when where why how than then so such not no yes can could should would
may might will shall do does did done has have had using use used based via per about over under
between among within across figure table fig tab paper section shows show shown results result""".split()
)

_ACRONYM = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?)\b")
_METRICNUM = re.compile(r"\b\d+(?:\.\d+)?\s?%?|\b[A-Z]{2,}@?\d*\b")


# --------------------------------------------------------------------------- #
# Corpus statistics
# --------------------------------------------------------------------------- #
def build_idf(texts: Sequence[str]) -> Dict[str, float]:
    """Standard smoothed IDF over the chunk corpus (token -> idf)."""
    n = len(texts)
    df: Counter = Counter()
    for t in texts:
        for tok in set(_tok(t)):
            df[tok] += 1
    return {tok: math.log((n + 1) / (dfi + 0.5)) for tok, dfi in df.items()}


# --------------------------------------------------------------------------- #
# 2 · BM25-aware lexical anchor extraction
# --------------------------------------------------------------------------- #
def extract_terms(
    chunk: Dict, idf: Dict[str, float], *, top_idf: int = 8, max_terms: int = 18
) -> List[str]:
    """
    Pull the lexical anchors a BM25-aware question should try to include:
    section-title words, figure/table labels, caption keywords, metric/number
    tokens, dataset/method/entity names (capitalised / acronym tokens), and the
    highest-IDF content words of the chunk. Order = most distinctive first;
    de-duplicated case-insensitively.
    """
    md = chunk.get("metadata", {})
    text = chunk.get("text", "")
    terms: List[str] = []
    seen = set()

    def add(t: str):
        t = t.strip().strip(".,;:()[]\"'")
        key = t.lower()
        if len(t) < 2 or key in seen or key in _STOP:
            return
        seen.add(key)
        terms.append(t)

    # section title words + figure/table label + caption
    for w in _tok(md.get("section_heading", "")):
        add(w)
    fid = md.get("fig_id", "")
    if fid:
        # e.g. ".../9-Table1-1.png" -> "Table1"
        m = re.search(r"([A-Za-z]+\d+)", fid.rsplit("/", 1)[-1])
        if m:
            add(m.group(1))
    caption = md.get("caption", "")

    # acronyms / capitalised entities (dataset / method / model names)
    for m in _ACRONYM.findall(caption + " " + text[:1200]):
        if (
            any(c.isupper() for c in m[1:])
            or m.isupper()
            or (m[0].isupper() and len(m) > 2)
        ):
            add(m)

    # metric / numeric anchors
    for m in _METRICNUM.findall(caption + " " + text[:800]):
        add(m)

    # caption keywords (high-idf content words in the caption)
    cap_toks = [w for w in _tok(caption) if w not in _STOP and len(w) > 2]
    for w in sorted(set(cap_toks), key=lambda w: -idf.get(w, 0.0))[:6]:
        add(w)

    # highest-IDF content words of the whole chunk
    body = [w for w in _tok(text) if w not in _STOP and len(w) > 2]
    for w in sorted(set(body), key=lambda w: -idf.get(w, 0.0))[:top_idf]:
        add(w)

    return terms[:max_terms]


# --------------------------------------------------------------------------- #
# 3 · Question filtering (answerable / specific / non-duplicate / grounded)
# --------------------------------------------------------------------------- #
_GENERIC_PATTERNS = [
    re.compile(r"^what (is|are) (the )?(main )?(purpose|goal|topic|focus|idea|point)"),
    re.compile(r"^what (does|do) (the|this) (passage|paper|section|text|authors?)"),
    re.compile(r"^what (is|are) (discussed|described|presented|mentioned|shown)"),
    re.compile(r"^(what|which) (is|are) (this|the passage|the text) about"),
    re.compile(r"^what can (be|we) (concluded|inferred|learned)"),
]


def _content_tokens(text: str) -> set:
    return {w for w in _tok(text) if w not in _STOP and len(w) > 2}


def _is_generic(q: str) -> bool:
    ql = q.strip().lower()
    return any(p.search(ql) for p in _GENERIC_PATTERNS)


def _near_duplicate(q: str, kept: List[str], thr: float = 0.86) -> bool:
    ql = q.lower()
    for k in kept:
        if SequenceMatcher(None, ql, k.lower()).ratio() >= thr:
            return True
    return False


def filter_questions(
    chunk_text: str, questions: List[str], *, min_overlap: int = 2, keep: int = 10
) -> Dict:
    """
    Heuristic (non-LLM) quality filter. Keep a question only if it:
      * shares >= min_overlap content tokens with the chunk (answerable / grounded,
        i.e. no unsupported new entities dominating it),
      * is not one of the generic templated forms,
      * is not a near-duplicate of an already-kept question.
    Returns {kept, dropped_generic, dropped_ungrounded, dropped_duplicate}.
    The round-trip retrieval check is applied separately (H.roundtrip_keep) by the
    orchestrator because it needs the embedded chunk matrix.
    """
    chunk_tokens = _content_tokens(chunk_text)
    kept: List[str] = []
    dg = du = dd = 0
    for q in questions:
        if len(kept) >= keep:
            break
        if _is_generic(q):
            dg += 1
            continue
        qt = _content_tokens(q)
        if len(qt & chunk_tokens) < min_overlap:
            du += 1
            continue
        if _near_duplicate(q, kept):
            dd += 1
            continue
        kept.append(q)
    return {
        "kept": kept,
        "dropped_generic": dg,
        "dropped_ungrounded": du,
        "dropped_duplicate": dd,
    }


# --------------------------------------------------------------------------- #
# 4 · Separate question index (question vectors -> parent chunk)
# --------------------------------------------------------------------------- #
class QuestionIndex:
    """
    Dense index built ONLY over generated questions. Each question vector is
    tagged with its parent chunk. At query time we score every question, take the
    BEST (max) similarity per parent chunk, and rank parents by that score. This
    keeps generated questions in a SEPARATE representation instead of polluting
    the original chunk vector.
    """

    def __init__(self, chunk_ids: List[str], questions_by_chunk: Dict[str, List[str]]):
        self.chunk_ids = chunk_ids
        self.idx_of = {c: i for i, c in enumerate(chunk_ids)}
        flat_q, owner = [], []
        for cid in chunk_ids:
            for q in questions_by_chunk.get(cid, []):
                flat_q.append(q)
                owner.append(self.idx_of[cid])
        self.n_questions = len(flat_q)
        self.owner = np.asarray(owner, dtype=np.int64)
        self.qvecs = _embed(flat_q) if flat_q else np.zeros((0, 1), np.float32)

    def scores(self, qv: np.ndarray) -> np.ndarray:
        """Best question similarity per parent chunk (0 if a chunk has none)."""
        out = np.zeros(len(self.chunk_ids), dtype=np.float32)
        if self.n_questions == 0:
            return out
        sims = self.qvecs @ qv
        # scatter-max into parents
        np.maximum.at(out, self.owner, sims)
        return out

    def ranks(self, qv: np.ndarray) -> np.ndarray:
        return _ranks_from_scores(self.scores(qv))

    def size_mb(self) -> float:
        # float32 vectors; report the raw vector payload in MB
        return round(self.qvecs.nbytes / 1048576, 3) if self.n_questions else 0.0


# --------------------------------------------------------------------------- #
# Generic RRF fusion evaluation over arbitrary rankers
# --------------------------------------------------------------------------- #
# A ranker is a callable(question_text, query_vec_or_None) -> ranks aligned to
# chunk_ids (1 = best). Wrap dense / bm25 / question indexes into this shape.
Ranker = Callable[[str, Optional[np.ndarray]], np.ndarray]


def dense_ranker(index) -> Ranker:
    return lambda q, qv: index.ranks(qv)


def bm25_ranker(index) -> Ranker:
    return lambda q, qv: index.ranks(q)


def qindex_ranker(index: "QuestionIndex") -> Ranker:
    return lambda q, qv: index.ranks(qv)


def eval_fusion(
    name: str,
    chunk_ids: List[str],
    queries: List[Dict],
    k_values: List[int],
    rankers: List[Ranker],
    *,
    needs_qv: bool = True,
) -> Dict:
    """RRF-fuse the given rankers. Mirrors H.eval_condition's metric handling."""
    from embeddings import get_embedder

    emb = get_embedder()
    per_query, lat = [], []
    maxk = max(k_values)
    for q in queries:
        t0 = time.perf_counter()
        qv = None
        if needs_qv:
            qv = H._san(np.asarray(emb.embed_query(q["question"]), np.float32))
        rrf = np.zeros(len(chunk_ids), dtype=np.float64)
        for r in rankers:
            ranks = r(q["question"], qv)
            rrf += 1.0 / (RRF_K + ranks)
        order = np.argsort(-rrf)[:maxk]
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = [chunk_ids[i] for i in order]
        m = _query_metrics(ranked, q["gold_chunk_ids"], k_values)
        per_query.append(m)

    return _aggregate(name, per_query, lat, k_values)


def eval_rerank(
    name: str,
    chunk_ids: List[str],
    queries: List[Dict],
    k_values: List[int],
    first_stage: List[Ranker],
    text_by_id: Dict[str, str],
    reranker: "CrossEncoderReranker",
    *,
    first_top_n: int = 50,
) -> Dict:
    """
    First-stage RRF fusion -> top-N candidate parent chunks -> cross-encoder
    rerank(query, chunk_text) -> final ranking. Latency includes rerank time.
    """
    from embeddings import get_embedder

    emb = get_embedder()
    per_query, lat = [], []
    for q in queries:
        t0 = time.perf_counter()
        qv = H._san(np.asarray(emb.embed_query(q["question"]), np.float32))
        rrf = np.zeros(len(chunk_ids), dtype=np.float64)
        for r in first_stage:
            rrf += 1.0 / (RRF_K + r(q["question"], qv))
        cand_idx = list(np.argsort(-rrf)[:first_top_n])
        cand_ids = [chunk_ids[i] for i in cand_idx]
        pairs = [(q["question"], text_by_id[c]) for c in cand_ids]
        scores = reranker.score(pairs)
        reordered = [cand_ids[i] for i in np.argsort(-np.asarray(scores))]
        lat.append((time.perf_counter() - t0) * 1000)
        m = _query_metrics(reordered, q["gold_chunk_ids"], k_values)
        per_query.append(m)
    return _aggregate(name, per_query, lat, k_values)


def _aggregate(
    name: str, per_query: List[Dict], lat: List[float], k_values: List[int]
) -> Dict:
    def avg(key):
        return round(sum(x[key] for x in per_query) / max(len(per_query), 1), 4)

    def p95(v):
        v = sorted(v)
        return (
            round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 3) if v else 0.0
        )

    metrics = {f"hit@{k}": avg(f"hit@{k}") for k in k_values}
    metrics["mrr"] = avg("mrr")
    metrics["ndcg@10"] = avg("ndcg@10")
    return {"condition": name, "metrics": metrics, "search_p95_ms": p95(lat)}


# --------------------------------------------------------------------------- #
# 5 · Cross-encoder reranker (graceful, cached model)
# --------------------------------------------------------------------------- #
class CrossEncoderReranker:
    """Thin wrapper around a sentence-transformers CrossEncoder.

    Loads once; if the model cannot be loaded (offline / missing), `available`
    is False and callers should skip the rerank conditions rather than crash.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.available = False
        self.model = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"  [reranker] unavailable ({model_name}): {e}", flush=True)

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not self.available or not pairs:
            return [0.0] * len(pairs)
        return list(self.model.predict(pairs, batch_size=64, show_progress_bar=False))
