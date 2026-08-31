"""
QASPER dataset loading, chunking, query + gold-label construction.

Source
------
Canonical AllenAI QASPER v0.3 dev split (``qasper-dev-v0.3.json``), a dict keyed
by arXiv paper id. Each paper has ``title``, ``abstract``, ``full_text`` (list of
``{section_name, paragraphs:[str]}``) and ``qas`` (questions, each with one
answer per annotator).

What this module produces (all reproducible, cached to disk)
------------------------------------------------------------
* selected_papers.json : the fixed 15-paper sample (seeded), for reproducibility.
* chunks.jsonl         : one chunk per kept paragraph (+ the abstract) pooled
                         across all selected papers (cross-paper retrieval).
                         Figures/tables and math-or-markup-heavy scraps dropped.
* queries.jsonl        : the human QASPER questions for those papers, skipping
                         unanswerable ones and any whose gold evidence does not
                         land on a kept text chunk. Each query carries:
                           - gold_chunk_ids : union of evidence paragraphs across
                             annotators, mapped to chunk ids (relevance labels);
                           - answer_type    : extractive / abstractive / boolean
                             (majority over annotators), for the per-type breakdown.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import qasper_config as C

_WS = re.compile(r"\s+")
# LaTeX / math markup indicators, for scoring how formula-heavy a paper is.
_MATH = re.compile(
    r"\$[^$]*\$|\\frac|\\sum|\\sqrt|\\alpha|\\beta|\\theta|\\lambda|\\mathbf|"
    r"\\mathcal|\\times|\\leq|\\geq|\^\{|_\{|\\begin\{"
)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Normalise for evidence<->paragraph matching (whitespace + case)."""
    return _WS.sub(" ", text or "").strip().lower()


def _is_float_evidence(s: str) -> bool:
    """QASPER marks figure/table evidence with a 'FLOAT SELECTED:' prefix."""
    return s.strip().startswith("FLOAT SELECTED")


def _alpha_ratio(text: str) -> float:
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    alpha = sum(c.isalpha() for c in non_space)
    return alpha / len(non_space)


def keep_paragraph(text: str) -> bool:
    """Keep body text; drop empties, figure/table markers, and math/markup scraps."""
    t = (text or "").strip()
    if not t or _is_float_evidence(t):
        return False
    if len(t.split()) < C.MIN_CHUNK_WORDS:
        return False
    if _alpha_ratio(t) < C.MIN_ALPHA_RATIO:
        return False
    return True


# --------------------------------------------------------------------------- #
# Load + select papers
# --------------------------------------------------------------------------- #
def _load_json(path) -> Dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_dev() -> Dict[str, dict]:
    if not C.DEV_JSON.exists():
        raise FileNotFoundError(
            f"{C.DEV_JSON} not found. Download qasper-train-dev-v0.3.tgz from "
            "the AllenAI QASPER release into data/raw/qasper/ and extract it."
        )
    return _load_json(C.DEV_JSON)


def load_pool() -> Dict[str, dict]:
    """Pooled train+dev papers (dev wins on the rare id collision), each tagged
    with its source split. QASPER papers are float-heavy, so a text-only sample
    needs the larger pool to find enough qualifying papers."""
    pool: Dict[str, dict] = {}
    if C.TRAIN_JSON.exists():
        for pid, p in _load_json(C.TRAIN_JSON).items():
            p["_split"] = "train"
            pool[pid] = p
    for pid, p in load_dev().items():
        p["_split"] = "dev"
        pool[pid] = p
    return pool


def formula_score(paper: dict) -> float:
    """Fraction of paragraphs that are math/formula-heavy (>=2 LaTeX markers)."""
    paras = [q for s in paper.get("full_text", []) for q in s.get("paragraphs", [])]
    if not paras:
        return 0.0
    return sum(1 for x in paras if len(_MATH.findall(x)) >= 2) / len(paras)


def _usable_text_query_count(pid: str, paper: dict) -> int:
    """Number of answerable questions with at least one kept TEXT gold chunk."""
    _, norm = build_corpus([pid], {pid: paper})
    qs, _, _ = build_queries([pid], {pid: paper}, norm)
    return len(qs)


def select_text_only_papers(pool: Dict[str, dict]) -> List[str]:
    """Papers that are pure text: no figures/tables, formula-free, and enough
    answerable text questions. Deterministic (criteria-based, sorted)."""
    keep = []
    for pid, p in pool.items():
        if len(p.get("figures_and_tables") or []) > C.TEXT_ONLY_MAX_FLOATS:
            continue
        if formula_score(p) > C.TEXT_ONLY_MAX_FORMULA:
            continue
        if _usable_text_query_count(pid, p) >= C.TEXT_ONLY_MIN_QUERIES:
            keep.append(pid)
    return sorted(keep)


def select_text_answerable_papers(dev: Dict[str, dict]) -> List[str]:
    """Papers where the retrieval TASK is text-only: formula-light and with enough
    fully-text-answerable questions (float paragraphs are dropped from the corpus,
    and float-dependent questions are excluded by build_queries). Qualifying set is
    deterministic; NUM_PAPERS are then seed-sampled from it."""
    qualifying = []
    for pid, p in dev.items():
        if formula_score(p) > C.TEXT_ANSWERABLE_MAX_FORMULA:
            continue
        if _usable_text_query_count(pid, p) >= C.TEXT_ANSWERABLE_MIN_QUERIES:
            qualifying.append(pid)
    qualifying.sort()
    if C.NUM_PAPERS <= 0 or C.NUM_PAPERS >= len(qualifying):
        return qualifying
    return sorted(random.Random(C.SELECTION_SEED).sample(qualifying, C.NUM_PAPERS))


def select_papers(pool: Optional[Dict[str, dict]] = None) -> List[str]:
    """Select paper ids per SELECTION_MODE and cache the list (reproducible)."""
    pool = (
        pool
        if pool is not None
        else (load_pool() if C.SELECTION_MODE == "text_only" else load_dev())
    )
    if C.SELECTION_MODE == "text_only":
        selected = select_text_only_papers(pool)
        criteria = {
            "max_floats": C.TEXT_ONLY_MAX_FLOATS,
            "max_formula_fraction": C.TEXT_ONLY_MAX_FORMULA,
            "min_text_queries": C.TEXT_ONLY_MIN_QUERIES,
        }
    elif C.SELECTION_MODE == "text_answerable":
        selected = select_text_answerable_papers(pool)
        criteria = {
            "max_formula_fraction": C.TEXT_ANSWERABLE_MAX_FORMULA,
            "min_text_queries": C.TEXT_ANSWERABLE_MIN_QUERIES,
            "seed": C.SELECTION_SEED,
            "num_papers": C.NUM_PAPERS,
            "note": "corpus=text-only paragraphs; only fully-text-answerable "
            "questions evaluated (float-dependent questions dropped)",
        }
    else:
        all_ids = sorted(pool.keys())
        if C.NUM_PAPERS <= 0 or C.NUM_PAPERS >= len(all_ids):
            selected = all_ids
        else:
            selected = sorted(
                random.Random(C.SELECTION_SEED).sample(all_ids, C.NUM_PAPERS)
            )
        criteria = {"seed": C.SELECTION_SEED, "num_papers": C.NUM_PAPERS}
    with C.SELECTED_PAPERS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "selection_mode": C.SELECTION_MODE,
                "criteria": criteria,
                "num_papers": len(selected),
                "paper_ids": selected,
                "paper_splits": {
                    pid: pool[pid].get("_split", "dev") for pid in selected
                },
            },
            fh,
            indent=2,
        )
    return selected


# --------------------------------------------------------------------------- #
# Chunk building
# --------------------------------------------------------------------------- #
def _paper_chunks(paper_id: str, paper: dict):
    """Yield (chunk_dict_or_None, raw_text, kept_bool) for abstract + every
    paragraph, so callers can build a gold index over *all* paragraphs while
    only keeping the retained ones as corpus chunks."""
    # Abstract as a chunk source.
    abstract = (paper.get("abstract") or "").strip()
    kept = keep_paragraph(abstract)
    chunk = None
    if kept:
        chunk = {
            "chunk_id": f"{paper_id}::abstract",
            "paper_id": paper_id,
            "section": "Abstract",
            "kind": "abstract",
            "text": abstract,
        }
    yield chunk, abstract, kept

    for si, sec in enumerate(paper.get("full_text", [])):
        sec_name = sec.get("section_name") or f"Section {si}"
        for pi, para in enumerate(sec.get("paragraphs", [])):
            raw = (para or "").strip()
            kept = keep_paragraph(raw)
            chunk = None
            if kept:
                chunk = {
                    "chunk_id": f"{paper_id}::s{si}::p{pi}",
                    "paper_id": paper_id,
                    "section": sec_name,
                    "kind": "paragraph",
                    "text": raw,
                }
            yield chunk, raw, kept


def build_corpus(paper_ids: List[str], dev: Dict[str, dict]):
    """Return (chunks, norm_index) where norm_index maps normalised paragraph
    text -> (chunk_id or None, kept). Used to resolve evidence to chunks and to
    diagnose evidence that fell on filtered/figure paragraphs."""
    chunks: List[dict] = []
    norm_index: Dict[str, Tuple[Optional[str], bool]] = {}
    for pid in paper_ids:
        for chunk, raw, kept in _paper_chunks(pid, dev[pid]):
            key = _norm(raw)
            if key and key not in norm_index:
                norm_index[key] = (chunk["chunk_id"] if chunk else None, kept)
            if chunk is not None:
                chunks.append(chunk)
    return chunks, norm_index


# --------------------------------------------------------------------------- #
# Answer type + gold evidence
# --------------------------------------------------------------------------- #
def _annotator_type(ans: dict) -> str:
    """Classify one annotator's answer into QASPER's answer-type buckets."""
    if ans.get("unanswerable"):
        return "unanswerable"
    if ans.get("yes_no") is not None:
        return "boolean"
    if ans.get("extractive_spans"):
        return "extractive"
    if (ans.get("free_form_answer") or "").strip():
        return "abstractive"
    return "unanswerable"


# Tie-break priority when annotators disagree on answer type.
_TYPE_PRIORITY = {"extractive": 0, "abstractive": 1, "boolean": 2}


def _resolve_answer_type(types: List[str]) -> Optional[str]:
    """Majority vote over annotators, ignoring 'unanswerable'. None if all
    annotators marked the question unanswerable."""
    votes = [t for t in types if t != "unanswerable"]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    winners = [t for t, n in counts.items() if n == top]
    return min(winners, key=lambda t: _TYPE_PRIORITY.get(t, 9))


def _gold_for_question(qa: dict, norm_index):
    """Union of evidence paragraphs across annotators mapped to chunk ids.

    Returns (gold_chunk_ids, diag) where diag counts evidence strings that hit a
    kept chunk / a filtered-or-figure paragraph / nothing at all."""
    gold, diag = set(), Counter()
    for ann in qa.get("answers", []):
        ans = ann.get("answer", {})
        for ev in ans.get("evidence", []):
            if _is_float_evidence(ev):
                diag["float"] += 1
                continue
            hit = norm_index.get(_norm(ev))
            if hit is None:
                diag["unmatched"] += 1
            elif hit[0] is not None:
                gold.add(hit[0])
                diag["kept"] += 1
            else:
                diag["filtered"] += 1
    return sorted(gold), diag


def build_queries(paper_ids: List[str], dev: Dict[str, dict], norm_index):
    """Build eval queries. Skip unanswerable questions and questions whose gold
    evidence does not land on any kept text chunk. In text_only/text_answerable
    modes also drop questions that rely on any figure/table evidence, so every
    evaluated question is fully text-answerable."""
    queries: List[dict] = []
    skipped = Counter()
    ev_diag = Counter()
    for pid in paper_ids:
        for qa in dev[pid].get("qas", []):
            types = [
                _annotator_type(a.get("answer", {})) for a in qa.get("answers", [])
            ]
            atype = _resolve_answer_type(types)
            gold, diag = _gold_for_question(qa, norm_index)
            ev_diag.update(diag)
            if atype is None:
                skipped["unanswerable"] += 1
                continue
            if C.REQUIRE_FULLY_TEXT_QUERIES and diag.get("float", 0) > 0:
                skipped["needs_figure_or_table"] += 1
                continue
            if not gold:
                skipped["no_text_gold"] += 1
                continue
            queries.append(
                {
                    "query_id": qa["question_id"],
                    "paper_id": pid,
                    "question": qa["question"].strip(),
                    "answer_type": atype,
                    "gold_chunk_ids": gold,
                    "n_gold": len(gold),
                    "n_annotators": len(qa.get("answers", [])),
                }
            )
    return queries, skipped, ev_diag


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
    """Build (or load) the corpus + queries and return a summary dict."""
    if not force and C.CHUNKS_PATH.exists() and C.QUERIES_PATH.exists():
        chunks, queries = load_chunks(), load_queries()
    else:
        pool = load_pool() if C.SELECTION_MODE == "text_only" else load_dev()
        paper_ids = select_papers(pool)
        chunks, norm_index = build_corpus(paper_ids, pool)
        queries, skipped, ev_diag = build_queries(paper_ids, pool, norm_index)
        _write_jsonl(C.CHUNKS_PATH, chunks)
        _write_jsonl(C.QUERIES_PATH, queries)
        summary = _summarize(paper_ids, chunks, queries)
        summary["skipped_questions"] = dict(skipped)
        summary["evidence_matching"] = dict(ev_diag)
        with (C.PROCESSED_DIR / "dataset_summary.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2)
        return summary
    return _summarize(
        json.load(C.SELECTED_PAPERS_PATH.open())["paper_ids"], chunks, queries
    )


def _summarize(paper_ids, chunks, queries) -> dict:
    by_type = Counter(q["answer_type"] for q in queries)
    gold_sizes = [q["n_gold"] for q in queries]
    return {
        "num_papers": len(paper_ids),
        "num_chunks": len(chunks),
        "num_abstract_chunks": sum(1 for c in chunks if c["kind"] == "abstract"),
        "num_queries": len(queries),
        "queries_by_answer_type": dict(by_type),
        "avg_gold_per_query": round(sum(gold_sizes) / max(len(gold_sizes), 1), 3),
        "max_gold_per_query": max(gold_sizes, default=0),
        "chunks_per_paper_avg": round(len(chunks) / max(len(paper_ids), 1), 1),
    }


if __name__ == "__main__":
    import pprint

    s = build_all(force=True)
    pprint.pp(s)
