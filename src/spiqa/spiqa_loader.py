"""
SPIQA dataset loader + schema normaliser.

SPIQA ships three *different* per-split JSON schemas. This module downloads only
the small split JSONs (never the full train JSON or the multi-GB image zips) and
normalises whichever schema into one unified in-memory structure the rest of the
pipeline consumes:

    Paper
      paper_id : str
      split    : str
      title    : str
      abstract : str | ""
      sections : List[Section]           # section_name + ordered paragraphs
      figures  : List[Figure]            # id, caption, kind (figure/table), type
      questions: List[QAItem]            # question, answer, evidence text,
                                         # referred figure/table ids (gold)

Schemas handled
---------------
test-B : flat — passages[] (paragraphs, no section names), all_figures_tables{},
         figure_types{}, parallel Q lists (question/composition/evidential_info/
         referred_figures_tables/Is_figure|table_in_evidence).
test-C : arxiv_id, paper_title, abstract, full_text[]{section_name,paragraphs[]},
         figures_and_tables[]{file,caption}, question[], answer[]{free_form_answer,
         evidence,...}, referred_figures_tables[].
test-A / val : paper_id, all_figures{fname:{caption,content_type,figure_type}},
         qa[]{question,answer,explanation}. No body text (image-QA split) and no
         per-question gold figure ids — supported as caption-only (documented).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import spiqa_config as C


# --------------------------------------------------------------------------- #
# Unified data model
# --------------------------------------------------------------------------- #
@dataclass
class Figure:
    fig_id: str  # the image filename key used across SPIQA
    caption: str
    kind: str  # "table" or "figure"
    fig_type: str = ""  # schematic / plot / photograph / table / ...


@dataclass
class Section:
    name: str
    paragraphs: List[str]


@dataclass
class QAItem:
    question_id: str
    question: str
    answer: str
    evidence_text: str  # supporting context text (may be "")
    referred_figures: List[str]  # gold figure/table ids (may be [])
    question_type: str = ""
    section: str = ""


@dataclass
class Paper:
    paper_id: str
    split: str
    title: str
    abstract: str
    sections: List[Section]
    figures: List[Figure]
    questions: List[QAItem]

    @property
    def has_body_text(self) -> bool:
        return any(s.paragraphs for s in self.sections)


# --------------------------------------------------------------------------- #
# Download (small splits only)
# --------------------------------------------------------------------------- #
def download_split(split: str) -> Path:
    """Download one split JSON into RAW_DIR (cached). Returns the local path."""
    if split not in C.SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choices: {list(C.SPLIT_FILES)}")
    local = C.RAW_DIR / Path(C.SPLIT_FILES[split]).name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(C.HF_REPO_ID, C.SPLIT_FILES[split], repo_type="dataset")
    shutil.copy(src, local)
    return local


# --------------------------------------------------------------------------- #
# Schema normalisers
# --------------------------------------------------------------------------- #
def _fig_kind(fig_type: str, fig_id: str) -> str:
    t = (fig_type or "").lower()
    if "table" in t or "table" in fig_id.lower():
        return "table"
    return "figure"


def _normalise_testB(pid: str, p: dict, split: str) -> Paper:
    figs_types = p.get("figure_types", {}) or {}
    figures = [
        Figure(
            fig_id=fid,
            caption=str(cap or ""),
            kind=_fig_kind(figs_types.get(fid, ""), fid),
            fig_type=str(figs_types.get(fid, "")),
        )
        for fid, cap in (p.get("all_figures_tables", {}) or {}).items()
    ]
    # test-B has no section names — treat each passage as its own paragraph in a
    # single unnamed section (chunker will merge them by token budget).
    sections = [
        Section(name="", paragraphs=[str(x) for x in p.get("passages", []) if x])
    ]

    questions: List[QAItem] = []
    q = p.get("question", []) or []
    for i in range(len(q)):
        ev = p.get("evidential_info", [])
        ev_text = ""
        if i < len(ev) and isinstance(ev[i], list):
            ev_text = " ".join(
                str(e.get("context", "")) for e in ev[i] if isinstance(e, dict)
            )
        ref = p.get("referred_figures_tables", [])
        refs = ref[i] if i < len(ref) and isinstance(ref[i], list) else []
        questions.append(
            QAItem(
                question_id=str(p.get("question_id", [i])[i])
                if i < len(p.get("question_id", []))
                else str(i),
                question=str(q[i]),
                answer=str(p.get("composition", [""] * len(q))[i])
                if i < len(p.get("composition", []))
                else "",
                evidence_text=ev_text,
                referred_figures=[str(r) for r in refs],
                question_type=str(p.get("question_type", [""] * len(q))[i])
                if i < len(p.get("question_type", []))
                else "",
                section=str(p.get("question_section", [""] * len(q))[i])
                if i < len(p.get("question_section", []))
                else "",
            )
        )
    return Paper(
        paper_id=pid,
        split=split,
        title=str(p.get("title", "")),
        abstract="",
        sections=sections,
        figures=figures,
        questions=questions,
    )


def _normalise_testC(pid: str, p: dict, split: str) -> Paper:
    figures = [
        Figure(
            fig_id=str(ft.get("file", "")),
            caption=str(ft.get("caption", "")),
            kind=_fig_kind(str(ft.get("content_type", "")), str(ft.get("file", ""))),
            fig_type=str(ft.get("content_type", "")),
        )
        for ft in (p.get("figures_and_tables", []) or [])
        if ft.get("file")
    ]
    sections = [
        Section(
            name=str(sec.get("section_name", "")),
            paragraphs=[str(x) for x in sec.get("paragraphs", []) if x],
        )
        for sec in (p.get("full_text", []) or [])
    ]
    questions: List[QAItem] = []
    q = p.get("question", []) or []
    for i in range(len(q)):
        ans = p.get("answer", [])
        ans_i = ans[i] if i < len(ans) else {}
        if isinstance(ans_i, dict):
            answer = str(ans_i.get("free_form_answer", "") or "")
            evd = ans_i.get("evidence", [])
            ev_text = (
                " ".join(str(e) for e in evd) if isinstance(evd, list) else str(evd)
            )
        else:
            answer, ev_text = str(ans_i), ""
        ref = p.get("referred_figures_tables", [])
        refs = ref[i] if i < len(ref) and isinstance(ref[i], list) else []
        questions.append(
            QAItem(
                question_id=f"{pid}_q{i}",
                question=str(q[i]),
                answer=answer,
                evidence_text=ev_text,
                referred_figures=[str(r) for r in refs],
            )
        )
    return Paper(
        paper_id=pid,
        split=split,
        title=str(p.get("paper_title", "")),
        abstract=str(p.get("abstract", "")),
        sections=sections,
        figures=figures,
        questions=questions,
    )


def _normalise_testA_val(pid: str, p: dict, split: str) -> Paper:
    figures = []
    for fid, meta in (p.get("all_figures", {}) or {}).items():
        meta = meta or {}
        figures.append(
            Figure(
                fig_id=str(fid),
                caption=str(meta.get("caption", "")),
                kind=_fig_kind(str(meta.get("content_type", "")), str(fid)),
                fig_type=str(meta.get("figure_type", "")),
            )
        )
    questions = []
    for i, qa in enumerate(p.get("qa", []) or []):
        questions.append(
            QAItem(
                question_id=f"{pid}_q{i}",
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                evidence_text=str(qa.get("explanation", "")),
                referred_figures=[],
            )
        )
    # These splits carry no body text — only figure captions.
    return Paper(
        paper_id=pid,
        split=split,
        title="",
        abstract="",
        sections=[],
        figures=figures,
        questions=questions,
    )


def _detect_and_normalise(pid: str, p: dict, split: str) -> Paper:
    if "passages" in p:
        return _normalise_testB(pid, p, split)
    if "full_text" in p:
        return _normalise_testC(pid, p, split)
    if "qa" in p and "all_figures" in p:
        return _normalise_testA_val(pid, p, split)
    raise ValueError(f"Unrecognised SPIQA paper schema for {pid}: keys={list(p)[:8]}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_papers(
    split: str = C.DEFAULT_SPLIT, max_papers: Optional[int] = None
) -> List[Paper]:
    """Download (if needed) and load a split into normalised Paper objects."""
    path = download_split(split)
    raw = json.load(open(path, encoding="utf-8"))
    papers: List[Paper] = []
    for pid, p in raw.items():
        papers.append(_detect_and_normalise(pid, p, split))
        if max_papers and len(papers) >= max_papers:
            break
    return papers


def save_samples(papers: List[Paper], n: int = 3) -> Path:
    """Write a few normalised examples to disk for manual inspection."""
    out = (
        C.PROCESSED_DIR / f"{papers[0].split}_samples.json"
        if papers
        else C.PROCESSED_DIR / "samples.json"
    )
    sample = [asdict(p) for p in papers[:n]]
    json.dump(sample, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def split_stats(papers: List[Paper]) -> dict:
    return {
        "split": papers[0].split if papers else None,
        "num_papers": len(papers),
        "num_questions": sum(len(p.questions) for p in papers),
        "num_figures": sum(len(p.figures) for p in papers),
        "num_paragraphs": sum(len(s.paragraphs) for p in papers for s in p.sections),
        "papers_with_body_text": sum(1 for p in papers if p.has_body_text),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load + inspect a SPIQA split.")
    ap.add_argument("--split", default=C.DEFAULT_SPLIT, choices=list(C.SPLIT_FILES))
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    papers = load_papers(args.split, args.max_papers)
    print("STATS:", json.dumps(split_stats(papers), indent=2))
    out = save_samples(papers, args.samples)
    print("Saved samples ->", out)
