"""
Config for the DocBank generated-question-enrichment retrieval experiment.

Research question
-----------------
Does generated-question-only retrieval work better on structured/layout-heavy
scientific text from DocBank, and does increasing the number of generated
questions per chunk (q5/q10/q13/q15) improve retrieval enough to justify the
extra generation and embedding cost?

Dataset access (small sample only — never the full 500K pages)
--------------------------------------------------------------
DocBank ships token-level layout annotations. The redistributable parquet mirror
(maveriq/DocBank) is token-level and ANONYMISED (no document/page id), so pages
cannot be grouped into documents there. The original release
(liminghao1630/DocBank) stores per-page .txt files whose FILENAMES encode the
arXiv document id + page number, e.g.
``1.tar_1401.0001.gz_infoingames..._arxiv_24.txt``. We stream only the zip
central directory + the handful of small .txt files for our 15 selected
documents via HTTP range requests (remotezip) — a few MB, never the 3.17 GB zip.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

_SRC_DIR = Path(__file__).resolve().parent.parent
_PEERQA_DIR = _SRC_DIR / "peerqa"  # reuse doc2query gen + metric helpers
for _p in (str(_SRC_DIR), str(_PEERQA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as base_config  # noqa: E402

PROJECT_ROOT = base_config.PROJECT_ROOT
RAW_DIR = base_config.RAW_DIR / "docbank"
TXT_DIR = RAW_DIR / "txt"  # cached per-page .txt files
PROCESSED_DIR = base_config.PROCESSED_DIR / "docbank"
RESULTS_DIR = base_config.RESULTS_DIR / "docbank"
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "docbank"

for _d in (RAW_DIR, TXT_DIR, PROCESSED_DIR, RESULTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset source
# --------------------------------------------------------------------------- #
TXT_ZIP_URL = (
    "https://huggingface.co/datasets/liminghao1630/DocBank/"
    "resolve/main/DocBank_500K_txt.zip"
)
NAMES_CACHE = RAW_DIR / "txt_filenames.txt"  # cached zip namelist

NUM_DOCS = int(os.getenv("DOCBANK_NUM_DOCS", "15"))
# Select documents with a page count in this band (substantive but manageable).
MIN_PAGES = int(os.getenv("DOCBANK_MIN_PAGES", "6"))
MAX_PAGES = int(os.getenv("DOCBANK_MAX_PAGES", "8"))

# DocBank layout labels -> coarse source/layout type used in the experiment.
LAYOUT_MAP = {
    "title": "title",
    "section": "section",
    "author": "paragraph",
    "abstract": "paragraph",
    "paragraph": "paragraph",
    "list": "paragraph",
    "table": "table",
    "equation": "equation",
    "caption": "caption",
    "figure": "figure",
    "reference": "reference",
    "footer": "paragraph",
    "date": "paragraph",
}
# Non-text placeholder tokens DocBank emits for graphical elements.
PLACEHOLDER_TOKENS = {
    "##LTLine##",
    "##LTFigure##",
    "##LTImage##",
    "##LTRect##",
    "##LTCurve##",
    "##LTChar##",
}

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE_TOKENS = int(os.getenv("DOCBANK_CHUNK_SIZE", "500"))
CHUNK_MAX_TOKENS = int(os.getenv("DOCBANK_CHUNK_MAX", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("DOCBANK_CHUNK_OVERLAP", "100"))
TIKTOKEN_ENCODING = "cl100k_base"

# --------------------------------------------------------------------------- #
# Question generation / conditions
# --------------------------------------------------------------------------- #
QUESTION_CONDITIONS: List[int] = [
    int(x) for x in os.getenv("DOCBANK_Q_CONDITIONS", "0,5,10,13,15").split(",")
]
MAX_QUESTIONS = max(QUESTION_CONDITIONS)
GEN_CONDITIONS = [n for n in QUESTION_CONDITIONS if n > 0]

# Synthetic eval QA per document.
QA_PER_DOC = int(os.getenv("DOCBANK_QA_PER_DOC", "8"))

LLM_MODEL = os.getenv("DOCBANK_LLM_MODEL", base_config.OPENAI_CHAT_MODEL)
LLM_TEMPERATURE = float(os.getenv("DOCBANK_LLM_TEMPERATURE", "0.3"))
LLM_WORKERS = int(os.getenv("DOCBANK_LLM_WORKERS", "12"))

TOP_K = 10
K_VALUES = sorted({1, 5, 10, TOP_K})

PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def condition_name(n: int) -> str:
    return "baseline" if n == 0 else f"q{n}"


def collection_name(n: int) -> str:
    return f"docbank_{condition_name(n)}"
