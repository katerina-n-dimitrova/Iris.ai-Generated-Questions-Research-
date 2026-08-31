"""
Central configuration for the PeerQA "generated-questions-only retrieval"
experiment.

Research question
-----------------
If we index ONLY LLM-generated questions per chunk (doc2query), never the chunk
text itself, how well do the generated questions retrieve the correct evidence
chunks for the real PeerQA reviewer questions — and how does the number of
generated questions (5 vs 10 vs 13) trade off retrieval quality against
generation cost, embedding count, and index storage?

Indexing strategy (questions-only doc2query, multi-vector)
----------------------------------------------------------
* Unlike the SPIQA sibling experiment, the original chunk vector is NOT stored.
  Each condition's collection holds ONLY generated-question embeddings.
* Each generated question is one embedding, linked back to its ``parent_chunk_id``
  via metadata. At retrieval time a question hit is resolved to its parent chunk
  and the *chunk text* is returned as the evidence (never the question).
* A separate ``baseline`` collection embeds the chunk TEXT (one vector/chunk,
  standard dense RAG) purely as the reference point for "does questions-only
  help vs. embedding the text".

Dataset note
------------
PeerQA (Baumgärtner et al., 2025; arXiv:2502.13668) bundles reviewer questions +
author-annotated sentence-level evidence over scientific papers. The
redistributable ``papers.jsonl`` carries full sentence-segmented text for the
nlpeer + egu corpora (90 papers); the openreview papers (ICLR/NeurIPS) are only
available via the ``papers-all`` config which fetches full text from
arXiv/OpenReview at load time. This experiment uses the text-available papers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Make the parent src/ importable so we reuse the repo's battle-tested
# ``config`` (paths/dotenv/chroma) and ``embeddings`` modules.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as base_config  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (isolated under the shared data/results trees)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = base_config.PROJECT_ROOT

RAW_DIR = base_config.RAW_DIR / "peerqa"
PROCESSED_DIR = base_config.PROCESSED_DIR / "peerqa"
RESULTS_DIR = base_config.RESULTS_DIR / "peerqa"

# Dedicated LOCAL (on-disk) Chroma dir — kept local regardless of the shared
# .env CHROMA_MODE=cloud so we can measure per-condition on-disk index size.
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "peerqa"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset source
# --------------------------------------------------------------------------- #
# Official PeerQA data release (redistributable subset: qa.jsonl + papers.jsonl).
DATA_ZIP_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/"
    "peerqa-data-v1.0.zip?sequence=5&isAllowed=y"
)

# How many papers to use in the subset run (spec: 10 if slow, 15 if manageable).
NUM_PAPERS = int(os.getenv("PEERQA_NUM_PAPERS", "15"))

# --------------------------------------------------------------------------- #
# Chunking config — token-budgeted packing of the pre-segmented sentences.
# --------------------------------------------------------------------------- #
CHUNK_SIZE_TOKENS = int(os.getenv("PEERQA_CHUNK_SIZE", "500"))  # target
CHUNK_MAX_TOKENS = int(os.getenv("PEERQA_CHUNK_MAX", "600"))  # hard cap
CHUNK_OVERLAP_TOKENS = int(os.getenv("PEERQA_CHUNK_OVERLAP", "100"))
TIKTOKEN_ENCODING = os.getenv("PEERQA_TIKTOKEN_ENCODING", "cl100k_base")

# --------------------------------------------------------------------------- #
# Question-generation config
# --------------------------------------------------------------------------- #
# Conditions = number of generated questions per chunk. We generate the MAX (13)
# once per chunk and take the first-k subset for 5 and 10 (clean count ablation,
# isolates the number of questions, ~1 LLM call/chunk). 0 == chunk-text baseline.
QUESTION_CONDITIONS: List[int] = [
    int(x) for x in os.getenv("PEERQA_Q_CONDITIONS", "0,5,10,13").split(",")
]
MAX_QUESTIONS = max(QUESTION_CONDITIONS)  # 13
GEN_QUESTION_CONDITIONS = [n for n in QUESTION_CONDITIONS if n > 0]

LLM_MODEL = os.getenv("PEERQA_LLM_MODEL", base_config.OPENAI_CHAT_MODEL)
LLM_TEMPERATURE = float(os.getenv("PEERQA_LLM_TEMPERATURE", "0.3"))
LLM_MAX_WORKERS = int(os.getenv("PEERQA_LLM_WORKERS", "8"))

# --------------------------------------------------------------------------- #
# Retrieval / evaluation config
# --------------------------------------------------------------------------- #
TOP_K = int(os.getenv("PEERQA_TOP_K", "10"))
K_VALUES = sorted({1, 5, 10, TOP_K})


# --------------------------------------------------------------------------- #
# Condition <-> collection-name helpers
# --------------------------------------------------------------------------- #
def condition_name(n_questions: int) -> str:
    return "baseline" if n_questions == 0 else f"q{n_questions}"


def collection_name(n_questions: int) -> str:
    """Chroma collection name, e.g. 'peerqa_q10' / 'peerqa_baseline'."""
    return f"peerqa_{condition_name(n_questions)}"


# Estimated OpenAI pricing (USD per 1M tokens) for the cost column.
PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
