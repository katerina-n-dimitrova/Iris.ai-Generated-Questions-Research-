"""
Central configuration for the SPIQA "number of generated questions per chunk"
experiment.

Research question
-----------------
How does the number of generated questions per chunk affect retrieval quality
and latency in a RAG system?

We compare 5 conditions on the SPIQA dataset:

    baseline : 0 generated questions (original chunk vector only)
    q1       : 1  generated question  per chunk (multi-vector)
    q10      : 10 generated questions per chunk
    q50      : 50 generated questions per chunk
    q100     : 100 generated questions per chunk

Indexing strategy (multi-vector doc2query)
------------------------------------------
* The original chunk is always stored with its own embedding.
* Each generated question is stored as its *own* embedding, linked back to the
  original ``chunk_id`` via metadata.
* At retrieval time a question hit is mapped back to its parent chunk, and the
  *original chunk text* is used as the retrieved evidence (never the question).

Everything the pipeline needs to agree on lives here. Every important knob is
overridable from the environment (``SPIQA_*`` vars) or via CLI flags in
``run_experiment.py`` so nothing is hard-coded in the logic modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Make the parent src/ importable so we can reuse the existing, battle-tested
# ``config`` (paths, dotenv loading, chroma client) and ``embeddings`` modules
# rather than duplicating them.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as base_config  # noqa: E402  (re-exports dotenv-loaded env)

# --------------------------------------------------------------------------- #
# Paths (isolated under the shared data/results trees)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = base_config.PROJECT_ROOT

RAW_DIR = base_config.RAW_DIR / "spiqa"  # downloaded split JSONs
PROCESSED_DIR = base_config.PROCESSED_DIR / "spiqa"  # chunks, questions, samples
RESULTS_DIR = base_config.RESULTS_DIR / "spiqa"  # metrics, latency, tables

# Dedicated persistent Chroma dir for this experiment. We deliberately keep this
# LOCAL (on-disk) even if the shared .env sets CHROMA_MODE=cloud, because Stage 8
# needs to measure on-disk index size / storage overhead, which is only possible
# with a local PersistentClient.
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "spiqa"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset / split config
# --------------------------------------------------------------------------- #
HF_REPO_ID = "google/spiqa"

# Only the small splits — we never pull the full train JSON (208 MB) or the
# multi-GB image zips. Filenames within the HF dataset repo:
SPLIT_FILES = {
    "test-B": "test-B/SPIQA_testB.json",  # 65 papers  — richest (default)
    "val": "train_val/SPIQA_val.json",  # 200 papers — figure-QA (caption only)
    "test-A": "test-A/SPIQA_testA.json",  # 118 papers — figure-QA (caption only)
    "test-C": "test-C/SPIQA_testC.json",  # 314 papers — full_text + figures
}

# Default split to run first (small + fully text-bearing + gold evidence labels).
DEFAULT_SPLIT = "test-B"

# --------------------------------------------------------------------------- #
# Chunking config
# --------------------------------------------------------------------------- #
# Token-based chunking via tiktoken. Defaults chosen per the experiment spec.
CHUNK_SIZE_TOKENS = int(os.getenv("SPIQA_CHUNK_SIZE", "416"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("SPIQA_CHUNK_OVERLAP", "100"))
TIKTOKEN_ENCODING = os.getenv("SPIQA_TIKTOKEN_ENCODING", "cl100k_base")

# --------------------------------------------------------------------------- #
# Question-generation config
# --------------------------------------------------------------------------- #
# The number of generated questions defining each condition. 0 == baseline.
# We generate the MAX once per chunk and derive the smaller conditions as the
# first-k subset (clean ablation that isolates *count*, not prompt wording, and
# avoids paying 4x the LLM cost). Set SPIQA_INDEPENDENT_GEN=1 to instead
# generate each condition with its own prompt.
QUESTION_CONDITIONS: List[int] = [
    int(x) for x in os.getenv("SPIQA_Q_CONDITIONS", "0,1,10,50,100").split(",")
]
MAX_QUESTIONS = max(QUESTION_CONDITIONS)  # 100 by default
INDEPENDENT_GENERATION = os.getenv("SPIQA_INDEPENDENT_GEN", "0") == "1"

# LLM used for question generation (OpenAI-compatible; see llm_adapter.py).
LLM_MODEL = os.getenv("SPIQA_LLM_MODEL", base_config.OPENAI_CHAT_MODEL)
LLM_TEMPERATURE = float(os.getenv("SPIQA_LLM_TEMPERATURE", "0.3"))
LLM_MAX_WORKERS = int(os.getenv("SPIQA_LLM_WORKERS", "8"))

# --------------------------------------------------------------------------- #
# Retrieval / evaluation config
# --------------------------------------------------------------------------- #
TOP_K = int(os.getenv("SPIQA_TOP_K", "10"))  # retrieve depth
K_VALUES = sorted({1, 5, 10, TOP_K})  # report Hit@k / Recall@k at these


# --------------------------------------------------------------------------- #
# Condition <-> collection-name helpers
# --------------------------------------------------------------------------- #
def condition_name(n_questions: int) -> str:
    """Map a question count to its condition/collection label."""
    return "baseline" if n_questions == 0 else f"q{n_questions}"


def collection_name(split: str, n_questions: int) -> str:
    """Chroma collection name, e.g. 'spiqa_testB_q10'. Slashes removed (Chroma
    forbids them and requires 3-63 char [a-zA-Z0-9._-] names)."""
    safe_split = split.replace("-", "").replace("_", "")
    return f"spiqa_{safe_split}_{condition_name(n_questions)}"


# Estimated OpenAI pricing (USD per 1M tokens) for cost reporting. Update if you
# switch models; only used for the cost estimate column.
PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
