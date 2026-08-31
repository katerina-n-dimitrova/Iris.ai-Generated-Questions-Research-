"""
Central configuration for the QASPER question-generation-strategy study.

Research programme
------------------
For each paragraph-chunk of a scientific paper an LLM generates synthetic
questions the chunk can answer (doc2query / HyDE-style enrichment). The
questions are embedded and indexed; a user query is matched against them and
routed back to the parent chunk. This project isolates *what kind* of questions
to generate. This module holds everything the arms must agree on so that only
the generation prompt (and, for two explicit exceptions, the index layout)
varies between arms.

Fixed factors (identical across ALL arms)
-----------------------------------------
* Chunking      : one paragraph = one chunk (QASPER evidence is paragraph-level).
* Question budget: exactly ``QUESTION_BUDGET`` (10) generated questions/chunk.
* Generator     : one LLM + temperature for every arm; only the prompt changes.
* Embedder      : one sentence-embedding model everywhere (repo default).
* Representation: DENSE INDEX HOLDS ONLY GENERATED-QUESTION VECTORS. Each question
  is one vector pointing back to its parent chunk; a chunk's dense score is the
  MAX similarity over its own questions alone. The chunk text is NEVER embedded
  in an enrichment arm -- it is used only for BM25 and for evaluation.
  Chunk embeddings survive in EXACTLY TWO arms: the B0 baseline (no questions to
  embed) and the Experiment-4 variant (f) (questions+chunk in one index, built
  to directly test whether dropping the chunk vector is the right call).
* Sparse side   : BM25 over chunk text, with generated questions appended to the
  chunk's BM25 document in enrichment arms.
* Fusion        : Reciprocal Rank Fusion over dense + BM25 rankings, k=60.

Baselines
---------
* B0 : no enrichment -- plain chunk vector + BM25(chunk text) + RRF.
* B1 : naive enrichment -- "generate 10 questions this passage answers", no other
  constraints (industry default). Every experiment is measured vs B0 and B1.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Make the parent src/ importable so we reuse the repo's battle-tested
# ``config`` (paths / dotenv / OpenAI client) and ``embeddings`` modules.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as base_config  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (isolated under the shared data / results / chroma trees)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = base_config.PROJECT_ROOT

RAW_DIR = base_config.RAW_DIR / "qasper"
PROCESSED_DIR = base_config.PROCESSED_DIR / "qasper"
RESULTS_DIR = base_config.RESULTS_DIR / "qasper"
RANKINGS_DIR = RESULTS_DIR / "rankings"  # saved rankings, per arm/mode
GEN_DIR = PROCESSED_DIR / "generated"  # generated-question caches
REPORT_DIR = RESULTS_DIR / "report"
# Dedicated LOCAL Chroma dir (kept local regardless of shared CHROMA_MODE).
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "qasper"

for _d in (
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    RANKINGS_DIR,
    GEN_DIR,
    REPORT_DIR,
    CHROMA_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical AllenAI QASPER v0.3 dev split (downloaded into RAW_DIR).
DEV_JSON = RAW_DIR / "qasper-dev-v0.3.json"
TRAIN_JSON = (
    RAW_DIR / "qasper-train-v0.3.json"
)  # used only for out-of-sample style exemplars (Exp 5)

# --------------------------------------------------------------------------- #
# Dataset selection (reproducible)
# --------------------------------------------------------------------------- #
# SELECTION_MODE:
#   "text_answerable" (default) -- the corpus is TEXT (float paragraphs dropped) and
#      every EVALUATED question is fully text-answerable (all its gold evidence is
#      text -- no question needs a table/figure), on formula-light papers. QASPER
#      papers are float-heavy, so a strict "no floats in the PDF" filter is too small;
#      this makes the retrieval TASK text-only while keeping enough papers for power.
#      NUM_PAPERS papers are seed-sampled from the qualifying dev pool (~59 papers).
#   "text_only" -- strict: papers with ZERO figures/tables, formula-free (pooled
#      train+dev). Only ~4 QASPER papers qualify -> tiny, underpowered.
#   "random_dev" -- NUM_PAPERS random dev papers, chunk-level float filtering only.
SELECTION_MODE = os.getenv("QASPER_SELECTION", "text_answerable")

SELECTION_SEED = int(os.getenv("QASPER_SEED", "42"))
NUM_PAPERS = int(os.getenv("QASPER_NUM_PAPERS", "10"))  # text_answerable / random_dev

# In text_only + text_answerable modes, keep ONLY fully-text-answerable questions
# (drop any question with figure/table evidence) so the retrieval task is text-only.
REQUIRE_FULLY_TEXT_QUERIES = SELECTION_MODE in ("text_only", "text_answerable")

# text_only-mode thresholds (a paper qualifies iff it clears all three).
TEXT_ONLY_MAX_FLOATS = int(os.getenv("QASPER_MAX_FLOATS", "0"))  # figures+tables
TEXT_ONLY_MAX_FORMULA = float(
    os.getenv("QASPER_MAX_FORMULA", "0.05")
)  # math-para fraction
TEXT_ONLY_MIN_QUERIES = int(os.getenv("QASPER_MIN_QUERIES", "4"))  # usable text queries

# text_answerable-mode thresholds.
TEXT_ANSWERABLE_MAX_FORMULA = float(os.getenv("QASPER_TA_MAX_FORMULA", "0.10"))
TEXT_ANSWERABLE_MIN_QUERIES = int(os.getenv("QASPER_TA_MIN_QUERIES", "4"))

SELECTED_PAPERS_PATH = PROCESSED_DIR / "selected_papers.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
QUERIES_PATH = PROCESSED_DIR / "queries.jsonl"

# --------------------------------------------------------------------------- #
# Paragraph filtering (discard figures / tables / math-or-markup-heavy scraps)
# --------------------------------------------------------------------------- #
MIN_CHUNK_WORDS = int(os.getenv("QASPER_MIN_WORDS", "6"))
MIN_ALPHA_RATIO = float(os.getenv("QASPER_MIN_ALPHA", "0.5"))  # alpha / non-space chars

# --------------------------------------------------------------------------- #
# Question generation (fixed generator for every arm)
# --------------------------------------------------------------------------- #
QUESTION_BUDGET = int(os.getenv("QASPER_BUDGET", "10"))
LLM_MODEL = os.getenv("QASPER_LLM_MODEL", base_config.OPENAI_CHAT_MODEL)
LLM_TEMPERATURE = float(os.getenv("QASPER_LLM_TEMPERATURE", "0.3"))
LLM_MAX_WORKERS = int(os.getenv("QASPER_LLM_WORKERS", "8"))

PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

# --------------------------------------------------------------------------- #
# Retrieval / fusion / evaluation
# --------------------------------------------------------------------------- #
RRF_K = int(os.getenv("QASPER_RRF_K", "60"))
K_VALUES = [1, 5, 10]
NDCG_K = 10
MRR_K = 10
# How deep to keep each ranking on disk (must exceed max k comfortably).
RANK_DEPTH = int(os.getenv("QASPER_RANK_DEPTH", "100"))
# Over-fetch factor for the dense (multi-vector) search so that, after
# deduplicating question hits to their parent chunks, we still have RANK_DEPTH
# distinct chunks.
DENSE_OVERFETCH = int(os.getenv("QASPER_DENSE_OVERFETCH", "40"))

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")

# Bootstrap confidence intervals over queries.
BOOTSTRAP_N = int(os.getenv("QASPER_BOOTSTRAP_N", "1000"))
BOOTSTRAP_SEED = int(os.getenv("QASPER_BOOTSTRAP_SEED", "123"))
BOOTSTRAP_CI = float(os.getenv("QASPER_BOOTSTRAP_CI", "0.95"))

# QASPER answer-type buckets we break results down by.
ANSWER_TYPES = ("extractive", "abstractive", "boolean")


# --------------------------------------------------------------------------- #
# Arm registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arm:
    """One experimental arm.

    name          : short id, used for cache/collection/ranking filenames.
    kind          : 'baseline' (no questions) or 'enrichment' (questions).
    prompt        : key into qasper_generate.PROMPTS (None for B0).
    embeds_chunk  : whether the chunk TEXT is embedded into the dense index.
                    MUST be False for every enrichment arm except the explicit
                    Experiment-4 variant (f). Enforced by ``validate_arm``.
    embeds_questions: whether generated-question vectors go in the dense index.
    bm25_appends_questions: append generated questions to the BM25 document.
    description   : human-readable summary for reports.
    """

    name: str
    kind: str
    prompt: Optional[str]
    embeds_chunk: bool
    embeds_questions: bool
    bm25_appends_questions: bool
    description: str
    experiment: str = "baselines"
    # Experiment-4 explicit index-placement overrides (else derived from the flags).
    #   dense_mode: chunk | questions | qa_pairs | concat_single | chunk+questions | none
    #   bm25_mode : chunk | chunk+questions | chunk+keywords | none
    #   source_arm: which arm's generated questions to reuse (default: self)
    dense_mode: Optional[str] = None
    bm25_mode: Optional[str] = None
    source_arm: Optional[str] = None


def dense_mode(arm: "Arm") -> str:
    """Effective dense-index content mode (explicit override or legacy flags)."""
    if arm.dense_mode:
        return arm.dense_mode
    if arm.embeds_chunk and arm.embeds_questions:
        return "chunk+questions"
    if arm.embeds_chunk:
        return "chunk"
    if arm.embeds_questions:
        return "questions"
    return "none"


def bm25_mode(arm: "Arm") -> str:
    """Effective BM25-document mode (explicit override or legacy flags)."""
    if arm.bm25_mode:
        return arm.bm25_mode
    return "chunk+questions" if arm.bm25_appends_questions else "chunk"


def source_arm(arm: "Arm") -> str:
    """Which arm's generated questions this arm reuses (default: itself)."""
    return arm.source_arm or arm.name


# Dense modes that place a chunk-text vector in the index (the fixed-factor
# invariant restricts these to B0 and the Experiment-4 variants that test it).
_CHUNK_DENSE_MODES = {"chunk", "chunk+questions", "concat_single"}


def validate_arm(arm: "Arm") -> None:
    """Guard the fixed-factor invariant: enrichment arms are questions-only in the
    dense index. Chunk-text vectors are allowed only for B0 and the Experiment-4
    variants explicitly built to test whether the chunk vector should be kept
    (E4b uses a chunk-only dense as the keyword-BM25 control; E4e/E4f test concat
    and questions+chunk)."""
    chunk_ok = {"B0", "E4b", "E4e", "E4f"}
    if dense_mode(arm) in _CHUNK_DENSE_MODES and arm.name not in chunk_ok:
        raise ValueError(
            f"Arm {arm.name!r} places a chunk vector in the dense index but only "
            f"{sorted(chunk_ok)} may (fixed-factor invariant)."
        )
    if dense_mode(arm) == "none" and arm.kind == "enrichment":
        raise ValueError(f"Arm {arm.name!r} would build an empty dense index.")


# The two baselines are all this deliverable runs; experiments 1-5 register more
# arms later (each just adds a prompt + an Arm entry -- indexing/retrieval/eval
# are untouched).
ARMS: Dict[str, Arm] = {
    "B0": Arm(
        name="B0",
        kind="baseline",
        prompt=None,
        embeds_chunk=True,
        embeds_questions=False,
        bm25_appends_questions=False,
        description="No enrichment: plain chunk vector + BM25(chunk text) + RRF.",
    ),
    "B1": Arm(
        name="B1",
        kind="enrichment",
        prompt="naive",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Naive enrichment: 'generate 10 questions this passage "
        "answers' (questions-only dense index + BM25 chunk+questions).",
    ),
    # ---- Experiment 1 — semantic question type (Cao & Wang 2021 ontology) ---- #
    "E1": Arm(
        name="E1",
        kind="enrichment",
        prompt="typed",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Type-stratified: 10 questions covering distinct Cao&Wang "
        "semantic types (JUDGMENTAL skipped), allocated to the real-"
        "query type distribution.",
        experiment="exp1_semantic_type",
    ),
    # ---- Experiment 2 — scope (local vs summary) ---- #
    "E2a": Arm(
        name="E2a",
        kind="enrichment",
        prompt="local",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Scope=local: 10 questions each answerable from a single "
        "sentence, spread across the chunk.",
        experiment="exp2_scope",
    ),
    "E2b": Arm(
        name="E2b",
        kind="enrichment",
        prompt="summary",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Scope=summary: 10 questions each requiring the whole "
        "chunk to answer.",
        experiment="exp2_scope",
    ),
    "E2c": Arm(
        name="E2c",
        kind="enrichment",
        prompt="mixed",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Scope=mixed: 8 local + 2 summary questions.",
        experiment="exp2_scope",
    ),
    # ---- Experiment 3 — explicitness ---- #
    "E3a": Arm(
        name="E3a",
        kind="enrichment",
        prompt="explicit",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Explicit: 10 questions whose answers are literally "
        "stated in the chunk.",
        experiment="exp3_explicitness",
    ),
    "E3b": Arm(
        name="E3b",
        kind="enrichment",
        prompt="implicit_mix",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Explicit+implicit: 5 explicit + 5 implicit (paraphrased, "
        "no content-word reuse) to bridge vocabulary mismatch.",
        experiment="exp3_explicitness",
    ),
    # ---- Experiment 4 — surface form & index placement (same B1 facts, placed
    #      differently). E4a/c/e/f reuse B1's NL questions; E4b/E4d generate. ---- #
    "E4a": Arm(
        name="E4a",
        kind="enrichment",
        prompt=None,
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=False,
        dense_mode="questions",
        bm25_mode="chunk",
        source_arm="B1",
        description="NL questions → dense only; chunk-only BM25.",
        experiment="exp4_surface_form",
    ),
    "E4b": Arm(
        name="E4b",
        kind="enrichment",
        prompt="keywords",
        embeds_chunk=True,
        embeds_questions=False,
        bm25_appends_questions=False,
        dense_mode="chunk",
        bm25_mode="chunk+keywords",
        source_arm="B1",
        description="Keyword variants → BM25 only; chunk-only dense.",
        experiment="exp4_surface_form",
    ),
    "E4c": Arm(
        name="E4c",
        kind="enrichment",
        prompt=None,
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=False,
        dense_mode="questions",
        bm25_mode="chunk+keywords",
        source_arm="B1",
        description="Both: NL questions → dense + keyword variants → BM25.",
        experiment="exp4_surface_form",
    ),
    "E4d": Arm(
        name="E4d",
        kind="enrichment",
        prompt="qa_pairs",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=False,
        dense_mode="qa_pairs",
        bm25_mode="chunk",
        source_arm="E4d",
        description="Q+A pairs: embed concatenated 'Q:…A:…' in the dense index.",
        experiment="exp4_surface_form",
    ),
    "E4e": Arm(
        name="E4e",
        kind="enrichment",
        prompt=None,
        embeds_chunk=True,
        embeds_questions=False,
        bm25_appends_questions=True,
        dense_mode="concat_single",
        bm25_mode="chunk+questions",
        source_arm="B1",
        description="doc2query concat: NL questions appended to chunk, embedded "
        "as ONE vector (Doc2Query++ says this hurts dense).",
        experiment="exp4_surface_form",
    ),
    "E4f": Arm(
        name="E4f",
        kind="enrichment",
        prompt=None,
        embeds_chunk=True,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_mode="chunk+questions",
        bm25_mode="chunk+questions",
        source_arm="B1",
        description="Questions + chunk: both chunk vector and question vectors "
        "in one index (tests whether dropping the chunk vector is right).",
        experiment="exp4_surface_form",
    ),
    # ---- Experiment 5 — style match. Zero-shot arm ≡ B1 (naive). E5b is few-shot
    #      with 8 real QASPER questions from papers OUTSIDE the selected set. ---- #
    "E5b": Arm(
        name="E5b",
        kind="enrichment",
        prompt="fewshot",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Few-shot: imitate the style/length/specificity of 8 real "
        "QASPER questions from non-selected papers (no leakage).",
        experiment="exp5_style_match",
    ),
}

for _a in ARMS.values():
    validate_arm(_a)

BASELINE_ARMS: List[str] = ["B0", "B1"]


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def collection_name(arm: str) -> str:
    """Chroma collection name for an arm's dense index."""
    return f"qasper_{arm}"


def gen_cache_path(arm: str) -> Path:
    """JSONL cache of generated questions for an arm (resumable)."""
    return GEN_DIR / f"questions_{arm}.jsonl"


def ranking_path(arm: str, mode: str) -> Path:
    """Saved ranking (per query -> ranked chunk ids) for an arm and mode."""
    return RANKINGS_DIR / f"{arm}__{mode}.jsonl"


def metrics_path(arm: str) -> Path:
    return RESULTS_DIR / f"metrics_{arm}.json"


def run_config_signature() -> dict:
    """Everything needed to reproduce a run, for logging into result files."""
    from embeddings import embedding_signature

    return {
        "selection_mode": SELECTION_MODE,
        "selection_seed": SELECTION_SEED,
        "num_papers": NUM_PAPERS,
        "question_budget": QUESTION_BUDGET,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "embedding_model": embedding_signature(),
        "rrf_k": RRF_K,
        "k_values": K_VALUES,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci": BOOTSTRAP_CI,
        "min_chunk_words": MIN_CHUNK_WORDS,
        "min_alpha_ratio": MIN_ALPHA_RATIO,
    }
