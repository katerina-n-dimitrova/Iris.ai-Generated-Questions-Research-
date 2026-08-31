"""
Central configuration for the MultiHop-RAG question-generation-strategy study.

Research programme
------------------
For each paragraph-chunk of a news article an LLM generates synthetic questions
the chunk can answer (doc2query / HyPE-style enrichment). The questions are
embedded and indexed; a user query is matched against them and routed back to
the parent chunk. This project isolates *what kind* of questions to generate on
MultiHop-RAG (Tang & Yang 2024), with a hybrid **dense + BM25 + RRF** retriever.
This module holds everything the arms must agree on so that only the generation
prompt (and, for a few explicit exceptions, the index layout) varies.

Fixed factors (identical across ALL arms)
-----------------------------------------
* Chunking      : article body split into paragraph chunks (blank-line split;
  paragraphs shorter than ``MERGE_MIN_TOKENS`` merged with a neighbour).
* Question budget: exactly ``QUESTION_BUDGET`` (10) generated questions/chunk.
* Generator     : one LLM + temperature for every arm; only the prompt changes.
* Embedder      : one sentence-embedding model everywhere (repo default).
* Representation: DENSE INDEX HOLDS ONLY GENERATED-QUESTION VECTORS. Each question
  is one vector pointing back to its parent chunk; a chunk's dense score is the
  MAX similarity over its own questions alone. The chunk text is NEVER embedded
  in an enrichment arm -- it is used only for BM25 and for gold-label evaluation.
  Chunk embeddings survive in EXACTLY the B0 baseline and the explicitly-marked
  Experiment-4 variants (b), (e), (f). Enforced by ``validate_arm``.
* Sparse side   : BM25 over chunk text, with generated questions appended to the
  chunk's BM25 document in enrichment arms. (Experiment 4 varies this.)
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
from dataclasses import dataclass
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

RAW_DIR = base_config.RAW_DIR / "multihoprag"
PROCESSED_DIR = base_config.PROCESSED_DIR / "multihoprag"
RESULTS_DIR = base_config.RESULTS_DIR / "multihoprag"
RANKINGS_DIR = RESULTS_DIR / "rankings"  # saved rankings, per arm/mode
GEN_DIR = PROCESSED_DIR / "generated"  # generated-question caches
REPORT_DIR = RESULTS_DIR / "report"
# Dedicated LOCAL Chroma dir (kept local regardless of the shared CHROMA_MODE,
# which the parent project points at Chroma Cloud).
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "multihoprag"

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

# Canonical HuggingFace yixuantt/MultiHopRAG files (downloaded into RAW_DIR).
CORPUS_JSON = RAW_DIR / "corpus.json"  # 609 news articles
QUERIES_JSON = RAW_DIR / "MultiHopRAG.json"  # ~2,556 queries (301 null)

# --------------------------------------------------------------------------- #
# Dataset selection (reproducible, query-first)
# --------------------------------------------------------------------------- #
# Each query's gold evidence spans 2-4 articles, so a random article sample
# leaves almost no fully-answerable query. Instead we select QUERY-FIRST:
#   1. drop null queries (they have no evidence by design),
#   2. shuffle the rest (fixed seed),
#   3. greedily accept a query iff adding all its evidence articles keeps the
#      selected-article set within ARTICLE_BUDGET, until the set hits the budget,
#   4. one more pass: also keep every remaining query whose evidence articles are
#      ALL already inside the selected set.
# The corpus = exactly those articles; the eval queries = the kept queries.
# Scale up later by changing ONE value: ARTICLE_BUDGET.
ARTICLE_BUDGET = int(os.getenv("MHRAG_ARTICLE_BUDGET", "10"))
SELECTION_SEED = int(os.getenv("MHRAG_SEED", "42"))

# Article identity: the corpus has unique titles AND unique urls; url is the
# canonical, stable id used to join evidence <-> corpus.
ARTICLE_ID_FIELD = "url"

SELECTED_ARTICLES_PATH = PROCESSED_DIR / "selected_articles.json"
SELECTED_QUERIES_PATH = PROCESSED_DIR / "selected_query_ids.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
QUERIES_PATH = PROCESSED_DIR / "queries.jsonl"

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Split the (cleaned) article body on blank lines into paragraphs; merge any
# paragraph shorter than MERGE_MIN_TOKENS (whitespace tokens) into a neighbour so
# tiny fragments/headers don't become standalone chunks.
MERGE_MIN_TOKENS = int(os.getenv("MHRAG_MERGE_MIN_TOKENS", "40"))
# Gold matching: an evidence 'fact' is located in a chunk by normalised-whitespace
# substring, else by best fuzzy ratio over the chunk's sentences (>= this).
GOLD_FUZZY_THRESHOLD = float(os.getenv("MHRAG_GOLD_FUZZY", "0.9"))

# --------------------------------------------------------------------------- #
# Question generation (fixed generator for every arm)
# --------------------------------------------------------------------------- #
QUESTION_BUDGET = int(os.getenv("MHRAG_BUDGET", "10"))
LLM_MODEL = os.getenv("MHRAG_LLM_MODEL", base_config.OPENAI_CHAT_MODEL)
LLM_TEMPERATURE = float(os.getenv("MHRAG_LLM_TEMPERATURE", "0.3"))
LLM_MAX_WORKERS = int(os.getenv("MHRAG_LLM_WORKERS", "8"))

PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

# --------------------------------------------------------------------------- #
# Retrieval / fusion / evaluation
# --------------------------------------------------------------------------- #
RRF_K = int(os.getenv("MHRAG_RRF_K", "60"))
# Primary metric is Evidence Recall@k (queries are multi-evidence, 2-4 gold
# chunks each); report at k = 2, 5, 10.
K_VALUES = [2, 5, 10]
NDCG_K = 10
MRR_K = 10
# How deep to keep each ranking on disk (must exceed max k comfortably).
RANK_DEPTH = int(os.getenv("MHRAG_RANK_DEPTH", "100"))
# Over-fetch factor for the dense (multi-vector) search so that, after
# deduplicating question hits to their parent chunks, we still have RANK_DEPTH
# distinct chunks.
DENSE_OVERFETCH = int(os.getenv("MHRAG_DENSE_OVERFETCH", "40"))

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")

# Bootstrap confidence intervals over queries.
BOOTSTRAP_N = int(os.getenv("MHRAG_BOOTSTRAP_N", "1000"))
BOOTSTRAP_SEED = int(os.getenv("MHRAG_BOOTSTRAP_SEED", "123"))
BOOTSTRAP_CI = float(os.getenv("MHRAG_BOOTSTRAP_CI", "0.95"))

# MultiHop-RAG's own query-type labels we break results down by.
# (Dataset uses '<type>_query'; we store the short form.)
QUERY_TYPES = ("inference", "comparison", "temporal")


# --------------------------------------------------------------------------- #
# Arm registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arm:
    """One experimental arm.

    name          : short id, used for cache/collection/ranking filenames.
    kind          : 'baseline' (no questions) or 'enrichment' (questions).
    prompt        : key into mhrag_generate.PROMPTS for this arm's OWN generation
                    cache (None if the arm generates nothing / reuses another).
    gen_kind      : how this arm's own content is generated/parsed:
                    'questions' | 'questions_typed' | 'keywords' | 'qa' |
                    'atoms' | 'atom_questions' | None.
    embeds_chunk  : whether the chunk TEXT is embedded into the dense index.
                    MUST be False for every enrichment arm except the explicit
                    Experiment-4 variants (b), (e), (f). Enforced by validate_arm.
    embeds_questions: whether multi-vector (question/atom) vectors go in dense.
    bm25_appends_questions: append the bm25-source content to the BM25 document.
    concat_questions_into_chunk: Exp-4(e) doc2query — append questions to the
                    chunk text and embed as ONE vector (needs embeds_chunk=True).
    dense_source  : arm name whose gen cache feeds the dense multi-vectors
                    (None -> use this arm's own cache when it has one).
    bm25_source   : arm name whose gen cache feeds the BM25 append terms.
    bm25_text_source: 'chunk' (default) or 'questions' (BM25 over questions only).
    only_modes    : restrict which retrieval modes are produced (Exp-4 a/b).
    description   : human-readable summary for reports.
    experiment    : which experiment this arm belongs to (grouping in reports).
    """

    name: str
    kind: str
    prompt: Optional[str]
    embeds_chunk: bool
    embeds_questions: bool
    bm25_appends_questions: bool
    description: str
    gen_kind: Optional[str] = "questions"
    concat_questions_into_chunk: bool = False
    dense_source: Optional[str] = None
    bm25_source: Optional[str] = None
    bm25_text_source: str = "chunk"
    only_modes: Optional[tuple] = None
    experiment: str = "baselines"

    def resolved_dense_source(self) -> Optional[str]:
        if not self.embeds_questions and not self.concat_questions_into_chunk:
            return None
        return self.dense_source or (self.name if self.prompt else None)

    def resolved_bm25_source(self) -> Optional[str]:
        if not self.bm25_appends_questions:
            return None
        return self.bm25_source or (self.name if self.prompt else None)


# Only B0 and the Experiment-4 chunk-embedding variants may embed chunk text.
CHUNK_EMBED_OK = {"B0", "E4b", "E4e", "E4f"}


def validate_arm(arm: "Arm") -> None:
    """Guard the fixed-factor invariant: only B0 and the Exp-4 variants (b,e,f)
    may embed chunk text; every other enrichment arm is questions-only dense."""
    if arm.embeds_chunk and arm.name not in CHUNK_EMBED_OK:
        raise ValueError(
            f"Arm {arm.name!r} sets embeds_chunk=True but only "
            f"{sorted(CHUNK_EMBED_OK)} may embed chunk text (fixed-factor "
            "invariant: enrichment arms are questions-only in the dense index)."
        )
    if arm.kind == "enrichment" and not arm.embeds_questions and not arm.embeds_chunk:
        raise ValueError(f"Arm {arm.name!r} would build an empty dense index.")


ARMS: Dict[str, Arm] = {
    # ---- Baselines (reused everywhere) ---- #
    "B0": Arm(
        name="B0",
        kind="baseline",
        prompt=None,
        gen_kind=None,
        embeds_chunk=True,
        embeds_questions=False,
        bm25_appends_questions=False,
        description="No enrichment: plain chunk vector + BM25(chunk text) + RRF.",
    ),
    "B1": Arm(
        name="B1",
        kind="enrichment",
        prompt="naive",
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Naive enrichment: 'generate 10 questions this passage "
        "answers' (questions-only dense + BM25 chunk+questions).",
    ),
    # ---- Experiment 1 — semantic question type (Cao & Wang 2021 ontology) ---- #
    "E1": Arm(
        name="E1",
        kind="enrichment",
        prompt="typed",
        gen_kind="questions_typed",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Type-stratified: 10 questions covering distinct Cao&Wang "
        "semantic types (JUDGMENTAL skipped), 1/type + extra COMPARISON.",
        experiment="exp1_semantic_type",
    ),
    # ---- Experiment 2 — scope (local / summary / mixed) ---- #
    "E2a": Arm(
        name="E2a",
        kind="enrichment",
        prompt="local",
        gen_kind="questions",
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
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Scope=summary: 10 questions each requiring the whole chunk.",
        experiment="exp2_scope",
    ),
    "E2c": Arm(
        name="E2c",
        kind="enrichment",
        prompt="mixed",
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Scope=mixed: 8 local + 2 summary questions.",
        experiment="exp2_scope",
    ),
    # ---- Experiment 3 — explicitness (explicit vs implicit/vocab-bridging) ---- #
    "E3a": Arm(
        name="E3a",
        kind="enrichment",
        prompt="explicit",
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Explicit: 10 questions whose answers are literally stated.",
        experiment="exp3_explicitness",
    ),
    "E3b": Arm(
        name="E3b",
        kind="enrichment",
        prompt="explicit_implicit",
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="5 explicit + 5 implicit (paraphrased, avoid chunk vocab) "
        "to bridge query-document vocabulary mismatch.",
        experiment="exp3_explicitness",
    ),
    # ---- Experiment 4 — surface form & index placement ---- #
    # (a) NL questions -> dense only; chunk-only BM25 (no appended questions).
    "E4a": Arm(
        name="E4a",
        kind="enrichment",
        prompt=None,
        gen_kind=None,
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=False,
        dense_source="B1",
        description="NL questions -> dense (reuse B1); BM25 = chunk text only "
        "(isolates the effect of appending questions to BM25).",
        experiment="exp4_surface_form",
    ),
    # (b) keyword queries -> BM25 only; chunk-only dense (chunk embeddings).
    "E4b": Arm(
        name="E4b",
        kind="enrichment",
        prompt="keyword",
        gen_kind="keywords",
        embeds_chunk=True,
        embeds_questions=False,
        bm25_appends_questions=True,
        bm25_source="E4b",
        description="Short keyword queries -> BM25 (appended); dense = chunk "
        "vectors only (no questions to embed).",
        experiment="exp4_surface_form",
    ),
    # (c) both: NL questions -> dense + keyword variants -> BM25.
    "E4c": Arm(
        name="E4c",
        kind="enrichment",
        prompt=None,
        gen_kind=None,
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_source="B1",
        bm25_source="E4b",
        description="NL questions -> dense (reuse B1) + keyword variants -> "
        "BM25 (reuse E4b).",
        experiment="exp4_surface_form",
    ),
    # (d) Q&A pairs embedded ("Q: .. A: ..") in the dense index.
    "E4d": Arm(
        name="E4d",
        kind="enrichment",
        prompt="qa",
        gen_kind="qa",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        bm25_source="E4d",
        description="Question+answer pairs: embed concatenated 'Q: .. A: ..' "
        "strings in the dense index.",
        experiment="exp4_surface_form",
    ),
    # (e) doc2query concat: questions appended to chunk, embedded as ONE vector.
    "E4e": Arm(
        name="E4e",
        kind="enrichment",
        prompt=None,
        gen_kind=None,
        embeds_chunk=True,
        embeds_questions=False,
        concat_questions_into_chunk=True,
        bm25_appends_questions=True,
        dense_source="B1",
        bm25_source="B1",
        description="Doc2Query-style: questions (reuse B1) appended to chunk "
        "text, embedded as a SINGLE vector (concatenation).",
        experiment="exp4_surface_form",
    ),
    # (f) questions + chunk embedding together in the dense index (max over both).
    "E4f": Arm(
        name="E4f",
        kind="enrichment",
        prompt=None,
        gen_kind=None,
        embeds_chunk=True,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_source="B1",
        bm25_source="B1",
        description="Questions (reuse B1) + chunk vector both in the dense "
        "index; chunk score = max over both.",
        experiment="exp4_surface_form",
    ),
    # ---- Experiment 5 — style match (zero-shot vs few-shot) ---- #
    # (a) zero-shot == the naive B1 prompt (reuse its cache; shown for contrast).
    "E5a": Arm(
        name="E5a",
        kind="enrichment",
        prompt=None,
        gen_kind=None,
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_source="B1",
        bm25_source="B1",
        description="Zero-shot (== B1 naive prompt), shown as the style-match control.",
        experiment="exp5_style_match",
    ),
    # (b) few-shot with 8 REAL MultiHop-RAG queries from OUTSIDE the eval set and
    #     the selected articles (no leakage — asserted in mhrag_style).
    "E5b": Arm(
        name="E5b",
        kind="enrichment",
        prompt="fewshot",
        gen_kind="questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        description="Few-shot: imitate the style of 8 real out-of-sample "
        "MultiHop-RAG queries (Promptagator/UDAPDR).",
        experiment="exp5_style_match",
    ),
    # ---- Experiment 6 — atomic units (Raina & Gales 2024, arXiv:2405.12363) ---- #
    # 2x2 with the chunk-level cells supplied by B0 (chunk statement) and B1
    # (chunk question). Retrieval scoring is already max-over-vectors.
    "E6as": Arm(
        name="E6as",
        kind="enrichment",
        prompt="atoms",
        gen_kind="atoms",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_source="E6as",
        bm25_source="E6as",
        description="Atom STATEMENTS: chunk decomposed into stand-alone "
        "atomic facts, embedded directly (multi-vector).",
        experiment="exp6_atomic_units",
    ),
    "E6aq": Arm(
        name="E6aq",
        kind="enrichment",
        prompt="atom_questions",
        gen_kind="atom_questions",
        embeds_chunk=False,
        embeds_questions=True,
        bm25_appends_questions=True,
        dense_source="E6aq",
        bm25_source="E6aq",
        description="Atom QUESTIONS: closed-answer questions generated over "
        "the atoms (chunk as context) — Raina & Gales' method.",
        experiment="exp6_atomic_units",
    ),
}

for _a in ARMS.values():
    validate_arm(_a)

BASELINE_ARMS: List[str] = ["B0", "B1"]
# Convenience groupings for the umbrella runner.
EXPERIMENT_ARMS: Dict[str, List[str]] = {
    "exp1": ["E1"],
    "exp2": ["E2a", "E2b", "E2c"],
    "exp3": ["E3a", "E3b"],
    "exp4": ["E4a", "E4b", "E4c", "E4d", "E4e", "E4f"],
    "exp5": ["E5a", "E5b"],
    "exp6": ["E6as", "E6aq"],
}
ALL_ARMS: List[str] = BASELINE_ARMS + [
    a for arms in EXPERIMENT_ARMS.values() for a in arms
]


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def collection_name(arm: str) -> str:
    """Chroma collection name for an arm's dense index."""
    return f"mhrag_{arm}"


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
        "dataset": "yixuantt/MultiHopRAG",
        "article_budget": ARTICLE_BUDGET,
        "selection_seed": SELECTION_SEED,
        "question_budget": QUESTION_BUDGET,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "embedding_model": embedding_signature(),
        "rrf_k": RRF_K,
        "k_values": K_VALUES,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci": BOOTSTRAP_CI,
        "merge_min_tokens": MERGE_MIN_TOKENS,
        "gold_fuzzy_threshold": GOLD_FUZZY_THRESHOLD,
    }
