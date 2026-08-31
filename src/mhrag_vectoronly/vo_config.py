"""
Central config for the MultiHop-RAG dense-VECTOR-SEARCH-ONLY 15-article pilot.

This experiment compares two indexed vector representations of the SAME chunks:
    Condition A (baseline)  : one embedding per original chunk.
    Condition B (generated) : 10 generated-question embeddings per chunk, each a
                              separate vector pointing back to its parent chunk;
                              a chunk's score = MAX cosine over its questions.

Everything is DENSE cosine similarity search in a LOCAL ChromaDB. There is NO
BM25 / sparse / keyword / hybrid / RRF anywhere in this package.

Knobs live in ``config/mhrag_vectoronly.yaml``. Credentials come only from the
project ``.env`` via the repo's ``config`` module (never stored here).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# Reuse the repo's battle-tested config (paths / dotenv / OpenAI client) and the
# shared embeddings module by putting src/ on the path.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as base_config  # noqa: E402

PROJECT_ROOT = base_config.PROJECT_ROOT
CONFIG_PATH = PROJECT_ROOT / "config" / "mhrag_vectoronly.yaml"

with CONFIG_PATH.open(encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)

# --------------------------------------------------------------------------- #
# Flattened knobs
# --------------------------------------------------------------------------- #
SEED: int = int(CFG["random_seed"])

ARTICLE_ID_FIELD: str = CFG["dataset"]["article_id_field"]
ARTICLE_COUNT: int = int(CFG["dataset"]["article_count"])
INCLUDE_NULL: bool = bool(CFG["dataset"]["include_null_questions"])

CHUNK_SIZE: int = int(CFG["chunking"]["chunk_size_tokens"])
CHUNK_OVERLAP: int = int(CFG["chunking"]["chunk_overlap_tokens"])
MIN_CHUNK: int = int(CFG["chunking"]["minimum_chunk_tokens"])
TOKENIZER: str = CFG["chunking"]["tokenizer"]

QUESTIONS_PER_CHUNK: int = int(CFG["question_generation"]["questions_per_chunk"])
GEN_MAX_RETRIES: int = int(CFG["question_generation"]["maximum_retries"])
NEAR_DUP_THRESHOLD: float = float(
    CFG["question_generation"]["near_duplicate_threshold"]
)

TOP_K_VALUES = list(CFG["retrieval"]["top_k_values"])
RANK_DEPTH: int = int(CFG["retrieval"]["rank_depth"])
CANDIDATE_MULTIPLIER: int = int(
    CFG["retrieval"]["generated_question_candidate_multiplier"]
)
SIMILARITY_METRIC: str = CFG["retrieval"]["similarity_metric"]
PARENT_SCORE_METHOD: str = CFG["retrieval"]["parent_chunk_score_method"]

ANSWER_TOP_K: int = int(CFG["answer_generation"]["top_k"])
ANSWER_ENABLED: bool = bool(CFG["answer_generation"]["enabled"])

BOOTSTRAP_RESAMPLES: int = int(CFG["evaluation"]["bootstrap_resamples"])

NAMESPACE: str = CFG["chromadb"]["namespace"]
BASELINE_COLLECTION: str = CFG["chromadb"]["baseline_collection"]
GENQ_COLLECTION: str = CFG["chromadb"]["generated_question_collection"]

GEN_TEMPERATURE: float = float(CFG["models"]["generation_temperature"])

# --------------------------------------------------------------------------- #
# Paths — isolated caches under the shared data / results trees
# --------------------------------------------------------------------------- #
RAW_DIR = base_config.RAW_DIR / "multihoprag"  # reuse existing download
CORPUS_JSON = RAW_DIR / "corpus.json"
QUERIES_JSON = RAW_DIR / "MultiHopRAG.json"

DATA_DIR = base_config.PROCESSED_DIR / "mhrag_vectoronly"
RESULTS_DIR = base_config.RESULTS_DIR / "mhrag_vectoronly"
RANKINGS_DIR = RESULTS_DIR / "rankings"
REPORT_HTML = PROJECT_ROOT / "report" / "multihoprag_vectoronly_results.html"

# Dedicated LOCAL Chroma dir — kept local regardless of the shared CHROMA_MODE
# (the project's .env points CHROMA_MODE at the cloud; this pilot never uses it).
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / "mhrag_vectoronly"

for _d in (DATA_DIR, RESULTS_DIR, RANKINGS_DIR, CHROMA_DIR, REPORT_HTML.parent):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Data artifact paths (§3, §4, §6, §7, §14) ---------------------------- #
PILOT_ARTICLES = DATA_DIR / "pilot_15_articles.json"
PILOT_ELIGIBLE = DATA_DIR / "pilot_eligible_queries.jsonl"
PILOT_EXCLUDED = DATA_DIR / "pilot_excluded_queries.jsonl"
PILOT_REPORT = DATA_DIR / "pilot_subset_report.json"
PROCESSED_ARTICLES = DATA_DIR / "processed_articles.jsonl"
PREPROCESS_REPORT = DATA_DIR / "preprocessing_report.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
GENQ_PATH = DATA_DIR / "generated_questions.jsonl"
GOLD_MAPPING = DATA_DIR / "gold_chunk_mapping.jsonl"

# ---- Results artifact paths ---------------------------------------------- #
GENQ_QUALITY = RESULTS_DIR / "generated_question_quality_report.json"
GENQ_FAILURES = RESULTS_DIR / "generated_question_failures.jsonl"
GOLD_REPORT = RESULTS_DIR / "gold_alignment_report.json"
UNRESOLVED_GOLD = RESULTS_DIR / "unresolved_gold_evidence.jsonl"
BASELINE_RANKINGS = RESULTS_DIR / "baseline_retrieval_results.jsonl"
GENQ_RANKINGS = RESULTS_DIR / "generated_question_retrieval_results.jsonl"
OVERALL_METRICS = RESULTS_DIR / "overall_metrics.csv"
PER_QUERY_METRICS = RESULTS_DIR / "per_query_metrics.csv"
METRICS_BY_TYPE = RESULTS_DIR / "metrics_by_question_type.csv"
METRICS_BY_DOCCOUNT = RESULTS_DIR / "metrics_by_document_count.csv"
PAIRED_JSON = RESULTS_DIR / "paired_comparison.json"
GENERATION_RESULTS = RESULTS_DIR / "generation_results.jsonl"
ANSWER_METRICS_JSON = RESULTS_DIR / "answer_metrics.json"
FAILURE_JSONL = RESULTS_DIR / "failure_analysis.jsonl"
FAILURE_SUMMARY = RESULTS_DIR / "failure_analysis_summary.md"
LATENCY_CSV = RESULTS_DIR / "latency_results.csv"
EXPERIMENT_SUMMARY = RESULTS_DIR / "experiment_summary.md"
METRICS_JSON = RESULTS_DIR / "metrics.json"  # machine-readable rollup

# --------------------------------------------------------------------------- #
# Condition identifiers
# --------------------------------------------------------------------------- #
COND_A = "baseline"  # original chunk vectors
COND_B = "generated"  # generated-question vectors
CONDITIONS = (COND_A, COND_B)


# --------------------------------------------------------------------------- #
# Local Chroma client + collection helpers
# --------------------------------------------------------------------------- #
def chroma_client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def reset_collection(name: str):
    client = chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": SIMILARITY_METRIC})


def get_collection(name: str):
    return chroma_client().get_collection(name)


def gen_model() -> str:
    return base_config.OPENAI_CHAT_MODEL


def openai_client():
    return base_config.get_openai_client()
