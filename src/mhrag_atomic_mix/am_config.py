"""
Config for the MultiHop-RAG 10-article atomic+chunk-level mixed-question pilot.

Condition A: one vector per original chunk.
Condition E: POOLED atomic-fact questions + broader chunk-level questions, each a
             separate vector pointing back to its parent chunk; chunk score = MAX
             cosine over its questions.

Dense cosine similarity in a LOCAL ChromaDB only — no BM25/sparse/hybrid/rerank.
Isolated namespace/paths so the 15-article mhrag_vectoronly experiment is intact.
Credentials come only from the project .env (via the repo `config` module).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_SRC_DIR = Path(__file__).resolve().parent.parent
_VO_DIR = _SRC_DIR / "mhrag_vectoronly"  # reuse the 15-article harness modules
for _p in (str(_SRC_DIR), str(_VO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as base_config  # noqa: E402

PROJECT_ROOT = base_config.PROJECT_ROOT
CONFIG_PATH = PROJECT_ROOT / "config" / "mhrag_atomic_chunk_mix_10.yaml"
with CONFIG_PATH.open(encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)

# ---- knobs ---------------------------------------------------------------- #
SEED = int(CFG["random_seed"])
ARTICLE_ID_FIELD = CFG["dataset"]["article_id_field"]
ARTICLE_COUNT = int(CFG["dataset"]["article_count"])

CHUNK_SIZE = int(CFG["chunking"]["chunk_size_tokens"])
CHUNK_OVERLAP = int(CFG["chunking"]["chunk_overlap_tokens"])
MIN_CHUNK = int(CFG["chunking"]["minimum_chunk_tokens"])
TOKENIZER = CFG["chunking"]["tokenizer"]

GEN = CFG["question_generation"]
ALLOW_SECOND_ATOMIC = bool(GEN["allow_second_atomic_question"])
DEFAULT_CHUNK_LEVEL_Q = int(GEN["default_chunk_level_questions"])
MAX_CHUNK_LEVEL_Q = int(GEN["maximum_chunk_level_questions"])
MAX_TOTAL_Q = int(GEN["maximum_total_questions_per_chunk"])
GEN_MAX_RETRIES = int(GEN["maximum_retries"])

FILT = CFG["question_filtering"]
PARENT_TOPK = int(FILT["parent_retrieval_top_k"])
REQUIRE_POS_MARGIN = bool(FILT["require_positive_confusion_margin"])
MIN_MARGIN = float(FILT["min_confusion_margin"])
NEAR_DUP_THRESHOLD = float(FILT["near_duplicate_threshold"])

TOP_K_VALUES = list(CFG["retrieval"]["top_k_values"])
RANK_DEPTH = int(CFG["retrieval"]["rank_depth"])
CANDIDATE_MULTIPLIER = int(CFG["retrieval"]["candidate_multiplier"])
SIMILARITY_METRIC = CFG["retrieval"]["similarity_metric"]

ANSWER_TOP_K = int(CFG["answer_generation"]["top_k"])
ANSWER_ENABLED = bool(CFG["answer_generation"]["enabled"])
BOOTSTRAP_RESAMPLES = int(CFG["evaluation"]["bootstrap_resamples"])

NAMESPACE = CFG["chromadb"]["namespace"]
BASELINE_COLLECTION = CFG["chromadb"]["baseline_collection"]
MIXED_COLLECTION = CFG["chromadb"]["mixed_question_collection"]
GEN_TEMPERATURE = float(CFG["models"]["generation_temperature"])

# ---- paths ---------------------------------------------------------------- #
RAW_DIR = base_config.RAW_DIR / "multihoprag"
CORPUS_JSON = RAW_DIR / "corpus.json"
QUERIES_JSON = RAW_DIR / "MultiHopRAG.json"

DATA_DIR = base_config.PROCESSED_DIR / NAMESPACE
RESULTS_DIR = base_config.RESULTS_DIR / NAMESPACE
RANKINGS_DIR = RESULTS_DIR / "rankings"
REPORT_HTML = PROJECT_ROOT / "report" / f"{NAMESPACE}_results.html"
CHROMA_DIR = base_config.CHROMA_PERSIST_DIR / NAMESPACE

for _d in (DATA_DIR, RESULTS_DIR, RANKINGS_DIR, CHROMA_DIR, REPORT_HTML.parent):
    _d.mkdir(parents=True, exist_ok=True)

# data artifacts
PILOT_ARTICLES = DATA_DIR / "pilot_10_articles.json"
ELIGIBLE = DATA_DIR / "eligible_queries.jsonl"
EXCLUDED = DATA_DIR / "excluded_queries.jsonl"
SUBSET_REPORT = DATA_DIR / "subset_report.json"
PROCESSED_ARTICLES = DATA_DIR / "processed_articles.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
ATOMS_PATH = DATA_DIR / "generated_atoms.jsonl"
QUESTIONS_RAW = DATA_DIR / "generated_questions_raw.jsonl"
QUESTIONS_FILTERED = DATA_DIR / "generated_questions_filtered.jsonl"
GOLD_MAPPING = DATA_DIR / "gold_chunk_mapping.jsonl"

# results artifacts
GEN_QUALITY = RESULTS_DIR / "generation_quality_report.json"
FILTER_REPORT = RESULTS_DIR / "filtering_report.json"
REJECTED = RESULTS_DIR / "rejected_questions.jsonl"
GOLD_REPORT = RESULTS_DIR / "gold_alignment_report.json"
UNRESOLVED_GOLD = RESULTS_DIR / "unresolved_gold_evidence.jsonl"
BASELINE_RANKINGS = RESULTS_DIR / "baseline_retrieval_results.jsonl"
MIXED_RANKINGS = RESULTS_DIR / "mixed_question_retrieval_results.jsonl"
PER_QUERY_METRICS = RESULTS_DIR / "per_query_metrics.csv"
OVERALL_METRICS = RESULTS_DIR / "overall_metrics.csv"
METRICS_BY_TYPE = RESULTS_DIR / "metrics_by_question_type.csv"
METRICS_BY_DOCCOUNT = RESULTS_DIR / "metrics_by_document_count.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "generated_question_type_diagnostics.csv"
ANSWER_RESULTS = RESULTS_DIR / "answer_generation_results.jsonl"
ANSWER_METRICS_JSON = RESULTS_DIR / "answer_metrics.json"
FAILURE_JSONL = RESULTS_DIR / "failure_analysis.jsonl"
LATENCY_CSV = RESULTS_DIR / "latency_results.csv"
EXPERIMENT_SUMMARY = RESULTS_DIR / "experiment_summary.md"
METRICS_JSON = RESULTS_DIR / "metrics.json"
INDEX_STATS = RESULTS_DIR / "index_stats.json"
RETRIEVAL_LATENCY = RESULTS_DIR / "retrieval_latency.json"
DIAGNOSTICS_JSON = RESULTS_DIR / "diagnostics.json"

# condition identifiers (kept as baseline/generated so reused vo_metrics funcs work)
COND_A = "baseline"  # original chunk vectors
COND_E = "generated"  # pooled atomic + chunk-level question vectors
CONDITIONS = (COND_A, COND_E)


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
