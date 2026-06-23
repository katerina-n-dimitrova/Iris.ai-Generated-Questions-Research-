"""
Central configuration for the context-enrichment-RAG experiment.

Everything that the rest of the pipeline needs to agree on lives here:
paths, env-driven knobs, the dataset registry, and a couple of tiny helpers
for talking to OpenAI. Import from this module rather than re-reading the
environment in every script so that behaviour stays consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives in <project>/src, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (no-op if the file is missing).
load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


DATA_DIR = _resolve(os.getenv("DATA_DIR", "./data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = _resolve(os.getenv("RESULTS_DIR", "./results"))
LATENCY_LOG_DIR = RESULTS_DIR / "latency_logs"
RETRIEVAL_METRICS_DIR = RESULTS_DIR / "retrieval_metrics"
ANSWER_METRICS_DIR = RESULTS_DIR / "answer_metrics"

CHROMA_PERSIST_DIR = _resolve(os.getenv("CHROMA_PERSIST_DIR", "./chroma_indexes"))

# ChromaDB connection mode: "local" (on-disk) or "cloud" (hosted Chroma Cloud).
CHROMA_MODE = os.getenv("CHROMA_MODE", "local").lower()
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")

# --------------------------------------------------------------------------- #
# OpenAI / model config
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "context_enrichment_rag")

# --------------------------------------------------------------------------- #
# Experiment knobs
# --------------------------------------------------------------------------- #
MAX_DATASET_SAMPLES = int(os.getenv("MAX_DATASET_SAMPLES", "300"))
TOP_K = int(os.getenv("TOP_K", "5"))

# k values we report retrieval metrics at. TOP_K is always included.
RETRIEVAL_K_VALUES = sorted({5, 10, TOP_K})

CONDITIONS = ("baseline", "enriched")


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset used in the experiment."""

    name: str               # short key, e.g. "scifact"
    hf_id: str              # Hugging Face dataset id
    hf_config: Optional[str]  # HF config name, if any
    input_type: str         # structured_text | unstructured_text | table | chart
    description: str

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.name

    def processed_path(self, condition: str) -> Path:
        return PROCESSED_DIR / f"{self.name}_{condition}.jsonl"

    def collection_name(self, condition: str) -> str:
        # Chroma collection names must be 3-63 chars; the prefix keeps them
        # unique if several experiments share one persist dir.
        return f"{self.name}_{condition}"


DATASETS: Dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        # allenai/scifact uses a (now-unsupported) loading script; the BeIR
        # mirror is parquet-based with the same BEIR corpus/queries/qrels layout.
        name="scifact",
        hf_id="BeIR/scifact",
        hf_config="corpus",
        input_type="structured_text",
        description="Structured scientific text (SciFact via BeIR mirror).",
    ),
    "nfcorpus": DatasetSpec(
        name="nfcorpus",
        hf_id="BeIR/nfcorpus",
        hf_config="corpus",
        input_type="unstructured_text",
        description="Unstructured biomedical text (BEIR NFCorpus).",
    ),
    "wikitablequestions": DatasetSpec(
        # stanfordnlp/wikitablequestions uses a loading script; lighteval mirror
        # is parquet-based with id/question/answers/table{header,rows}.
        name="wikitablequestions",
        hf_id="lighteval/wikitablequestions",
        hf_config=None,
        input_type="table",
        description="Tables (WikiTableQuestions via lighteval parquet mirror).",
    ),
    "chartqa": DatasetSpec(
        name="chartqa",
        hf_id="HuggingFaceM4/ChartQA",
        hf_config=None,
        input_type="chart",
        description="Charts / graphs (ChartQA).",
    ),
}

ALL_DATASETS = list(DATASETS.keys())


# --------------------------------------------------------------------------- #
# OpenAI helpers
# --------------------------------------------------------------------------- #
def get_openai_client():
    """Return an OpenAI client, raising a clear error if the key is missing."""
    from openai import OpenAI

    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def get_chroma_client():
    """
    Return a ChromaDB client based on CHROMA_MODE.

    - "local" (default): on-disk PersistentClient at CHROMA_PERSIST_DIR.
      No credentials required.
    - "cloud": hosted Chroma Cloud via CloudClient. Requires CHROMA_API_KEY,
      CHROMA_TENANT, and CHROMA_DATABASE to be set in .env.
    """
    import chromadb

    if CHROMA_MODE == "cloud":
        missing = [k for k, v in {
            "CHROMA_API_KEY": CHROMA_API_KEY,
            "CHROMA_TENANT": CHROMA_TENANT,
            "CHROMA_DATABASE": CHROMA_DATABASE,
        }.items() if not v or v.startswith("your_")]
        if missing:
            raise RuntimeError(
                "CHROMA_MODE=cloud but missing credentials: "
                + ", ".join(missing)
                + ". Fill them in .env or set CHROMA_MODE=local."
            )
        return chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )

    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def chroma_signature() -> str:
    """Short string identifying the active Chroma backend, for logs."""
    if CHROMA_MODE == "cloud":
        return f"cloud:{CHROMA_TENANT}/{CHROMA_DATABASE}"
    return f"local:{CHROMA_PERSIST_DIR.name}"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        CHROMA_PERSIST_DIR,
        LATENCY_LOG_DIR,
        RETRIEVAL_METRICS_DIR,
        ANSWER_METRICS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# Create directories on import so downstream scripts can assume they exist.
ensure_dirs()
