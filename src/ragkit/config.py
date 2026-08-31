"""
Central configuration for the context-enrichment-RAG experiments.

Everything that the rest of the pipeline needs to agree on lives here:
the project root, `.env` loading, and the OpenAI chat/embedding settings.
Import from this module rather than re-reading the environment in every
script so that behaviour stays consistent.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives in <project>/src/ragkit, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the project root (no-op if the file is missing).
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# OpenAI / model config
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_openai_client():
    """Return an OpenAI client, raising a clear error if the key is missing."""
    from openai import OpenAI

    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=OPENAI_API_KEY)
