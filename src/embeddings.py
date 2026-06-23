"""
Pluggable embedding backends.

Default backend is a Hugging Face sentence-embedding model
(Octen/Octen-Embedding-0.6B) loaded via sentence-transformers. OpenAI
embeddings remain available by setting EMBEDDING_BACKEND=openai in .env.

We compute embeddings ourselves (rather than via Chroma's embedding_function)
so we can time query-embedding latency separately from Chroma search latency,
which the latency logs require.

Configuration (.env):
    EMBEDDING_BACKEND      huggingface | openai      (default: huggingface)
    HF_EMBEDDING_MODEL     default: Octen/Octen-Embedding-0.6B
    HF_EMBEDDING_DEVICE    cpu | cuda | mps          (default: auto)
    HF_QUERY_PROMPT_NAME   optional prompt name for query encoding
    OPENAI_EMBEDDING_MODEL used only when backend=openai
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Sequence

import config

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "Octen/Octen-Embedding-0.6B")
HF_EMBEDDING_DEVICE = os.getenv("HF_EMBEDDING_DEVICE", "").strip() or None
HF_QUERY_PROMPT_NAME = os.getenv("HF_QUERY_PROMPT_NAME", "").strip() or None
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", "32"))


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class HFEmbedder(BaseEmbedder):
    """sentence-transformers backend (default: Octen-Embedding-0.6B)."""

    def __init__(self, model_name: str = HF_EMBEDDING_MODEL,
                 device: Optional[str] = HF_EMBEDDING_DEVICE):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        # trust_remote_code is needed for many recent embedding models that
        # ship custom pooling/architecture code.
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._query_prompt = HF_QUERY_PROMPT_NAME

    def _encode(self, texts: Sequence[str], is_query: bool) -> List[List[float]]:
        kwargs = dict(
            batch_size=HF_BATCH_SIZE,
            normalize_embeddings=True,   # cosine-ready
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Some MTEB models expect a query prompt; pass it only if configured
        # and only for queries.
        if is_query and self._query_prompt:
            kwargs["prompt_name"] = self._query_prompt
        vecs = self.model.encode(list(texts), **kwargs)
        return vecs.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings backend (text-embedding-3-*)."""

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = config.OPENAI_EMBEDDING_MODEL):
        self.name = model_name
        self.client = config.get_openai_client()
        self.dim = self._DIMS.get(model_name, 1536)

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.name, input=list(texts))
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        # batch to stay well under request limits
        out: List[List[float]] = []
        B = 128
        for i in range(0, len(texts), B):
            out.extend(self._embed(texts[i:i + B]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Return the configured embedder (cached so the HF model loads once)."""
    if EMBEDDING_BACKEND == "openai":
        print(f"[embeddings] backend=openai model={config.OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbedder()
    print(f"[embeddings] backend=huggingface model={HF_EMBEDDING_MODEL}")
    return HFEmbedder()


def embedding_signature() -> str:
    """Short string identifying the active embedding model, for logs."""
    emb = get_embedder()
    return f"{EMBEDDING_BACKEND}:{emb.name}:dim{emb.dim}"
