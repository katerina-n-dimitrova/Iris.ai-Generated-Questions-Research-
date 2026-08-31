"""
Pluggable embedding backends.

Default backend is a Hugging Face sentence-embedding model
(Octen/Octen-Embedding-0.6B) loaded via sentence-transformers. OpenAI
embeddings remain available by setting EMBEDDING_BACKEND=openai in .env, and
the hosted Iris embedding service via EMBEDDING_BACKEND=iris.

We compute embeddings ourselves (rather than via Chroma's embedding_function)
so we can time query-embedding latency separately from Chroma search latency,
which the latency logs require.

Configuration (.env):
    EMBEDDING_BACKEND      huggingface | openai | iris   (default: huggingface)
    HF_EMBEDDING_MODEL     default: Octen/Octen-Embedding-0.6B
    HF_EMBEDDING_DEVICE    cpu | cuda | mps          (default: auto)
    HF_QUERY_PROMPT_NAME   optional prompt name for query encoding
    OPENAI_EMBEDDING_MODEL used only when backend=openai
    IRIS_EMBEDDINGS_URL    used only when backend=iris
                           (default: https://llm-api-dev.iris.ai/embeddings/generate/)
    IRIS_API_KEY           optional bearer token for the Iris service
"""

from __future__ import annotations

import os
import random
import time
from functools import lru_cache
from typing import List, Optional, Sequence

from ragkit import config

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "Octen/Octen-Embedding-0.6B")
HF_EMBEDDING_DEVICE = os.getenv("HF_EMBEDDING_DEVICE", "").strip() or None
HF_QUERY_PROMPT_NAME = os.getenv("HF_QUERY_PROMPT_NAME", "").strip() or None
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", "32"))

IRIS_EMBEDDINGS_URL = os.getenv(
    "IRIS_EMBEDDINGS_URL", "https://llm-api-dev.iris.ai/embeddings/generate/"
)
IRIS_API_KEY = os.getenv("IRIS_API_KEY", "").strip()
IRIS_BATCH_SIZE = int(os.getenv("IRIS_BATCH_SIZE", "64"))
IRIS_TIMEOUT = int(os.getenv("IRIS_TIMEOUT", "120"))
IRIS_MAX_RETRIES = int(os.getenv("IRIS_MAX_RETRIES", "6"))


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class HFEmbedder(BaseEmbedder):
    """sentence-transformers backend (default: Octen-Embedding-0.6B)."""

    def __init__(
        self,
        model_name: str = HF_EMBEDDING_MODEL,
        device: Optional[str] = HF_EMBEDDING_DEVICE,
    ):
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
            normalize_embeddings=True,  # cosine-ready
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
            out.extend(self._embed(texts[i : i + B]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class IrisEmbedder(BaseEmbedder):
    """Hosted Iris embedding service (llm-api-dev.iris.ai/embeddings/generate/).

    POSTs {"texts": [...]} and receives {"result": [[...vector...], ...]},
    one vector per input text. The vector dimension (384 for the current model)
    is discovered from the first response rather than hard-coded.
    """

    def __init__(self, url: str = IRIS_EMBEDDINGS_URL, api_key: str = IRIS_API_KEY):
        import requests

        self.name = url
        self.url = url
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.update(headers)
        self.dim = 0  # filled in after the first successful call

    def _post(self, texts: Sequence[str]) -> List[List[float]]:
        import requests

        expected = len(texts)
        for attempt in range(1, IRIS_MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    self.url, json={"texts": list(texts)}, timeout=IRIS_TIMEOUT
                )
                resp.raise_for_status()
                data = resp.json()
                vecs = data.get("result") if isinstance(data, dict) else data
                if not isinstance(vecs, list) or len(vecs) != expected:
                    raise ValueError(
                        f"Iris returned {len(vecs) if isinstance(vecs, list) else 'invalid'} "
                        f"vectors for {expected} texts"
                    )
                if vecs and (not isinstance(vecs[0], list) or not vecs[0]):
                    raise ValueError("Iris returned an invalid vector payload")
                if vecs and not self.dim:
                    self.dim = len(vecs[0])
                return vecs
            except (requests.RequestException, ValueError, KeyError) as error:
                if attempt == IRIS_MAX_RETRIES:
                    raise
                delay = min(30.0, 2 ** (attempt - 1)) + random.uniform(0, 0.75)
                print(
                    f"[iris-embedding:retry] attempt={attempt}/{IRIS_MAX_RETRIES} "
                    f"delay={delay:.1f}s error={error}",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Iris embedding retry loop exited unexpectedly")

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), IRIS_BATCH_SIZE):
            out.extend(self._post(texts[i : i + IRIS_BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Return the configured embedder (cached so the HF model loads once)."""
    if EMBEDDING_BACKEND == "openai":
        print(f"[embeddings] backend=openai model={config.OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbedder()
    if EMBEDDING_BACKEND == "iris":
        print(f"[embeddings] backend=iris url={IRIS_EMBEDDINGS_URL}")
        emb = IrisEmbedder()
        # Probe once so `dim` is known before indexing starts (callers that
        # build Chroma collections read emb.dim up front).
        emb.embed_query("dimension probe")
        return emb
    print(f"[embeddings] backend=huggingface model={HF_EMBEDDING_MODEL}")
    return HFEmbedder()


def embedding_signature() -> str:
    """Short string identifying the active embedding model, for logs."""
    emb = get_embedder()
    return f"{EMBEDDING_BACKEND}:{emb.name}:dim{emb.dim}"
