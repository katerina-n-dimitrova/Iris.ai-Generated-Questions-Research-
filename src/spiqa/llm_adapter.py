"""
LLM adapter behind an OpenAI-compatible interface.

Two backends are supported and selected via ``SPIQA_LLM_BACKEND``:

    openai   (default) : the official ``openai`` Python package.
    litellm            : the optional ``litellm`` wrapper, which speaks the same
                         OpenAI chat-completions schema but can route to many
                         providers.

Design rules honoured here:
* The API key is ALWAYS read from the environment (``OPENAI_API_KEY``), never
  hard-coded. We fail fast with a clear message if it is missing/placeholder.
* A single ``chat()`` entry point returns ``(text, usage)`` so callers can track
  token usage / cost uniformly regardless of backend.
* An optional ``base_url`` (``OPENAI_BASE_URL``) is honoured so the same code can
  talk to any OpenAI-compatible gateway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import spiqa_config as C


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self


def _require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is not set (or is still the placeholder). "
            "Add it to the project .env — never hard-code it in source."
        )
    return key


class LLMAdapter:
    """Minimal OpenAI-compatible chat wrapper with pluggable backend."""

    def __init__(self, model: str = C.LLM_MODEL, backend: Optional[str] = None):
        self.model = model
        self.backend = (backend or os.getenv("SPIQA_LLM_BACKEND", "openai")).lower()
        self.base_url = os.getenv("OPENAI_BASE_URL") or None
        self._api_key = _require_api_key()

        if self.backend == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        elif self.backend == "litellm":
            import litellm  # noqa: F401  (imported for its side effects/availability)

            self._litellm = litellm
        else:
            raise ValueError(
                f"Unknown SPIQA_LLM_BACKEND={self.backend!r} "
                "(expected 'openai' or 'litellm')."
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = C.LLM_TEMPERATURE,
        max_tokens: int = 1500,
    ) -> Tuple[str, Usage]:
        """Send a chat-completion request. Returns (content, usage)."""
        if self.backend == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            usage = Usage(
                getattr(u, "prompt_tokens", 0) or 0,
                getattr(u, "completion_tokens", 0) or 0,
            )
            return text.strip(), usage

        # litellm — same schema; it reads OPENAI_API_KEY from the env itself.
        resp = self._litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
            base_url=self.base_url,
        )
        text = resp["choices"][0]["message"]["content"] or ""
        u = resp.get("usage", {}) or {}
        usage = Usage(
            u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
        )
        return text.strip(), usage

    def signature(self) -> str:
        return f"{self.backend}:{self.model}"


@lru_cache(maxsize=4)
def get_llm(model: str = C.LLM_MODEL, backend: Optional[str] = None) -> LLMAdapter:
    """Return a cached adapter (so the client is constructed once)."""
    return LLMAdapter(model=model, backend=backend)


def check_llm(live: bool = False) -> str:
    """
    Verify the LLM is reachable. With ``live=False`` we only confirm the key is
    present and the client constructs. With ``live=True`` we make one tiny real
    call to prove end-to-end connectivity.
    """
    llm = get_llm()
    if not live:
        return f"OK (constructed) backend={llm.signature()} (no live call)"
    text, usage = llm.chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
        temperature=0.0,
    )
    return f"OK (live) backend={llm.signature()} reply={text!r} tokens={usage.total_tokens}"
