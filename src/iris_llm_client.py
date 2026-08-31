"""OpenAI-compatible client for the Iris development LLM endpoint."""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI


IRIS_LLM_BASE_URL = os.getenv("IRIS_LLM_BASE_URL", "https://chat-api-dev.iris.ai/v1/")
IRIS_LLM_API_KEY = os.getenv("IRIS_LLM_API_KEY", "EMPTY")
IRIS_LLM_MODEL = os.getenv("IRIS_LLM_MODEL", "Qwen/Qwen3.5-4B")
IRIS_LLM_TIMEOUT = float(os.getenv("IRIS_LLM_TIMEOUT", "180"))
IRIS_LLM_MAX_TOKENS = int(os.getenv("IRIS_LLM_MAX_TOKENS", "8192"))


@lru_cache(maxsize=1)
def get_iris_llm_client() -> OpenAI:
    """Return a client configured for Iris's OpenAI-compatible chat API."""
    return OpenAI(
        base_url=IRIS_LLM_BASE_URL,
        api_key=IRIS_LLM_API_KEY,
        timeout=IRIS_LLM_TIMEOUT,
        max_retries=0,
    )


def stream_iris_chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = IRIS_LLM_MAX_TOKENS,
    seed: int | None = None,
    json_mode: bool = False,
) -> str:
    """Collect one streamed response.

    Streaming sends response bytes before Iris's 60-second gateway deadline.
    Concurrent callers are still continuously batched by vLLM's scheduler.
    """
    kwargs = {
        "model": IRIS_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if seed is not None:
        kwargs["seed"] = seed
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    parts = []
    for event in get_iris_llm_client().chat.completions.create(**kwargs):
        if not event.choices:
            continue
        content = event.choices[0].delta.content
        if content:
            parts.append(content)
    return "".join(parts)


if __name__ == "__main__":
    print(
        stream_iris_chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Explain prefix tuning in two sentences.",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
    )
