"""LLM factory. Any OpenAI-compatible endpoint works (Groq, Gemini, Ollama, vLLM...)."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings

log = logging.getLogger("scholarcrew")


def make_llm(settings: Settings, temperature: float = 0.2, max_retries: int = 6) -> BaseChatModel:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=temperature,
        timeout=120,
        # Free tiers cap tokens per minute. The client honors the provider's retry-after header,
        # so more retries means waiting out the limit instead of failing the request.
        max_retries=max_retries,
    )


_DAILY_LIMIT = re.compile(r"per day|\bTPD\b|\bRPD\b|quota", re.IGNORECASE)


def key_unusable(exc: Exception) -> bool:
    """True when switching to another key could help: the key is rejected, or its daily quota is
    spent. A per-minute limit is not one of these: waiting a moment fixes it, and the client
    already does that."""
    # The client library's classes carry a vendor prefix in some versions (OpenAIRateLimitError),
    # so match on how the name ends.
    name = type(exc).__name__
    if name.endswith(("AuthenticationError", "PermissionDeniedError")):
        return True
    return name.endswith("RateLimitError") and bool(_DAILY_LIMIT.search(str(exc)))


class KeyPool:
    """Several models for the same endpoint, one per API key. Calls go to the first key that
    works; when a key is rejected or out of daily quota, the pool moves on to the next one and
    repeats the call. A spent key is not tried again for the life of the pool."""

    def __init__(self, llms: list):
        if not llms:
            raise ValueError("KeyPool needs at least one model")
        self._llms, self._current = llms, 0

    @property
    def keys_left(self) -> int:
        return len(self._llms) - self._current

    def invoke(self, *args, **kwargs):
        while True:
            try:
                return self._llms[self._current].invoke(*args, **kwargs)
            except Exception as exc:  # only key problems are handled here
                if not key_unusable(exc) or self._current + 1 >= len(self._llms):
                    raise
                log.warning("API key %d of %d is unusable (%s); switching to the next",
                            self._current + 1, len(self._llms), type(exc).__name__)
                self._current += 1


def make_llm_pool(settings: Settings, temperature: float = 0.2) -> KeyPool:
    """A model that falls back through every key in the settings. Fewer retries per key than the
    app uses, so a spent daily quota moves on at once instead of waiting it out."""
    return KeyPool([
        make_llm(replace(settings, api_key=key), temperature, max_retries=2)
        for key in settings.all_keys
    ])
