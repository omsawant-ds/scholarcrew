"""LLM factory. Any OpenAI-compatible endpoint works (Groq, Gemini, Ollama, vLLM...)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings


def make_llm(settings: Settings, temperature: float = 0.2) -> BaseChatModel:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=temperature,
        timeout=120,
        # Free tiers cap tokens per minute. The client honors the provider's retry-after header,
        # so more retries means waiting out the limit instead of failing the request.
        max_retries=6,
    )
