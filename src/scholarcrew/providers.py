"""LLM provider presets and a preflight check, used by the bring-your-own-key web app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .config import Settings
from .llm import make_llm


@dataclass(frozen=True)
class Provider:
    base_url: str
    default_model: str
    key_url: str


PROVIDERS: dict[str, Provider] = {
    "Groq (free)": Provider(
        "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "https://console.groq.com/keys"
    ),
    "Google Gemini (free)": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash",
        "https://aistudio.google.com/apikey",
    ),
    "Custom (OpenAI-compatible)": Provider("", "", ""),
}


def settings_for(base_url: str, api_key: str, model: str) -> Settings:
    """Build Settings from user input. The key lives only in this object, never on disk."""
    return Settings(
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        model=model.strip(),
        max_steps=8,
        reports_dir=Path("reports"),
    )


def problems(settings: Settings) -> str | None:
    """Return a human-readable problem with the inputs, or None if they look usable."""
    if not settings.api_key:
        return "Paste your API key in the sidebar to start."
    if not settings.base_url.startswith(("http://", "https://")):
        return "The base URL must start with http:// or https://"
    if not settings.model:
        return "Enter a model name."
    return None


def check_llm(settings: Settings, llm: BaseChatModel | None = None) -> str | None:
    """Make one tiny call so a bad key or model name fails loudly before the agents run.

    Returns an error message, or None if the model answered.
    """
    try:
        (llm or make_llm(settings)).invoke([HumanMessage(content="Reply with the single word: ok")])
    except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
        message = str(exc).replace(settings.api_key, "***") if settings.api_key else str(exc)
        return f"{type(exc).__name__}: {message[:300]}"
    return None
