"""Analyst: extracts key findings from the gathered sources."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CrewState
from .common import ask, format_sources

SYSTEM = (
    "You are a research analyst. From the numbered sources, extract the key findings that answer "
    "the question. Group them by theme, note where sources agree or disagree, and flag weak "
    "evidence. Cite sources inline as [1], [2]. Use only the sources given. Do not invent facts."
)


def make_analyst(llm: BaseChatModel) -> Callable[[CrewState], dict]:
    def analyst(state: CrewState) -> dict:
        sources = state.get("sources", [])
        user = f"Question: {state['question']}\n\nSources:\n{format_sources(sources)}"
        notes = ask(llm, SYSTEM, user).strip()
        if not notes:
            # Degrade gracefully: hand the writer the raw material instead of nothing.
            notes = "Analysis unavailable. Raw sources:\n" + format_sources(sources, 300)
        return {"notes": notes, "log": ["analyst: notes ready"]}

    return analyst
