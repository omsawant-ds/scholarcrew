"""Analyst: extracts key findings from the gathered sources."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..grounding import normalize_citations
from ..state import CrewState
from .common import ask, format_sources

SYSTEM = (
    "You are a research analyst. From the numbered sources, extract the key findings that answer "
    "the question. Group them by theme, note where sources agree or disagree, and flag weak "
    "evidence. Cite sources inline as [1], [2]. Use only the sources given. Do not invent facts.\n"
    "Rules for numbers: state a figure (a percentage, score or measurement) only if it appears "
    "word for word in that source's excerpt, and cite that source next to it. Never calculate, "
    "round, convert or estimate figures. If a source gives no figure, describe its finding in "
    "words. Never describe results the excerpt does not state."
)


def make_analyst(llm: BaseChatModel) -> Callable[[CrewState], dict]:
    def analyst(state: CrewState) -> dict:
        sources = state.get("sources", [])
        user = f"Question: {state['question']}\n\nSources:\n{format_sources(sources)}"
        notes = normalize_citations(ask(llm, SYSTEM, user)).strip()
        if not notes:
            # Degrade gracefully: hand the writer the raw material instead of nothing.
            notes = "Analysis unavailable. Raw sources:\n" + format_sources(sources, 300)
        return {"notes": notes, "log": ["analyst: notes ready"]}

    return analyst
