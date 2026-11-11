"""Writer: produces the final report. The reference list is built from real sources, not the LLM."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CrewState
from .common import ask

SYSTEM = (
    "You are a technical writer. Write a clear Markdown report that answers the question using "
    "ONLY the analyst's notes. Structure: a title, a short summary, 2-4 themed sections, and a "
    "'Limitations' section. Keep the inline citations like [1] from the notes. Do not write a "
    "references list; it is added automatically.\n"
    "Rules for numbers: use a figure only if it appears in the notes, copy it exactly, and keep "
    "it next to the citation it came with. Never add figures, benchmarks or results of your own. "
    "Cite every statement of fact."
)


def render_references(sources: list[dict]) -> str:
    lines = [f"[{s['id']}] {s['cite']}" for s in sources]
    return "## References\n\n" + "\n".join(lines)


def make_writer(llm: BaseChatModel) -> Callable[[CrewState], dict]:
    def writer(state: CrewState) -> dict:
        sources = state.get("sources", [])
        body = ask(llm, SYSTEM, f"Question: {state['question']}\n\nNotes:\n{state.get('notes', '')}")
        body = body.strip() or f"# {state['question']}\n\n{state.get('notes', '')}"
        report = f"{body}\n\n{render_references(sources)}\n"
        return {"report": report, "log": ["writer: report drafted"]}

    return writer
