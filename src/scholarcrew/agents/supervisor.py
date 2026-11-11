"""Supervisor: decides which agent runs next, with validation and a deterministic fallback."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CrewState
from .common import ask, parse_json

AGENTS = ("researcher", "analyst", "writer")

SYSTEM = (
    "You are the supervisor of a research team: researcher (finds sources), analyst (extracts "
    "key findings), writer (writes the final report). Given the progress so far, choose who "
    'works next. Reply with JSON only: {"next": "researcher" | "analyst" | "writer" | "FINISH"}.'
)


def fallback_next(state: CrewState) -> str:
    """Fixed pipeline order. Used when the LLM answer is missing, invalid or repeats finished work."""
    if not state.get("sources"):
        return "researcher"
    if not state.get("notes"):
        return "analyst"
    if not state.get("report"):
        return "writer"
    return "FINISH"


def _done(state: CrewState, agent: str) -> bool:
    key = {"researcher": "sources", "analyst": "notes", "writer": "report"}[agent]
    return bool(state.get(key))


def make_supervisor(llm: BaseChatModel, max_steps: int) -> Callable[[CrewState], dict]:
    def supervisor(state: CrewState) -> dict:
        steps = state.get("steps", 0) + 1
        expected = fallback_next(state)

        if steps > max_steps:
            # Out of budget: write whatever we have, or stop.
            choice = "writer" if state.get("sources") and not state.get("report") else "FINISH"
            return {"next": choice, "steps": steps, "log": [f"supervisor: step cap hit -> {choice}"]}

        progress = (
            f"Question: {state['question']}\n"
            f"Sources found: {len(state.get('sources', []))}\n"
            f"Analysis done: {bool(state.get('notes'))}\n"
            f"Report written: {bool(state.get('report'))}"
        )
        parsed = parse_json(ask(llm, SYSTEM, progress))
        choice = parsed.get("next") if isinstance(parsed, dict) else None

        valid = choice == "FINISH" and expected == "FINISH"
        valid = valid or (choice in AGENTS and not _done(state, choice))
        if not valid:
            choice = expected
        return {"next": choice, "steps": steps, "log": [f"supervisor -> {choice}"]}

    return supervisor
