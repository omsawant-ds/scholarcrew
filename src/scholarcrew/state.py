"""Shared state passed between agents in the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Source(TypedDict):
    id: int
    kind: str  # "arxiv" | "web"
    title: str
    url: str
    text: str
    cite: str


class CrewState(TypedDict, total=False):
    question: str
    sources: list[Source]
    notes: str
    report: str
    next: str
    steps: int
    # `operator.add` makes LangGraph append log lines from every node instead of overwriting.
    log: Annotated[list[str], operator.add]
