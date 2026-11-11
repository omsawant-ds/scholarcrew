"""LangGraph wiring: supervisor <-> researcher / analyst / writer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .agents import make_analyst, make_researcher, make_supervisor, make_writer
from .config import Settings
from .state import CrewState
from .tools import save_report, search_arxiv, search_web


def build_graph(
    llm: BaseChatModel,
    max_steps: int = 8,
    arxiv_search: Callable = search_arxiv,
    web_search: Callable = search_web,
):
    graph = StateGraph(CrewState)
    graph.add_node("supervisor", make_supervisor(llm, max_steps))
    graph.add_node("researcher", make_researcher(llm, arxiv_search, web_search))
    graph.add_node("analyst", make_analyst(llm))
    graph.add_node("writer", make_writer(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "FINISH": END},
    )
    for agent in ("researcher", "analyst", "writer"):
        graph.add_edge(agent, "supervisor")
    return graph.compile()


def stream_crew(
    question: str, llm: BaseChatModel, settings: Settings, **tools: Callable
) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, state_update) as each agent finishes. Used by the CLI and the web UI."""
    app = build_graph(llm, settings.max_steps, **tools)
    config = {"recursion_limit": settings.max_steps * 2 + 6}
    for event in app.stream({"question": question, "steps": 0, "log": []}, config, stream_mode="updates"):
        yield from event.items()


def run_crew(
    question: str, llm: BaseChatModel, settings: Settings, **tools: Callable
) -> tuple[str, Path | None]:
    """Run the whole crew. Returns (report_markdown, saved_path)."""
    report = ""
    for _node, update in stream_crew(question, llm, settings, **tools):
        report = update.get("report", report)
    path = save_report(question, report, settings.reports_dir) if report else None
    return report, path
