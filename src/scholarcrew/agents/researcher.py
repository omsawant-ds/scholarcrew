"""Researcher: plans search queries, then gathers sources from arXiv and the web."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CrewState, Source
from ..tools.arxiv import Paper
from ..tools.web import WebResult
from .common import ask, parse_json

SYSTEM = (
    "You are a research assistant. Turn the user's question into 2 or 3 short, distinct search "
    'queries for finding papers and articles. Reply with JSON only: {"queries": ["...", "..."]}.'
)


def plan_queries(llm: BaseChatModel, question: str) -> list[str]:
    parsed = parse_json(ask(llm, SYSTEM, question))
    queries = parsed.get("queries") if isinstance(parsed, dict) else None
    if isinstance(queries, list):
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if cleaned:
            return cleaned[:3]
    return [question]


def make_researcher(
    llm: BaseChatModel,
    arxiv_search: Callable[[str, int], list[Paper]],
    web_search: Callable[[str, int], list[WebResult]],
    per_query: int = 4,
) -> Callable[[CrewState], dict]:
    def researcher(state: CrewState) -> dict:
        queries = plan_queries(llm, state["question"])
        sources: list[Source] = []
        seen: set[str] = set()

        def add(kind: str, title: str, url: str, text: str, cite: str) -> None:
            if url and url not in seen:
                seen.add(url)
                sources.append(
                    Source(id=len(sources) + 1, kind=kind, title=title, url=url, text=text, cite=cite)
                )

        # arXiv asks clients to send one request at a time, so its queries stay in sequence. They
        # run alongside the web searches, which roughly halves the wait. Results are added in
        # the same order as a sequential run, so source numbering does not change.
        with ThreadPoolExecutor(max_workers=2) as pool:
            arxiv_job = pool.submit(lambda: [arxiv_search(q, per_query) for q in queries])
            web_job = pool.submit(lambda: [web_search(q, per_query) for q in queries])
            arxiv_results, web_results = arxiv_job.result(), web_job.result()

        for papers, pages in zip(arxiv_results, web_results):
            for p in papers:
                add("arxiv", p.title, p.url, p.summary, p.cite())
            for w in pages:
                add("web", w.title, w.url, w.snippet, f"{w.title}. {w.url}")

        return {"sources": sources, "log": [f"researcher: {len(sources)} sources from {queries}"]}

    return researcher
