"""Researcher: plans search queries, then gathers sources from arXiv and the web."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for
from itertools import zip_longest

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CrewState, Source
from ..tools.arxiv import Paper
from ..tools.web import WebResult
from .common import ask, parse_json

SYSTEM = (
    "You are a research assistant. Turn the user's question into 2 or 3 short, distinct search "
    'queries for finding papers and articles. Reply with JSON only: {"queries": ["...", "..."]}.'
)


# The most time the searches may take before the researcher works with what has arrived. The
# search services vary a lot (arXiv in particular), and a slow one should cost a few sources,
# not half a minute.
SEARCH_BUDGET_SECONDS = 8.0


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
    search_budget: float = SEARCH_BUDGET_SECONDS,
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
        # run alongside the web searches. Each job records results as they arrive, so if the time
        # budget runs out the researcher keeps whatever finished instead of waiting for the rest.
        arxiv_results: list = []
        web_results: list = []

        def collect(search: Callable, into: list) -> None:
            for q in queries:
                into.append(search(q, per_query))

        pool = ThreadPoolExecutor(max_workers=2)
        jobs = [pool.submit(collect, arxiv_search, arxiv_results),
                pool.submit(collect, web_search, web_results)]
        _, late = wait_for(jobs, timeout=search_budget)
        pool.shutdown(wait=False)  # a straggler finishes (or times out) on its own, unobserved
        notes = f" (stopped waiting after {search_budget:.0f}s)" if late else ""

        # Added in the same order as a sequential run, so source numbering does not change.
        for papers, pages in zip_longest(list(arxiv_results), list(web_results), fillvalue=[]):
            for p in papers:
                add("arxiv", p.title, p.url, p.summary, p.cite())
            for w in pages:
                add("web", w.title, w.url, w.snippet, f"{w.title}. {w.url}")

        return {"sources": sources, "log": [f"researcher: {len(sources)} sources from {queries}{notes}"]}

    return researcher
