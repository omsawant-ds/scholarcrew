"""Web search tool. Uses DuckDuckGo via the `ddgs` package (free, no key)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebResult:
    title: str
    snippet: str
    url: str


def search_web(query: str, max_results: int = 5) -> list[WebResult]:
    """Return web results for a query. Returns [] if the search backend fails."""
    try:
        from ddgs import DDGS

        raw = DDGS().text(query, max_results=max_results)
    except Exception:  # noqa: BLE001 - search backends fail in many ways; degrade gracefully
        return []
    return [
        WebResult(
            title=r.get("title", ""),
            snippet=r.get("body", ""),
            url=r.get("href", ""),
        )
        for r in raw or []
    ]
