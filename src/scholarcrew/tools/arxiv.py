"""arXiv search tool. Uses the public arXiv API (free, no key)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class Paper:
    title: str
    authors: list[str]
    summary: str
    url: str
    published: str

    def cite(self) -> str:
        authors = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        return f"{authors} ({self.published[:4]}). {self.title}. {self.url}"


def _text(entry: ET.Element, tag: str) -> str:
    return (entry.findtext(f"a:{tag}", default="", namespaces=_NS) or "").strip()


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    return [
        Paper(
            title=" ".join(_text(entry, "title").split()),
            authors=[
                (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
                for a in entry.findall("a:author", _NS)
            ],
            summary=" ".join(_text(entry, "summary").split()),
            url=_text(entry, "id"),
            published=_text(entry, "published"),
        )
        for entry in root.findall("a:entry", _NS)
    ]


_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "to", "using", "what", "which", "who", "why", "with", "can", "could", "should", "would", "researchers", "researcher"]
)


def build_query(query: str, operator: str = "AND", max_terms: int = 6) -> str:
    """Turn a natural-language question into arXiv keyword terms.

    A raw sentence matches loosely (arXiv ranks any word in any field), so drop stopwords and
    require the remaining keywords together.
    """
    terms = [t for t in re.findall(r"[A-Za-z0-9-]+", query) if t.lower() not in _STOPWORDS]
    terms = list(dict.fromkeys(terms))[:max_terms] or [query]
    return f" {operator} ".join(f"all:{t}" for t in terms)


def _fetch(search_query: str, max_results: int, client: httpx.Client | None) -> list[Paper] | None:
    """Papers for one query. [] means arXiv answered with no matches; None means it did not answer."""
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        if client is not None:
            resp = client.get(ARXIV_API, params=params, timeout=12)
        else:
            resp = httpx.get(ARXIV_API, params=params, timeout=12, follow_redirects=True)
        resp.raise_for_status()
        return parse_feed(resp.text)
    except (httpx.HTTPError, ET.ParseError):
        return None


def search_arxiv(query: str, max_results: int = 5, client: httpx.Client | None = None) -> list[Paper]:
    """Return the most relevant arXiv papers for a query. Returns [] on network errors.

    Tries all keywords together first; if that finds nothing, retries matching any keyword. If
    arXiv does not answer at all (a timeout or an error) it gives up at once: a second request to
    a service that is already slow only doubles the wait.
    """
    papers = _fetch(build_query(query), max_results, client)
    if papers is None:
        return []
    return papers or _fetch(build_query(query, "OR"), max_results, client) or []
