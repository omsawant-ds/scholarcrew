"""arXiv search tool. Uses the public arXiv API (free, no key)."""

from __future__ import annotations

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


def search_arxiv(query: str, max_results: int = 5, client: httpx.Client | None = None) -> list[Paper]:
    """Return the most relevant arXiv papers for a query. Returns [] on network errors."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        if client is not None:
            resp = client.get(ARXIV_API, params=params, timeout=20)
        else:
            resp = httpx.get(ARXIV_API, params=params, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        return parse_feed(resp.text)
    except (httpx.HTTPError, ET.ParseError):
        return []
