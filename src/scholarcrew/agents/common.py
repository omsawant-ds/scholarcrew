"""Helpers shared by agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger("scholarcrew")


def ask(llm: BaseChatModel, system: str, user: str) -> str:
    """One LLM call. Returns text; returns '' if the call fails so callers can fall back.

    Failures are logged as warnings so a wrong model name or bad key is visible, not silent.
    """
    try:
        reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001 - provider errors vary; agents have fallbacks
        log.warning("LLM call failed (%s): %s", type(exc).__name__, str(exc)[:200])
        return ""
    content = reply.content
    return content if isinstance(content, str) else str(content)


def parse_json(text: str) -> Any | None:
    """Pull the first JSON object/array out of an LLM reply. Free models often add prose or fences."""
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def format_sources(sources: list[dict], limit_chars: int = 700) -> str:
    return "\n\n".join(
        f"[{s['id']}] ({s['kind']}) {s['title']}\n{s['text'][:limit_chars]}\n{s['url']}"
        for s in sources
    )
