"""LLM-as-judge scoring for faithfulness and relevance.

Known limitation: a judge from the same model family as the generator tends to rate it
generously. Use `--judge-model` to judge with a different model when you can.
"""

from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel

from ..agents.common import ask, parse_json

FAITH_SYSTEM = (
    "You are a strict fact-checker. For each numbered claim, decide whether the cited source "
    "excerpts explicitly support it. A claim is supported only if the excerpts state it or clearly "
    "imply it. Do not use outside knowledge. Reply with JSON only: "
    '{"verdicts": [{"id": 1, "supported": true}, {"id": 2, "supported": false}]}'
)

RELEVANCE_SYSTEM = (
    "You are grading a research report. Rate from 1 to 5 how directly and completely it answers "
    "the question: 5 = fully answers it with a clear structure and specific details, 3 = partly "
    "answers it or is vague, 1 = off-topic or empty. Reply with JSON only: "
    '{"score": 4, "reason": "one short sentence"}'
)


_pause = time.sleep  # a module-level name so tests can replace it


def _ask_json(llm: BaseChatModel, system: str, user: str, retries: int = 2, wait: float = 20.0):
    """Ask for JSON. If the reply is empty or unparseable, wait and try again.

    Empty replies usually mean the provider rate-limited the call (free tiers cap tokens per
    minute), so waiting a little is what actually helps.
    """
    for attempt in range(retries + 1):
        parsed = parse_json(ask(llm, system, user))
        if parsed is not None:
            return parsed
        if attempt < retries:
            _pause(wait)
    return None


def judge_faithfulness(llm: BaseChatModel, claims: list[dict], sources: list[dict]) -> float | None:
    """Share of cited claims the excerpts support, or None if the judge could not answer."""
    if not claims:
        return None
    cited = {i for claim in claims for i in claim["cited"]}
    excerpts = "\n\n".join(
        f"[{s['id']}] {s['title']}\n{s['text'][:1200]}" for s in sources if s["id"] in cited
    )
    listing = "\n".join(
        f"{n}. {c['text']} (cites {', '.join(f'[{i}]' for i in c['cited'])})"
        for n, c in enumerate(claims, 1)
    )
    parsed = _ask_json(llm, FAITH_SYSTEM, f"Source excerpts:\n{excerpts}\n\nClaims:\n{listing}")
    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list):
        return None
    answered = [v for v in verdicts if isinstance(v, dict) and isinstance(v.get("supported"), bool)]
    if len(answered) < max(1, len(claims) // 2):
        return None  # too few verdicts to trust
    return sum(v["supported"] for v in answered) / len(answered)


def judge_relevance(llm: BaseChatModel, question: str, report: str) -> float | None:
    """Relevance on a 0-1 scale (a 1-5 rating rescaled), or None if the judge failed."""
    parsed = _ask_json(llm, RELEVANCE_SYSTEM, f"Question: {question}\n\nReport:\n{report[:6000]}")
    score = parsed.get("score") if isinstance(parsed, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
        return None
    return (score - 1) / 4
