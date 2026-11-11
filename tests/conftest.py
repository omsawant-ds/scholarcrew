"""Test helpers: a scripted fake LLM and fake tools, so tests need no network or API key."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from scholarcrew.tools.arxiv import Paper
from scholarcrew.tools.web import WebResult


class FakeLLM:
    """Answers based on the system prompt, like a tiny scripted model."""

    def __init__(self, supervisor_reply: str | None = None, fail: bool = False):
        self.supervisor_reply = supervisor_reply
        self.fail = fail
        self.calls: list[str] = []

    def invoke(self, messages):
        if self.fail:
            raise RuntimeError("provider down")
        system = messages[0].content
        self.calls.append(system[:20])
        if "supervisor" in system:
            return AIMessage(content=self.supervisor_reply or '{"next": "researcher"}')
        if "search" in system and "queries" in system:
            return AIMessage(content='```json\n{"queries": ["rag hallucination", "grounding"]}\n```')
        if "technical writer" in system:
            return AIMessage(content="# Report\n\nGrounding helps [1][2].")
        return AIMessage(content="- Retrieval grounding reduces errors [1]\n- Web agrees [2]")


def fake_arxiv(query: str, n: int) -> list[Paper]:
    return [Paper("Grounded RAG", ["A. Author", "B. Author"], "Grounding cuts errors.",
                  "http://arxiv.org/abs/1", "2024-01-01T00:00:00Z")]


def fake_web(query: str, n: int) -> list[WebResult]:
    return [WebResult("Blog on RAG", "Tips for RAG.", "https://example.com/rag")]
