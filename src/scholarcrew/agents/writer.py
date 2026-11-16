"""Writer: produces the final report. The reference list is built from real sources, not the LLM."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..grounding import (
    EVIDENCE_THRESHOLD,
    linewise_citations,
    normalize_citations,
    repair_citations,
    verify_citations,
)
from ..state import CrewState
from .common import ask

SYSTEM = (
    "You are a technical writer. Write a clear Markdown report that answers the question using "
    "ONLY the analyst's notes. Structure: a title, a short summary, 2-4 themed sections, and a "
    "'Limitations' section. Keep the inline citations like [1] from the notes. Do not write a "
    "references list; it is added automatically.\n"
    "Rules for numbers: use a figure only if it appears in the notes, copy it exactly, and keep "
    "it next to the citation it came with. Never add figures, benchmarks or results of your own. "
    "Cite every statement of fact."
)


# Added for the app only. `SYSTEM` above is what the fine-tuning data and the writer comparison
# used, so it stays as it was and their results remain reproducible.
LAYOUT = (
    "\nCitation layout: cite a statement only when it states a fact taken from the notes, and end "
    "it with the citation(s) for that statement only, like 'Fact [2].' Do not cite your own "
    "comparisons, conclusions, summaries or transitions: write those without a citation and keep "
    "them short. Never put a citation in the middle of a sentence, and never join two facts that "
    "have different sources into one sentence. Refer to sources only through the bracket numbers: "
    "do not write 'the notes', 'the sources' or 'source 3' in the text."
)
APP_SYSTEM = SYSTEM + LAYOUT


def writer_input(question: str, notes: str) -> str:
    """The user message the writer receives. Fine-tuning data uses this exact format."""
    return f"Question: {question}\n\nNotes:\n{notes}"


def render_references(sources: list[dict]) -> str:
    lines = [f"[{s['id']}] {s['cite']}" for s in sources]
    # A blank line between entries: a single line break is shown as a space by Markdown, which ran
    # the whole list together into one paragraph in the app and on GitHub.
    return "## References\n\n" + "\n\n".join(lines)


def make_writer(llm: BaseChatModel, strict: bool = False) -> Callable[[CrewState], dict]:
    def writer(state: CrewState) -> dict:
        sources = state.get("sources", [])
        notes = state.get("notes", "")
        question = state["question"]
        # Models sometimes cite as 【3】; rewrite every style as plain [3] so reports look consistent.
        body = normalize_citations(ask(llm, APP_SYSTEM, writer_input(question, notes))).strip()
        body = body or f"# {question}\n\n{notes}"
        # Keep each statement's citations on that statement (one set, at its end), then point them
        # at the source whose text actually matches it. Models often cite [1] for facts from elsewhere.
        body = repair_citations(linewise_citations(body), sources)
        update: dict = {}
        log = "writer: report drafted"
        if strict:
            strict_body, counts = verify_citations(body, sources, EVIDENCE_THRESHOLD)
            if counts["kept"]:
                body = strict_body
                total = counts["kept"] + counts["dropped"]
                log = f"writer: strict mode kept {counts['kept']} of {total} cited statements"
            else:  # nothing matched, so a strict report would be empty: show the report as written
                log = "writer: strict mode found no matching statements, showing the full report"
            update["evidence"] = counts
        report = f"{body}\n\n{render_references(sources)}\n"
        return {"report": report, "log": [log], **update}

    return writer
