"""Writer: produces the final report. The reference list is built from real sources, not the LLM.

After drafting, the writer checks its own draft with the same rules the evaluation scores by. If
the draft cites nothing, or states a figure its cited source does not contain, it gets one
targeted revision, and the revision is kept only if it is actually better.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from ..grounding import find_issues, normalize_citations, problem_count
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

REVISE_SYSTEM = (
    "You are a technical writer correcting a draft research report. Fix only the listed problems "
    "and change nothing else. Use only facts and figures that appear in the notes, and cite them "
    "with the source numbers used there. Reply with the complete corrected Markdown report, "
    "without a references list."
)


def render_references(sources: list[dict]) -> str:
    lines = [f"[{s['id']}] {s['cite']}" for s in sources]
    return "## References\n\n" + "\n".join(lines)


def make_writer(llm: BaseChatModel) -> Callable[[CrewState], dict]:
    def writer(state: CrewState) -> dict:
        sources = state.get("sources", [])
        notes = state.get("notes", "")
        question = state["question"]
        body = normalize_citations(ask(llm, SYSTEM, f"Question: {question}\n\nNotes:\n{notes}")).strip()
        body = body or f"# {question}\n\n{notes}"
        log = ["writer: report drafted"]

        issues = find_issues(body, sources)
        if issues:
            before = problem_count(body, sources)
            prompt = (
                f"Question: {question}\n\nNotes:\n{notes}\n\nDraft report:\n{body}\n\n"
                "Problems to fix:\n" + "\n".join(f"- {issue}" for issue in issues)
            )
            revised = normalize_citations(ask(llm, REVISE_SYSTEM, prompt)).strip()
            after = problem_count(revised, sources) if revised else before
            if revised and after < before:
                body = revised
                log.append(f"writer: revised the draft ({before} problems down to {after})")
            else:
                log.append(f"writer: revision did not help, kept the draft ({before} problems)")

        report = f"{body}\n\n{render_references(sources)}\n"
        return {"report": report, "log": log}

    return writer
