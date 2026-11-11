"""Reference writers to compare a fine-tuned writer against.

The writer mostly reshapes the analyst's notes into a report. If a writer that does nothing but
hand the notes through scores about as well as a tuned model, the tuned model has not learned
anything worth keeping. This baseline makes that visible.
"""

from __future__ import annotations

from ..grounding import normalize_citations


def copy_notes_writer(row: dict) -> str:
    """Return the analyst's notes unchanged as the report body (citations normalized)."""
    notes = normalize_citations(row["notes"]).strip()
    return notes if notes.startswith("#") else f"# {row['question']}\n\n{notes}"
