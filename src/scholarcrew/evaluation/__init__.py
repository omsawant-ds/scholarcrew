"""Automated evaluation: score ScholarCrew reports on faithfulness, relevance and citations."""

from .metrics import deterministic_scores, extract_claims
from .runner import aggregate, check_thresholds, load_questions, run, to_markdown

__all__ = [
    "aggregate",
    "check_thresholds",
    "deterministic_scores",
    "extract_claims",
    "load_questions",
    "run",
    "to_markdown",
]
