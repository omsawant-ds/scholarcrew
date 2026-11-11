"""Deterministic report metrics. No model calls, so they are fast, free and reproducible.

Citation parsing and statistic checking live in `scholarcrew.grounding`, shared with the writer.
"""

from __future__ import annotations

import re

from ..grounding import (
    check_statistics,
    cite_ids,
    split_report,
    strip_cites,
    units,
    word_count,
)


def extract_claims(report: str, cap: int = 12) -> list[dict]:
    """Cited statements to fact-check, evenly sampled down to `cap` to bound judge cost."""
    body, _ = split_report(report)
    found = []
    for unit in units(body):
        ids = sorted(set(cite_ids(unit)))
        if ids and word_count(unit) >= 6:
            found.append({"text": strip_cites(unit).strip(), "cited": ids})
    if len(found) > cap:
        step = len(found) / cap
        found = [found[int(i * step)] for i in range(cap)]
    return found


def citation_scores(report: str, n_sources: int) -> dict[str, float]:
    """How well the report uses citations.

    citation_validity: share of cited source numbers that point at a real retrieved source.
    citation_coverage: share of substantive statements that carry a citation.
    source_utilization: share of retrieved sources the report actually cites.
    """
    body, _ = split_report(report)
    markers = cite_ids(body)
    valid = [m for m in markers if 1 <= m <= n_sources]
    statements = [u for u in units(body) if word_count(u) >= 8]
    cited = [u for u in statements if cite_ids(u)]
    return {
        "citation_validity": len(valid) / len(markers) if markers else 0.0,
        "citation_coverage": len(cited) / len(statements) if statements else 0.0,
        "source_utilization": len(set(valid)) / n_sources if n_sources else 0.0,
    }


def numeric_grounding(report: str, sources: list[dict]) -> float | None:
    """Share of statistics in the report that appear in the source(s) the statement cites.

    A cheap, judge-free check for invented figures. A statistic with no citation counts as
    ungrounded. Returns None when the report states no statistics.
    """
    total, unsupported = check_statistics(report, sources)
    return (total - len(unsupported)) / total if total else None


def structure_score(report: str) -> float:
    """Fraction of basic structure checks passed: title, 2+ sections, references, enough length."""
    body, refs = split_report(report)
    checks = [
        bool(re.search(r"^#\s+\S", report, re.MULTILINE)),
        len(re.findall(r"^#{2,3}\s+\S", body, re.MULTILINE)) >= 2,
        bool(refs.strip()),
        len(report.split()) >= 250,
    ]
    return sum(checks) / len(checks)


def deterministic_scores(report: str, n_sources: int) -> dict[str, float]:
    if not report.strip():
        return {
            "citation_validity": 0.0,
            "citation_coverage": 0.0,
            "source_utilization": 0.0,
            "structure": 0.0,
        }
    return {**citation_scores(report, n_sources), "structure": structure_score(report)}
