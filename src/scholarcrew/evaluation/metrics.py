"""Deterministic report metrics. No model calls, so they are fast, free and reproducible."""

from __future__ import annotations

import re

# A citation group such as [3], [1, 4], [2][5] (two groups) or a range like [18-20] (any dash).
GROUP = re.compile(r"\[(\d+(?:\s*[,;\-‐-—]\s*\d+)*)\]")
RANGE = re.compile(r"(\d+)\s*[\-‐-—]\s*(\d+)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
REFERENCES = re.compile(r"^#{1,6}\s*References\s*$", re.MULTILINE | re.IGNORECASE)
EMPHASIS = re.compile(r"\*\*|\*|`")
MAX_RANGE = 25


def cite_ids(text: str) -> list[int]:
    """Every source number cited in `text`, with ranges and comma lists expanded."""
    ids: list[int] = []
    for group in GROUP.findall(text):
        for part in re.split(r"\s*[,;]\s*", group):
            span = RANGE.fullmatch(part)
            if span:
                low, high = int(span[1]), int(span[2])
                ids += range(low, high + 1) if 0 < high - low <= MAX_RANGE else [low, high]
            elif part.isdigit():
                ids.append(int(part))
    return ids


def strip_cites(text: str) -> str:
    return GROUP.sub("", text)


def split_report(report: str) -> tuple[str, str]:
    """Split a report into (body, references section)."""
    match = REFERENCES.search(report)
    return (report[: match.start()], report[match.start() :]) if match else (report, "")


def _word_count(text: str) -> int:
    return len(strip_cites(text).split())


def units(body: str) -> list[str]:
    """Break the report body into statements: sentences, bullets, and whole table rows.

    A table row is one statement on purpose. Reports often put the claim in one cell and its
    citation in another, so splitting by cell would separate them.
    """
    out: list[str] = []
    lines = [ln.strip() for ln in body.splitlines()]
    for i, line in enumerate(lines):
        if not line or HEADING.match(line) or TABLE_SEP.match(line) or set(line) <= set("-*_ "):
            continue
        if line.startswith("|"):
            if i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1]):
                continue  # header row
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            out.append(EMPHASIS.sub("", " | ".join(cells)))
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*\[(])", line):
            sentence = EMPHASIS.sub("", sentence).strip(" -*>\t")
            if sentence:
                out.append(sentence)
    return out


def extract_claims(report: str, cap: int = 12) -> list[dict]:
    """Cited statements to fact-check, evenly sampled down to `cap` to bound judge cost."""
    body, _ = split_report(report)
    found = []
    for unit in units(body):
        ids = sorted(set(cite_ids(unit)))
        if ids and _word_count(unit) >= 6:
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
    statements = [u for u in units(body) if _word_count(u) >= 8]
    cited = [u for u in statements if cite_ids(u)]
    return {
        "citation_validity": len(valid) / len(markers) if markers else 0.0,
        "citation_coverage": len(cited) / len(statements) if statements else 0.0,
        "source_utilization": len(set(valid)) / n_sources if n_sources else 0.0,
    }


_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_DECIMAL = re.compile(r"(?<![\w.\-‐-—])\d+\.\d+(?!\w)")


def statistics_in(text: str) -> list[str]:
    """Figures that look like reported results: percentages and decimals (not years or list numbers).

    Decimals glued to a name with a hyphen, such as the 3.5 in a model name, are ignored.
    """
    text = strip_cites(text)
    found = _PERCENT.findall(text) + _DECIMAL.findall(text)
    return list(dict.fromkeys(found))


def _present(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", text) is not None


def numeric_grounding(report: str, sources: list[dict]) -> float | None:
    """Share of statistics in the report that appear in the source(s) the statement cites.

    A cheap, judge-free check for invented figures. A statistic with no citation counts as
    ungrounded. Returns None when the report states no statistics.
    """
    body, _ = split_report(report)
    text_by_id = {s["id"]: f"{s['title']} {s['text']}" for s in sources}
    total = grounded = 0
    for unit in units(body):
        cited_text = " ".join(text_by_id.get(i, "") for i in cite_ids(unit))
        for number in statistics_in(unit):
            total += 1
            grounded += _present(number, cited_text)
    return grounded / total if total else None


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
