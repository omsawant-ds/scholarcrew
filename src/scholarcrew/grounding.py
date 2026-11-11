"""Checks that a report's citations and statistics are supported by its sources.

These checks are shared on purpose. The writer uses them to inspect its own draft, and the
evaluation pipeline uses the same code to score the final report, so the system is held to exactly
the standard it is measured by.
"""

from __future__ import annotations

import re

# A citation group such as [3], [1, 4], [2][5] (two groups) or a range like [18-20] (any dash).
# Models sometimes write full-width brackets such as 【3】 or ［3］ and full-width commas, and
# those must count as citations too: missing them once made cited reports look uncited.
GROUP = re.compile(
    r"[\[［【]\s*(\d+(?:\s*[,;，、\-‐-—]\s*\d+)*)\s*[\]］】]"
)
RANGE = re.compile(r"(\d+)\s*[\-‐-—]\s*(\d+)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
REFERENCES = re.compile(r"^#{1,6}\s*References\s*$", re.MULTILINE | re.IGNORECASE)
EMPHASIS = re.compile(r"\*\*|\*|`")
MAX_RANGE = 25

_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_DECIMAL = re.compile(r"(?<![\w.\-‐-—])\d+\.\d+(?!\w)")


def cite_ids(text: str) -> list[int]:
    """Every source number cited in `text`, with ranges and comma lists expanded."""
    ids: list[int] = []
    for group in GROUP.findall(text):
        for part in re.split(r"\s*[,;，、]\s*", group):
            span = RANGE.fullmatch(part)
            if span:
                low, high = int(span[1]), int(span[2])
                ids += range(low, high + 1) if 0 < high - low <= MAX_RANGE else [low, high]
            elif part.isdigit():
                ids.append(int(part))
    return ids


def strip_cites(text: str) -> str:
    return GROUP.sub("", text)


def normalize_citations(text: str) -> str:
    """Rewrite any bracket style (full-width, spaced) as plain ASCII citations like [3] or [1, 4]."""
    def plain(match: re.Match) -> str:
        inner = match.group(1).replace("，", ",").replace("、", ",")
        return f"[{inner}]"

    return GROUP.sub(plain, text)


def split_report(report: str) -> tuple[str, str]:
    """Split a report into (body, references section)."""
    match = REFERENCES.search(report)
    return (report[: match.start()], report[match.start() :]) if match else (report, "")


def word_count(text: str) -> int:
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


def statistics_in(text: str) -> list[str]:
    """Figures that look like reported results: percentages and decimals (not years or list numbers).

    Decimals glued to a name with a hyphen, such as the 3.5 in a model name, are ignored.
    """
    text = strip_cites(text)
    found = _PERCENT.findall(text) + _DECIMAL.findall(text)
    return list(dict.fromkeys(found))


def present(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", text) is not None


def check_statistics(report: str, sources: list[dict]) -> tuple[int, list[tuple[str, str]]]:
    """Return (how many statistics the report states, the ones its cited sources do not contain).

    Each unsupported statistic is a (figure, statement) pair. A statistic in a statement with no
    citation counts as unsupported.
    """
    body, _ = split_report(report)
    text_by_id = {s["id"]: f"{s['title']} {s['text']}" for s in sources}
    total, unsupported = 0, []
    for unit in units(body):
        cited_text = " ".join(text_by_id.get(i, "") for i in cite_ids(unit))
        for number in statistics_in(unit):
            total += 1
            if not present(number, cited_text):
                unsupported.append((number, strip_cites(unit)))
    return total, unsupported


def problem_count(report: str, sources: list[dict]) -> int:
    """How many grounding problems a report has: no citations at all, plus each unsupported figure."""
    body, _ = split_report(report)
    return (0 if cite_ids(body) else 1) + len(check_statistics(report, sources)[1])


def find_issues(report: str, sources: list[dict], max_figures: int = 8) -> list[str]:
    """Plain-language problems to hand back to the writer for a targeted revision."""
    body, _ = split_report(report)
    issues = []
    if not cite_ids(body):
        issues.append(
            "The report cites no sources. Put a citation such as [3] after every statement of "
            "fact, using only source numbers that appear in the notes."
        )
    for number, sentence in check_statistics(report, sources)[1][:max_figures]:
        issues.append(
            f'The figure {number} is not in the source cited for this statement: "{sentence[:180]}". '
            "Remove the figure or describe the finding in words."
        )
    return issues
