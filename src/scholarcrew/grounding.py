"""Checks that a report's citations and statistics are supported by its sources.

Citation parsing and statistic checking live here so the agents and the evaluation pipeline agree
on what counts as a citation. The agents rewrite every bracket style as plain [n]; the evaluation
scores reports with the same parser.
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


_STOPWORDS = frozenset(
    ["the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by", "is", "are", "was", "were", "be", "been", "that", "this", "these", "those", "it", "its", "as", "at", "from", "which", "can", "may", "also", "not", "than", "more", "most", "such", "using", "used", "use", "based", "into", "over", "between", "their", "they", "has", "have", "had", "but", "how", "what", "when", "where", "while", "across", "both", "each", "other", "our", "we"]
)
# A statement is moved to another source only if that source matches this much of its words, and
# an existing citation is kept if it matches at least this share of what the best source does.
REPAIR_MIN_MATCH = 0.2
REPAIR_KEEP_SHARE = 0.6


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def repair_citations(body: str, sources: list[dict]) -> str:
    """Point each cited statement at the source whose text matches it best.

    Writers, especially small ones, often cite source [1] for a fact that came from another source.
    A citation should lead to the text that supports the statement, so for every statement with a
    citation this finds the source sharing the most of its words (the title and the excerpt the
    writer was given). The best source is added, and a citation that matches clearly less than the
    best one is dropped. A statement that matches no source well (commentary, for example) is left
    alone, so nothing is invented. Headings, table rows and the references list are not touched.
    """
    text, references = split_report(body)
    by_source = {s["id"]: content_words(f"{s['title']} {s['text'][:1200]}") for s in sources}
    lines = []
    for line in text.split("\n"):
        if not line.strip() or HEADING.match(line) or line.lstrip().startswith("|"):
            lines.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*\[(])", line)
        lines.append(" ".join(_repair_sentence(sent, by_source) for sent in sentences))
    return "\n".join(lines) + references


def _repair_sentence(sentence: str, by_source: dict[int, set[str]]) -> str:
    cited = sorted(set(cite_ids(sentence)))
    words = content_words(strip_cites(sentence))
    if not cited or not words or not by_source:
        return sentence
    match = {i: len(words & w) / len(words) for i, w in by_source.items()}
    best = max(match, key=match.get)
    if match[best] < REPAIR_MIN_MATCH:
        return sentence
    chosen = sorted({i for i in cited if match.get(i, 0) >= REPAIR_KEEP_SHARE * match[best]} | {best})
    if chosen == cited:
        return sentence
    # The new citation takes the place of the last run of old ones, so "([3])" becomes "([2])" and
    # "[1], [2], [3]" becomes one marker. Earlier runs in the same statement are removed, and the
    # gaps and empty brackets they leave closed.
    marker = "[" + ", ".join(map(str, chosen)) + "]"
    run = _only_citation_run(sentence)
    if run is None:  # several claims, each with its own citation: matching the whole statement would mix them
        return sentence
    out = sentence[: run[0]] + marker + sentence[run[1] :]
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"(\S)[ \t]{2,}(\S)", r"\1 \2", out)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", out)


def _citation_runs(sentence: str) -> list[tuple[int, int]]:
    """Start and end of each run of citations: citations next to each other, separated only by commas."""
    runs: list[list[int]] = []
    for group in GROUP.finditer(sentence):
        if runs and re.fullmatch(r"[\s,;]*", sentence[runs[-1][1] : group.start()]):
            runs[-1][1] = group.end()
        else:
            runs.append([group.start(), group.end()])
    return [(start, end) for start, end in runs]


def _only_citation_run(sentence: str) -> tuple[int, int] | None:
    """The citations in a statement, if they form one run. None if there are two or more."""
    runs = _citation_runs(sentence)
    return runs[0] if len(runs) == 1 else None


_SENTENCE_BREAK = r"(?<=[.!?])\s+(?=[A-Z0-9*\[(])"
_LEAD = re.compile(r"\s*(?:[-*+>]|\d+[.)])?\s*")  # a bullet or list number at the start of a line
_JOINER = re.compile(r"^[\s,;:]*(?:(?:and|but|while|whereas|as well as|also|then)\s+)?", re.IGNORECASE)


def _with_marker_at_end(text: str, marker: str, stop: str) -> str:
    """`text` with `marker` placed before its final stop. `stop` is added if the text has none."""
    text = text.rstrip()
    end = re.search(r"([.!?]+[\"')\]]*)$", text)
    if end:
        return f"{text[: end.start()].rstrip()} {marker}{end.group(1)}"
    return f"{text} {marker}{stop}"


def _split_statement(sentence: str) -> list[str]:
    """Give each claim in a sentence its own citations at its own end.

    A sentence with one run of citations is left alone. Several runs naming the same sources are
    merged into one at the end. Several runs naming different sources mean several claims, so the
    sentence is split after each run: "A [1] and B [2]." becomes "A [1]." and "B [2]."
    """
    runs = _citation_runs(sentence)
    if len(runs) < 2:
        return [sentence]
    ids = [frozenset(cite_ids(sentence[a:b])) for a, b in runs]
    if len(set(ids)) == 1:
        bare = sentence
        for a, b in reversed(runs):
            bare = bare[:a] + bare[b:]
        bare = re.sub(r"(\S)[ \t]{2,}(\S)", r"\1 \2", bare)
        bare = re.sub(r"[ \t]+([.,;:!?])", r"\1", bare)
        return [_with_marker_at_end(bare, sentence[runs[-1][0] : runs[-1][1]], "")]
    pieces, previous = [], 0
    for a, b in runs:
        pieces.append([sentence[previous:a], sentence[a:b]])
        previous = b
    pieces[-1][0] += sentence[previous:]  # words after the last citation belong to the last claim
    out = []
    for index, (text, marker) in enumerate(pieces):
        text = _JOINER.sub("", text, count=1) if index else text
        if len(text.split()) < 2:  # nothing worth splitting out: leave the sentence as written
            return [sentence]
        text = re.sub(r"[ \t]{2,}", " ", text.strip())
        text = re.sub(r"[A-Za-z]", lambda m: m.group(0).upper(), text, count=1)
        out.append(_with_marker_at_end(text, marker, "" if index == len(pieces) - 1 else "."))
    return out


def linewise_citations(body: str) -> str:
    """Keep every statement's citations on that statement, so a citation can never belong to a
    neighbouring claim. See `_split_statement`. Headings, table rows and the references list are
    not touched."""
    text, references = split_report(body)
    lines = []
    for line in text.split("\n"):
        if not line.strip() or HEADING.match(line) or line.lstrip().startswith("|"):
            lines.append(line)
            continue
        lead = _LEAD.match(line).group(0)
        parts = [p for s in re.split(_SENTENCE_BREAK, line[len(lead) :]) for p in _split_statement(s)]
        lines.append(lead + " ".join(parts))
    return "\n".join(lines) + references


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
