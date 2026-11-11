"""The writer checks its own draft and revises once. Offline: scripted model, no network."""

from langchain_core.messages import AIMessage

from scholarcrew.agents.writer import make_writer
from scholarcrew.evaluation.metrics import numeric_grounding
from scholarcrew.grounding import (
    check_statistics,
    cite_ids,
    find_issues,
    normalize_citations,
    problem_count,
)

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking improved faithfulness by 13% over fixed-size chunks."},
    {"id": 2, "kind": "web", "title": "Guide", "url": "u2", "cite": "c2",
     "text": "A practical guide to chunking with no reported results."},
]

GOOD = "# T\n\nLate chunking improved faithfulness by 13 % over fixed-size chunks in the study [1].\n"
UNCITED = "# T\n\nChunking strategy matters a great deal for retrieval quality in practice today.\n"
INVENTED = "# T\n\nLate chunking raised faithfulness from 0.71 to 0.84 in the reported experiments [1].\n"


class Script:
    """Returns the given replies in order and remembers every system prompt it was called with."""

    def __init__(self, *replies):
        self.replies, self.systems = list(replies), []

    def invoke(self, messages):
        self.systems.append(messages[0].content)
        return AIMessage(content=self.replies[min(len(self.systems) - 1, len(self.replies) - 1)])


def run_writer(llm):
    return make_writer(llm)({"question": "q?", "notes": "notes [1]", "sources": SOURCES})


def test_grounding_helpers_agree_with_each_other():
    total, bad = check_statistics(INVENTED, SOURCES)
    assert total == 2 and [n for n, _ in bad] == ["0.71", "0.84"]
    assert problem_count(INVENTED, SOURCES) == 2
    assert problem_count(UNCITED, SOURCES) == 1
    assert problem_count(GOOD, SOURCES) == 0
    assert numeric_grounding(INVENTED, SOURCES) == 0.0  # the metric and the guard use one definition


def test_full_width_citation_brackets_count_as_citations():
    """Regression: 【13】 was invisible to the parser, so cited reports looked uncited and every
    figure in them looked unsupported."""
    text = "Source-aware reranking raised precision from 0.48 to 0.72 on a health corpus【1】."
    assert cite_ids(text) == [1]
    assert cite_ids("a［2］ b【3，4】 c [5、6] d【7-9】") == [2, 3, 4, 5, 6, 7, 8, 9]
    sources = [{"id": 1, "kind": "arxiv", "title": "t", "url": "u", "cite": "c",
                "text": "Precision@5 rose from 0.48 to 0.72."}]
    assert check_statistics("# T\n\n" + text + "\n", sources) == (2, [])  # grounded, not flagged
    assert problem_count("# T\n\n" + text + "\n", sources) == 0


def test_normalize_citations_rewrites_every_style_as_plain_ascii():
    assert normalize_citations("a【1】 b［2，3］ c [ 4 ] d [5]") == "a[1] b[2,3] c [4] d [5]"
    assert normalize_citations("no citations, a [note] and 2024") == "no citations, a [note] and 2024"


def test_writer_normalizes_citation_brackets_in_the_report():
    llm = Script("# T\n\nLate chunking improved faithfulness by 13 % over fixed-size chunks【1】.\n")
    report = run_writer(llm)["report"]
    assert "[1]" in report and "【" not in report and len(llm.systems) == 1  # clean: no revision


def test_find_issues_names_each_problem_in_plain_language():
    issues = find_issues(INVENTED, SOURCES)
    assert len(issues) == 2 and "0.71" in issues[0] and "0.84" in issues[1]
    assert "cites no sources" in find_issues(UNCITED, SOURCES)[0]
    assert find_issues(GOOD, SOURCES) == []


def test_a_clean_draft_costs_no_extra_model_call():
    llm = Script(GOOD)
    result = run_writer(llm)
    assert len(llm.systems) == 1
    assert result["log"] == ["writer: report drafted"]


def test_an_uncited_draft_is_revised_once_and_the_better_version_is_kept():
    llm = Script(UNCITED, GOOD)
    result = run_writer(llm)
    assert len(llm.systems) == 2 and "correcting a draft" in llm.systems[1]
    assert "13 %" in result["report"] and "great deal" not in result["report"]
    assert "revised the draft (1 problems down to 0)" in result["log"][1]
    assert result["report"].rstrip().endswith("c2")  # references still come from the real sources


def test_invented_figures_trigger_a_revision_that_names_them():
    seen = {}

    class Spy(Script):
        def invoke(self, messages):
            if len(self.systems) == 1:
                seen["prompt"] = messages[1].content
            return super().invoke(messages)

    run_writer(Spy(INVENTED, GOOD))
    assert "0.71" in seen["prompt"] and "0.84" in seen["prompt"] and "Draft report" in seen["prompt"]


def test_a_revision_that_is_no_better_is_discarded():
    worse = "# T\n\nIt gets 9.5 and 7.2 and 1.1 percent better [1].\n"
    llm = Script(INVENTED, worse)
    result = run_writer(llm)
    assert "0.71" in result["report"] and "9.5" not in result["report"]
    assert "revision did not help" in result["log"][1]


def test_a_failed_revision_call_keeps_the_draft():
    class Flaky(Script):
        def invoke(self, messages):
            if len(self.systems) >= 1:
                self.systems.append("x")
                raise RuntimeError("429")
            return super().invoke(messages)

    result = run_writer(Flaky(INVENTED))
    assert "0.71" in result["report"] and "revision did not help" in result["log"][1]
