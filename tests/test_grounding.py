"""Shared citation and statistic checks, and the writer's citation normalization. Offline."""

from langchain_core.messages import AIMessage

from scholarcrew.agents.analyst import make_analyst
from scholarcrew.agents.writer import make_writer
from scholarcrew.evaluation.metrics import numeric_grounding
from scholarcrew.grounding import check_statistics, cite_ids, normalize_citations

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking improved faithfulness by 13% over fixed-size chunks."},
    {"id": 2, "kind": "web", "title": "Guide", "url": "u2", "cite": "c2",
     "text": "A practical guide to chunking with no reported results."},
]

INVENTED = "# T\n\nLate chunking raised faithfulness from 0.71 to 0.84 in the reported experiments [1].\n"


class Script:
    """A model that returns one canned reply and counts how often it was called."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=self.reply)


def test_statistic_check_and_the_metric_share_one_definition():
    total, bad = check_statistics(INVENTED, SOURCES)
    assert total == 2 and [n for n, _ in bad] == ["0.71", "0.84"]
    assert numeric_grounding(INVENTED, SOURCES) == 0.0


def test_full_width_citation_brackets_count_as_citations():
    """Regression: 【13】 was invisible to the parser, so cited reports looked uncited and every
    figure in them looked unsupported."""
    text = "Source-aware reranking raised precision from 0.48 to 0.72 on a health corpus【1】."
    assert cite_ids(text) == [1]
    assert cite_ids("a［2］ b【3，4】 c [5、6] d【7-9】") == [2, 3, 4, 5, 6, 7, 8, 9]
    sources = [{"id": 1, "kind": "arxiv", "title": "t", "url": "u", "cite": "c",
                "text": "Precision@5 rose from 0.48 to 0.72."}]
    assert check_statistics("# T\n\n" + text + "\n", sources) == (2, [])  # grounded, not flagged


def test_normalize_citations_rewrites_every_style_as_plain_ascii():
    assert normalize_citations("a【1】 b［2，3］ c [ 4 ] d [5]") == "a[1] b[2,3] c [4] d [5]"
    assert normalize_citations("no citations, a [note] and 2024") == "no citations, a [note] and 2024"


def test_writer_normalizes_citation_brackets_in_the_report():
    llm = Script("# T\n\nLate chunking improved faithfulness by 13 % over fixed-size chunks【1】.\n")
    report = make_writer(llm)({"question": "q?", "notes": "n", "sources": SOURCES})["report"]
    assert "[1]" in report and "【" not in report
    assert llm.calls == 1  # the writer makes a single model call
    assert report.rstrip().endswith("c2")  # references still come from the real sources


def test_analyst_normalizes_citation_brackets_in_its_notes():
    notes = make_analyst(Script("Chunking helps【1】 and also【2，1】."))(
        {"question": "q?", "sources": SOURCES})["notes"]
    assert notes == "Chunking helps[1] and also[2,1]."
