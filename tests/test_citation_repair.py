"""Pointing each cited statement at the source that matches it best. Offline."""

from langchain_core.messages import AIMessage

from scholarcrew.agents.writer import make_writer
from scholarcrew.finetune.compare import build_parser, build_rows
from scholarcrew.grounding import cite_ids, repair_citations

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking embeds the whole document first and then splits it, which keeps context."},
    {"id": 2, "kind": "arxiv", "title": "KV cache eviction", "url": "u2", "cite": "c2",
     "text": "Smart eviction policies for the KV cache raise serving throughput under long contexts."},
]
WRONG = "# T\n\nSmart eviction policies for the KV cache raise serving throughput [1].\n"


class Script:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _messages):
        return AIMessage(content=self.reply)


def test_a_citation_that_points_at_the_wrong_source_is_moved_to_the_matching_one():
    fixed = repair_citations(WRONG, SOURCES)
    assert "throughput [2]." in fixed and "[1]" not in fixed


def test_a_correct_citation_is_left_exactly_as_written():
    body = "# T\n\nLate chunking embeds the whole document first and then splits it [1].\n"
    assert repair_citations(body, SOURCES) == body


def test_a_second_source_that_also_matches_is_kept_and_the_best_one_is_added():
    sources = [
        {"id": 1, "kind": "arxiv", "title": "A", "url": "u", "cite": "c",
         "text": "Chunking strategies change retrieval quality for long documents."},
        {"id": 2, "kind": "arxiv", "title": "B", "url": "u", "cite": "c",
         "text": "Chunking strategies change retrieval quality for long documents and queries."},
    ]
    claim = "Chunking strategies change retrieval quality for long documents and queries [1]."
    fixed = repair_citations(claim, sources)  # source 1 matches well, source 2 matches best
    assert cite_ids(fixed) == [1, 2]


def test_commentary_that_matches_no_source_is_not_given_a_new_citation():
    body = "# T\n\nThe evidence is thin and a larger comparison would be needed to settle it [1].\n"
    assert repair_citations(body, SOURCES) == body


def test_headings_tables_uncited_text_and_the_reference_list_are_untouched():
    body = (
        "# Smart eviction policies for the KV cache\n\n"
        "Smart eviction policies for the KV cache raise serving throughput.\n\n"
        "| Smart eviction policies | raise throughput [1] |\n\n"
        "## References\n\n[1] c1\n[2] c2\n"
    )
    assert repair_citations(body, SOURCES) == body


def test_repair_is_stable_and_changes_only_the_citation_numbers():
    body = ("Smart eviction policies for the KV cache raise serving throughput [1]. "
            "Late chunking embeds the whole document first and then splits it [2].")
    once = repair_citations(body, SOURCES)
    assert repair_citations(once, SOURCES) == once
    assert cite_ids(once) == [2, 1]
    strip = lambda text: text.replace("[1]", "[x]").replace("[2]", "[x]")
    assert strip(once) == strip(body)


def test_a_citation_inside_brackets_is_replaced_in_place_and_leaves_no_empty_brackets():
    """Regression: removing "[1]" from "([1])" left "()" and moved the citation to the end."""
    body = "Smart eviction policies for the KV cache raise serving throughput ([1]); other work differs."
    fixed = repair_citations(body, SOURCES)
    assert "throughput ([2]);" in fixed and "()" not in fixed


def test_several_citations_in_one_statement_collapse_into_one_at_the_last_position():
    body = "Smart eviction [1] policies for the KV cache raise serving throughput [1] under long contexts."
    fixed = repair_citations(body, SOURCES)
    assert fixed == "Smart eviction policies for the KV cache raise serving throughput [2] under long contexts."


def test_no_sources_means_nothing_to_repair():
    assert repair_citations(WRONG, []) == WRONG


def test_the_writer_repairs_citations_before_adding_the_references():
    out = make_writer(Script(WRONG))({"question": "q", "notes": "n", "sources": SOURCES})["report"]
    assert "throughput [2]." in out and out.rstrip().endswith("[2] c2")


def test_the_comparison_can_score_with_repaired_citations():
    frozen = [{"id": "ev-01", "topic": "RAG", "question": "q?", "sources": SOURCES, "notes": "n [1]."}]
    generation = [{"id": "ev-01", "variant": "base", "sample": 0, "body": WRONG, "seconds": 1.0}]

    def base_report(rows):
        return next(r for r in rows if r["variant"] == "base")["report"]

    assert "throughput [1]." in base_report(build_rows(frozen, generation))
    assert "throughput [2]." in base_report(build_rows(frozen, generation, repair=True))
    assert build_parser().parse_args(["score", "--repair"]).repair is True
