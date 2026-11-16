"""Every statement keeps its own citations, at its own end. Offline."""

from langchain_core.messages import AIMessage

from scholarcrew.agents.writer import APP_SYSTEM, SYSTEM, make_writer
from scholarcrew.grounding import linewise_citations

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking embeds the whole document first and then splits it, which keeps context."},
    {"id": 2, "kind": "arxiv", "title": "KV cache eviction", "url": "u2", "cite": "c2",
     "text": "Smart eviction policies for the KV cache raise serving throughput under long contexts."},
]


class Script:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _messages):
        return AIMessage(content=self.reply)


def test_two_claims_with_different_sources_are_split_so_each_keeps_its_citation():
    text = "Eviction policies raise throughput [2] and late chunking keeps context [1]."
    assert linewise_citations(text) == "Eviction policies raise throughput [2]. Late chunking keeps context [1]."


def test_words_after_the_last_citation_stay_with_the_last_claim():
    text = "A big claim about caches [1] and a second claim about chunking [2] under load."
    assert linewise_citations(text) == "A big claim about caches [1]. A second claim about chunking under load [2]."


def test_a_bullet_keeps_its_marker():
    text = "* A claim about caches [1] and another claim about chunking [2]."
    assert linewise_citations(text) == "* A claim about caches [1]. Another claim about chunking [2]."


def test_repeated_citations_of_the_same_sources_are_merged_at_the_end():
    text = "Smart eviction [2] policies raise throughput [2] under long contexts."
    assert linewise_citations(text) == "Smart eviction policies raise throughput under long contexts [2]."


def test_a_statement_with_one_run_of_citations_is_untouched():
    for text in ("As shown in [1], eviction matters.", "Eviction policies raise throughput [1, 2].",
                 "Eviction policies raise throughput [1], [2]."):
        assert linewise_citations(text) == text


def test_headings_tables_and_the_reference_list_are_untouched():
    text = ("# Title [1]\n\n| Claim | Source |\n|---|---|\n| A claim [1] and a claim [2] | x |\n\n"
            "## References\n\n[1] c1 [2]\n")
    assert linewise_citations(text) == text


def test_it_is_stable_and_never_changes_the_wording():
    text = "Eviction policies raise throughput [2] and late chunking keeps context [1]. More text here."
    once = linewise_citations(text)
    assert linewise_citations(once) == once

    def words(t):  # the joining "and" is dropped on purpose when a sentence is split
        found = (w.lower().strip(".") for w in t.replace("[1]", "").replace("[2]", "").split())
        return sorted(w for w in found if w not in ("", "and"))

    assert words(once) == words(text)


def test_a_fragment_too_short_to_stand_alone_is_not_split_out():
    text = "[1] Caches matter [2]."
    assert linewise_citations(text) == text


def test_the_writer_splits_two_claims_and_then_points_each_at_its_best_source():
    reply = ("# T\n\nSmart eviction policies for the KV cache raise serving throughput [1] and late "
             "chunking embeds the whole document first [2].\n")
    out = make_writer(Script(reply))({"question": "q", "notes": "n", "sources": SOURCES})["report"]
    assert "serving throughput [2]." in out and "document first [1]." in out


def test_the_app_prompt_adds_the_layout_rules_and_the_training_prompt_is_unchanged():
    assert "Citation layout" in APP_SYSTEM and APP_SYSTEM.startswith(SYSTEM)
    assert "Do not cite your own" in APP_SYSTEM  # own comparisons and conclusions stay uncited
    assert "Citation layout" not in SYSTEM  # fine-tuning data and the comparison keep using this one
