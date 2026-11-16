"""Strict mode keeps only statements whose cited source has matching text. Offline."""

from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew import graph, providers
from scholarcrew.agents.writer import make_writer
from scholarcrew.cli import build_parser
from scholarcrew.config import Settings
from scholarcrew.grounding import evidence_score, statement_is_verified, verify_citations

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking embeds the whole document first and then splits it. It improved faithfulness by 13% over fixed-size chunks."},
    {"id": 2, "kind": "arxiv", "title": "KV cache", "url": "u2", "cite": "c2",
     "text": "Smart eviction policies for the KV cache raise serving throughput under long contexts."},
]
GOOD = "Late chunking embeds the whole document first and then splits it [1]."
BAD = "Quantum annealers dramatically outperform classical optimizers on every benchmark [1]."


class Script:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _messages):
        return AIMessage(content=self.reply)


def test_a_sentence_of_the_source_that_matches_scores_high_and_an_unrelated_one_low():
    assert evidence_score("Late chunking embeds the whole document first.", SOURCES[0]["text"]) > 0.8
    assert evidence_score("Quantum annealers outperform classical optimizers.", SOURCES[0]["text"]) < 0.2


def test_numbers_must_appear_in_the_cited_source():
    texts = [SOURCES[0]["text"]]
    assert statement_is_verified("Late chunking improved faithfulness by 13% over fixed-size chunks.", texts, 0.4)
    assert not statement_is_verified("Late chunking improved faithfulness by 47% over fixed-size chunks.", texts, 0.4)


def test_unsupported_statements_are_dropped_and_everything_else_is_kept():
    body = f"# Title\n\n{GOOD} {BAD}\n\nAn uncited remark about the field.\n\n| A | B [1] |\n\n## References\n\n[1] c1\n"
    out, counts = verify_citations(body, SOURCES, 0.4)
    assert counts == {"kept": 1, "dropped": 1}
    assert GOOD in out and "annealers" not in out
    assert "An uncited remark" in out and "| A | B [1] |" in out and "[1] c1" in out and "# Title" in out


def test_a_line_whose_only_statement_is_dropped_disappears():
    out, _ = verify_citations(f"* {BAD}\n* {GOOD}\n", SOURCES, 0.4)
    assert out.strip() == f"* {GOOD}"


def test_the_writer_only_filters_in_strict_mode():
    reply = f"# T\n\n{GOOD} {BAD}\n"
    state = {"question": "q", "notes": "n", "sources": SOURCES}
    normal = make_writer(Script(reply))(state)
    strict = make_writer(Script(reply), strict=True)(state)
    assert "annealers" in normal["report"] and "evidence" not in normal
    assert "annealers" not in strict["report"] and GOOD in strict["report"]
    assert strict["evidence"] == {"kept": 1, "dropped": 1}
    assert "kept 1 of 2" in strict["log"][0]


def test_a_strict_report_is_never_empty_when_nothing_matches():
    state = {"question": "q", "notes": "n", "sources": SOURCES}
    out = make_writer(Script(f"# T\n\n{BAD}\n"), strict=True)(state)
    assert "annealers" in out["report"] and "showing the full report" in out["log"][0]


def test_the_setting_reaches_the_writer_through_the_graph():
    settings = Settings("https://x", "k", "m", strict_evidence=True)
    updates = dict(graph.stream_crew("q?", FakeLLM(), settings, arxiv_search=fake_arxiv, web_search=fake_web))
    assert "evidence" in updates["writer"]
    plain = dict(graph.stream_crew("q?", FakeLLM(), Settings("https://x", "k", "m"),
                                   arxiv_search=fake_arxiv, web_search=fake_web))
    assert "evidence" not in plain["writer"]


def test_settings_and_command_line_carry_the_choice():
    assert providers.settings_for("https://x", "k", "m", strict=True).strict_evidence is True
    assert providers.settings_for("https://x", "k", "m").strict_evidence is False
    assert build_parser().parse_args(["q", "--strict"]).strict is True
    assert build_parser().parse_args(["q"]).strict is False


def test_the_app_offers_strict_mode_off_by_default():
    app = AppTest.from_file(str(Path(__file__).parent.parent / "streamlit_app.py"), default_timeout=30).run()
    box = next(c for c in app.checkbox if "Strict" in c.label)
    assert box.value is False
    assert "90%" in box.help and "not proof" in box.help
