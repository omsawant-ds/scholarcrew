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
    body = f"# Title\n\n{GOOD} {BAD}\n\nAn uncited remark about the field.\n\n## References\n\n[1] c1\n"
    out, counts = verify_citations(body, SOURCES, 0.4)
    assert counts == {"kept": 1, "dropped": 1}
    assert GOOD in out and "annealers" not in out
    assert "An uncited remark" in out and "[1] c1" in out and "# Title" in out


TABLE = (
    "| Method | What it does | Evidence |\n|---|---|---|\n"
    "| Late chunking | Embeds the whole document first and then splits it | Source one [1] |\n"
    "| Quantum annealing | Dramatically outperforms every classical optimizer | Source one [1] |\n"
    "| Uncited row | Nothing to check here | none |\n"
)


def test_table_rows_are_checked_one_by_one_and_uncited_rows_stay():
    out, counts = verify_citations(f"# T\n\n{TABLE}\n## References\n\n[1] c1\n", SOURCES, 0.4)
    assert counts == {"kept": 1, "dropped": 1}
    assert "Late chunking" in out and "Quantum annealing" not in out and "Uncited row" in out
    assert "| Method | What it does | Evidence |" in out and "|---|---|---|" in out  # header stays


def test_a_table_with_no_rows_left_is_removed_entirely():
    table = ("| Method | Evidence |\n|---|---|\n"
             "| Quantum annealing outperforms optimizers | Source [1] |\n")
    out, counts = verify_citations(f"# T\n\n{table}\n{GOOD}\n", SOURCES, 0.4)
    assert counts == {"kept": 1, "dropped": 1}
    assert "| Method |" not in out and "|---|" not in out and GOOD in out


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
    assert out["evidence"] == {"kept": 0, "dropped": 1, "fallback": True}  # the app words its caption from this


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


class WriterSaysNonsense(FakeLLM):
    """A model whose report states something its source does not."""

    def invoke(self, messages):
        if "technical writer" in messages[0].content:
            return AIMessage(content="# Report\n\nQuantum annealers dramatically outperform every classical optimizer [1].")
        return super().invoke(messages)


def _run_app(monkeypatch, model, strict):
    real = graph.stream_crew
    monkeypatch.setattr(providers, "make_llm", lambda s: model)
    monkeypatch.setattr(
        graph, "stream_crew",
        lambda q, m, s: real(q, model, s, arxiv_search=fake_arxiv, web_search=fake_web),
    )
    app = AppTest.from_file(str(Path(__file__).parent.parent / "streamlit_app.py"), default_timeout=30).run()
    next(c for c in app.checkbox if "Strict" in c.label).set_value(strict).run()
    next(w for w in app.text_input if w.label == "API key").set_value("test-key").run()
    next(w for w in app.text_input if w.label == "Research question").set_value("Does RAG reduce hallucination?").run()
    next(b for b in app.button if b.label == "Research").click().run()
    return app


def test_the_caption_says_so_when_strict_mode_matched_nothing_and_shows_the_full_report(monkeypatch):
    app = _run_app(monkeypatch, WriterSaysNonsense(), strict=True)
    captions = [c.value for c in app.caption]
    assert any("could not match any cited statement" in c for c in captions)
    assert not any(c.startswith("Strict mode:") for c in captions)
    assert "annealers" in app.session_state["result"]["report"]


def test_the_caption_counts_the_statements_when_strict_mode_filtered_some(monkeypatch):
    app = _run_app(monkeypatch, FakeLLM(), strict=True)
    assert any(c.value.startswith("Strict mode:") and "cited statements shown" in c.value for c in app.caption)


def test_the_limitations_section_is_never_filtered_but_other_sections_are():
    remark = "The routing description lacks quantitative results, so its strength is speculative [2]."
    body = (f"# T\n\n## Findings\n\n{GOOD} {BAD}\n\n## Limitations\n\n- {remark}\n- {BAD}\n\n"
            f"## More findings\n\n{BAD}\n\n## References\n\n[1] c1\n")
    out, counts = verify_citations(body, SOURCES, 0.4)
    assert counts == {"kept": 1, "dropped": 2}  # the two outside Limitations; the two inside are not counted
    assert remark in out and out.count("annealers") == 1  # only the one inside Limitations survives
    assert "## More findings" in out and "[1] c1" in out


def test_limitations_is_recognised_at_any_heading_level_and_ends_at_the_next_section():
    body = f"# T\n\n### Limitations\n\n{BAD}\n\n#### A detail\n\n{BAD}\n\n### Next\n\n{BAD}\n"
    out, counts = verify_citations(body, SOURCES, 0.4)
    assert out.count("annealers") == 2 and counts == {"kept": 0, "dropped": 1}  # subsections stay, Next is filtered
