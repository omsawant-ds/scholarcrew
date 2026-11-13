"""UI tests for the Streamlit app, run with Streamlit's AppTest and a fake LLM (no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew import graph, llm, providers

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def field(at: AppTest, label: str):
    return next(w for w in at.text_input if w.label == label)


@pytest.fixture
def offline(monkeypatch):
    """Swap the real LLM and search tools for fakes."""
    real = graph.stream_crew
    monkeypatch.setattr(llm, "make_llm", lambda s: FakeLLM())
    monkeypatch.setattr(providers, "make_llm", lambda s: FakeLLM())
    monkeypatch.setattr(
        graph, "stream_crew",
        lambda q, m, s: real(q, m, s, arxiv_search=fake_arxiv, web_search=fake_web),
    )


def test_loads_cleanly_and_guides_the_user_to_add_a_key():
    at = new_app()
    assert not at.exception
    assert any("Get started" in m.value and "API key" in m.value for m in at.markdown)
    assert any("ScholarCrew" in m.value for m in at.markdown)


def test_page_contains_no_emojis(offline):
    """Design rule: plain text only. Checks every label and block of text the page renders."""
    import re

    emoji = re.compile("[\U0001f300-\U0001faff☀-➿⭐⬆⏩-⏿←-⇿]")
    at = new_app()
    field(at, "API key").set_value("test-key").run()
    field(at, "Research question").set_value("Does RAG reduce hallucination?").run()
    next(b for b in at.button if b.label == "Research").click().run()

    texts = [m.value for m in at.markdown] + [b.label for b in at.button] + [t.label for t in at.tabs]
    texts += [c.value for c in at.caption] + [s.value for s in at.subheader]
    assert not [t for t in texts if emoji.search(t)]


def test_example_button_fills_the_question():
    at = new_app()
    next(b for b in at.button if b.label == "Hallucination in RAG").click().run()
    assert field(at, "Research question").value.startswith("How do researchers reduce")
    assert not at.exception


def test_research_without_a_key_does_nothing_harmful():
    at = new_app()
    field(at, "Research question").set_value("anything").run()
    next(b for b in at.button if b.label == "Research").click().run()
    assert not at.exception
    assert "result" not in at.session_state or not at.session_state["result"]


def test_full_run_shows_metrics_tabs_and_cited_report(offline):
    at = new_app()
    field(at, "API key").set_value("test-key").run()
    field(at, "Research question").set_value("Does RAG reduce hallucination?").run()
    next(b for b in at.button if b.label == "Research").click().run()

    assert not at.exception and not at.error
    result = at.session_state["result"]
    assert "## References" in result["report"]
    assert len(result["sources"]) == 2
    assert [m.label for m in at.metric] == ["Sources", "arXiv papers", "Web pages", "Time"]
    assert [t.label for t in at.tabs] == ["Report", "Sources", "Agent log"]


def test_bad_key_shows_a_clear_error_and_no_report(monkeypatch):
    class Boom:
        def invoke(self, _):
            raise RuntimeError("401 invalid key test-key")

    monkeypatch.setattr(providers, "make_llm", lambda s: Boom())
    at = new_app()
    field(at, "API key").set_value("test-key").run()
    field(at, "Research question").set_value("q").run()
    next(b for b in at.button if b.label == "Research").click().run()

    assert any("RuntimeError" in e.value and "test-key" not in e.value for e in at.error)
    assert not at.session_state["result"]


def test_preset_address_and_model_are_read_only_but_custom_is_editable():
    at = new_app()
    assert field(at, "Base URL").disabled and field(at, "Model").disabled  # Groq preset
    assert field(at, "Model").value == "openai/gpt-oss-120b"

    at.selectbox[0].select("Custom (OpenAI-compatible)").run()
    assert not field(at, "Base URL").disabled and not field(at, "Model").disabled
