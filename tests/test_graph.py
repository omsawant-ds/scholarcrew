from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew.config import Settings
from scholarcrew.graph import run_crew, stream_crew


def settings(tmp_path, max_steps=8):
    return Settings("http://x", "k", "m", max_steps=max_steps, reports_dir=tmp_path)


def test_full_pipeline_produces_cited_report(tmp_path):
    report, path = run_crew("Does RAG reduce hallucination?", FakeLLM(), settings(tmp_path),
                            arxiv_search=fake_arxiv, web_search=fake_web)
    assert "Grounding helps" in report
    assert "## References" in report
    assert "http://arxiv.org/abs/1" in report and "https://example.com/rag" in report
    assert path is not None and path.read_text(encoding="utf-8") == report


def test_agents_run_in_order(tmp_path):
    nodes = [n for n, _ in stream_crew("q", FakeLLM(), settings(tmp_path),
                                       arxiv_search=fake_arxiv, web_search=fake_web)]
    agents = [n for n in nodes if n != "supervisor"]
    assert agents == ["researcher", "analyst", "writer"]


def test_garbage_supervisor_output_falls_back(tmp_path):
    llm = FakeLLM(supervisor_reply="I think we should probably maybe write?? lol")
    report, _ = run_crew("q", llm, settings(tmp_path), arxiv_search=fake_arxiv, web_search=fake_web)
    assert "## References" in report


def test_llm_outage_still_terminates_with_a_report(tmp_path):
    report, _ = run_crew("q", FakeLLM(fail=True), settings(tmp_path),
                         arxiv_search=fake_arxiv, web_search=fake_web)
    assert "## References" in report


def test_no_sources_does_not_loop_forever(tmp_path):
    empty = lambda q, n: []
    report, path = run_crew("q", FakeLLM(), settings(tmp_path, max_steps=4),
                            arxiv_search=empty, web_search=empty)
    assert path is None or "## References" in report
