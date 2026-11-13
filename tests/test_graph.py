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


def test_searches_run_side_by_side_without_changing_source_order():
    import threading
    import time

    from scholarcrew.agents import make_researcher

    started = []

    def slow_arxiv(query, n):
        started.append(("arxiv", threading.get_ident(), time.perf_counter()))
        time.sleep(0.2)
        return fake_arxiv(query, n)

    def slow_web(query, n):
        started.append(("web", threading.get_ident(), time.perf_counter()))
        time.sleep(0.2)
        return fake_web(query, n)

    began = time.perf_counter()
    out = make_researcher(FakeLLM(), slow_arxiv, slow_web)({"question": "q?"})
    elapsed = time.perf_counter() - began

    assert [s["kind"] for s in out["sources"]] == ["arxiv", "web"]  # same order as before
    assert [s["id"] for s in out["sources"]] == [1, 2]
    assert len({thread for _, thread, _ in started}) == 2  # two worker threads
    assert elapsed < 0.7  # four 0.2s searches in sequence would take at least 0.8s


def test_supervisor_skips_the_model_once_everything_is_done():
    from scholarcrew.agents import make_supervisor

    llm = FakeLLM()
    done = {"question": "q?", "sources": [{"id": 1}], "notes": "n", "report": "r", "steps": 3}
    update = make_supervisor(llm, max_steps=8)(done)
    assert update["next"] == "FINISH" and llm.calls == []
