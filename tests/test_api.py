import json

from fastapi.testclient import TestClient

from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew import api
from scholarcrew.graph import stream_crew as real_stream


def client(monkeypatch):
    monkeypatch.setattr(api, "make_llm", lambda settings: FakeLLM())
    monkeypatch.setattr(
        api, "stream_crew",
        lambda q, llm, s: real_stream(q, llm, s, arxiv_search=fake_arxiv, web_search=fake_web),
    )
    api._hits.clear()
    return TestClient(api.app)


def test_health(monkeypatch):
    assert client(monkeypatch).get("/health").json() == {"status": "ok"}


def test_research_streams_agents_then_report(monkeypatch):
    res = client(monkeypatch).get("/api/research", params={"q": "Does RAG help?"})
    events = [json.loads(line) for line in res.text.splitlines() if line]
    assert [e["agent"] for e in events if e["agent"] != "supervisor"] == [
        "researcher", "analyst", "writer",
    ]
    assert "## References" in events[-2]["report"]


def test_rejects_empty_and_long_questions(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/api/research", params={"q": "  "}).status_code == 400
    assert c.get("/api/research", params={"q": "x" * 301}).status_code == 400


def test_rate_limit(monkeypatch):
    c = client(monkeypatch)
    codes = [c.get("/api/research", params={"q": "hi"}).status_code for _ in range(api.RATE_LIMIT + 1)]
    assert codes[-1] == 429
