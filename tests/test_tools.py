from scholarcrew.agents.common import parse_json
from scholarcrew.tools.arxiv import parse_feed
from scholarcrew.tools.reports import save_report

FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>A  Great
      Paper</title>
    <summary>It works.</summary>
    <author><name>Ada Lovelace</name></author>
  </entry>
</feed>"""


def test_parse_feed():
    (paper,) = parse_feed(FEED)
    assert paper.title == "A Great Paper"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.cite().startswith("Ada Lovelace (2024). A Great Paper.")


def test_parse_json_handles_fences_and_prose():
    assert parse_json('Sure! ```json\n{"next": "writer"}\n``` done') == {"next": "writer"}
    assert parse_json("no json here") is None


def test_save_report(tmp_path):
    path = save_report("What is RAG?", "# hi", tmp_path / "out")
    assert path.read_text(encoding="utf-8") == "# hi"
    assert "what-is-rag" in path.name
