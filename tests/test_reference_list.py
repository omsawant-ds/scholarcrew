"""The reference list shows one reference per line. Offline."""

from scholarcrew.agents.writer import render_references
from scholarcrew.evaluation.metrics import deterministic_scores
from scholarcrew.grounding import split_report

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "A", "url": "u1", "cite": "Author One (2024). A. http://arxiv.org/abs/1", "text": "t"},
    {"id": 2, "kind": "web", "title": "B", "url": "u2", "cite": "B - Site. https://example.com/b", "text": "t"},
]


def test_each_reference_is_its_own_paragraph_so_markdown_does_not_join_them():
    text = render_references(SOURCES)
    assert text == "## References\n\n[1] Author One (2024). A. http://arxiv.org/abs/1\n\n[2] B - Site. https://example.com/b"
    assert "\n[2]" not in text.replace("\n\n[2]", "")  # no entry follows another on the very next line


def test_the_references_are_still_split_off_from_the_report_body():
    body = "# Title\n\n## Section\n\nA fact [1]. Another [2].\n\n"
    report = body + render_references(SOURCES) + "\n"
    found_body, references = split_report(report)
    assert "[1] Author One" not in found_body and references.startswith("## References")


def test_the_structure_check_still_sees_a_references_list():
    long_text = " ".join(["Careful chunking choices change retrieval for the model [1]."] * 30)
    report = f"# Title\n\n## One\n\n{long_text}\n\n## Two\n\n{long_text}\n\n" + render_references(SOURCES) + "\n"
    assert deterministic_scores(report, len(SOURCES))["structure"] == 1.0
