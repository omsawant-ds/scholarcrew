"""Source lists: duplicates of a paper are dropped and messy web titles are tidied. Offline."""

from conftest import FakeLLM
from scholarcrew.agents.researcher import make_researcher, same_title, tidy_title
from scholarcrew.tools.arxiv import Paper
from scholarcrew.tools.web import WebResult

PAPER_TITLE = "LoRA vs Full Fine-tuning: An Illusion of Equivalence"


def run(papers, pages):
    researcher = make_researcher(FakeLLM(), lambda q, n: papers, lambda q, n: pages)
    return researcher({"question": "q?"})["sources"]


def paper(title=PAPER_TITLE, url="http://arxiv.org/abs/1"):
    return Paper(title, ["A. Author"], "An abstract.", url, "2024-01-01T00:00:00Z")


def test_a_web_page_about_a_paper_already_found_is_dropped():
    sources = run(
        [paper()],
        [
            WebResult(f"{PAPER_TITLE} - arXiv.org", "Mirror.", "https://arxiv.org/html/1"),
            WebResult(f"{PAPER_TITLE} | OpenReview", "Listing.", "https://openreview.net/x"),
            WebResult("A practical guide to LoRA", "Guide.", "https://example.com/lora"),
        ],
    )
    assert [(s["kind"], s["title"]) for s in sources] == [
        ("arxiv", PAPER_TITLE),
        ("web", "A practical guide to LoRA"),
    ]
    assert [s["id"] for s in sources] == [1, 2]  # numbering stays continuous


def test_two_papers_are_never_treated_as_duplicates_of_each_other():
    sources = run([paper(), paper("Hybrid-LoRA: Bridging Full Fine-Tuning and Low-Rank Adaptation", "http://arxiv.org/abs/2")], [])
    assert len(sources) == 2


def test_short_or_unrelated_titles_are_kept():
    assert not same_title("RAG tips", "RAG tips and tricks for production")  # too short to compare
    assert not same_title("A practical guide to LoRA", PAPER_TITLE)
    assert same_title(PAPER_TITLE, f"{PAPER_TITLE} - arXiv.org")


def test_a_runaway_web_title_is_shortened_to_one_line():
    junk = (PAPER_TITLE + " ") * 4 + "- Related links GeeksforGeeks"
    tidy = tidy_title(junk)
    assert len(tidy) <= 123 and "\n" not in tidy and tidy.endswith("...")
    assert tidy_title("  A short   title\n") == "A short title"
