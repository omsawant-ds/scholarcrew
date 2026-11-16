"""Web results that cannot help (dictionary pages, other languages, no shared words) are left out. Offline."""

from conftest import FakeLLM
from scholarcrew.agents.researcher import is_off_topic_web, make_researcher
from scholarcrew.tools.web import WebResult

QUESTION = "How do multi-agent LLM systems coordinate, and where do they fail?"


def test_dictionary_and_definition_sites_are_off_topic():
    for url in ("https://www.merriam-webster.com/dictionary/failure",
                "https://dictionary.cambridge.org/dictionary/english/failure",
                "https://www.thefreedictionary.com/challenges", "https://en.wiktionary.org/wiki/fail"):
        assert is_off_topic_web(QUESTION, "FAILURE Definition & Meaning", "failure of agent systems", url)


def test_a_title_in_another_script_is_off_topic():
    assert is_off_topic_web(QUESTION, "challenges是什么意思_challenges的翻译", "multi-agent systems", "https://x.com/a")


def test_a_page_sharing_no_word_with_the_question_is_off_topic_and_a_related_one_is_not():
    assert is_off_topic_web(QUESTION, "Best pasta recipes", "Cook the noodles for ten minutes.", "https://x.com/a")
    assert not is_off_topic_web(QUESTION, "Failure modes in multi-agent LLM systems", "A study.", "https://x.com/b")


def test_a_question_without_key_words_does_not_filter_anything():
    assert not is_off_topic_web("q?", "Blog on RAG", "Tips for RAG.", "https://example.com/rag")


def test_the_researcher_leaves_them_out_of_the_source_list():
    pages = [
        WebResult("Failure - definition", "The state of not working.", "https://www.merriam-webster.com/dictionary/failure"),
        WebResult("Why multi-agent LLM systems fail", "Coordination breaks down.", "https://example.com/fail"),
    ]
    sources = make_researcher(FakeLLM(), lambda q, n: [], lambda q, n: pages)({"question": QUESTION})["sources"]
    assert [s["title"] for s in sources] == ["Why multi-agent LLM systems fail"]
