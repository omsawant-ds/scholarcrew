"""Production touches in the web app: the author credit, the accuracy note, readable errors. Offline."""

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from scholarcrew.providers import explain_error

APP = Path(__file__).parent.parent / "streamlit_app.py"


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_the_author_is_credited_in_the_header_and_the_footer():
    at = new_app()
    assert not at.exception
    pages = [m.value for m in at.markdown]
    assert sum("Built by" in m and "Om Sawant" in m for m in pages) >= 2  # hero and footer


def test_the_page_tells_readers_the_reports_are_written_by_ai_from_abstracts():
    note = next(m.value for m in new_app().markdown if "written by AI" in m.value)
    assert "abstracts" in note and "Check the cited sources" in note


def test_the_model_choice_is_labelled_as_a_model_not_a_provider():
    at = new_app()
    assert at.selectbox[0].label == "Model"


def test_the_settings_button_is_never_hidden():
    """Regression: hiding the whole toolbar also hid the button that reopens the settings panel,
    so a visitor who closed it (easy to do on a phone) could not get it back."""
    css = APP.read_text(encoding="utf-8")
    hidden = re.search(r"#MainMenu[^{]*\{", css).group(0)  # the list of things that are hidden
    assert '"stToolbar"' not in hidden and "stExpandSidebarButton" in css


def test_the_three_agents_stay_on_one_row_on_a_phone():
    css = APP.read_text(encoding="utf-8")
    phone = css.split("@media (max-width: 640px)")[1]
    assert "grid-template-columns: 1fr" not in phone  # the old rule that stacked them


def test_common_errors_get_a_plain_explanation():
    assert "did not accept this key" in explain_error("AuthenticationError: Error code: 401 - Invalid API Key")
    assert "free limit" in explain_error("RateLimitError: Error code: 429 - Rate limit reached")
    assert "Groq fast" in explain_error("429 tokens per day (TPD)")
    assert "internet connection" in explain_error("APIConnectionError: Connection error.")
    assert "does not offer this model" in explain_error("NotFoundError: 404 - The model `x` does not exist")
    assert explain_error("something odd").startswith("Something went wrong")


def test_an_error_message_never_blames_the_wrong_thing_for_a_rejected_key():
    assert "limit" not in explain_error("401 Invalid API Key").lower()
