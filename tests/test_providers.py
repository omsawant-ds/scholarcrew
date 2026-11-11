from conftest import FakeLLM
from scholarcrew.providers import PROVIDERS, check_llm, problems, settings_for


def test_presets_are_complete():
    for name in ("Groq (free)", "Google Gemini (free)"):
        p = PROVIDERS[name]
        assert p.base_url.startswith("https://") and p.default_model and p.key_url.startswith("https://")


def test_problems_flags_missing_inputs():
    assert "API key" in problems(settings_for("https://x", "", "m"))
    assert "http" in problems(settings_for("nonsense", "k", "m"))
    assert "model" in problems(settings_for("https://x", "k", ""))
    assert problems(settings_for("https://x", "k", "m")) is None


def test_settings_are_trimmed():
    s = settings_for(" https://x ", " key ", " m ")
    assert (s.base_url, s.api_key, s.model) == ("https://x", "key", "m")


def test_check_llm_ok_and_failure_never_leaks_key():
    s = settings_for("https://x", "sekret-key", "m")
    assert check_llm(s, FakeLLM()) is None

    class Boom:
        def invoke(self, _):
            raise RuntimeError("401 bad key sekret-key")

    error = check_llm(s, Boom())
    assert "RuntimeError" in error and "sekret-key" not in error and "***" in error


def test_app_loads_and_asks_for_a_key():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).parent.parent / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any("API key" in i.value for i in at.info)
    assert at.button[0].disabled
