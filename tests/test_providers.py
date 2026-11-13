from conftest import FakeLLM
from scholarcrew.providers import PROVIDERS, check_llm, problems, settings_for


def test_presets_are_complete():
    for name in ("Groq (free)", "Groq fast (free)", "Google Gemini (free)"):
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


def test_fast_groq_option_is_a_different_model_on_the_same_endpoint():
    """A smaller model has its own daily quota, so it helps when the main one is busy or spent."""
    main, fast = PROVIDERS["Groq (free)"], PROVIDERS["Groq fast (free)"]
    assert fast.base_url == main.base_url and fast.key_url == main.key_url
    assert fast.default_model == "openai/gpt-oss-20b" and fast.default_model != main.default_model
