"""The command line. Offline."""

from scholarcrew.cli import main
from scholarcrew.config import Settings


def test_without_a_key_it_says_where_to_get_one_instead_of_failing_obscurely(monkeypatch, capsys):
    monkeypatch.setattr("scholarcrew.config._load_dotenv", lambda: None)  # never read a real .env
    for name in ("LLM_API_KEY", "LLM_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    assert main(["a question"]) == 2
    assert "console.groq.com/keys" in capsys.readouterr().err


def test_defaults_point_at_groq(monkeypatch):
    monkeypatch.setattr("scholarcrew.config._load_dotenv", lambda: None)
    for name in ("LLM_API_KEY", "LLM_API_KEYS", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.model == "openai/gpt-oss-120b" and settings.all_keys == ()


def test_a_list_of_keys_alone_is_enough(monkeypatch):
    """Regression: with only LLM_API_KEYS set the main key was empty, so the command line failed."""
    monkeypatch.setattr("scholarcrew.config._load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEYS", "first, second")
    settings = Settings.from_env()
    assert settings.api_key == "first" and settings.all_keys == ("first", "second")
