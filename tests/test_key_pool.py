"""Several API keys: switch when one is rejected or out of daily quota. Offline."""

import pytest

from scholarcrew.config import Settings
from scholarcrew.llm import KeyPool, key_unusable
from scholarcrew.providers import check_llm


class RateLimitError(Exception):
    """Same class name as the provider client's, which is how the pool recognizes it."""


class AuthenticationError(Exception):
    pass


DAILY = RateLimitError("Rate limit reached on tokens per day (TPD): Limit 200000, Used 199990")
PER_MINUTE = RateLimitError("Rate limit reached on tokens per minute (TPM): Limit 8000")


class Model:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, fail, 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.name


def test_settings_lists_the_main_key_first_without_blanks_or_repeats(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "main")
    monkeypatch.setenv("LLM_API_KEYS", " second , ,main,third")
    assert Settings.from_env().all_keys == ("main", "second", "third")
    monkeypatch.delenv("LLM_API_KEYS")
    assert Settings.from_env().all_keys == ("main",)


def test_only_daily_limits_and_rejected_keys_count_as_unusable():
    assert key_unusable(DAILY) and key_unusable(AuthenticationError("bad key"))
    assert not key_unusable(PER_MINUTE)  # waiting fixes this one, not a new key
    assert not key_unusable(RuntimeError("tokens per day"))  # wrong kind of error


def test_pool_moves_to_the_next_key_and_repeats_the_call():
    first, second = Model("a", fail=DAILY), Model("b")
    pool = KeyPool([first, second])
    assert pool.invoke("hi") == "b"
    assert first.calls == 1 and second.calls == 1


def test_a_spent_key_is_not_tried_again():
    first, second = Model("a", fail=DAILY), Model("b")
    pool = KeyPool([first, second])
    pool.invoke("x")
    pool.invoke("y")
    assert first.calls == 1 and second.calls == 2 and pool.keys_left == 1


def test_pool_raises_when_every_key_is_spent():
    pool = KeyPool([Model("a", fail=DAILY), Model("b", fail=AuthenticationError("bad key"))])
    with pytest.raises(AuthenticationError):
        pool.invoke("x")


def test_other_errors_are_not_swallowed_and_do_not_switch_keys():
    first, second = Model("a", fail=PER_MINUTE), Model("b")
    pool = KeyPool([first, second])
    with pytest.raises(RateLimitError):
        pool.invoke("x")
    assert second.calls == 0


def test_check_llm_hides_every_key_in_its_error_message():
    settings = Settings(base_url="https://x", api_key="key-one", model="m", api_keys=("key-two",))
    bad = Model("a", fail=RuntimeError("rejected key-one and key-two"))
    error = check_llm(settings, bad)
    assert "key-one" not in error and "key-two" not in error and "***" in error


def test_vendor_prefixed_error_names_are_recognized():
    """Regression: the real client raises OpenAIRateLimitError / OpenAIAuthenticationError, and
    matching only the bare names meant a spent key was never skipped."""
    class OpenAIRateLimitError(Exception):
        pass

    class OpenAIAuthenticationError(Exception):
        pass

    assert key_unusable(OpenAIRateLimitError("Rate limit reached on tokens per day (TPD)"))
    assert not key_unusable(OpenAIRateLimitError("Rate limit reached on tokens per minute (TPM)"))
    assert key_unusable(OpenAIAuthenticationError("Invalid API Key"))
