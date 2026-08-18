"""A transient OpenAI 500 cost a whole logged turn (empty question and answer).
Retryable statuses must be retried; client errors must not be.

Note: the plan named a single `LLMError`; the module actually raises
`OllamaError` and `OpenAIError`, so the retry helper keys off the HTTP status
in the message and both classes are exercised here.
"""
from __future__ import annotations

import pytest

from aiu_chat.agent import llm


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise llm.OpenAIError("OpenAI API returned HTTP 500: server error")
        return "fine"

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    assert llm._with_retries(flaky) == "fine"
    assert calls["n"] == 3


def test_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def always_500(*a, **k):
        calls["n"] += 1
        raise llm.OpenAIError("OpenAI API returned HTTP 503: unavailable")

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    with pytest.raises(llm.OpenAIError):
        llm._with_retries(always_500)
    assert calls["n"] == 3


def test_does_not_retry_client_error(monkeypatch):
    calls = {"n": 0}

    def bad_request(*a, **k):
        calls["n"] += 1
        raise llm.OpenAIError("OpenAI API returned HTTP 400: bad request")

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    with pytest.raises(llm.OpenAIError):
        llm._with_retries(bad_request)
    assert calls["n"] == 1


def test_retries_ollama_errors_too(monkeypatch):
    """The local provider gets the same treatment."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise llm.OllamaError("Ollama /api/chat returned HTTP 502: bad gateway")
        return "ok"

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    assert llm._with_retries(flaky) == "ok"
    assert calls["n"] == 2


def test_transport_failure_is_retried(monkeypatch):
    """"Could not reach the OpenAI API" is transient too — turn 15's sibling."""
    calls = {"n": 0}

    def unreachable(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise llm.OpenAIError("Could not reach the OpenAI API (timeout).")
        return "ok"

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    assert llm._with_retries(unreachable) == "ok"
    assert calls["n"] == 2
