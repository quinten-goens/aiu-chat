"""Tests for the LLM clients, mocking HTTP so no server/key is needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiu_chat.agent.llm import (
    Message,
    OllamaClient,
    OllamaError,
    OpenAIClient,
    build_client,
)


def test_build_client_selects_provider_and_model():
    from aiu_chat import config

    for tier, spec in config.MODEL_TIERS.items():
        c = build_client(tier)
        assert c.model == spec["model"]
        if spec.get("provider") == "openai":
            assert isinstance(c, OpenAIClient)
        else:
            assert isinstance(c, OllamaClient)
            assert c.think is False
            assert c.num_ctx == spec["num_ctx"]
        # embeddings always use the Ollama embedding model, regardless of provider
        assert c.embedding_model == config.EMBEDDING_MODEL


def test_build_client_unknown_falls_back_to_default():
    from aiu_chat import config

    c = build_client("does-not-exist")
    assert c.model == config.MODEL_TIERS[config.DEFAULT_TIER]["model"]


def _mock_response(status=200, payload=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.text = str(payload)
    return resp


@patch("aiu_chat.agent.llm.requests.post")
def test_chat_returns_content(mock_post):
    mock_post.return_value = _mock_response(
        payload={"message": {"role": "assistant", "content": "hello"}}
    )
    client = OllamaClient(host="http://x", model="m")
    out = client.chat([Message("user", "hi")])
    assert out == "hello"
    # temperature 0 + correct endpoint + perf-critical options
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/chat")
    body = kwargs["json"]
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_ctx"] == 8192  # capped context (avoids 256K blowup)
    assert body["think"] is False  # thinking off by default (avoids multi-min stalls)


@patch("aiu_chat.agent.llm.requests.post")
def test_chat_json_mode_sets_format(mock_post):
    mock_post.return_value = _mock_response(payload={"message": {"content": '{"a": 1}'}})
    client = OllamaClient(host="http://x", model="m")
    out = client.chat_json([Message("user", "hi")])
    assert out == {"a": 1}
    assert mock_post.call_args.kwargs["json"]["format"] == "json"


@patch("aiu_chat.agent.llm.requests.post")
def test_chat_json_invalid_raises(mock_post):
    mock_post.return_value = _mock_response(payload={"message": {"content": "not json"}})
    client = OllamaClient(host="http://x", model="m")
    with pytest.raises(OllamaError):
        client.chat_json([Message("user", "hi")])


@patch("aiu_chat.agent.llm.requests.post")
def test_embed_returns_vector(mock_post):
    mock_post.return_value = _mock_response(payload={"embedding": [0.1, 0.2, 0.3]})
    client = OllamaClient(host="http://x", embedding_model="e")
    assert client.embed("text") == [0.1, 0.2, 0.3]
    assert mock_post.call_args.args[0].endswith("/api/embeddings")


@patch("aiu_chat.agent.llm.requests.post")
def test_http_error_raises(mock_post):
    mock_post.return_value = _mock_response(status=500, payload={"error": "boom"})
    client = OllamaClient(host="http://x")
    with pytest.raises(OllamaError):
        client.chat([Message("user", "hi")])


# --- OpenAI client ---------------------------------------------------------

def _openai_response(status=200, content="hello"):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.text = content
    return resp


@patch("aiu_chat.agent.llm.requests.post")
def test_openai_chat_calls_api_and_returns_content(mock_post):
    from aiu_chat.agent.llm import OpenAIClient

    mock_post.return_value = _openai_response(content="42 flights")
    c = OpenAIClient(model="gpt-5.4-mini", api_key="sk-test")
    out = c.chat([Message("user", "hi")])
    assert out == "42 flights"
    args, kwargs = mock_post.call_args
    assert args[0] == OpenAIClient.API_URL
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["json"]["model"] == "gpt-5.4-mini"


@patch("aiu_chat.agent.llm.requests.post")
def test_openai_chat_json_sets_response_format(mock_post):
    from aiu_chat.agent.llm import OpenAIClient

    mock_post.return_value = _openai_response(content='{"route": "data"}')
    c = OpenAIClient(model="m", api_key="sk-test")
    assert c.chat_json([Message("user", "hi")]) == {"route": "data"}
    assert mock_post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}


def test_openai_chat_without_key_raises():
    from aiu_chat.agent.llm import OpenAIClient, OpenAIError

    c = OpenAIClient(model="m", api_key="")
    with pytest.raises(OpenAIError):
        c.chat([Message("user", "hi")])


@patch("aiu_chat.agent.llm.requests.post")
def test_openai_embed_uses_ollama(mock_post):
    """OpenAI client's embeddings still hit Ollama's /api/embeddings."""
    from aiu_chat.agent.llm import OpenAIClient

    mock_post.return_value = _mock_response(payload={"embedding": [0.5]})
    c = OpenAIClient(model="m", api_key="sk-test", embedding_model="e")
    assert c.embed("x") == [0.5]
    assert mock_post.call_args.args[0].endswith("/api/embeddings")
