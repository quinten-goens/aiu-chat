"""Tests for the Ollama client, mocking HTTP so no server is needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiu_chat.agent.llm import Message, OllamaClient, OllamaError


def test_from_tier_selects_model_and_keeps_embeddings():
    from aiu_chat import config

    for tier, spec in config.MODEL_TIERS.items():
        c = OllamaClient.from_tier(tier, think=True, num_ctx=16384)
        assert c.model == spec["model"]
        assert c.think is True
        assert c.num_ctx == 16384
        # embeddings stay on the embedding model regardless of chat tier
        assert c.embedding_model == config.EMBEDDING_MODEL


def test_from_tier_unknown_falls_back_to_default():
    from aiu_chat import config

    c = OllamaClient.from_tier("does-not-exist")
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
