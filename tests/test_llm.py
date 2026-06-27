"""Tests for the Ollama client, mocking HTTP so no server is needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiu_chat.agent.llm import Message, OllamaClient, OllamaError


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
    # temperature 0 + correct endpoint
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/chat")
    assert kwargs["json"]["options"]["temperature"] == 0.0


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
