"""LLM clients (chat + embeddings).

Two interchangeable chat providers — local Ollama and cloud OpenAI — exposing the
same small interface the rest of the agent depends on: `chat()`, `chat_json()`,
and `embed()`. Embeddings ALWAYS go through Ollama (nomic-embed-text) so they
match the existing vector index, regardless of which chat provider is used.

Use `build_client(tier)` to get the right client for a configured mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from aiu_chat import config


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class OpenAIError(RuntimeError):
    """Raised when the OpenAI API is unreachable or returns an error."""


# Embeddings match the deployment's index: Ollama nomic-embed-text (local, 768)
# or OpenAI text-embedding-3-small (cloud, 1536). The dimension must match the
# index the documents were embedded with, so the provider is config-driven.
def embed_text(text: str, *, model: str | None = None, timeout: int = 60) -> list[float]:
    """Return the embedding vector for a string, via the configured provider."""
    if config.EMBEDDING_PROVIDER == "openai":
        return _embed_openai(text, model=model or config.EMBEDDING_MODEL, timeout=timeout)
    return _embed_ollama(text, model=model or config.EMBEDDING_MODEL, timeout=timeout)


def _embed_ollama(text: str, *, model: str, timeout: int) -> list[float]:
    host = config.OLLAMA_HOST.rstrip("/")
    try:
        resp = requests.post(
            f"{host}/api/embeddings", json={"model": model, "prompt": text}, timeout=timeout
        )
    except requests.RequestException as exc:
        raise OllamaError(
            f"Could not reach Ollama for embeddings at {host}. Is `ollama serve` running? ({exc})"
        ) from exc
    if resp.status_code != 200:
        raise OllamaError(f"Ollama /api/embeddings returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()["embedding"]
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaError(f"Unexpected embedding response: {resp.text[:200]!r}") from exc


def _embed_openai(text: str, *, model: str, timeout: int) -> list[float]:
    if not config.OPENAI_KEY:
        raise OpenAIError("OPENAI_KEY is not set (needed for cloud embeddings).")
    try:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            json={"model": model, "input": text}, timeout=timeout,
            headers={"Authorization": f"Bearer {config.OPENAI_KEY}",
                     "Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        raise OpenAIError(f"Could not reach the OpenAI embeddings API ({exc}).") from exc
    if resp.status_code != 200:
        raise OpenAIError(f"OpenAI embeddings returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpenAIError(f"Unexpected OpenAI embeddings response: {resp.text[:200]!r}") from exc


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
        think: bool | None = None,
    ):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.MODEL_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        # Cap the context window. Some models (e.g. qwen3.5:9b) default to a huge
        # 256K context, which inflates memory to ~20GB and makes generation crawl.
        # Our prompts are small, so a modest window is much faster.
        self.num_ctx = num_ctx or config.OLLAMA_NUM_CTX
        # Reasoning models burn minutes on hidden chain-of-thought per call;
        # disabled by default for deterministic SQL/JSON generation.
        self.think = config.OLLAMA_THINK if think is None else think

    # --- chat --------------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion and return the assistant text.

        temperature defaults to 0 — for SQL/JSON generation we want determinism.
        json_mode asks Ollama to constrain output to valid JSON.
        """
        payload = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "think": self.think,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"

        data = self._post("/api/chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected chat response shape: {data!r}") from exc

    def chat_json(
        self, messages: list[Message], *, temperature: float = 0.0
    ) -> dict:
        """Chat in JSON mode and parse the result into a dict."""
        raw = self.chat(messages, temperature=temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Model did not return valid JSON: {raw!r}") from exc

    # --- embeddings --------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single string (configured provider)."""
        return embed_text(text, model=self.embedding_model, timeout=self.timeout)

    # --- internals ---------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.host}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running? ({exc})"
            ) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama {path} returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


class OpenAIClient:
    """OpenAI chat client over the REST API (same interface as OllamaClient).

    Chat goes to OpenAI; embeddings delegate to Ollama so they match the index.
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        timeout: int | None = None,
    ):
        self.model = model or config.MODEL_NAME
        self.api_key = api_key or config.OPENAI_KEY
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.timeout = timeout or 120

    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        if not self.api_key:
            raise OpenAIError("OPENAI_KEY is not set.")
        payload: dict = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                self.API_URL, json=payload, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
        except requests.RequestException as exc:
            raise OpenAIError(f"Could not reach the OpenAI API ({exc}).") from exc
        if resp.status_code != 200:
            # Some models reject temperature != default; retry once without it.
            if resp.status_code == 400 and "temperature" in resp.text.lower():
                payload.pop("temperature", None)
                resp = requests.post(
                    self.API_URL, json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                )
            if resp.status_code != 200:
                raise OpenAIError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenAIError(f"Unexpected OpenAI response: {resp.text[:300]!r}") from exc

    def chat_json(self, messages: list[Message], *, temperature: float = 0.0) -> dict:
        raw = self.chat(messages, temperature=temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAIError(f"Model did not return valid JSON: {raw!r}") from exc

    def embed(self, text: str) -> list[float]:
        """Embeddings still go through Ollama to match the vector index."""
        return embed_text(text, model=self.embedding_model)


def build_client(tier: str | None = None):
    """Build the right chat client for a configured mode (provider-aware)."""
    tier = tier or config.DEFAULT_TIER
    spec = config.MODEL_TIERS.get(tier) or config.MODEL_TIERS[config.DEFAULT_TIER]
    if spec.get("provider") == "openai":
        return OpenAIClient(model=spec["model"])
    return OllamaClient(model=spec["model"], think=False, num_ctx=spec.get("num_ctx"))
