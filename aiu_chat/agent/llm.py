"""Thin Ollama client (chat + embeddings).

Talks to a local Ollama server over its native HTTP API. Kept deliberately small:
the rest of the agent depends only on `chat()`, `chat_json()`, and `embed()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from aiu_chat import config


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


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

    @classmethod
    def from_tier(cls, tier: str | None = None, *, num_ctx: int | None = None) -> "OllamaClient":
        """Build a client for a named mode (fast/smart).

        Thinking is forced off (the prompts scaffold the reasoning and it's much
        faster). Context window comes from the tier unless explicitly overridden.
        """
        tier = tier or config.DEFAULT_TIER
        spec = config.MODEL_TIERS.get(tier) or config.MODEL_TIERS[config.DEFAULT_TIER]
        return cls(model=spec["model"], think=False, num_ctx=num_ctx or spec.get("num_ctx"))

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
        """Return the embedding vector for a single string."""
        data = self._post(
            "/api/embeddings", {"model": self.embedding_model, "prompt": text}
        )
        try:
            return data["embedding"]
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected embedding response: {data!r}") from exc

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
