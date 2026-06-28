"""Central configuration, read from environment with sensible defaults.

Everything the app touches (Ollama endpoint, model names, data paths, retrieval
knobs) is configurable here so nothing is hard-coded across modules.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Repo root = two levels up from this file (aiu_chat/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load a local .env if present so credentials (e.g. NOP) are picked up without a
# manual export. Real environment variables take precedence (override=False).
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass
DATA_DIR = Path(os.getenv("AIU_DATA_DIR", REPO_ROOT / "data"))
PARQUET_DIR = Path(os.getenv("AIU_PARQUET_DIR", DATA_DIR / "parquet"))
DUCKDB_PATH = Path(os.getenv("AIU_DUCKDB_PATH", DATA_DIR / "aiu.duckdb"))
CATALOG_PATH = Path(os.getenv("AIU_CATALOG_PATH", DATA_DIR / "catalog.json"))

# --- Ollama ----------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("AIU_MODEL_NAME", "qwen3.5:9b")
EMBEDDING_MODEL = os.getenv("AIU_EMBEDDING_MODEL", "nomic-embed-text")

# --- OpenAI (optional cloud provider) --------------------------------------
OPENAI_KEY = os.getenv("OPENAI_KEY", "").strip()


def openai_enabled() -> bool:
    return bool(OPENAI_KEY)


# Selectable modes. Local modes (Ollama/Qwen3.5) are always available; OpenAI
# modes appear only when an OPENAI_KEY is set. Each tier names its provider and
# model. Local thinking is always off; embeddings always use Ollama (to match
# the existing nomic-embed-text vector index) regardless of the chat provider.
_LOCAL_TIERS = {
    "fast": {
        "provider": "ollama",
        "model": os.getenv("AIU_MODEL_FAST", "qwen3.5:4b"),
        "label": "⚡ Fast · local (qwen3.5:4b)",
        "blurb": "Quicker responses, runs on your machine. Good for everyday questions.",
        "num_ctx": int(os.getenv("AIU_FAST_NUM_CTX", "16384")),
    },
    "smart": {
        "provider": "ollama",
        "model": os.getenv("AIU_MODEL_SMART", MODEL_NAME),
        "label": "🧠 Smart · local (qwen3.5:9b)",
        "blurb": "Stronger local reasoning. Better on harder, multi-step questions.",
        "num_ctx": int(os.getenv("AIU_SMART_NUM_CTX", "8192")),
    },
}

_OPENAI_TIERS = {
    "gpt_nano": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_NANO", "gpt-5.4-nano"),
        "label": "☁️ GPT nano · cloud (fast & cheap)",
        "blurb": "OpenAI's smallest GPT-5 model. Fast and inexpensive.",
    },
    "gpt_mini": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MINI", "gpt-5.4-mini"),
        "label": "☁️ GPT mini · cloud (balanced)",
        "blurb": "Balanced OpenAI model — a good cloud default.",
    },
    "gpt_max": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MAX", "gpt-5.5"),
        "label": "☁️ GPT max · cloud (most capable)",
        "blurb": "OpenAI's most capable general model. Best quality, higher cost.",
    },
}

# OpenAI tiers are only offered when a key is configured.
MODEL_TIERS = {**_LOCAL_TIERS, **(_OPENAI_TIERS if openai_enabled() else {})}
DEFAULT_TIER = os.getenv("AIU_MODEL_TIER", "fast")
# Context window cap. Some models default to a 256K context that inflates memory
# to ~20GB and slows generation dramatically; our prompts are small.
OLLAMA_NUM_CTX = int(os.getenv("AIU_OLLAMA_NUM_CTX", "8192"))
# HTTP timeout (seconds). Generous to absorb a cold model load on first request.
OLLAMA_TIMEOUT = int(os.getenv("AIU_OLLAMA_TIMEOUT", "180"))
# Disable "thinking" mode for reasoning models (e.g. qwen3.5). Our prompts do the
# reasoning scaffolding; hidden chain-of-thought adds minutes per query for no
# benefit on deterministic SQL/JSON generation. Set to "1"/"true" to re-enable.
OLLAMA_THINK = os.getenv("AIU_OLLAMA_THINK", "false").lower() in ("1", "true", "yes")

# --- Retrieval -------------------------------------------------------------
# With a large PDF-inclusive corpus, give the model a few more candidates.
TOP_K = int(os.getenv("AIU_TOP_K", "8"))
# Embedding vector dimension (nomic-embed-text = 768). Must match the model.
EMBEDDING_DIM = int(os.getenv("AIU_EMBEDDING_DIM", "768"))
# Minimum similarity (cosine) for a retrieved chunk to be considered relevant.
# 0.5 trims weak/irrelevant matches that otherwise dilute concept answers.
MIN_SIMILARITY = float(os.getenv("AIU_MIN_SIMILARITY", "0.5"))

# --- Safety / limits -------------------------------------------------------
# Hard cap on rows a generated query may return (also nudges the model to
# aggregate rather than dump raw rows).
MAX_RESULT_ROWS = int(os.getenv("AIU_MAX_RESULT_ROWS", "5000"))

# --- NOP content (PocketBase) ----------------------------------------------
PB_NOP_URL = os.getenv("PB_NOP_URL", "https://aiu-nop.pockethost.io").rstrip("/")
PB_NOP_USER_EMAIL = os.getenv("PB_NOP_USER_EMAIL", "")
PB_NOP_USER_PASSWORD = os.getenv("PB_NOP_USER_PASSWORD", "")

# --- EUROCONTROL Data App API ----------------------------------------------
# NOTE: this API is D-1 (yesterday's daily figures), not real-time.
DATAAPP_BASE = os.getenv("AIU_DATAAPP_BASE", "https://api-data-app.eurocontrol.int/api").rstrip("/")

# --- EUROCONTROL NM live API -----------------------------------------------
# The genuinely real-time Network Manager API behind .../performance/live.html.
NM_LIVE_BASE = os.getenv(
    "AIU_NM_LIVE_BASE", "https://int-api.nsv.eurocontrol.int/eurocontrol"
).rstrip("/")


def nop_configured() -> bool:
    """True if NOP PocketBase credentials are present."""
    return bool(PB_NOP_USER_EMAIL and PB_NOP_USER_PASSWORD)


def ensure_dirs() -> None:
    """Create local data directories if they do not yet exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
