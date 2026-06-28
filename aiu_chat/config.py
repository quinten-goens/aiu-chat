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

# On Streamlit Cloud, secrets are exposed via st.secrets (not as env vars). Mirror
# them into the environment so the os.getenv-based config below picks them up.
# Real env vars still win (we only set keys that aren't already present).
try:  # pragma: no cover - only runs under Streamlit
    import streamlit as _st

    for _k, _v in dict(_st.secrets).items():
        if isinstance(_v, (str, int, float, bool)) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:
    pass
# --- Deployment mode -------------------------------------------------------
# LOCAL=true  -> runs on your machine: chat + embeddings via local Ollama.
# LOCAL=false -> cloud (e.g. Streamlit Cloud): chat + embeddings via OpenAI only
#                (no Ollama). Set OPENAI_KEY for the cloud deployment.
LOCAL = os.getenv("LOCAL", "true").lower() in ("1", "true", "yes")

DATA_DIR = Path(os.getenv("AIU_DATA_DIR", REPO_ROOT / "data"))
PARQUET_DIR = Path(os.getenv("AIU_PARQUET_DIR", DATA_DIR / "parquet"))
CATALOG_PATH = Path(os.getenv("AIU_CATALOG_PATH", DATA_DIR / "catalog.json"))
# Vector index lives in its own DuckDB per deployment (different embedding dims):
# local uses nomic (768), cloud uses OpenAI (1536) — they are not interchangeable.
_DEFAULT_DUCKDB = DATA_DIR / ("aiu.duckdb" if LOCAL else "aiu_cloud.duckdb")
DUCKDB_PATH = Path(os.getenv("AIU_DUCKDB_PATH", _DEFAULT_DUCKDB))

# --- Ollama ----------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("AIU_MODEL_NAME", "qwen3.5:9b")

# --- Embeddings (provider depends on the deployment) -----------------------
# Local: Ollama nomic-embed-text (768-dim). Cloud: OpenAI text-embedding-3-small
# (1536-dim). The dimension MUST match the index that was built, so it is derived
# from the deployment rather than hand-set.
if LOCAL:
    EMBEDDING_PROVIDER = "ollama"
    EMBEDDING_MODEL = os.getenv("AIU_LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_DIM = int(os.getenv("AIU_LOCAL_EMBEDDING_DIM", "768"))
else:
    EMBEDDING_PROVIDER = "openai"
    EMBEDDING_MODEL = os.getenv("AIU_CLOUD_EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(os.getenv("AIU_CLOUD_EMBEDDING_DIM", "1536"))

OPENAI_KEY = os.getenv("OPENAI_KEY", "").strip()


def openai_enabled() -> bool:
    return bool(OPENAI_KEY)


# --- Selectable chat modes (depend on the deployment) ----------------------
# Local: Ollama/Qwen3.5 tiers. Cloud: OpenAI GPT tiers. Each tier names its
# provider + model; local thinking is always off.
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
        "label": "⚡ Fast · GPT nano",
        "blurb": "OpenAI's smallest GPT-5 model. Fast and inexpensive.",
    },
    "gpt_mini": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MINI", "gpt-5.4-mini"),
        "label": "🧠 Balanced · GPT mini",
        "blurb": "Balanced OpenAI model — a good default.",
    },
    "gpt_max": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MAX", "gpt-5.5"),
        "label": "🚀 Max · GPT (most capable)",
        "blurb": "OpenAI's most capable general model. Best quality, higher cost.",
    },
}

if LOCAL:
    MODEL_TIERS = dict(_LOCAL_TIERS)
    DEFAULT_TIER = os.getenv("AIU_MODEL_TIER", "fast")
else:
    MODEL_TIERS = dict(_OPENAI_TIERS)
    DEFAULT_TIER = os.getenv("AIU_MODEL_TIER", "gpt_mini")
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
# (EMBEDDING_DIM is derived from the deployment above.)
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
