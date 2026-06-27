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
DATA_DIR = Path(os.getenv("AIU_DATA_DIR", REPO_ROOT / "data"))
PARQUET_DIR = Path(os.getenv("AIU_PARQUET_DIR", DATA_DIR / "parquet"))
DUCKDB_PATH = Path(os.getenv("AIU_DUCKDB_PATH", DATA_DIR / "aiu.duckdb"))
CATALOG_PATH = Path(os.getenv("AIU_CATALOG_PATH", DATA_DIR / "catalog.json"))

# --- Ollama ----------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("AIU_MODEL_NAME", "qwen3.5:9b")
EMBEDDING_MODEL = os.getenv("AIU_EMBEDDING_MODEL", "nomic-embed-text")
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
TOP_K = int(os.getenv("AIU_TOP_K", "5"))
# Embedding vector dimension (nomic-embed-text = 768). Must match the model.
EMBEDDING_DIM = int(os.getenv("AIU_EMBEDDING_DIM", "768"))
# Minimum similarity (cosine) for a retrieved chunk to be considered relevant.
MIN_SIMILARITY = float(os.getenv("AIU_MIN_SIMILARITY", "0.4"))

# --- Safety / limits -------------------------------------------------------
# Hard cap on rows a generated query may return (also nudges the model to
# aggregate rather than dump raw rows).
MAX_RESULT_ROWS = int(os.getenv("AIU_MAX_RESULT_ROWS", "5000"))


def ensure_dirs() -> None:
    """Create local data directories if they do not yet exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
