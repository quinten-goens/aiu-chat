# AIU Chat

A **fully local, offline** AI chatbot that answers questions about European Air
Navigation Services (ANS) performance using the open data and reference content
published by EUROCONTROL's Aviation Intelligence Unit (AIU) at
[ansperformance.eu](https://ansperformance.eu).

It is a **hybrid agentic RAG** system: a local LLM (via Ollama) routes each
question to either **text-to-SQL over DuckDB/Parquet** (for numbers) or **vector
retrieval over documents** (for concepts), and can visualize results as charts.
See [CLAUDE.md](CLAUDE.md) for the full architecture and design decisions.

> **Unofficial.** This project is not affiliated with or endorsed by EUROCONTROL.
> Data © EUROCONTROL AIU (ansperformance.eu).

## Stack

- **UI:** Streamlit
- **LLM serving:** Ollama (chat + embeddings), OpenAI-compatible API
- **Default chat model:** `qwen3.5:9b` · **Embeddings:** `nomic-embed-text`
- **Data + vectors:** DuckDB (Parquet for data, VSS extension for doc vectors)
- **Charts:** Plotly from a validated LLM chart spec

## Setup

```bash
# 1. Local model runtime (one-time)
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# 3. Ingest data (downloads from ansperformance.eu / eurocontrol.int)
python -m aiu_chat.ingest.download_datasets
python -m aiu_chat.ingest.build_catalog
```

## Status

Early development. Built in vertical slices (see CLAUDE.md). Current milestone:
single-dataset ingestion + a read-only, sandboxed text-to-SQL tool over DuckDB.

## Tests

```bash
pytest
```
