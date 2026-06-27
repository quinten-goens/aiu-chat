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

Early development, built in vertical slices (see CLAUDE.md).

Working today: single-dataset ingestion → typed Parquet + semantic catalog; a
read-only, sandboxed text-to-SQL tool over DuckDB; grounded narration via a local
Ollama model; auto-generated Plotly charts from a validated chart spec; a gold
evaluation set; a CLI; and a Streamlit web UI.

Try it (after ingesting data and pulling a model):

```bash
# web UI
streamlit run app/streamlit_app.py

# or the CLI
aiu-chat-cli
# you> which 5 states had the most CO2 emissions in 2024?

# run the gold evaluation set against your local model
python -m aiu_chat.eval.runner
```

## Tests

```bash
pytest
```

## Troubleshooting

- **First request hangs / times out.** Reasoning models (e.g. `qwen3.5`) default
  to "thinking" mode and can spend minutes on hidden chain-of-thought. This app
  disables it by default (`AIU_OLLAMA_THINK=false`); ensure you're on a build
  that sends `think: false`.
- **High memory / slow generation.** Some models default to a 256K context
  (~20 GB). This app caps it (`AIU_OLLAMA_NUM_CTX=8192`). Check with `ollama ps`.
- **Wrong model tag.** Set `AIU_MODEL_NAME` in `.env` to match `ollama list`.
