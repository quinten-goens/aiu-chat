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

The full hybrid RAG agent works end-to-end on a single dataset (CO2 emissions by
state) plus the reference docs. The main remaining work is breadth: registering
the rest of the ~14 EUROCONTROL datasets (each is a `DatasetSpec` with semantic
notes) and ingesting linked methodology PDFs for deeper concept answers.

Built in vertical slices (see CLAUDE.md).

Working today — the full hybrid agent:

- **Data path:** read-only, sandboxed text-to-SQL over DuckDB/Parquet, with a
  semantic catalog so the model respects units/granularity; grounded narration.
- **Concept path:** vector retrieval (DuckDB VSS) over scraped reference pages,
  answered strictly from the excerpts with sources.
- **Router + orchestrator:** classifies each question (data / concept / both),
  rewrites follow-ups into standalone questions, and combines results.
- **Charts:** auto Plotly charts from a validated LLM chart spec.
- **Trustworthiness:** a gold eval set, schema-drift checks, and a scheduled
  refresh script.
- **Interfaces:** a CLI and a Streamlit web UI.

Try it (after ingesting data and pulling models):

```bash
# web UI
streamlit run app/streamlit_app.py

# or the CLI
aiu-chat-cli
# you> which 5 states had the most CO2 emissions in 2024?
# you> what about 2020?            (follow-ups work)
# you> what does ATFM stand for?   (concept questions work)

# run the gold evaluation set against your local model
python -m aiu_chat.eval.runner
```

## Refresh (scheduled)

Re-download data, rebuild the catalog (fails loudly on schema drift), and rebuild
the document index in one command:

```bash
bash scripts/refresh.sh
```

To run monthly via launchd on macOS, see
`scripts/com.aiuchat.refresh.plist.example`.

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
