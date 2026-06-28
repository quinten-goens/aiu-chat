# Deploying to Streamlit Cloud (OpenAI-only)

The app runs in two modes, switched by the `LOCAL` env var:

- **`LOCAL=true`** (default) — your machine: chat + embeddings via local **Ollama**
  (the ⚡ Fast / 🧠 Smart modes), data read from the local Parquet files +
  `data/aiu.duckdb`.
- **`LOCAL=false`** — cloud: chat + embeddings via **OpenAI**, everything served
  from a single self-contained `data/aiu_cloud.duckdb` (datasets *and* the vector
  index baked in). No Ollama, no Parquet directory.

## What ships to the cloud

- The code.
- **`data/aiu_cloud.duckdb`** — the self-contained DB (datasets + OpenAI-embedded
  vector index + acronyms). It is ~216 MB, tracked with **Git LFS**.
- `data/catalog.json` — the dataset schema/catalogue.
- `requirements.txt` — runtime deps (no Ollama).

## One-time: build & commit the cloud DB

Run locally (needs the local Parquet data + an `OPENAI_KEY`; embeds ~8k chunks):

```bash
LOCAL=false OPENAI_KEY=sk-... python -m aiu_chat.ingest.build_cloud_db
```

It writes `data/aiu_cloud.duckdb`. Git LFS already tracks it (see
`.gitattributes`); commit and push:

```bash
git add data/aiu_cloud.duckdb data/catalog.json
git commit -m "Update cloud DuckDB"
git push      # uploads the big file via LFS
```

## Deploy on share.streamlit.io

1. Connect the GitHub repo; set the main file to **`app/streamlit_app.py`**.
2. In **Settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):
   ```toml
   LOCAL = "false"
   OPENAI_KEY = "sk-..."
   PB_NOP_URL = "https://aiu-nop.pockethost.io"
   PB_NOP_USER_EMAIL = "..."
   PB_NOP_USER_PASSWORD = "..."
   ```
3. Deploy. Streamlit Cloud pulls the LFS file automatically.

## Notes

- **Embeddings differ by mode** (local nomic = 768-dim, cloud OpenAI = 1536-dim),
  so the two DuckDB files are not interchangeable — that's why cloud uses its own
  `aiu_cloud.duckdb`.
- **Live sources** (NOP / Data App / NM live) work in the cloud as long as those
  EUROCONTROL/PocketBase endpoints are reachable from Streamlit Cloud.
- **Refresh the data**: re-run the build script and push the updated LFS file.
- **Git LFS quota**: GitHub's free tier is 1 GB storage / 1 GB bandwidth per
  month — fine for one ~216 MB file with occasional updates.
