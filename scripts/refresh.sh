#!/usr/bin/env bash
# Full data refresh: download datasets, rebuild the catalog (failing loudly on
# schema drift), and rebuild the document index. Intended for cron / launchd.
#
# Usage: scripts/refresh.sh
# Env:   AIU_PYTHON  - python interpreter to use (default: .venv/bin/python)
set -euo pipefail

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${AIU_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "error: python interpreter not found at ${PYTHON}" >&2
    echo "       set AIU_PYTHON or create the venv (python3 -m venv .venv)" >&2
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Refresh starting (repo: ${REPO_ROOT})"

log "Step 1/3: downloading datasets -> Parquet"
"${PYTHON}" -m aiu_chat.ingest.download_datasets

log "Step 2/3: rebuilding catalog (strict — fails on schema drift)"
"${PYTHON}" -m aiu_chat.ingest.build_catalog --strict

log "Step 3/3: rebuilding document index"
"${PYTHON}" -m aiu_chat.ingest.build_index

log "Refresh complete."
