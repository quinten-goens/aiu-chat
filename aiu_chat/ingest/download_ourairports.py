"""Download OurAirports reference CSVs (airports + countries) for entity enrichment.

Run: python -m aiu_chat.ingest.download_ourairports

OurAirports data is **public domain** ("released to the Public Domain … credit
appreciated but not required"). It is used only to enrich the entity/knowledge
layer (IATA codes, aliases, municipality, ISO country map) — it is never exposed
as queryable performance data.

A snapshot is written under data/ourairports/ and committed so ingestion stays
reproducible offline (the network is only touched here). Re-run to refresh.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from aiu_chat import config

USER_AGENT = "aiu-chat/0.1 (local research tool; OurAirports public-domain data)"
REQUEST_TIMEOUT = 60

# The canonical mirror (GitHub-hosted, stable since Nov 2021).
BASE_URL = "https://davidmegginson.github.io/ourairports-data"
FILES = ("airports.csv", "countries.csv")


def ourairports_dir() -> Path:
    return config.DATA_DIR / "ourairports"


def _get_with_retry(url: str, attempts: int = 3, backoff: float = 2.0) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        if attempt < attempts:
            time.sleep(backoff * attempt)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_exc}")


def download(dest_dir: Path | None = None) -> list[Path]:
    """Fetch the OurAirports CSVs into dest_dir. Returns the written paths."""
    dest_dir = dest_dir or ourairports_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in FILES:
        url = f"{BASE_URL}/{name}"
        print(f"Downloading {url} ...")
        resp = _get_with_retry(url)
        if not resp.content.strip():
            raise RuntimeError(f"empty response for {url}")
        out = dest_dir / name
        out.write_bytes(resp.content)
        print(f"  + {name}: {len(resp.content):,} bytes")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OurAirports reference CSVs.")
    parser.add_argument("--dest", default=None, help="Destination dir (default: data/ourairports/).")
    args = parser.parse_args(argv)
    dest = Path(args.dest) if args.dest else None
    paths = download(dest)
    print(f"Wrote {len(paths)} file(s) to {paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
