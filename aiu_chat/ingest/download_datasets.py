"""Download EUROCONTROL AIU CSV datasets and convert them to local Parquet.

Run: python -m aiu_chat.ingest.download_datasets

This is the only step that touches the network. It is a polite client: it sets a
User-Agent, downloads year-by-year, and tolerates years that are not (yet)
published. CSVs are converted to a single typed Parquet file per dataset via
DuckDB (columnar, fast to query).
"""
from __future__ import annotations

import argparse
import bz2
import datetime as dt
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import requests

from aiu_chat import config
from aiu_chat.ingest.datasets import DATASETS, DatasetSpec

USER_AGENT = "aiu-chat/0.1 (local research tool; +https://ansperformance.eu)"
REQUEST_TIMEOUT = 60


class DatasetNotAvailable(Exception):
    """A year is legitimately not published (HTTP 404)."""


def _get_with_retry(url: str, attempts: int = 3, backoff: float = 2.0):
    """GET with retries on transient errors. Raises on 404 (DatasetNotAvailable)
    or after exhausting retries, so failures are loud rather than silent."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code == 404:
                raise DatasetNotAvailable(url)
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        if attempt < attempts:
            time.sleep(backoff * attempt)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_exc}")


def _download_year(spec: DatasetSpec, year: int, dest_dir: Path) -> Path | None:
    """Download one year's CSV. Returns the local path, or None if not published.

    Raises on a transient/server failure (after retries) so a flaky network can't
    silently shrink a dataset.
    """
    url = spec.url_for_year(year)
    try:
        resp = _get_with_retry(url)
    except DatasetNotAvailable:
        print(f"  - {year}: not published (404)")
        return None

    if not resp.content.strip():
        print(f"  - {year}: empty response, skipping")
        return None

    content = resp.content
    if spec.compressed:
        try:
            content = bz2.decompress(content)
        except OSError as exc:
            print(f"  ! {year}: bz2 decompress failed ({exc}), skipping", file=sys.stderr)
            return None

    # Normalise to clean UTF-8. Some files contain stray non-UTF-8 bytes (e.g. an
    # airport name in latin-1) that otherwise break DuckDB's CSV reader. Decoding
    # as UTF-8 with latin-1 fallback then re-encoding yields valid UTF-8 without
    # dropping rows.
    content = _to_utf8(content)

    out = dest_dir / f"{spec.key}_{year}.csv"
    out.write_bytes(content)
    print(f"  + {year}: {len(resp.content):,} bytes ({'bz2' if spec.compressed else 'csv'})")
    return out


def _to_utf8(raw: bytes) -> bytes:
    try:
        raw.decode("utf-8")
        return raw  # already valid
    except UnicodeDecodeError:
        # latin-1 maps every byte, so this never fails; it preserves accented
        # characters reasonably for European place names.
        return raw.decode("latin-1").encode("utf-8")


def download_dataset(spec: DatasetSpec, to_year: int | None = None) -> Path:
    """Download all available years for a dataset and write one Parquet file.

    Returns the path to the written Parquet file.
    """
    config.ensure_dirs()
    to_year = to_year or dt.date.today().year

    print(f"Downloading '{spec.key}' ({spec.first_year}-{to_year})...")
    with tempfile.TemporaryDirectory(prefix="aiu_csv_") as tmp:
        tmp_dir = Path(tmp)
        csv_paths = [
            p
            for year in range(spec.first_year, to_year + 1)
            if (p := _download_year(spec, year, tmp_dir)) is not None
        ]
        if not csv_paths:
            raise RuntimeError(f"No CSV files downloaded for '{spec.key}'.")

        parquet_path = config.PARQUET_DIR / f"{spec.key}.parquet"
        glob = str(tmp_dir / f"{spec.key}_*.csv").replace("'", "''")
        out = str(parquet_path).replace("'", "''")
        con = duckdb.connect()  # in-memory; we only need it to transcode
        try:
            rows = _transcode_to_parquet(con, glob, out, parquet_path)
        finally:
            con.close()

    print(f"  -> wrote {parquet_path} ({rows:,} rows)")
    return parquet_path


def _transcode_to_parquet(con, glob: str, out: str, parquet_path: Path) -> int:
    """Union yearly CSVs into one Parquet, robust to cross-year type drift.

    Paths are internal (temp dir + config), not user input; DuckDB's COPY ... TO
    can't bind file targets so we inline them after escaping single quotes.

    First try a typed read with a full scan. If a column's type clashes between
    years (e.g. BIGINT in one file, DOUBLE in another), fall back to reading
    every column as VARCHAR — that always parses, and types are applied at query
    time via casts. We never silently drop rows.
    """
    typed = (
        f"SELECT * FROM read_csv_auto('{glob}', union_by_name=true, sample_size=-1)"
    )
    fallback = (
        f"SELECT * FROM read_csv('{glob}', union_by_name=true, "
        f"all_varchar=true, sample_size=-1)"
    )
    for select in (typed, fallback):
        try:
            con.execute(f"COPY ({select}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            (rows,) = con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]
            ).fetchone()
            if select is fallback:
                print("    (note: read as text due to cross-year type drift)")
            return rows
        except duckdb.Error as exc:
            if select is fallback:
                raise
            print(f"    typed read failed ({str(exc)[:60]}...); retrying as text")
    return 0  # unreachable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download AIU datasets to Parquet.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset to download (default: all registered).",
    )
    parser.add_argument(
        "--to-year", type=int, default=None, help="Last year to attempt (default: current year)."
    )
    args = parser.parse_args(argv)

    targets = DATASETS.values() if args.dataset == "all" else [DATASETS[args.dataset]]
    for spec in targets:
        download_dataset(spec, to_year=args.to_year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
