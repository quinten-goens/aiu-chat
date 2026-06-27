"""Download EUROCONTROL AIU CSV datasets and convert them to local Parquet.

Run: python -m aiu_chat.ingest.download_datasets

This is the only step that touches the network. It is a polite client: it sets a
User-Agent, downloads year-by-year, and tolerates years that are not (yet)
published. CSVs are converted to a single typed Parquet file per dataset via
DuckDB (columnar, fast to query).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path

import duckdb
import requests

from aiu_chat import config
from aiu_chat.ingest.datasets import DATASETS, DatasetSpec

USER_AGENT = "aiu-chat/0.1 (local research tool; +https://ansperformance.eu)"
REQUEST_TIMEOUT = 60


def _download_year(spec: DatasetSpec, year: int, dest_dir: Path) -> Path | None:
    """Download one year's CSV. Returns the local path, or None if unavailable."""
    url = spec.url_for_year(year)
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        print(f"  ! {year}: request failed ({exc})", file=sys.stderr)
        return None

    if resp.status_code == 404:
        print(f"  - {year}: not published (404), skipping")
        return None
    if resp.status_code != 200:
        print(f"  ! {year}: HTTP {resp.status_code}, skipping", file=sys.stderr)
        return None
    if not resp.content.strip():
        print(f"  - {year}: empty response, skipping")
        return None

    out = dest_dir / f"{spec.key}_{year}.csv"
    out.write_bytes(resp.content)
    print(f"  + {year}: {len(resp.content):,} bytes")
    return out


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
        # Union all yearly CSVs into one typed Parquet via DuckDB. union_by_name
        # guards against column-order drift between years. Paths are internal
        # (temp dir + config), not user input, and DuckDB's COPY ... TO does not
        # accept bound parameters for the file targets, so we inline them after
        # escaping single quotes.
        glob = str(tmp_dir / f"{spec.key}_*.csv").replace("'", "''")
        out = str(parquet_path).replace("'", "''")
        con = duckdb.connect()  # in-memory; we only need it to transcode
        try:
            con.execute(
                f"""
                COPY (
                    SELECT * FROM read_csv_auto('{glob}', union_by_name=true)
                ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            (rows,) = con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]
            ).fetchone()
        finally:
            con.close()

    print(f"  -> wrote {parquet_path} ({rows:,} rows)")
    return parquet_path


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
