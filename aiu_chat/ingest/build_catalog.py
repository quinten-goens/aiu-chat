"""Build the schema catalog from downloaded Parquet files.

Run: python -m aiu_chat.ingest.build_catalog

The catalog (data/catalog.json) is the single source of truth for:
  * which tables/columns exist (introspected from the actual Parquet, so it
    reflects reality, not assumptions);
  * the semantic notes (units, granularity, gotchas) from the dataset registry,
    which get injected into the SQL-generation prompt;
  * the data "as-of" date (max FLIGHT_MONTH / latest year) per dataset.

It also drives schema-drift detection: a refresh that finds columns not matching
the registry should fail loudly rather than silently break queries.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb

from aiu_chat import config
from aiu_chat.ingest.datasets import DATASETS, DatasetSpec


def _parquet_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> list[dict]:
    """Introspect column names and DuckDB types from a Parquet file."""
    rows = con.execute(
        "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
        [str(parquet_path)],
    ).fetchall()
    return [{"name": name, "type": dtype} for name, dtype in rows]


def _as_of(con: duckdb.DuckDBPyConnection, parquet_path: Path, columns: list[str]) -> str | None:
    """Best-effort 'data as-of' date for the dataset."""
    if "FLIGHT_MONTH" in columns:
        (val,) = con.execute(
            "SELECT MAX(FLIGHT_MONTH) FROM read_parquet(?)", [str(parquet_path)]
        ).fetchone()
        return str(val) if val is not None else None
    if "YEAR" in columns and "MONTH" in columns:
        row = con.execute(
            "SELECT MAX(YEAR), MAX(MONTH) FROM read_parquet(?)", [str(parquet_path)]
        ).fetchone()
        return f"{row[0]}-{row[1]:02d}" if row[0] is not None else None
    return None


def build_dataset_entry(
    con: duckdb.DuckDBPyConnection, spec: DatasetSpec
) -> tuple[dict, list[str]]:
    """Build one catalog entry. Returns (entry, drift_warnings)."""
    parquet_path = config.PARQUET_DIR / f"{spec.key}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet for '{spec.key}' not found at {parquet_path}. "
            f"Run `python -m aiu_chat.ingest.download_datasets` first."
        )

    actual = _parquet_columns(con, parquet_path)
    actual_names = [c["name"] for c in actual]
    spec_by_name = {c.name: c for c in spec.columns}

    # Schema-drift detection: compare registry expectations vs. reality.
    warnings: list[str] = []
    missing = [n for n in spec_by_name if n not in actual_names]
    extra = [n for n in actual_names if n not in spec_by_name]
    if missing:
        warnings.append(f"{spec.key}: columns in registry but MISSING from data: {missing}")
    if extra:
        warnings.append(f"{spec.key}: columns in data not described in registry: {extra}")

    # Merge actual types with registry semantics.
    columns = []
    for col in actual:
        meta = spec_by_name.get(col["name"])
        columns.append(
            {
                "name": col["name"],
                "type": col["type"],
                "description": meta.description if meta else "",
                "unit": meta.unit if meta else None,
                "note": meta.note if meta else None,
            }
        )

    entry = {
        "table": spec.key,
        "title": spec.title,
        "description": spec.description,
        "granularity": spec.granularity,
        "parquet_path": str(parquet_path),
        "as_of": _as_of(con, parquet_path, actual_names),
        "columns": columns,
    }
    return entry, warnings


def build_catalog(strict: bool = False) -> dict:
    """Build and write the full catalog. If strict, raise on any schema drift."""
    con = duckdb.connect()
    try:
        datasets = []
        all_warnings: list[str] = []
        for spec in DATASETS.values():
            entry, warnings = build_dataset_entry(con, spec)
            datasets.append(entry)
            all_warnings.extend(warnings)
    finally:
        con.close()

    catalog = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "EUROCONTROL Aviation Intelligence Unit (ansperformance.eu)",
        "datasets": datasets,
    }

    if all_warnings:
        print("Schema-drift warnings:")
        for w in all_warnings:
            print(f"  ! {w}")
        if strict:
            raise RuntimeError("Schema drift detected (strict mode). See warnings above.")

    config.ensure_dirs()
    config.CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote catalog with {len(datasets)} dataset(s) -> {config.CATALOG_PATH}")
    return catalog


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the schema catalog.")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on any schema drift (use in scheduled refresh)."
    )
    args = parser.parse_args(argv)
    build_catalog(strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
