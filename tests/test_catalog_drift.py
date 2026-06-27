"""Tests for schema-drift detection in the catalog builder."""
from __future__ import annotations

import duckdb
import pytest

from aiu_chat.ingest.build_catalog import build_dataset_entry
from aiu_chat.ingest.datasets import ColumnSpec, DatasetSpec


def _make_parquet(tmp_path, columns_sql, name="co2_emissions_by_state"):
    parquet = tmp_path / f"{name}.parquet"
    con = duckdb.connect()
    try:
        safe = str(parquet).replace("'", "''")
        con.execute(f"COPY (SELECT {columns_sql}) TO '{safe}' (FORMAT PARQUET)")
    finally:
        con.close()
    return parquet


def _spec(tmp_path, expected_cols):
    return DatasetSpec(
        key="co2_emissions_by_state",
        title="t", description="d", filename_pattern="x_{year}.csv",
        first_year=2024,
        columns=[ColumnSpec(c, "INTEGER", "") for c in expected_cols],
    )


def test_no_drift_when_columns_match(tmp_path, monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path)
    _make_parquet(tmp_path, "1 AS YEAR, 2 AS MONTH")
    spec = _spec(tmp_path, ["YEAR", "MONTH"])

    con = duckdb.connect()
    try:
        entry, warnings = build_dataset_entry(con, spec)
    finally:
        con.close()
    assert warnings == []
    assert {c["name"] for c in entry["columns"]} == {"YEAR", "MONTH"}


def test_missing_column_is_flagged(tmp_path, monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path)
    _make_parquet(tmp_path, "1 AS YEAR")           # data has only YEAR
    spec = _spec(tmp_path, ["YEAR", "MONTH"])       # registry expects MONTH too

    con = duckdb.connect()
    try:
        _, warnings = build_dataset_entry(con, spec)
    finally:
        con.close()
    assert any("MISSING" in w and "MONTH" in w for w in warnings)


def test_extra_column_is_flagged(tmp_path, monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path)
    _make_parquet(tmp_path, "1 AS YEAR, 2 AS SURPRISE")  # data has an undocumented col
    spec = _spec(tmp_path, ["YEAR"])

    con = duckdb.connect()
    try:
        _, warnings = build_dataset_entry(con, spec)
    finally:
        con.close()
    assert any("not described" in w and "SURPRISE" in w for w in warnings)
