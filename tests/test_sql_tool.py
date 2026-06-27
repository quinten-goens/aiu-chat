"""Tests for the read-only SQL tool: functional correctness + adversarial safety.

These run against a small synthetic Parquet file so they need no network and no
downloaded data.
"""
from __future__ import annotations

import duckdb
import pytest

from aiu_chat.agent.catalog import Catalog, DatasetCatalogEntry
from aiu_chat.agent.sql_tool import UnsafeSQLError, run_sql, validate_sql


@pytest.fixture()
def catalog(tmp_path):
    """Build a tiny co2 parquet + matching catalog for tests."""
    parquet = tmp_path / "co2_emissions_by_state.parquet"
    con = duckdb.connect()
    try:
        safe = str(parquet).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT * FROM (VALUES
                    (2024, 1, 'FRANCE',  'LF', 100.0, 10),
                    (2024, 2, 'FRANCE',  'LF', 200.0, 20),
                    (2024, 1, 'GERMANY', 'ED', 300.0, 30),
                    (2024, 2, 'GERMANY', 'ED', 400.0, 40)
                ) AS t(YEAR, MONTH, STATE_NAME, STATE_CODE, CO2_QTY_TONNES, TF)
            ) TO '{safe}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

    return Catalog(
        datasets=[
            DatasetCatalogEntry(
                table="co2_emissions_by_state",
                title="CO2 Emissions by State",
                description="test",
                granularity="one row per state per month",
                parquet_path=str(parquet),
                as_of="2024-02-01",
                columns=[
                    {"name": "YEAR", "type": "BIGINT"},
                    {"name": "MONTH", "type": "BIGINT"},
                    {"name": "STATE_NAME", "type": "VARCHAR"},
                    {"name": "STATE_CODE", "type": "VARCHAR"},
                    {"name": "CO2_QTY_TONNES", "type": "DOUBLE"},
                    {"name": "TF", "type": "BIGINT"},
                ],
            )
        ]
    )


# --- Functional correctness ------------------------------------------------

def test_simple_aggregation(catalog):
    res = run_sql(
        "SELECT SUM(CO2_QTY_TONNES) AS total FROM co2_emissions_by_state "
        "WHERE STATE_NAME = 'FRANCE'",
        catalog=catalog,
    )
    assert res.row_count == 1
    assert res.dataframe["total"].iloc[0] == 300.0


def test_group_by(catalog):
    res = run_sql(
        "SELECT STATE_NAME, SUM(CO2_QTY_TONNES) AS total "
        "FROM co2_emissions_by_state GROUP BY STATE_NAME ORDER BY total DESC",
        catalog=catalog,
    )
    assert list(res.dataframe["STATE_NAME"]) == ["GERMANY", "FRANCE"]
    assert list(res.dataframe["total"]) == [700.0, 300.0]


def test_cte_query_is_allowed(catalog):
    res = run_sql(
        "WITH per_state AS ("
        "  SELECT STATE_NAME, SUM(CO2_QTY_TONNES) AS total "
        "  FROM co2_emissions_by_state GROUP BY STATE_NAME"
        ") SELECT MAX(total) AS top FROM per_state",
        catalog=catalog,
    )
    assert res.dataframe["top"].iloc[0] == 700.0


def test_row_cap_truncates(catalog):
    res = run_sql(
        "SELECT * FROM co2_emissions_by_state", catalog=catalog, max_rows=2
    )
    assert res.row_count == 2
    assert res.truncated is True


# --- Adversarial safety ----------------------------------------------------

@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE co2_emissions_by_state",
        "DELETE FROM co2_emissions_by_state",
        "UPDATE co2_emissions_by_state SET TF = 0",
        "INSERT INTO co2_emissions_by_state VALUES (2024,1,'X','X',1,1)",
        "CREATE TABLE evil AS SELECT 1",
        "ALTER TABLE co2_emissions_by_state ADD COLUMN x INT",
        "ATTACH 'evil.db' AS evil",
        "PRAGMA database_list",
        "SET memory_limit='1GB'",
        "INSTALL httpfs",
        "COPY co2_emissions_by_state TO '/tmp/leak.csv'",
        # multiple statements (second one is the payload)
        "SELECT 1; DROP TABLE co2_emissions_by_state",
        # filesystem read functions
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM glob('/etc/*')",
        # comment hiding a second statement
        "SELECT 1 -- comment\n; DROP TABLE co2_emissions_by_state",
    ],
)
def test_dangerous_sql_is_rejected(catalog, bad_sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(bad_sql, catalog=catalog)


def test_unknown_table_is_rejected(catalog):
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT * FROM secret_table", catalog=catalog)


def test_empty_sql_is_rejected(catalog):
    with pytest.raises(UnsafeSQLError):
        validate_sql("   ", catalog=catalog)
