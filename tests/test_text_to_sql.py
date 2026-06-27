"""Tests for the text-to-SQL pipeline, with a fake LLM client (no Ollama needed).

The fake client returns scripted responses, so we exercise the orchestration:
SQL extraction, safety rejection, CANNOT_ANSWER, and grounded narration over the
real synthetic Parquet fixture.
"""
from __future__ import annotations

import duckdb
import pytest

from aiu_chat.agent.catalog import Catalog, DatasetCatalogEntry
from aiu_chat.agent.text_to_sql import answer_data_question


@pytest.fixture()
def catalog(tmp_path):
    parquet = tmp_path / "co2_emissions_by_state.parquet"
    con = duckdb.connect()
    try:
        safe = str(parquet).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT * FROM (VALUES
                    (2024, 'FRANCE', 100.0),
                    (2024, 'GERMANY', 300.0)
                ) AS t(YEAR, STATE_NAME, CO2_QTY_TONNES)
            ) TO '{safe}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return Catalog(
        datasets=[
            DatasetCatalogEntry(
                table="co2_emissions_by_state", title="CO2", description="t",
                granularity="", parquet_path=str(parquet), as_of="2024-12-01",
                columns=[
                    {"name": "YEAR", "type": "BIGINT"},
                    {"name": "STATE_NAME", "type": "VARCHAR"},
                    {"name": "CO2_QTY_TONNES", "type": "DOUBLE"},
                ],
            )
        ]
    )


class FakeClient:
    """Returns queued responses for successive chat() calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.0, json_mode=False):
        self.calls.append(messages)
        return self._responses.pop(0)


def test_happy_path_executes_and_narrates(catalog):
    client = FakeClient([
        # 1st call: SQL generation (wrapped in a markdown fence to test cleaning)
        "```sql\nSELECT SUM(CO2_QTY_TONNES) AS total FROM co2_emissions_by_state\n```",
        # 2nd call: narration
        "The total CO2 is 400 tonnes.",
    ])
    ans = answer_data_question("total co2?", client=client, catalog=catalog)
    assert ans.ok is True
    assert ans.result.dataframe["total"].iloc[0] == 400.0
    assert "400" in ans.answer
    assert len(client.calls) == 2  # SQL + narration


def test_cannot_answer_short_circuits(catalog):
    client = FakeClient(["SELECT 'CANNOT_ANSWER' AS note"])
    ans = answer_data_question("what is the weather?", client=client, catalog=catalog)
    assert ans.ok is False
    assert ans.result is None
    assert len(client.calls) == 1  # no narration call


def test_unsafe_sql_is_rejected_before_execution(catalog):
    client = FakeClient(["DROP TABLE co2_emissions_by_state"])
    ans = answer_data_question("drop it", client=client, catalog=catalog)
    assert ans.ok is False
    assert "safety check" in ans.answer.lower()
    assert len(client.calls) == 1  # rejected before narration


def test_bad_sql_execution_surfaces_error(catalog):
    client = FakeClient(["SELECT nonexistent_col FROM co2_emissions_by_state"])
    ans = answer_data_question("bad", client=client, catalog=catalog)
    assert ans.ok is False
    assert "failed to execute" in ans.answer.lower()
