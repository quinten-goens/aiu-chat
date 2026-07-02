"""Tests for the entity builder (aiu_chat.ingest.build_entities).

Pure helpers are tested directly; the full build runs against tiny synthetic
parquet + OurAirports CSVs in a temp dir (no network, no real data), asserting
the two things the layer exists to fix: (1) state-name mismatch across tables
resolves to one canonical id with per-table filter values, and (2) airport name
variants + external aliases collapse to one ICAO entity.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from aiu_chat.ingest import build_entities as be
from aiu_chat.agent.catalog import Catalog, DatasetCatalogEntry


def test_norm_collapses_punctuation_and_case():
    assert be._norm("Paris-Charles-de-Gaulle") == "paris charles de gaulle"
    assert be._norm("Paris - Charles-de-Gaulle") == "paris charles de gaulle"
    assert be._norm("  LONDON  Heathrow ") == "london heathrow"


def test_state_iso_uses_overrides_then_countries():
    name_to_iso = {"UNITED KINGDOM": "GB", "FRANCE": "FR"}
    assert be._state_iso_for("United Kingdom", name_to_iso) == "GB"
    assert be._state_iso_for("CZECHIA", name_to_iso) == "CZ"  # override
    assert be._state_iso_for("Türkiye", name_to_iso) == "TR"  # override
    assert be._state_iso_for("Atlantis", name_to_iso) is None  # unknown


@pytest.fixture
def synthetic(tmp_path: Path):
    """A tiny fake dataset dir + OurAirports snapshot + catalog."""
    pq = tmp_path / "parquet"
    pq.mkdir()

    # State table with UPPERCASE names (like co2_emissions_by_state).
    co2 = pq / "co2.parquet"
    pd.DataFrame({"STATE_NAME": ["UNITED KINGDOM", "FRANCE"],
                  "STATE_CODE": ["EG", "LF"]}).to_parquet(co2)

    # Airport table with Title-Case state names + a name variant for LFPG.
    apt = pq / "apt.parquet"
    pd.DataFrame({
        "APT_ICAO": ["EGLL", "LFPG", "LFPG"],
        "APT_NAME": ["London - Heathrow", "Paris-Charles-de-Gaulle", "Paris - Charles-de-Gaulle"],
        "STATE_NAME": ["United Kingdom", "France", "France"],
    }).to_parquet(apt)

    # OurAirports snapshot.
    oa = tmp_path / "ourairports"
    oa.mkdir()
    (oa / "countries.csv").write_text(
        'id,code,name,continent,wikipedia_link,keywords\n'
        '1,"GB","United Kingdom","EU","",""\n'
        '2,"FR","France","EU","",""\n'
    )
    (oa / "airports.csv").write_text(
        'id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,'
        'iso_country,iso_region,municipality,scheduled_service,icao_code,iata_code,'
        'gps_code,local_code,home_link,wikipedia_link,keywords\n'
        '1,EGLL,large_airport,London Heathrow Airport,51.47,-0.46,83,EU,GB,GB-ENG,'
        'London,yes,EGLL,LHR,EGLL,,,,"LON, Londres"\n'
        '2,LFPG,large_airport,Charles de Gaulle International Airport,49.0,2.55,392,'
        'EU,FR,FR-IDF,Paris,yes,LFPG,CDG,LFPG,,,,"PAR, Roissy"\n'
    )

    catalog = Catalog(datasets=[
        DatasetCatalogEntry("co2_emissions_by_state", "CO2", "", "", str(co2), None, []),
        DatasetCatalogEntry("airport_traffic", "Traffic", "", "", str(apt), None, []),
    ])
    return catalog, oa


def _index(result):
    ents = {e["entity_id"]: e for e in result["entities"]}
    aliases = {}
    for a in result["aliases"]:
        aliases.setdefault(a["entity_id"], set()).add(a["alias"])
    return ents, aliases


def test_state_mismatch_resolves_with_per_table_filter_values(synthetic):
    catalog, oa = synthetic
    result = be.build_entities(catalog=catalog, oa_dir=oa, write_db=False, write_json=False)
    ents, aliases = _index(result)

    gb = ents["state:GB"]
    assert gb["canonical_name"] == "United Kingdom"
    # The smoking gun: different literal per table.
    assert gb["filter_values"]["co2_emissions_by_state"] == "UNITED KINGDOM"
    assert gb["filter_values"]["airport_traffic"] == "United Kingdom"
    # Aliases include ISO, ICAO 2-letter, and both name casings (normalised).
    assert {"gb", "united kingdom", "eg"} <= aliases["state:GB"]


def test_airport_name_variants_and_external_aliases_collapse(synthetic):
    catalog, oa = synthetic
    result = be.build_entities(catalog=catalog, oa_dir=oa, write_db=False, write_json=False)
    ents, aliases = _index(result)

    lfpg = ents["apt:LFPG"]
    assert lfpg["iata"] == "CDG"
    assert lfpg["state_id"] == "state:FR"
    # Both dataset name variants normalise to the same alias.
    assert "paris charles de gaulle" in aliases["apt:LFPG"]
    # External enrichment aliases present.
    assert "cdg" in aliases["apt:LFPG"]
    assert "roissy" in aliases["apt:LFPG"]

    egll = ents["apt:EGLL"]
    assert egll["iata"] == "LHR"
    assert egll["lat"] == 51.47
    assert {"lhr", "london", "londres", "lon"} <= aliases["apt:EGLL"]


def test_writes_duckdb_tables(synthetic, tmp_path):
    catalog, oa = synthetic
    db = tmp_path / "test.duckdb"
    be.build_entities(catalog=catalog, oa_dir=oa, duckdb_path=db, write_db=True,
                      out_dir=tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    try:
        (n_ent,) = con.execute("SELECT COUNT(*) FROM entities").fetchone()
        (n_al,) = con.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()
        (gb_name,) = con.execute(
            "SELECT canonical_name FROM entities WHERE entity_id='state:GB'"
        ).fetchone()
    finally:
        con.close()
    assert n_ent >= 4  # 2 states + 2 airports
    assert n_al > 0
    assert gb_name == "United Kingdom"


def test_missing_ourairports_snapshot_raises(tmp_path):
    catalog = Catalog(datasets=[])
    with pytest.raises(FileNotFoundError, match="OurAirports snapshot"):
        be.build_entities(catalog=catalog, oa_dir=tmp_path / "nope", write_db=False,
                          write_json=False)
