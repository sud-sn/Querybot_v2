"""
tests/test_business_meanings_are_kept_for_review.py

Discovery proposes what a warehouse's codes mean and keeps the proposals for
the admin. After the value index is built -- holding only what its privacy
gates let it keep -- core/business_meaning.py reads the discovered tables and
those values, and the store keeps each proposal with its evidence until the
admin decides. A later discovery refreshes the evidence, drops a suggestion no
longer found, and never touches a decision.

The value index now also holds a dimension's place columns (a city, a province
or state, a country). No naming role marked them, so the values that say a
profit center's CO is a country -- and "stock in Calgary" -- were never read.

The warehouse is a SQLite database behind the query boundary -- never a real
one. Synthetic tables; no customer data. `store` is imported where it is used:
modules collected after this one re-import it (see
tests/test_unknown_members_are_found_at_discovery.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest


def _table(own_key: str, *columns: tuple[str, str], masked: tuple[str, ...] = ()) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": True, "comment": ""} for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 10,
        "comment": "",
        "schema": "MART",
        "database": "",
        "masked_fields": list(masked),
    }


SCHEMA = {
    "MART.PFT_CTR_DMS": _table(
        "PFT_CTR_DMS_KEY", ("PFT_CTR_DMS_KEY", "int"), ("PFT_CTR_NM", "nvarchar"),
        ("PC_ADR_LIN_1", "nvarchar"), ("PC_CTY", "nvarchar"), ("PC_PRV", "nvarchar"),
        ("PC_PSL_CD", "nvarchar"), ("PC_CO", "nvarchar"),
    ),
    "MART.SLR_DMS": _table(
        "SLR_DMS_KEY", ("SLR_DMS_KEY", "int"), ("SLR_CD", "nvarchar"), ("SLR_NM", "nvarchar"),
        ("PYE_CD", "nvarchar"),
    ),
    "MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("QR", "int"),
        ("MTH", "int"), ("WK_OF_YR", "int"),
    ),
    "MART.ACME_RGN_DMS": _table(
        "ACME_RGN_DMS_KEY", ("ACME_RGN_DMS_KEY", "int"), ("ACME_RGN_DSC", "nvarchar"),
    ),
    "MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "int"), ("SLR_DMS_KEY", "int"),
        ("PFT_CTR_DMS_KEY", "int"), ("ON_HND_QTY", "decimal"), ("QXR_IND", "nvarchar"),
        ("SHP_CTY", "nvarchar"),
    ),
}


def _warehouse(directory) -> None:
    mart = sqlite3.connect(str(directory / "mart.db"))
    mart.executescript("""
        CREATE TABLE PFT_CTR_DMS (PFT_CTR_DMS_KEY INT, PFT_CTR_NM TEXT, PC_ADR_LIN_1 TEXT,
                                  PC_CTY TEXT, PC_PRV TEXT, PC_PSL_CD TEXT, PC_CO TEXT);
        INSERT INTO PFT_CTR_DMS VALUES
            (0, 'NULL value provided', NULL, NULL, NULL, NULL, NULL),
            (1, 'Branch 1', '1 King St', 'LONDON', 'ON', 'N6A 4N7', 'CA'),
            (2, 'Branch 2', '2 Main St', 'CALGARY', 'AB', 'T2H 0R3', 'CA'),
            (3, 'Branch 3', '3 Rue Est', 'LAVAL', 'QC', 'H7N 1A1', 'CA');
        CREATE TABLE SLR_DMS (SLR_DMS_KEY INT, SLR_CD TEXT, SLR_NM TEXT, PYE_CD TEXT);
        INSERT INTO SLR_DMS VALUES (1, 'S1', 'Bolt Works', 'S1');
        CREATE TABLE DT_DMS (DT_DMS_KEY INT, DMS_DT TEXT, YR INT, QR INT, MTH INT, WK_OF_YR INT);
        CREATE TABLE ACME_RGN_DMS (ACME_RGN_DMS_KEY INT, ACME_RGN_DSC TEXT);
        INSERT INTO ACME_RGN_DMS VALUES (1, 'EAST');
        CREATE TABLE ITM_BAL_DLY_FCT (ITM_BAL_DLY_FCT_KEY INT, SLR_DMS_KEY INT, PFT_CTR_DMS_KEY INT,
                                      ON_HND_QTY REAL, QXR_IND TEXT, SHP_CTY TEXT);
    """)
    mart.commit()
    mart.close()


def _run_query(directory):
    """core.schema.run_query, answered by the SQLite warehouse: what a
    warehouse returns for the SQL the product sends it."""
    def run(credentials, db_type, sql, max_rows=200):
        conn = sqlite3.connect(":memory:")
        conn.execute(f"ATTACH DATABASE '{directory / 'mart.db'}' AS MART")
        try:
            cursor = conn.execute(sql)
            names = [d[0] for d in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]
        finally:
            conn.close()
    return run


def _write_schema(schema_dir) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")


@pytest.fixture
def warehouse(tmp_path):
    _warehouse(tmp_path)
    return tmp_path


@pytest.fixture
def account(warehouse):
    """A client called Acme Supply, its schema discovered."""
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    with store.get_db() as conn:
        conn.execute("UPDATE client SET client_name=? WHERE account_id=?", ("Acme Supply", account_id))
    # An unregulated tenant: its value index keeps what the masking and PII
    # gates let through. A tenant with no posture is regulated (fail-closed).
    store.save_compliance_profile(account_id, mode="standard")
    schema_dir = warehouse / "schema"
    _write_schema(schema_dir)
    store.update_client_state(account_id, "SCHEMA_READY", {"schema_dir": str(schema_dir)})
    return account_id


def _index(account_id: str, warehouse) -> None:
    from core.value_index import build_value_index

    build_value_index(
        account_id, {}, "snowflake", str(warehouse / "schema"),
        run_query_fn=_run_query(warehouse), base_dir=str(warehouse / "clients"),
    )


def _propose(account_id: str, warehouse) -> dict:
    from core.business_meaning import propose_for_account

    return propose_for_account(account_id, base_dir=str(warehouse / "clients"))


def _kept(account_id: str, **kwargs) -> dict[tuple[str, str], dict]:
    import store

    return {(m["scope"], m["subject"]): m for m in store.list_business_meanings(account_id, **kwargs)}


class TestTheValueIndexHoldsPlaceColumns:

    def test_a_dimensions_city_province_and_country(self):
        from core.value_index import select_filterable_columns

        chosen = {(c["table"], c["column"]) for c in select_filterable_columns(SCHEMA)}
        for column in ("PC_CTY", "PC_PRV", "PC_CO"):
            assert ("PFT_CTR_DMS", column) in chosen
        assert ("ITM_BAL_DLY_FCT", "SHP_CTY") not in chosen

    def test_a_masked_place_column_is_not_indexed(self):
        from core.value_index import select_filterable_columns

        schema = dict(SCHEMA)
        schema["MART.PFT_CTR_DMS"] = dict(SCHEMA["MART.PFT_CTR_DMS"], masked_fields=["PC_CTY"])
        chosen = {(c["table"], c["column"]) for c in select_filterable_columns(schema)}
        assert ("PFT_CTR_DMS", "PC_CTY") not in chosen
        assert ("PFT_CTR_DMS", "PC_PRV") in chosen


class TestDiscoveryKeepsProposals:

    def test_what_the_codes_appear_to_mean_is_kept_with_its_evidence(self, account, warehouse):
        _index(account, warehouse)
        counts = _propose(account, warehouse)
        kept = _kept(account)
        country = kept[("column", "PC_CO")]
        assert (country["reading"], country["rule"], country["status"]) == (
            "profit center country", "values", "suggested")
        assert any("(CA)" in line for line in country["evidence"])
        assert country["found_in"] == ["PFT_CTR_DMS.PC_CO"]
        assert kept[("code", "PRV")]["reading"] == "province"
        # CTY holds place names in the profit center; on the fact nothing reads it.
        assert kept[("column", "PC_CTY")]["reading"] == "profit center city"
        assert (kept[("code", "CTY")]["reading"], kept[("code", "CTY")]["found_in"]) == (
            "", ["ITM_BAL_DLY_FCT.SHP_CTY"])
        assert kept[("code", "SLR")]["reading"] == "supplier"
        assert kept[("code", "SLR")]["synonyms"] == ["vendor", "seller"]
        assert kept[("code", "QR")]["reading"] == "quarter"
        assert (kept[("code", "QXR")]["reading"], kept[("code", "QXR")]["rule"]) == ("", "unread")
        assert counts["proposed"] == len(kept) and counts["unread"] == 2

    def test_the_tenants_own_name_is_not_a_code(self, account, warehouse):
        _index(account, warehouse)
        _propose(account, warehouse)
        assert not [key for key in _kept(account) if key[1] == "ACME"]

    def test_without_values_the_address_only_suggests_a_country(self, account, warehouse):
        _propose(account, warehouse)
        country = _kept(account)[("column", "PC_CO")]
        assert (country["reading"], country["rule"]) == ("profit center country", "address")
        assert country["confidence"] < 70

    def test_a_regulated_tenant_proposes_from_names_alone(self, account, warehouse):
        """Values come only from the value index, and a regulated tenant's index
        keeps nothing an admin has not cleared: no value is read around it."""
        import store

        store.save_compliance_profile(
            account, mode="regulated", industry="banking", policy_pack_key="",
            enforcement_mode="enforce")
        _index(account, warehouse)
        _propose(account, warehouse)
        country = _kept(account)[("column", "PC_CO")]
        assert country["rule"] == "address"
        assert not any("(CA)" in line for line in country["evidence"])

    def test_no_discovered_schema_proposes_nothing(self, account, warehouse):
        import store

        store.update_client_state(account, "SCHEMA_READY", {"schema_dir": str(warehouse / "none")})
        assert _propose(account, warehouse)["proposed"] == 0
        assert _kept(account) == {}


class TestTheAdminsDecisionsStand:

    def test_a_confirmed_reading_is_kept_when_discovery_runs_again(self, account, warehouse):
        import store

        _index(account, warehouse)
        _propose(account, warehouse)
        assert store.decide_business_meaning(
            account, "code", "SLR", "confirmed", reading="supplier", synonyms=["vendor"])
        _propose(account, warehouse)
        slr = _kept(account)[("code", "SLR")]
        assert (slr["status"], slr["decided_reading"], slr["decided_synonyms"]) == (
            "confirmed", "supplier", ["vendor"])
        assert slr["decided_at"]

    def test_an_edited_reading_is_the_one_kept(self, account, warehouse):
        import store

        _propose(account, warehouse)
        store.decide_business_meaning(account, "code", "QR", "confirmed", reading="calendar quarter")
        assert _kept(account)[("code", "QR")]["decided_reading"] == "calendar quarter"

    def test_a_rejected_reading_is_never_proposed_again(self, account, warehouse):
        import store

        _propose(account, warehouse)
        store.decide_business_meaning(account, "code", "QR", "rejected")
        _propose(account, warehouse)
        assert ("code", "QR") not in _kept(account, statuses={"suggested", "confirmed"})
        assert _kept(account)[("code", "QR")]["status"] == "rejected"

    def test_a_code_nothing_reads_is_confirmed_only_with_the_admins_reading(self, account, warehouse):
        import store

        _propose(account, warehouse)
        with pytest.raises(ValueError):
            store.decide_business_meaning(account, "code", "QXR", "confirmed")
        store.decide_business_meaning(account, "code", "QXR", "confirmed", reading="quick reorder")
        assert _kept(account)[("code", "QXR")]["decided_reading"] == "quick reorder"

    def test_a_suggestion_no_longer_found_goes_a_decision_stays(self, account, warehouse):
        import store

        _propose(account, warehouse)
        store.decide_business_meaning(account, "code", "QR", "confirmed")
        schema = {k: v for k, v in SCHEMA.items() if k not in {"MART.DT_DMS", "MART.ITM_BAL_DLY_FCT"}}
        (warehouse / "schema" / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        counts = _propose(account, warehouse)
        kept = _kept(account)
        assert ("code", "QXR") not in kept
        assert kept[("code", "QR")]["status"] == "confirmed"
        assert counts["dropped"] >= 1

    def test_an_unknown_status_or_subject_is_refused(self, account, warehouse):
        import store

        _propose(account, warehouse)
        with pytest.raises(ValueError):
            store.decide_business_meaning(account, "code", "QR", "approved")
        assert store.decide_business_meaning(account, "code", "NOPE", "rejected") is False

    def test_they_leave_with_the_client(self, account, warehouse):
        import store

        _propose(account, warehouse)
        assert _kept(account)
        store.delete_client(account)
        assert _kept(account) == {}


class TestDiscoveryDoesIt:

    def test_the_discovery_route_indexes_the_values_then_proposes(self, account, warehouse, monkeypatch):
        """From the admin's Discover button to the kept proposals: the route's
        own background job, with only the warehouse and its reader replaced."""
        import store
        from admin import routes

        monkeypatch.chdir(warehouse)
        with store.get_db() as conn:
            db_id = int(conn.execute(
                "INSERT INTO db_config (name, db_type, credentials_encrypted) VALUES (?,?,?)",
                (f"wh-{account}", "snowflake", "read-by-the-stub"),
            ).lastrowid)
        store.update_client_meta(account, db_config_id=db_id)

        def discover(creds, db_type, schema_dir, **_kwargs):
            _write_schema(warehouse / schema_dir)
            return len(SCHEMA)

        def saved_connection(db_id):
            return {"id": int(db_id), "db_type": "snowflake", "credentials": {"account": "a"}}

        tasks = routes.BackgroundTasks()
        request = MagicMock()
        request.headers = {"accept": "application/json"}
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "get_db_config", side_effect=saved_connection), \
                patch("store.get_db_config", side_effect=saved_connection), \
                patch("core.schema.discover_and_write", side_effect=discover), \
                patch("core.schema.run_query", side_effect=_run_query(warehouse)), \
                patch("core.relationship_validator.run_probe", side_effect=RuntimeError("no probes here")), \
                patch.object(routes, "_sync_all_log_exports_bg", return_value=None):
            asyncio.run(routes.admin_discover_schema(request, account, tasks))
            asyncio.run(tasks())
        country = _kept(account)[("column", "PC_CO")]
        assert (country["reading"], country["rule"]) == ("profit center country", "values")
        assert _kept(account)[("code", "SLR")]["reading"] == "supplier"
