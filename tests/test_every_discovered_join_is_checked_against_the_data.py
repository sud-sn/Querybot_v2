"""
Every join discovery suggests is checked against the warehouse's data.

Discovery proposes joins from names alone. Whether a key matches its dimension,
how often it is empty, whether the dimension repeats a key -- only the data
says, and the admin's "validate all" asked it only when someone pressed the
button. Until then every suggested edge was "untested": the resolver could not
tell a join that matches every row from one that matches none, and a key that
was empty on half the rows was joined INNER, dropping those rows from every
total grouped by that dimension.

Discovery now profiles each suggested join once the graph is populated: the
stored match, orphan and empty-key rates and the verdict feed the resolver's
ranking, and an unconfirmed INNER join whose key is empty or orphaned on some
rows becomes LEFT. A probe that cannot run leaves the edge untested instead of
marking it down; confirmed and manual edges are the admin's.

The warehouse is a SQLite database behind the connector boundary -- never a
real one. Synthetic tables in a mart's naming convention; no customer data.

`store` is imported where it is used, not at the top: modules collected after
this one re-import it, and the code under test resolves it at call time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from core.graph_resolver import find_join_path_with_diagnostics

FACT = "ITM_BAL_DLY_FCT"


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": False, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 10,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


# Every key is declared NOT NULL, so discovery suggests INNER for all of them;
# only the data can say otherwise.
SCHEMA = {
    f"WH.MART.{FACT}": _table(
        f"{FACT}_KEY", (f"{FACT}_KEY", "int"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("ITM_STK_STS_DMS_KEY", "int"), ("PDC_GRP_DMS_KEY", "int"), ("ON_HND_QTY", "decimal"),
    ),
    "WH.MART.ITM_DMS": _table("ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "varchar")),
    "WH.MART.WHS_DMS": _table("WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "varchar")),
    "WH.MART.ITM_STK_STS_DMS": _table(
        "ITM_STK_STS_DMS_KEY", ("ITM_STK_STS_DMS_KEY", "int"), ("ITM_STK_STS_CD", "varchar"),
    ),
    "WH.MART.PDC_GRP_DMS": _table(
        "PDC_GRP_DMS_KEY", ("PDC_GRP_DMS_KEY", "int"), ("PDC_GRP_CD", "varchar"),
    ),
}

# Ten balance rows. Items all match; two warehouses of ten are unknown (20%
# orphans); stock status is empty on half the rows and its dimension repeats
# a key; the product-group keys match no product group at all.
FACT_ROWS = [
    (n, 1 + n % 5, 99 if n >= 8 else 1 + n % 2, None if n % 2 else 1, 70 + n % 2, 10.0 * n)
    for n in range(10)
]


def _warehouse(directory) -> None:
    mart = sqlite3.connect(str(directory / "mart.db"))
    mart.executescript(f"""
        CREATE TABLE {FACT} ({FACT}_KEY INT, ITM_DMS_KEY INT, WHS_DMS_KEY INT,
                             ITM_STK_STS_DMS_KEY INT, PDC_GRP_DMS_KEY INT, ON_HND_QTY REAL);
        CREATE TABLE ITM_DMS (ITM_DMS_KEY INT, ITM_CD TEXT);
        CREATE TABLE WHS_DMS (WHS_DMS_KEY INT, WHS_CD TEXT);
        CREATE TABLE ITM_STK_STS_DMS (ITM_STK_STS_DMS_KEY INT, ITM_STK_STS_CD TEXT);
        CREATE TABLE PDC_GRP_DMS (PDC_GRP_DMS_KEY INT, PDC_GRP_CD TEXT);
        INSERT INTO ITM_DMS VALUES (1,'A'),(2,'B'),(3,'C'),(4,'D'),(5,'E');
        INSERT INTO WHS_DMS VALUES (1,'W1'),(2,'W2');
        INSERT INTO ITM_STK_STS_DMS VALUES (1,'OK'),(1,'OK-DUP');
        INSERT INTO PDC_GRP_DMS VALUES (1,'G1'),(2,'G2');
    """)
    mart.executemany(f"INSERT INTO {FACT} VALUES (?,?,?,?,?,?)", FACT_ROWS)
    mart.commit()
    mart.close()


def _connector(directory):
    """What the Snowflake connector returns, backed by the SQLite warehouse."""
    def connect(*_args, **_kwargs):
        conn = sqlite3.connect(str(directory / "main.db"))
        conn.execute(f"ATTACH DATABASE '{directory / 'mart.db'}' AS MART")
        return conn
    return connect


CREDENTIALS = {"account": "a", "user": "u", "password": "p", "warehouse": "w"}


def _saved_connection(db_id):
    """What get_db_config returns for the account's saved Snowflake connection.

    Read without decrypting. By the time a full-suite run collects this module
    others have repointed QUERYBOT_KEY_FILE and re-imported store, so a Fernet
    round trip can fail inside one test (tests/test_client_sources.py documents
    the same condition). Decryption is not what is tested here, and the
    connector the credentials would reach is the SQLite warehouse anyway.
    """
    return {"id": int(db_id), "db_type": "snowflake", "credentials": dict(CREDENTIALS)}


@contextmanager
def _warehouse_connection(connect):
    """The validator's connection: the saved config as it reads it, and the
    Snowflake connector replaced by `connect` (a function or an exception)."""
    with patch("store.get_db_config", side_effect=_saved_connection), \
            patch("core.schema._sf_connect", side_effect=connect) as connector:
        yield connector


def _write_schema(schema_dir) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")


@pytest.fixture
def warehouse(tmp_path):
    _warehouse(tmp_path)
    return tmp_path


@pytest.fixture
def account(warehouse):
    """A client on a (SQLite-backed) warehouse, its graph discovered."""
    import store
    from core.graph_autopopulate import auto_populate_from_schema

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    # The connection row itself; its credentials are read by _saved_connection.
    with store.get_db() as conn:
        db_id = int(conn.execute(
            "INSERT INTO db_config (name, db_type, credentials_encrypted) VALUES (?,?,?)",
            (f"wh-{account_id}", "snowflake", "read-by-_saved_connection"),
        ).lastrowid)
    store.update_client_meta(account_id, db_config_id=db_id)
    schema_dir = warehouse / "schema"
    _write_schema(schema_dir)
    auto_populate_from_schema(account_id, str(schema_dir))
    return account_id


def _edges(account_id: str) -> dict[str, dict]:
    import store

    return {
        rel["from_column"]: rel
        for rel in store.list_relationships(account_id, active_only=True)
        if rel["from_entity"] == FACT
    }


def _profile(account_id: str, warehouse) -> dict:
    from core.relationship_validator import profile_suggested_relationships

    with _warehouse_connection(_connector(warehouse)):
        return profile_suggested_relationships(account_id)


class TestTheDataDecides:

    def test_suggested_joins_start_untested_and_inner(self, account):
        edges = _edges(account)
        assert {edges[c]["validation_status"] for c in edges} == {"untested"}
        assert {edges[c]["join_type"] for c in edges} == {"INNER"}

    def test_a_key_that_matches_every_row_is_valid_and_stays_inner(self, account, warehouse):
        _profile(account, warehouse)
        item = _edges(account)["ITM_DMS_KEY"]
        assert (item["validation_status"], item["join_type"]) == ("valid", "INNER")
        assert item["match_rate"] == pytest.approx(100.0)

    def test_orphan_keys_make_the_join_left(self, account, warehouse):
        _profile(account, warehouse)
        warehouse_edge = _edges(account)["WHS_DMS_KEY"]
        assert warehouse_edge["orphan_rate"] == pytest.approx(20.0)
        assert (warehouse_edge["join_type"], warehouse_edge["optionality"]) == ("LEFT", "optional")
        assert warehouse_edge["validation_status"] == "warning"

    def test_empty_keys_make_the_join_left_and_a_repeated_key_is_a_warning(
        self, account, warehouse,
    ):
        _profile(account, warehouse)
        status = _edges(account)["ITM_STK_STS_DMS_KEY"]
        assert status["null_fk_rate"] == pytest.approx(50.0)
        assert status["join_type"] == "LEFT"
        assert status["validation_status"] == "warning"
        assert status["join_multiplicity"] == "one_to_many_or_many_to_many"

    def test_a_join_whose_column_has_gone_is_broken(self, account, warehouse):
        """Checked against the discovered schema before the data: a key the
        schema no longer has is broken, and the resolver drops the edge."""
        import store

        schema_dir = warehouse / "schema"
        store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
        shrunk = json.loads(json.dumps(SCHEMA))
        shrunk[f"WH.MART.{FACT}"]["columns"] = [
            c for c in shrunk[f"WH.MART.{FACT}"]["columns"] if c["name"] != "PDC_GRP_DMS_KEY"
        ]
        (schema_dir / "_schema.json").write_text(json.dumps(shrunk), encoding="utf-8")
        _profile(account, warehouse)
        assert _edges(account)["PDC_GRP_DMS_KEY"]["validation_status"] == "broken"

    def test_a_key_that_matches_nothing_is_recorded(self, account, warehouse):
        _profile(account, warehouse)
        assert _edges(account)["PDC_GRP_DMS_KEY"]["join_multiplicity"] == "zero_match"

    def test_the_summary_counts_what_happened(self, account, warehouse):
        summary = _profile(account, warehouse)
        assert summary["valid"] == 1 and summary["warning"] == 3
        assert summary["made_left"] == 2 and summary["not_profiled"] == 0


class TestTheResolverUsesIt:

    def test_a_join_that_matches_nothing_is_no_longer_a_path(self, account, warehouse):
        import store

        target = ["ITM_BAL_DLY_FCT", "PDC_GRP_DMS"]
        path, _ = find_join_path_with_diagnostics(target, store.get_full_graph(account))
        assert [step["from_column"] for step in path] == ["PDC_GRP_DMS_KEY"]
        _profile(account, warehouse)
        path, diagnostics = find_join_path_with_diagnostics(target, store.get_full_graph(account))
        assert path == [] and diagnostics["unreachable"] == ["PDC_GRP_DMS"]


class TestWhatIsNotTouched:

    def test_a_confirmed_join_is_the_admins(self, account, warehouse):
        import store

        rel_id = int(_edges(account)["WHS_DMS_KEY"]["id"])
        with store.get_db() as conn:
            conn.execute("UPDATE entity_relationships SET status='confirmed' WHERE id=?", (rel_id,))
        _profile(account, warehouse)
        warehouse_edge = _edges(account)["WHS_DMS_KEY"]
        assert (warehouse_edge["join_type"], warehouse_edge["validation_status"]) == (
            "INNER", "untested")

    def test_a_probe_that_cannot_run_marks_nothing_down(self, account, caplog):
        """And three of them end the run: a timed-out probe's query is still
        running in the warehouse, and the rest would only pile onto it."""
        from core.relationship_validator import profile_suggested_relationships

        with caplog.at_level(logging.WARNING, logger="querybot.relationship_validator"), \
                _warehouse_connection(ConnectionError("unreachable")) as connect:
            summary = profile_suggested_relationships(account)
        assert connect.call_count == 3 and summary["not_profiled"] == 3
        assert {e["validation_status"] for e in _edges(account).values()} == {"untested"}
        assert "3 probes could not run" in caplog.text

    def test_the_store_never_changes_a_confirmed_join(self, account):
        import store

        rel_id = int(_edges(account)["WHS_DMS_KEY"]["id"])
        with store.get_db() as conn:
            conn.execute("UPDATE entity_relationships SET status='confirmed' WHERE id=?", (rel_id,))
        assert store.set_suggested_join_type(
            account, rel_id, "LEFT", optionality="optional", reason="test",
        ) is False
        assert _edges(account)["WHS_DMS_KEY"]["join_type"] == "INNER"

    def test_one_run_is_bounded_and_says_what_it_left(self, account, warehouse, caplog):
        from core.relationship_validator import profile_suggested_relationships

        with caplog.at_level(logging.WARNING, logger="querybot.relationship_validator"), \
                _warehouse_connection(_connector(warehouse)):
            profile_suggested_relationships(account, limit=1)
        statuses = [e["validation_status"] for e in _edges(account).values()]
        assert len(statuses) - statuses.count("untested") == 1
        assert "profiling the first 1" in caplog.text

    def test_a_validator_that_raises_is_counted_the_same_way(self, account):
        from core import relationship_validator

        with patch.object(relationship_validator, "validate_relationship",
                          side_effect=RuntimeError("store unavailable")) as validate:
            summary = relationship_validator.profile_suggested_relationships(account)
        assert validate.call_count == 3 and summary["not_profiled"] == 3

    def test_a_second_run_asks_only_about_what_is_still_untested(self, account, warehouse):
        _profile(account, warehouse)
        assert _profile(account, warehouse) == {
            "valid": 0, "warning": 0, "broken": 0, "made_left": 0, "not_profiled": 0,
        }


class TestDiscoveryDoesIt:

    def test_the_discovery_route_profiles_the_joins_it_suggested(
        self, account, warehouse, monkeypatch,
    ):
        """From the admin's Discover button to the stored verdicts: the route's
        own background job, with only the warehouse and its reader replaced."""
        from admin import routes

        monkeypatch.chdir(warehouse)

        def discover(creds, db_type, schema_dir, **_kwargs):
            _write_schema(warehouse / schema_dir)
            return len(SCHEMA)

        tasks = routes.BackgroundTasks()
        request = MagicMock()
        request.headers = {"accept": "application/json"}
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "get_db_config", side_effect=_saved_connection), \
                patch("core.schema.discover_and_write", side_effect=discover), \
                _warehouse_connection(_connector(warehouse)), \
                patch.object(routes, "_sync_all_log_exports_bg", return_value=None):
            asyncio.run(routes.admin_discover_schema(request, account, tasks))
            asyncio.run(tasks())
        edges = _edges(account)
        assert edges["ITM_DMS_KEY"]["validation_status"] == "valid"
        assert edges["WHS_DMS_KEY"]["join_type"] == "LEFT"
