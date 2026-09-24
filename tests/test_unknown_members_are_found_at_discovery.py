"""
A dimension's "no value" and "no match" members are found at discovery.

A warehouse loads a fact row whose source value is empty, or matches no member
of a dimension, against a reserved member: key 0 "NO_VALUE -- NULL value
provided", key 777 "NO_MATCH -- Value provided does not match", -1 "Unknown".
Nothing in the product knew them, so they were ranked, counted and listed as
if they were members. Discovery now asks each dimension for its rows at the
keys such members use, keeps the ones whose text reads as one, and stores
them with the dimension for the admin to confirm or reject.

The warehouse is a SQLite database behind the connector boundary -- never a
real one. Synthetic tables in a mart's naming convention; no customer data.
`store` is imported where it is used: modules collected after this one
re-import it, and the code under test resolves it at call time.
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

from core.unknown_members import NOT_SPECIFIED, UNKNOWN, UNMATCHED, member_kind

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


SCHEMA = {
    f"WH.MART.{FACT}": _table(
        f"{FACT}_KEY", (f"{FACT}_KEY", "int"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("PTY_DMS_KEY", "int"), ("SLR_DMS_KEY", "nvarchar"), ("DT_DMS_KEY", "int"),
        ("PDC_GRP_DMS_KEY", "numeric(18,0)"), ("BRD_DMS_KEY", "int"), ("SRC_REF", "nvarchar"),
        ("ON_HND_QTY", "decimal"),
    ),
    # The warehouse's own convention, with an audit column that is not a name.
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "nvarchar"), ("ITM_NM", "nvarchar"),
        ("AZ_LST_UPD_USR", "nvarchar"),
    ),
    # A real warehouse sits at key 0; its type is "N/A". 999 is the unknown
    # one, and a size held as text is only digits.
    "WH.MART.WHS_DMS": _table(
        "WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "nvarchar"), ("WHS_DSC", "nvarchar"),
        ("WHS_TYP_CD", "nvarchar"), ("SQ_FT", "nvarchar"),
    ),
    # A French-labelled dimension.
    "WH.MART.PTY_DMS": _table(
        "PTY_DMS_KEY", ("PTY_DMS_KEY", "int"), ("PTY_CD", "nvarchar"), ("PTY_NM", "nvarchar"),
    ),
    # A text key: its members cannot be looked up at numeric keys.
    "WH.MART.SLR_DMS": _table("SLR_DMS_KEY", ("SLR_DMS_KEY", "nvarchar"), ("SLR_NM", "nvarchar")),
    # The date dimension keeps them too; 777's date is a placeholder.
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("DT_DSC", "nvarchar"),
        ("DAY_NM", "nvarchar"),
    ),
    # A decimal key, returned by the driver as 0.0.
    "WH.MART.PDC_GRP_DMS": _table(
        "PDC_GRP_DMS_KEY", ("PDC_GRP_DMS_KEY", "numeric(18,0)"), ("PDC_GRP_CD", "nvarchar"),
    ),
    # A column spelled in mixed case, as a quoted identifier keeps it.
    "WH.MART.BRD_DMS": _table("BRD_DMS_KEY", ("BRD_DMS_KEY", "int"), ("BrandName", "nvarchar")),
}


def _warehouse(directory) -> None:
    mart = sqlite3.connect(str(directory / "mart.db"))
    mart.executescript(f"""
        CREATE TABLE {FACT} ({FACT}_KEY INT, ITM_DMS_KEY INT, WHS_DMS_KEY INT, PTY_DMS_KEY INT,
                             SLR_DMS_KEY TEXT, DT_DMS_KEY INT, PDC_GRP_DMS_KEY REAL,
                             BRD_DMS_KEY INT, SRC_REF TEXT, ON_HND_QTY REAL);
        INSERT INTO {FACT} VALUES (0, 1, 1, 5, 'S1', 20240102, 1, 1, 'NULL value provided', 4);
        CREATE TABLE ITM_DMS (ITM_DMS_KEY INT, ITM_CD TEXT, ITM_NM TEXT, AZ_LST_UPD_USR TEXT);
        INSERT INTO ITM_DMS VALUES
            (0, 'NO_VALUE', 'NULL value provided', 'dbo'),
            (777, 'NO_MATCH', 'Value provided does not match', 'dbo'),
            (1, 'A100', 'Hex bolt', 'dbo'), (2, 'A200', 'Washer', 'dbo');
        CREATE TABLE WHS_DMS (WHS_DMS_KEY INT, WHS_CD TEXT, WHS_DSC TEXT, WHS_TYP_CD TEXT,
                              SQ_FT TEXT);
        INSERT INTO WHS_DMS VALUES
            (0, 'MAIN', 'Main warehouse', 'N/A', '12000'),
            (999, 'UNK', 'Unknown', '', '0'),
            (1, 'NORTH', 'North depot', 'DC', '8000');
        CREATE TABLE PTY_DMS (PTY_DMS_KEY INT, PTY_CD TEXT, PTY_NM TEXT);
        INSERT INTO PTY_DMS VALUES
            (-1, 'N/A', 'Non renseigné'), (-2, 'INCONNU', 'Inconnu'), (5, 'P5', 'Dupont SA');
        CREATE TABLE SLR_DMS (SLR_DMS_KEY TEXT, SLR_NM TEXT);
        INSERT INTO SLR_DMS VALUES ('0', 'NULL value provided'), ('S1', 'Acme');
        CREATE TABLE DT_DMS (DT_DMS_KEY INT, DMS_DT TEXT, DT_DSC TEXT, DAY_NM TEXT);
        INSERT INTO DT_DMS VALUES
            (0, NULL, 'NO_VALUE', 'NULL value provided'),
            (777, '1899-01-01', 'NO_MATCH', 'Value provided not match'),
            (20240102, '2024-01-02', '2 January 2024', 'Tuesday');
        CREATE TABLE PDC_GRP_DMS (PDC_GRP_DMS_KEY REAL, PDC_GRP_CD TEXT);
        INSERT INTO PDC_GRP_DMS VALUES (0.0, 'NO_VALUE'), (777.0, 'NO_MATCH'), (1.0, 'G1');
        CREATE TABLE BRD_DMS (BRD_DMS_KEY INT, BrandName TEXT);
        INSERT INTO BRD_DMS VALUES (-1, 'Unknown'), (1, 'Acme');
    """)
    mart.commit()
    mart.close()


class _Recorded:
    """A connection that notes each statement it runs."""

    def __init__(self, conn, statements: list[str]):
        self._conn, self._statements = conn, statements

    def cursor(self):
        cursor, statements = self._conn.cursor(), self._statements

        class _Cursor:
            def execute(self, sql, *args):
                statements.append(sql)
                return cursor.execute(sql, *args)

            def fetchmany(self, size):
                return cursor.fetchmany(size)

        return _Cursor()

    def close(self):
        self._conn.close()


def _connector(directory, statements: list[str] | None = None):
    """What the Snowflake connector returns, backed by the SQLite warehouse."""
    def connect(*_args, **_kwargs):
        conn = sqlite3.connect(str(directory / "main.db"))
        conn.execute(f"ATTACH DATABASE '{directory / 'mart.db'}' AS MART")
        return conn if statements is None else _Recorded(conn, statements)
    return connect


CREDENTIALS = {"account": "a", "user": "u", "password": "p", "warehouse": "w"}


def _saved_connection(db_id):
    """What get_db_config returns for the account's saved Snowflake connection,
    read without decrypting (tests/test_client_sources.py documents why a
    Fernet round trip can fail mid-suite). The credentials only reach the
    SQLite warehouse."""
    return {"id": int(db_id), "db_type": "snowflake", "credentials": dict(CREDENTIALS)}


@contextmanager
def _warehouse_connection(connect):
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
    """A client on a (SQLite-backed) warehouse, its schema discovered."""
    import store
    from core.graph_autopopulate import auto_populate_from_schema

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    with store.get_db() as conn:
        db_id = int(conn.execute(
            "INSERT INTO db_config (name, db_type, credentials_encrypted) VALUES (?,?,?)",
            (f"wh-{account_id}", "snowflake", "read-by-_saved_connection"),
        ).lastrowid)
    store.update_client_meta(account_id, db_config_id=db_id)
    schema_dir = warehouse / "schema"
    _write_schema(schema_dir)
    store.update_client_state(account_id, "SCHEMA_READY", {"schema_dir": str(schema_dir)})
    auto_populate_from_schema(account_id, str(schema_dir))
    return account_id


def _detect(account_id: str, warehouse) -> dict:
    from core.unknown_members import detect_unknown_members

    with _warehouse_connection(_connector(warehouse)):
        return detect_unknown_members(account_id)


def _found(account_id: str, **kwargs) -> dict[str, dict[str, str]]:
    """{dimension: {key: kind}} as stored."""
    import store

    found: dict[str, dict[str, str]] = {}
    for member in store.list_unknown_members(account_id, **kwargs):
        found.setdefault(member["entity_name"], {})[member["key_value"]] = member["kind"]
    return found


def _member(account_id: str, entity: str, key: str) -> dict:
    import store

    return next(
        m for m in store.list_unknown_members(account_id, include_rejected=True)
        if m["entity_name"] == entity and m["key_value"] == key
    )


def _change_warehouse(warehouse, sql: str) -> None:
    mart = sqlite3.connect(str(warehouse / "mart.db"))
    mart.execute(sql)
    mart.commit()
    mart.close()


class TestWhatReadsAsAnUnknownMember:

    @pytest.mark.parametrize("texts, kind", [
        (["NO_VALUE", "NULL value provided"], NOT_SPECIFIED),
        (["NO_MATCH", "Value provided does not match"], UNMATCHED),
        (["NO_MATCH", "Value provided not match"], UNMATCHED),
        (["UNK", "Unknown", ""], UNKNOWN),
        (["N/A", "Non renseigné"], NOT_SPECIFIED),
        (["INCONNU", "Inconnue"], UNKNOWN),
        # The most specific reading wins.
        (["NO_MATCH", "Unknown"], UNMATCHED),
        (["N/A", "Unknown"], NOT_SPECIFIED),
        # Nothing said is not a member; one real value makes a real member.
        (["", None, "0", "-"], None),
        (["MAIN", "Main warehouse", "N/A"], None),
        (["NORTH", "North depot"], None),
    ])
    def test_every_value_that_says_something_must_read_as_one(self, texts, kind):
        assert member_kind(texts) == kind


class TestDiscoveryFindsThem:

    def test_each_dimensions_members_are_stored_by_kind(self, account, warehouse):
        _detect(account, warehouse)
        found = _found(account)
        assert found["ITM_DMS"] == {"0": NOT_SPECIFIED, "777": UNMATCHED}
        assert found["DT_DMS"] == {"0": NOT_SPECIFIED, "777": UNMATCHED}
        assert found["PTY_DMS"] == {"-1": NOT_SPECIFIED, "-2": UNKNOWN}
        assert found["BRD_DMS"] == {"-1": UNKNOWN}

    def test_a_real_member_at_a_reserved_key_is_left_alone(self, account, warehouse):
        _detect(account, warehouse)
        assert _found(account)["WHS_DMS"] == {"999": UNKNOWN}

    def test_a_decimal_key_is_stored_as_it_is_written(self, account, warehouse):
        _detect(account, warehouse)
        assert _found(account)["PDC_GRP_DMS"] == {"0": NOT_SPECIFIED, "777": UNMATCHED}

    def test_a_text_key_is_not_probed_and_a_fact_is_not_a_dimension(self, account, warehouse):
        summary = _detect(account, warehouse)
        found = _found(account)
        assert "SLR_DMS" not in found and FACT not in found
        assert summary == {"dimensions": 6, "members": 10, "not_probed": 0}

    def test_a_table_two_entities_share_is_asked_once_and_both_know(self, account, warehouse):
        """The graph names the date dimension twice -- as itself and as the
        canonical date entity."""
        import store
        from core.unknown_members import detect_unknown_members

        with _warehouse_connection(_connector(warehouse)) as connect:
            detect_unknown_members(account)
        assert connect.call_count == 6
        names = [e["entity_name"] for e in store.list_entities(account, active_only=True)
                 if e["table_name"] == "DT_DMS"]
        found = _found(account)
        assert len(names) == 2
        assert all(found[name] == {"0": NOT_SPECIFIED, "777": UNMATCHED} for name in names)

    def test_the_probe_names_columns_as_discovered(self, account, warehouse):
        """A quoted identifier is case-sensitive in Snowflake and Oracle."""
        from core.unknown_members import detect_unknown_members

        statements: list[str] = []
        with _warehouse_connection(_connector(warehouse, statements)):
            detect_unknown_members(account)
        brand = next(sql for sql in statements if "BRD_DMS" in sql)
        assert '"BrandName"' in brand and '"BRD_DMS_KEY" IN (0, -1' in brand

    def test_the_stored_member_keeps_its_names_not_its_audit_columns(self, account, warehouse):
        _detect(account, warehouse)
        item = _member(account, "ITM_DMS", "0")
        assert item["key_column"] == "ITM_DMS_KEY"
        assert item["member_text"] == {"ITM_CD": "NO_VALUE", "ITM_NM": "NULL value provided"}
        assert item["status"] == "suggested"


class TestTheAdminDecides:

    def test_a_rejected_member_is_not_brought_back(self, account, warehouse):
        import store

        _detect(account, warehouse)
        assert store.set_unknown_member_status(account, "ITM_DMS", "777", "rejected") is True
        _detect(account, warehouse)
        assert _found(account)["ITM_DMS"] == {"0": NOT_SPECIFIED}
        assert _found(account, include_rejected=True)["ITM_DMS"]["777"] == UNMATCHED

    def test_a_confirmed_member_stays_confirmed_and_its_text_follows_the_warehouse(
        self, account, warehouse,
    ):
        import store

        _detect(account, warehouse)
        store.set_unknown_member_status(account, "WHS_DMS", "999", "confirmed")
        _change_warehouse(warehouse, "UPDATE WHS_DMS SET WHS_DSC='UNDEFINED' WHERE WHS_DMS_KEY=999")
        _detect(account, warehouse)
        member = _member(account, "WHS_DMS", "999")
        assert member["status"] == "confirmed"
        assert member["member_text"]["WHS_DSC"] == "UNDEFINED"

    def test_a_confirmed_member_outlives_its_row_a_suggestion_does_not(self, account, warehouse):
        import store

        _detect(account, warehouse)
        store.set_unknown_member_status(account, "PTY_DMS", "-1", "confirmed")
        _change_warehouse(warehouse, "DELETE FROM PTY_DMS WHERE PTY_DMS_KEY IN (-1, -2)")
        _detect(account, warehouse)
        assert _found(account)["PTY_DMS"] == {"-1": NOT_SPECIFIED}

    def test_a_suggestion_the_dimension_no_longer_has_is_dropped(self, account, warehouse):
        _detect(account, warehouse)
        _change_warehouse(warehouse, "DELETE FROM ITM_DMS WHERE ITM_DMS_KEY = 777")
        _detect(account, warehouse)
        assert _found(account)["ITM_DMS"] == {"0": NOT_SPECIFIED}

    def test_a_dimension_that_leaves_the_graph_takes_its_members_with_it(self, account, warehouse):
        import store

        _detect(account, warehouse)
        store.delete_entity(account, "ITM_DMS")
        assert "ITM_DMS" not in _found(account)
        kept = [f"MART.{t.split('.')[-1]}" for t in SCHEMA if not t.endswith(".WHS_DMS")]
        store.prune_entity_graph_to_tables(account, kept)
        assert "WHS_DMS" not in _found(account) and "PTY_DMS" in _found(account)
        store.delete_client(account)
        assert store.list_unknown_members(account) == []


class TestWhenTheWarehouseCannotAnswer:

    def test_failed_probes_store_nothing_and_three_end_the_pass(self, account, caplog):
        from core.unknown_members import detect_unknown_members

        with caplog.at_level(logging.WARNING, logger="querybot.unknown_members"), \
                _warehouse_connection(ConnectionError("unreachable")) as connect:
            summary = detect_unknown_members(account)
        assert connect.call_count == 3 and summary["not_probed"] == 3
        assert _found(account) == {}
        assert "stopped after 3 probes could not run" in caplog.text

    def test_no_connection_means_no_probe(self, account, caplog):
        import store
        from core.unknown_members import detect_unknown_members

        with store.get_db() as conn:
            conn.execute("UPDATE client SET db_config_id=NULL WHERE account_id=?", (account,))
        with caplog.at_level(logging.WARNING, logger="querybot.unknown_members"), \
                patch("core.schema._sf_connect") as connect:
            summary = detect_unknown_members(account)
        assert connect.call_count == 0 and summary["dimensions"] == 0
        assert "no warehouse connection" in caplog.text


class TestDiscoveryDoesIt:

    def test_the_discovery_route_finds_them(self, account, warehouse, monkeypatch):
        """From the admin's Discover button to the stored members: the route's
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
        assert _found(account)["ITM_DMS"] == {"0": NOT_SPECIFIED, "777": UNMATCHED}
