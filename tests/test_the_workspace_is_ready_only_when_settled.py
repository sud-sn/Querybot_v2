"""
The workspace says it is ready for questions only when its model is settled.

core/model_readiness.py orders what to model next; nothing said when the
modelling was done. The readiness gate does: READY only when every fact in
the model has a checked time axis, its joins are checked against the data or
confirmed, no business date is ambiguous, and every live metric is certified
-- and otherwise it lists what blocks it, each with what to do and where.

Each test starts where production starts -- discovery over a warehouse, the
semantic model built by the KB build's own builder -- and settles the model
through the admin's own routes and the join profiler, then reads the gate
through the readiness page and its API. The warehouse is a SQLite database
behind the connector boundary, never a real one. Synthetic tables in a mart's
naming convention; no customer data.

`store` is imported where it is used (see
tests/test_every_discovered_join_is_checked_against_the_data.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from core.readiness_gate import CHECKS, check_readiness

FACT = "SLS_TRX_FCT"


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


# A sales fact counted on two dates -- invoiced and shipped -- through one date table.
SCHEMA = {
    f"WH.MART.{FACT}": _table(
        f"{FACT}_KEY", (f"{FACT}_KEY", "int"), ("ITM_DMS_KEY", "int"), ("IVC_DT_DMS_KEY", "int"),
        ("SHP_DT_DMS_KEY", "int"), ("NET_SLS_AMT", "decimal"),
    ),
    "WH.MART.ITM_DMS": _table("ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "varchar")),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("CAL_DT", "date"), ("MTH_NM", "varchar"),
    ),
}


def _warehouse(directory) -> None:
    mart = sqlite3.connect(str(directory / "mart.db"))
    mart.executescript(f"""
        CREATE TABLE {FACT} ({FACT}_KEY INT, ITM_DMS_KEY INT, IVC_DT_DMS_KEY INT,
                             SHP_DT_DMS_KEY INT, NET_SLS_AMT REAL);
        CREATE TABLE ITM_DMS (ITM_DMS_KEY INT, ITM_CD TEXT);
        CREATE TABLE DT_DMS (DT_DMS_KEY INT, CAL_DT TEXT, MTH_NM TEXT);
        INSERT INTO ITM_DMS VALUES (1,'A'),(2,'B'),(3,'C');
        INSERT INTO DT_DMS VALUES (1,'2026-01-05','January'),(2,'2026-02-09','February');
    """)
    mart.executemany(f"INSERT INTO {FACT} VALUES (?,?,?,?,?)", [
        (n, 1 + n % 3, 1 + n % 2, 1 + (n + 1) % 2, 100.0 * n) for n in range(10)
    ])
    mart.commit()
    mart.close()


@contextmanager
def _warehouse_connection(directory):
    """The saved Snowflake connection, its connector replaced by the SQLite warehouse."""
    def connect(*_args, **_kwargs):
        conn = sqlite3.connect(str(directory / "main.db"))
        conn.execute(f"ATTACH DATABASE '{directory / 'mart.db'}' AS MART")
        return conn

    def saved(db_id):
        return {"id": int(db_id), "db_type": "snowflake",
                "credentials": {"account": "a", "user": "u", "password": "p", "warehouse": "w"}}

    with patch("store.get_db_config", side_effect=saved), \
            patch("core.schema._sf_connect", side_effect=connect):
        yield


def _request():
    request = MagicMock()
    request.query_params = {}
    request.headers = {}
    return request


@pytest.fixture
def warehouse(tmp_path):
    _warehouse(tmp_path)
    return tmp_path


@pytest.fixture
def account(warehouse):
    """A client discovered from the warehouse, its semantic model built."""
    import store
    from core.graph_autopopulate import auto_populate_from_schema
    from core.pipeline_context import save_state
    from core.semantic_model import write_semantic_model

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    with store.get_db() as conn:
        db_id = int(conn.execute(
            "INSERT INTO db_config (name, db_type, credentials_encrypted) VALUES (?,?,?)",
            (f"wh-{account_id}", "snowflake", "read-by-the-patched-reader"),
        ).lastrowid)
    store.update_client_meta(account_id, db_config_id=db_id)
    schema_dir = warehouse / "schema"
    schema_dir.mkdir()
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    auto_populate_from_schema(account_id, str(schema_dir))
    kb_dir = warehouse / "kb"
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir), account_id=account_id)
    save_state(account_id, "READY", {"schema_dir": str(schema_dir), "kb_dir": str(kb_dir)})
    return account_id


# ── The admin settling the model, through the product's own paths ──────────


def _admin(route, *args, **form):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval"):
        return asyncio.run(getattr(routes, route)(_request(), *args, **form))


def _profile_joins(account_id, warehouse) -> None:
    from core.relationship_validator import profile_suggested_relationships

    with _warehouse_connection(warehouse):
        profile_suggested_relationships(account_id)


def _approve_date(account_id, column) -> None:
    response = _admin(
        "date_role_approve", account_id, fact_table=f"MART.{FACT}", fact_column=column,
        dimension_table="MART.DT_DMS", dimension_key="DT_DMS_KEY", business_role="", name="",
        date_value_column="CAL_DT", date_key_type="surrogate_fk", synonyms="",
    )
    assert "saved=1" in response.headers["location"], response.headers["location"]


def _make_default(account_id, column) -> None:
    response = _admin("date_role_set_default", account_id, fact_table=f"MART.{FACT}",
                      fact_column=column, make_default="1")
    assert "error" not in response.headers["location"], response.headers["location"]


def _metric(account_id, name="Net Sales", sql="SUM(NET_SLS_AMT)") -> int:
    import store

    return store.save_metric(account_id, {
        "name": name, "formula_type": "expression", "sql_template": sql,
        "base_table": f"MART.{FACT}", "synonyms": "revenue",
    })


def _certify(account_id, metric_id) -> None:
    assert _admin("metric_certify", account_id, metric_id).headers["location"].endswith("saved=certified")


def _joins(account_id) -> dict[str, dict]:
    import store

    return {f"{r['from_column']}>{r['to_entity']}": r for r in store.list_relationships(account_id)}


def _settle(account_id, warehouse) -> int:
    """What an admin does to get to ready: the joins probed, the invoice date
    approved and made the default, a metric certified."""
    _profile_joins(account_id, warehouse)
    _approve_date(account_id, "IVC_DT_DMS_KEY")
    _make_default(account_id, "IVC_DT_DMS_KEY")
    metric_id = _metric(account_id)
    _certify(account_id, metric_id)
    return metric_id


def _problems(verdict, check=None) -> list[str]:
    return [b.problem for b in verdict.blockers if check is None or b.check == check]


class TestFromDiscoveryToReady:

    def test_just_discovered_it_is_not_ready_and_says_why(self, account):
        verdict = check_readiness(account)
        assert not verdict.ready and verdict.facts == (f"MART.{FACT}",)
        assert _problems(verdict, "time_axis") == [f"MART.{FACT} has no approved business date"]
        assert "ITM_DMS_KEY → ITM_DMS.ITM_DMS_KEY has not been checked against the data" in " ".join(
            _problems(verdict, "joins"))
        assert _problems(verdict, "metrics") == [f"MART.{FACT} has no metric"]

    def test_settled_it_is_ready(self, account, warehouse):
        _settle(account, warehouse)
        verdict = check_readiness(account)
        assert verdict.blockers == () and verdict.ready
        assert all(verdict.passed(check) for check in CHECKS)

    def test_a_workspace_with_no_fact_is_not_ready(self):
        import store

        store.init_db()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        verdict = check_readiness(account_id)
        assert not verdict.ready and _problems(verdict, "model") == ["No fact table is in the model"]


class TestTheTimeAxis:

    def test_a_confirmed_date_join_whose_tables_changed(self, account, warehouse):
        import store

        _settle(account, warehouse)
        _admin("graph_confirm_rel", account, _joins(account)["IVC_DT_DMS_KEY>Invoice Date"]["id"])
        assert check_readiness(account).ready
        store.flag_relationships_needing_review(account, {"DT_DMS"})
        verdict = check_readiness(account)
        assert _problems(verdict, "time_axis") == [
            f"The join from MART.{FACT}.IVC_DT_DMS_KEY to the date table changed tables since "
            "it was checked"]
        # Reported once, as the time axis, not again as a join.
        assert not any("IVC_DT_DMS_KEY" in problem for problem in _problems(verdict, "joins"))

    def test_a_date_join_the_admin_confirmed_needs_no_probe(self, account):
        _approve_date(account, "IVC_DT_DMS_KEY")
        _make_default(account, "IVC_DT_DMS_KEY")
        date_join = _joins(account)["IVC_DT_DMS_KEY>Invoice Date"]
        _admin("graph_confirm_rel", account, date_join["id"])
        assert _problems(check_readiness(account), "time_axis") == []

    def test_an_approved_date_with_no_join_to_its_date_table(self, account, warehouse):
        _settle(account, warehouse)
        _admin("graph_reject_rel", account, _joins(account)["IVC_DT_DMS_KEY>Invoice Date"]["id"])
        assert _problems(check_readiness(account), "time_axis") == [
            f"MART.{FACT}.IVC_DT_DMS_KEY has no join to the date table MART.DT_DMS"]


    def test_an_approved_date_that_names_no_date_column_is_no_time_axis(self, account, warehouse):
        from core.semantic_model import MODEL_JSON, load_semantic_model

        _settle(account, warehouse)
        kb_dir = warehouse / "kb"
        model = load_semantic_model(str(kb_dir))
        for role in [*model["date_roles"], *(r for t in model["tables"] for r in t.get("date_roles") or [])]:
            role["date_value_column"] = ""
        (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")
        assert _problems(check_readiness(account), "time_axis") == [
            f"MART.{FACT} has no approved business date"]


class TestTheJoins:

    def test_every_join_probed_against_the_data_passes(self, account, warehouse):
        _settle(account, warehouse)
        assert check_readiness(account).passed("joins")

    @pytest.mark.parametrize("status, multiplicity, problem", [
        ("broken", "", "is broken: a column it joins on is missing"),
        ("warning", "zero_match", "matches nothing: no key on one side is found on the other"),
    ])
    def test_a_join_the_data_refuted(self, account, warehouse, status, multiplicity, problem):
        import store

        _settle(account, warehouse)
        item = _joins(account)["ITM_DMS_KEY>ITM_DMS"]
        store.update_relationship_validation(account, item["id"], status, join_multiplicity=multiplicity)
        assert _problems(check_readiness(account), "joins") == [
            f"SLS_TRX_FCT.ITM_DMS_KEY → ITM_DMS.ITM_DMS_KEY {problem}"]

    def test_a_confirmed_join_whose_tables_changed(self, account, warehouse):
        import store

        _settle(account, warehouse)
        _admin("graph_confirm_rel", account, _joins(account)["ITM_DMS_KEY>ITM_DMS"]["id"])
        store.flag_relationships_needing_review(account, {"ITM_DMS"})
        assert _problems(check_readiness(account), "joins") == [
            "SLS_TRX_FCT.ITM_DMS_KEY → ITM_DMS.ITM_DMS_KEY changed tables since it was checked"]

    def test_a_join_to_a_table_taken_out_of_the_model_is_not_asked_for(self, account, warehouse):
        import store

        _settle(account, warehouse)
        store.update_relationship_validation(
            account, _joins(account)["ITM_DMS_KEY>ITM_DMS"]["id"], "broken")
        _admin("graph_reject_entity", account, "ITM_DMS")
        assert check_readiness(account).ready

    def test_a_join_the_admin_confirmed_or_rejected_blocks_nothing(self, account):
        joins = _joins(account)
        _admin("graph_confirm_rel", account, joins["ITM_DMS_KEY>ITM_DMS"]["id"])
        _admin("graph_reject_rel", account, joins["SHP_DT_DMS_KEY>Shipment Date"]["id"])
        problems = " ".join(_problems(check_readiness(account), "joins"))
        assert "ITM_DMS_KEY" not in problems and "SHP_DT_DMS_KEY" not in problems


class TestTheDates:

    def test_two_approved_dates_and_no_default_are_ambiguous(self, account, warehouse):
        _settle(account, warehouse)
        _approve_date(account, "SHP_DT_DMS_KEY")
        _admin("date_role_set_default", account, fact_table=f"MART.{FACT}",
               fact_column="IVC_DT_DMS_KEY", make_default="0")
        problems = _problems(check_readiness(account), "date_roles")
        assert problems == [
            f"MART.{FACT} has more than one business date and none is its default "
            "(asked of questions naming no metric, Net Sales)"]

    def test_choosing_the_default_settles_it(self, account, warehouse):
        _settle(account, warehouse)
        _approve_date(account, "SHP_DT_DMS_KEY")
        _make_default(account, "SHP_DT_DMS_KEY")
        assert check_readiness(account).ready


class TestTheMetrics:

    def test_a_live_metric_not_certified(self, account, warehouse):
        _settle(account, warehouse)
        _metric(account, "Gross Margin", "SUM(NET_SLS_AMT) * 0.3")
        assert _problems(check_readiness(account), "metrics") == ["Gross Margin is not certified"]

    def test_a_metric_whose_formula_does_not_validate(self, account, warehouse):
        _settle(account, warehouse)
        _metric(account, "Broken", "DELETE FROM MART.SLS_TRX_FCT")
        assert _problems(check_readiness(account), "metrics") == ["Broken's formula does not validate"]

    def test_a_certified_metric_whose_formula_changed(self, account, warehouse):
        import store

        metric_id = _settle(account, warehouse)
        store.update_metric(metric_id, {"sql_template": "SUM(NET_SLS_AMT) * 1.1"}, account_id=account)
        assert _problems(check_readiness(account), "metrics") == ["Net Sales is not certified"]

    def test_a_deprecated_metric_is_not_asked_for(self, account, warehouse):
        import store

        _settle(account, warehouse)
        store.deprecate_metric(_metric(account, "Old Sales", "SUM(NET_SLS_AMT)"), account)
        assert check_readiness(account).ready


class TestThePageAndTheApi:

    def _page(self, account):
        from admin import routes

        class _Url:
            path = f"/admin/clients/{account}/readiness"

        class _FakeRequest:
            url = _Url()
            query_params: dict = {}
            session = {"admin_id": "admin_user_1"}
            scope = {"type": "http"}
            cookies: dict = {}
            headers: dict = {}

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_resp", side_effect=lambda request, name, context: context):
            context = asyncio.run(routes.readiness_page(_request(), account))
        return routes.templates.get_template("client_readiness.html").render(
            request=_FakeRequest(), **context)

    def _api(self, account):
        from admin import routes

        with patch.object(routes, "_is_auth", return_value=True):
            return json.loads(asyncio.run(routes.model_readiness_api(_request(), account)).body)["gate"]

    def test_not_ready_it_says_what_to_settle_and_where(self, account):
        html = self._page(account)
        verdict = check_readiness(account)
        assert f"Not ready: {len(verdict.blockers)} things to settle" in html
        assert f"MART.{FACT} has no approved business date" in html
        assert f'href="/admin/clients/{account}/date-roles"' in html
        assert f'href="/admin/clients/{account}/graph"' in html
        gate = self._api(account)
        assert gate["ready"] is False and gate["checks"]["time_axis"] is False
        assert gate["blockers"][0]["where"] in {"date_roles", "graph", "metrics", "setup"}

    def test_ready_it_says_so(self, account, warehouse):
        _settle(account, warehouse)
        assert "Ready for questions" in self._page(account)
        gate = self._api(account)
        assert gate["ready"] is True and gate["blockers"] == []
        assert gate["checks"] == {check: True for check in CHECKS}

    def test_a_check_that_cannot_run_is_said(self, account):
        with patch("store.list_entities", side_effect=RuntimeError("store down")):
            verdict = check_readiness(account)
        assert not verdict.ready
        assert _problems(verdict) == ["The readiness check could not run"]
