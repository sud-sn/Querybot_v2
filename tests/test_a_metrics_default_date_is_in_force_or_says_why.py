"""
A metric's default date is in force when the form says "Saved", or the page says why not.

The metric form offers a Default time column. The date resolver reads that
setting only through an approved, complete date role on the metric's table, so
choosing a date whose role nobody had approved -- the usual state after a
knowledge-base build, where every discovered date is "generated" -- did
nothing. The page said "Saved successfully." and period questions went on
using another date, or asking the reader which one.

And the resolver looked for the role on every fact the question touched, not on
the metric's own table: a same-named key approved on another fact became the
metric's date.

Now saving the metric approves its date role in the same action when the role is
complete, and says so; otherwise nothing is approved and the page says why --
no role on that column, a key with no calendar mapped, a role someone
rejected, or a column that is a date on several tables when the metric names
no table. A deactivated metric approves nothing. The resolver takes the
metric's date from the metric's own table only.

The real admin routes, contract compile and date resolver, on a scratch store
(QUERYBOT_DB_PATH) and a synthetic model built by the real model writer.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from core.contextual_dates import _metric_default_time_role_bindings, resolve_contextual_date_binding
from core.semantic_contract import load_contract
from core.semantic_model import load_semantic_model, patch_date_role, write_semantic_model

SALES, RETURNS = "SALES.SALES_FACT", "SALES.RETURNS_FACT"
SCHEMA = {
    "SYNDB.SALES.SALES_FACT": {"database": "SYNDB", "schema": "SALES", "table": "SALES_FACT", "columns": [
        {"name": "SALES_FACT_KEY", "type": "bigint"}, {"name": "ORD_DT_KEY", "type": "bigint"},
        {"name": "SHP_DT", "type": "date"}, {"name": "NET_SLS_AMT", "type": "decimal"}]},
    "SYNDB.SALES.RETURNS_FACT": {"database": "SYNDB", "schema": "SALES", "table": "RETURNS_FACT", "columns": [
        {"name": "RETURNS_FACT_KEY", "type": "bigint"}, {"name": "ORD_DT_KEY", "type": "bigint"},
        {"name": "RTN_AMT", "type": "decimal"}]},
    "SYNDB.SALES.DT_DMS": {"database": "SYNDB", "schema": "SALES", "table": "DT_DMS", "columns": [
        {"name": "DT_DMS_KEY", "type": "bigint"}, {"name": "CAL_DT", "type": "date"},
        {"name": "YEAR", "type": "int"}, {"name": "MONTH", "type": "int"}]},
}
FORM_FIELDS = ("synonyms", "description", "result_format", "required_columns", "allowed_dimensions",
               "metric_builder_config", "example_questions", "grain", "category", "base_entity")


@pytest.fixture
def account(tmp_path):
    import store
    from core.pipeline_context import save_state

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    (tmp_path / "schema").mkdir()
    (tmp_path / "schema" / "_schema.json").write_text(json.dumps(SCHEMA))
    kb_dir = str(tmp_path / "kb")
    write_semantic_model(schema_dir=str(tmp_path / "schema"), kb_dir=kb_dir)
    save_state(account_id, "READY", {"kb_dir": kb_dir})
    return account_id, kb_dir


def _status(kb_dir, table, column):
    return next(role["status"] for role in load_semantic_model(kb_dir)["date_roles"]
                if role["fact_table"] == table and role["fact_column"] == column)


def _admin(call):
    """An admin form post; the evaluation run an approval starts is not under test."""
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_run_default_evals_async", new=MagicMock()), \
            patch("core.background_tasks.spawn"):
        response = asyncio.run(call(routes))
    assert response.status_code == 303
    return {key: values[0] for key, values in parse_qs(urlsplit(response.headers["location"]).query).items()}


def _create(account_id, default_time_column, base_table=SALES, name="Net Sales"):
    return _admin(lambda routes: routes.metric_create(
        MagicMock(), account_id, name=name, sql_template="SUM(NET_SLS_AMT)", formula_type="expression",
        default_time_column=default_time_column, base_table=base_table,
        **{field: "" for field in FORM_FIELDS}))


def _update(account_id, metric_id, default_time_column, is_active="1"):
    return _admin(lambda routes: routes.metric_update(
        MagicMock(), account_id, metric_id, name="Net Sales", sql_template="SUM(NET_SLS_AMT)",
        formula_type="expression", default_time_column=default_time_column, base_table=SALES,
        is_active=is_active, **{field: "" for field in FORM_FIELDS}))


def _stored_metric(account_id):
    import store

    return next(m for m in store.list_metrics(account_id, active_only=False) if m["name"] == "Net Sales")


def _runtime_date(account_id, kb_dir, question="net sales by month", facts=(SALES,)):
    """What a period question on the metric uses: the date roles as the runtime
    reads them -- the compiled contract -- and the metric as stored."""
    roles = (load_contract(kb_dir).get("model") or {}).get("date_roles") or []
    return resolve_contextual_date_binding(
        question, matched_metrics=[_stored_metric(account_id)], bindings=[],
        date_roles=list(roles), required_fact_tables=set(facts))


class TestSavingTheMetricPutsItsDateInForce:

    def test_a_complete_date_role_is_approved_with_the_save_and_period_questions_use_it(self, account):
        account_id, kb_dir = account
        assert _status(kb_dir, SALES, "ORD_DT_KEY") == "generated"
        query = _create(account_id, "ORD_DT_KEY")
        assert _status(kb_dir, SALES, "ORD_DT_KEY") == "approved"
        assert "Order Date" in query["notice"] and "ORD_DT_KEY" in query["notice"]
        assert "date_warning" not in query
        resolution = _runtime_date(account_id, kb_dir)
        assert resolution["status"] == "selected"
        assert resolution["binding"]["fact_column"] == "ORD_DT_KEY"
        assert resolution["binding"]["resolution_source"] == "metric_default_time_column"

    def test_the_edit_form_does_the_same(self, account):
        account_id, kb_dir = account
        _create(account_id, "")
        query = _update(account_id, _stored_metric(account_id)["id"], "SHP_DT")
        assert _status(kb_dir, SALES, "SHP_DT") == "approved"
        assert "Shipment Date" in query["notice"]
        assert _runtime_date(account_id, kb_dir)["binding"]["fact_column"] == "SHP_DT"

    def test_a_date_already_in_force_is_saved_without_a_note(self, account):
        account_id, kb_dir = account
        patch_date_role(kb_dir=kb_dir, fact_table=SALES, fact_column="ORD_DT_KEY", status="approved")
        query = _create(account_id, "ORD_DT_KEY")
        assert query["saved"] == "1" and "notice" not in query and "date_warning" not in query

    def test_the_other_questions_on_the_table_are_not_approved_with_it(self, account):
        account_id, kb_dir = account
        _create(account_id, "ORD_DT_KEY")
        assert _status(kb_dir, SALES, "SHP_DT") == "generated"
        assert _status(kb_dir, RETURNS, "ORD_DT_KEY") == "generated"


class TestWhenItCannotBeTheSaveSaysWhy:

    def test_a_key_with_no_calendar_mapped_is_not_in_force(self, account):
        account_id, kb_dir = account
        # Retyped as a key on the Date Roles page, with no calendar table given.
        patch_date_role(kb_dir=kb_dir, fact_table=SALES, fact_column="SHP_DT",
                        date_key_type="surrogate_fk", status="generated")
        query = _create(account_id, "SHP_DT")
        assert _status(kb_dir, SALES, "SHP_DT") == "generated"
        assert "not in force" in query["date_warning"] and "no calendar table" in query["date_warning"]
        assert "notice" not in query

    def test_a_rejected_date_role_stays_rejected(self, account):
        account_id, kb_dir = account
        patch_date_role(kb_dir=kb_dir, fact_table=SALES, fact_column="ORD_DT_KEY", status="rejected")
        query = _create(account_id, "ORD_DT_KEY")
        assert _status(kb_dir, SALES, "ORD_DT_KEY") == "rejected"
        assert "was rejected" in query["date_warning"]

    def test_a_column_that_is_not_a_date_on_the_table(self, account):
        account_id, _kb_dir = account
        query = _create(account_id, "NET_SLS_AMT")
        assert query["date_warning"] == (
            "Saved, but the default date is not in force: NET_SLS_AMT has no date role on "
            "SALES.SALES_FACT. Add it on the Date Roles page.")

    def test_with_no_table_named_a_date_on_several_tables_is_not_guessed(self, account):
        account_id, kb_dir = account
        query = _create(account_id, "ORD_DT_KEY", base_table="")
        assert "several tables" in query["date_warning"]
        assert _status(kb_dir, SALES, "ORD_DT_KEY") == _status(kb_dir, RETURNS, "ORD_DT_KEY") == "generated"

    def test_a_deactivated_metric_approves_nothing(self, account):
        account_id, kb_dir = account
        _create(account_id, "")
        query = _update(account_id, _stored_metric(account_id)["id"], "ORD_DT_KEY", is_active="0")
        assert _status(kb_dir, SALES, "ORD_DT_KEY") == "generated"
        assert "notice" not in query and "date_warning" not in query

    def test_the_metrics_page_shows_the_note_and_the_way_to_the_date_roles(self, account):
        from admin import routes

        account_id, _kb_dir = account
        warning = "Saved, but the default date is not in force: NET_SLS_AMT has no date role."
        request = MagicMock()
        request.query_params = {"saved": "1", "date_warning": warning, "notice": "Approved <b>x</b>"}
        with patch.object(routes, "_is_auth", return_value=True):
            page = asyncio.run(routes.metrics_page(request, account_id)).body.decode()
        assert warning in page
        assert f'href="/admin/clients/{account_id}/date-roles"' in page
        assert "Approved &lt;b&gt;x&lt;/b&gt;" in page
        assert "Saved successfully." not in page  # the warning already says it was saved


class TestTheMetricsDateIsOnItsOwnTable:

    SALES_METRIC = {"name": "Net Sales", "base_table": SALES, "default_time_column": "ORD_DT_KEY"}

    def test_a_same_named_key_approved_on_another_fact_is_not_this_metrics_date(self, account):
        _account_id, kb_dir = account
        patch_date_role(kb_dir=kb_dir, fact_table=RETURNS, fact_column="ORD_DT_KEY", status="approved")
        roles = load_semantic_model(kb_dir)["date_roles"]
        assert _metric_default_time_role_bindings(
            matched_metrics=[self.SALES_METRIC], date_roles=roles, fact_scope={SALES, RETURNS}) == []
        resolution = resolve_contextual_date_binding(
            "net sales and returns by month", matched_metrics=[self.SALES_METRIC], bindings=[],
            date_roles=roles, required_fact_tables={SALES, RETURNS})
        assert (resolution.get("binding") or {}).get("resolution_source") != "metric_default_time_column"

    def test_its_own_approved_date_is(self, account):
        _account_id, kb_dir = account
        for table in (SALES, RETURNS):
            patch_date_role(kb_dir=kb_dir, fact_table=table, fact_column="ORD_DT_KEY", status="approved")
        bindings = _metric_default_time_role_bindings(
            matched_metrics=[self.SALES_METRIC], date_roles=load_semantic_model(kb_dir)["date_roles"],
            fact_scope={SALES, RETURNS})
        assert [binding["fact_table"] for binding in bindings] == [SALES]

    def test_a_metric_naming_no_table_finds_its_date_on_the_questions_facts(self, account):
        _account_id, kb_dir = account
        for table in (SALES, RETURNS):
            patch_date_role(kb_dir=kb_dir, fact_table=table, fact_column="ORD_DT_KEY", status="approved")
        bindings = _metric_default_time_role_bindings(
            matched_metrics=[{"name": "Net Sales", "default_time_column": "ORD_DT_KEY"}],
            date_roles=load_semantic_model(kb_dir)["date_roles"], fact_scope={SALES})
        assert [binding["fact_table"] for binding in bindings] == [SALES]
