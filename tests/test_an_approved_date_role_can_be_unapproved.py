"""
An approved date role can be unapproved.

A date role approved by mistake -- a customer's first invoice date approved as
"Invoice Date" -- stayed approved: the page offered Edit, which saves the
mapping and keeps the approval, and a rebuild keeps every approved role. The
only way back was editing the knowledge base by hand.

Unapprove puts the role back in review: questions that name it no longer
require it, it is no longer a default, a metric's default time column no longer
reads it, and a rebuild does not bring the approval back. The page says what
still names the date.

The real model writer, admin routes and compiled contract on a synthetic schema.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from core.contextual_dates import resolve_contextual_date_binding
from core.semantic_contract import load_contract
from core.semantic_model import (
    find_default_date_roles,
    load_semantic_model,
    patch_date_role,
    set_default_date_role,
    unapprove_date_role,
    write_semantic_model,
)

SALES, RETURNS = "SALES.SALES_FACT", "SALES.RETURNS_FACT"
SCHEMA = {
    "SYNDB.SALES.SALES_FACT": {"database": "SYNDB", "schema": "SALES", "table": "SALES_FACT", "columns": [
        {"name": "SALES_FACT_KEY", "type": "bigint"}, {"name": "ORD_DT_KEY", "type": "bigint"},
        {"name": "INV_DT_KEY", "type": "bigint"}, {"name": "NET_SLS_AMT", "type": "decimal"}]},
    "SYNDB.SALES.RETURNS_FACT": {"database": "SYNDB", "schema": "SALES", "table": "RETURNS_FACT", "columns": [
        {"name": "RETURNS_FACT_KEY", "type": "bigint"}, {"name": "ORD_DT_KEY", "type": "bigint"},
        {"name": "RTN_AMT", "type": "decimal"}]},
    "SYNDB.SALES.DT_DMS": {"database": "SYNDB", "schema": "SALES", "table": "DT_DMS", "columns": [
        {"name": "DT_DMS_KEY", "type": "bigint"}, {"name": "CAL_DT", "type": "date"},
        {"name": "YEAR", "type": "int"}, {"name": "MONTH", "type": "int"}]},
}
METRIC_FORM = ("synonyms", "description", "result_format", "required_columns", "allowed_dimensions",
               "metric_builder_config", "example_questions", "grain", "category", "base_entity")


@pytest.fixture
def model_dirs(tmp_path):
    schema_dir, kb_dir = tmp_path / "schema", tmp_path / "kb"
    schema_dir.mkdir()
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA))
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
    return str(schema_dir), str(kb_dir)


@pytest.fixture
def account(model_dirs):
    import store
    from core.pipeline_context import save_state

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    save_state(account_id, "READY", {"kb_dir": model_dirs[1]})
    return account_id, model_dirs[1]


def _approve(kb_dir, column, table=SALES):
    assert patch_date_role(kb_dir=kb_dir, fact_table=table, fact_column=column, dimension_table="SALES.DT_DMS",
                           dimension_key="DT_DMS_KEY", business_role=column.lower(), date_value_column="CAL_DT",
                           status="approved")


def _entries(kb_dir, table, column):
    """The role as the model keeps it twice: in the top-level list and under its table."""
    model = load_semantic_model(kb_dir)
    top = [r for r in model["date_roles"] if r["fact_table"] == table and r["fact_column"] == column]
    own = [r for t in model["tables"] if t.get("qualified_name") == table
           for r in t.get("date_roles") or [] if r["fact_column"] == column]
    assert len(top) == 1 and len(own) == 1
    return top[0], own[0]


def _statuses(kb_dir, table, column):
    return {entry["status"] for entry in _entries(kb_dir, table, column)}


def _defaults(kb_dir):
    return [role["fact_column"] for role in find_default_date_roles(model=load_semantic_model(kb_dir))]


def _admin(call):
    """An admin form post; the evaluation run an approval starts is not under test."""
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_run_default_evals_async", new=MagicMock()), \
            patch("core.background_tasks.spawn"):
        response = asyncio.run(call(routes))
    assert response.status_code == 303
    return {key: values[0] for key, values in parse_qs(urlsplit(response.headers["location"]).query).items()}


def _unapprove(account_id, column, table=SALES):
    return _admin(lambda routes: routes.date_role_unapprove(
        MagicMock(), account_id, fact_table=table, fact_column=column))


def _create_metric(account_id, default_time_column):
    return _admin(lambda routes: routes.metric_create(
        MagicMock(), account_id, name="Net Sales", sql_template="SUM(NET_SLS_AMT)", formula_type="expression",
        default_time_column=default_time_column, base_table=SALES, **{field: "" for field in METRIC_FORM}))


def _runtime_date(account_id, kb_dir):
    """What "net sales by month" uses: the compiled contract's date roles and the metric as stored."""
    import store

    metric = next(m for m in store.list_metrics(account_id) if m["name"] == "Net Sales")
    roles = (load_contract(kb_dir).get("model") or {}).get("date_roles") or []
    return resolve_contextual_date_binding(
        "net sales by month", matched_metrics=[metric], bindings=[],
        date_roles=list(roles), required_fact_tables={SALES})


def _page(account_id, **query):
    from admin import routes

    request = MagicMock()
    request.query_params = query
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(routes.date_roles_page(request, account_id)).body.decode()


def _unapprove_form(account_id, table, column):
    return (f'action="/admin/clients/{account_id}/date-roles/unapprove"', f'name="fact_table"  value="{table}"',
            f'name="fact_column" value="{column}"')


class TestTheModel:

    def test_the_role_goes_back_to_review_in_both_places_the_model_keeps_it(self, model_dirs):
        _schema_dir, kb_dir = model_dirs
        _approve(kb_dir, "INV_DT_KEY")
        assert _statuses(kb_dir, SALES, "INV_DT_KEY") == {"approved"}
        assert unapprove_date_role(kb_dir, SALES, "INV_DT_KEY")
        assert _statuses(kb_dir, SALES, "INV_DT_KEY") == {"needs_review"}
        top, own = _entries(kb_dir, SALES, "INV_DT_KEY")
        # The mapping stays as the admin left it, so approving again is one click.
        assert (top["dimension_table"], top["date_value_column"]) == ("SALES.DT_DMS", "CAL_DT")
        assert own["business_role"] == "inv_dt_key"

    def test_a_default_goes_with_the_approval(self, model_dirs):
        _schema_dir, kb_dir = model_dirs
        _approve(kb_dir, "ORD_DT_KEY")
        _approve(kb_dir, "INV_DT_KEY")
        assert set_default_date_role(kb_dir, SALES, "INV_DT_KEY")
        assert _defaults(kb_dir) == ["INV_DT_KEY"]
        unapprove_date_role(kb_dir, SALES, "INV_DT_KEY")
        assert not any(entry.get("is_default") for entry in _entries(kb_dir, SALES, "INV_DT_KEY"))
        # The one approved role left is the table's date by itself, as it always is.
        assert _defaults(kb_dir) == ["ORD_DT_KEY"]

    def test_the_same_column_on_another_table_keeps_its_approval(self, model_dirs):
        _schema_dir, kb_dir = model_dirs
        _approve(kb_dir, "ORD_DT_KEY")
        _approve(kb_dir, "ORD_DT_KEY", table=RETURNS)
        unapprove_date_role(kb_dir, SALES, "ORD_DT_KEY")
        assert _statuses(kb_dir, SALES, "ORD_DT_KEY") == {"needs_review"}
        assert _statuses(kb_dir, RETURNS, "ORD_DT_KEY") == {"approved"}

    def test_a_role_that_is_not_approved_is_left_alone(self, model_dirs):
        _schema_dir, kb_dir = model_dirs
        written = (Path(kb_dir) / "_semantic_model.json").read_text()
        assert not unapprove_date_role(kb_dir, SALES, "ORD_DT_KEY")
        assert not unapprove_date_role(kb_dir, SALES, "NO_SUCH_KEY")
        assert (Path(kb_dir) / "_semantic_model.json").read_text() == written

    def test_a_rebuild_does_not_bring_the_approval_back(self, model_dirs):
        schema_dir, kb_dir = model_dirs
        _approve(kb_dir, "INV_DT_KEY")
        unapprove_date_role(kb_dir, SALES, "INV_DT_KEY")
        write_semantic_model(schema_dir=schema_dir, kb_dir=kb_dir)
        assert "approved" not in _statuses(kb_dir, SALES, "INV_DT_KEY")


class TestTheButton:

    def test_questions_stop_using_it_and_the_page_says_what_lost_its_date(self, account):
        account_id, kb_dir = account
        _create_metric(account_id, "ORD_DT_KEY")  # saving the metric approves its default date
        assert _statuses(kb_dir, SALES, "ORD_DT_KEY") == {"approved"}
        assert _runtime_date(account_id, kb_dir)["binding"]["fact_column"] == "ORD_DT_KEY"

        query = _unapprove(account_id, "ORD_DT_KEY")
        assert query["saved"] == "unapproved"
        assert "No longer the default date of Net Sales." in query["notice"]
        assert _statuses(kb_dir, SALES, "ORD_DT_KEY") == {"needs_review"}
        # What answers read -- the compiled contract -- has it too, not only the file.
        contract_roles = (load_contract(kb_dir).get("model") or {}).get("date_roles") or []
        assert [r["status"] for r in contract_roles if r["fact_table"] == SALES
                and r["fact_column"] == "ORD_DT_KEY"] == ["needs_review"]
        binding = _runtime_date(account_id, kb_dir).get("binding") or {}
        assert binding.get("resolution_source") != "metric_default_time_column"

    def test_a_metric_date_context_on_it_is_named(self, account):
        import store

        account_id, kb_dir = account
        _create_metric(account_id, "")
        _approve(kb_dir, "INV_DT_KEY")
        metric_id = next(m["id"] for m in store.list_metrics(account_id) if m["name"] == "Net Sales")
        saved = _admin(lambda routes: routes.date_context_save(
            MagicMock(), account_id, metric_id=metric_id, context_name="Billing",
            aliases="", role_identity=f"{SALES}||INV_DT_KEY", is_default=""))
        assert saved["saved"] == "context"
        query = _unapprove(account_id, "INV_DT_KEY")
        assert "Metric date contexts still using it: Net Sales: Billing." in query["notice"]
        assert "No longer the default date" not in query["notice"]

    def test_a_role_with_nothing_on_it_says_only_that_it_was_unapproved(self, account):
        account_id, kb_dir = account
        _approve(kb_dir, "INV_DT_KEY")
        assert _unapprove(account_id, "INV_DT_KEY") == {"saved": "unapproved"}

    def test_a_role_that_is_not_approved_is_an_error(self, account):
        account_id, _kb_dir = account
        query = _unapprove(account_id, "ORD_DT_KEY")
        assert query == {"error": "This date role is not approved."}

    def test_an_approved_row_offers_it_and_a_row_in_review_does_not(self, account):
        account_id, kb_dir = account
        _approve(kb_dir, "INV_DT_KEY")
        page = _page(account_id)
        assert all(part in page for part in _unapprove_form(account_id, SALES, "INV_DT_KEY"))
        assert 'data-confirm-label="Unapprove"' in page
        assert page.count('/date-roles/unapprove"') == 1   # ORD_DT_KEY is still in review

        _unapprove(account_id, "INV_DT_KEY")
        page = _page(account_id, saved="unapproved")
        assert "Date role unapproved. It is back in review" in page
        assert '/date-roles/unapprove"' not in page
