"""
An admin certifies a metric, and a change to its numbers takes that away.

Nothing recorded that anyone had checked a metric's numbers. metric_status is
set by the formula validator -- "validated" means the SQL compiles, not that
the total is right -- and rows older than that column read "published" by
default. So a readiness gate could not ask for key metrics to be certified:
there was no such thing.

The admin now certifies a metric on the registry once its numbers match a
figure the business already trusts. Only a live metric whose formula
validates can be. A change to what its numbers are -- the formula, the table,
the grain, the date they are counted on -- takes the certification away; a new
synonym, a description, or the edit form saved unchanged does not.
Deprecating a metric takes it away too.

A scratch store (QUERYBOT_DB_PATH); no real database.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.headers = {}
    return request


@pytest.fixture
def account():
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _metric(account_id, name="Net Sales", sql="SUM(NET_SLS_AMT)"):
    import store

    return store.save_metric(account_id, {
        "name": name, "formula_type": "expression", "sql_template": sql,
        "base_table": "MART.SLS_TRX_FCT", "synonyms": "revenue", "description": "Invoiced sales",
        "grain": "invoice line", "default_time_column": "IVC_DT_DMS_KEY",
    })


def _row(metric_id):
    import store

    return store.get_metric(metric_id)


def _post(route, account_id, metric_id):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(getattr(routes, route)(_request(), account_id, metric_id))


def _save_form(account_id, metric_id, **changes):
    """The registry's edit form, as the page fills it from the stored metric."""
    from admin import routes

    row = {**_row(metric_id), **changes}
    fields = (
        "name", "synonyms", "sql_template", "description", "formula_type", "result_format",
        "required_columns", "allowed_dimensions", "metric_builder_config", "example_questions",
        "grain", "category", "default_time_column", "base_table", "base_entity",
    )
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval"):
        response = asyncio.run(routes.metric_update(
            _request(), account_id, metric_id,
            **{key: str(row.get(key) or "") for key in fields}, is_active="1",
        ))
    assert response.status_code == 303 and "error" not in response.headers["location"]


class TestCertifying:

    def test_a_validated_metric_is_certified_by_the_admin(self, account):
        metric_id = _metric(account)
        assert _row(metric_id)["metric_status"] == "validated"
        response = _post("metric_certify", account, metric_id)
        assert response.headers["location"].endswith("saved=certified")
        row = _row(metric_id)
        assert row["certified_at"] and row["certified_by"] == "admin"

    def test_a_metric_whose_formula_fails_cannot_be(self, account):
        metric_id = _metric(account, name="Broken", sql="DELETE FROM MART.SLS_TRX_FCT")
        assert _row(metric_id)["metric_status"] == "draft"
        response = _post("metric_certify", account, metric_id)
        assert "error=" in response.headers["location"]
        assert _row(metric_id)["certified_at"] == ""

    def test_a_deprecated_metric_cannot_be(self, account):
        import store

        metric_id = _metric(account)
        store.deprecate_metric(metric_id, account)
        assert "error=" in _post("metric_certify", account, metric_id).headers["location"]

    def test_a_metric_switched_off_cannot_be(self, account):
        import store

        metric_id = _metric(account)
        store.update_metric(metric_id, {"is_active": 0}, account_id=account)
        assert _row(metric_id)["metric_status"] == "validated"
        assert "error=" in _post("metric_certify", account, metric_id).headers["location"]

    def test_another_workspaces_metric_is_not_touched(self, account):
        import store

        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        metric_id = _metric(other)
        assert "error=" in _post("metric_certify", account, metric_id).headers["location"]
        assert _row(metric_id)["certified_at"] == ""

    def test_the_admin_can_take_it_back(self, account):
        metric_id = _metric(account)
        _post("metric_certify", account, metric_id)
        assert _post("metric_uncertify", account, metric_id).headers["location"].endswith(
            "saved=uncertified")
        assert _row(metric_id)["certified_at"] == ""


class TestWhatTakesItAway:

    @pytest.mark.parametrize("change", [
        {"sql_template": "SUM(NET_SLS_AMT) - SUM(RTN_AMT)"},
        {"base_table": "MART.SLS_ORD_FCT"},
        {"grain": "invoice"},
        {"default_time_column": "SHP_DT_DMS_KEY"},
    ])
    def test_a_change_to_its_numbers(self, account, change):
        metric_id = _metric(account)
        _post("metric_certify", account, metric_id)
        _save_form(account, metric_id, **change)
        assert _row(metric_id)["certified_at"] == ""

    @pytest.mark.parametrize("change", [
        {}, {"synonyms": "revenue, net revenue"}, {"description": "Sales net of discounts"},
    ])
    def test_not_a_synonym_a_description_or_a_save_that_changes_nothing(self, account, change):
        metric_id = _metric(account)
        _post("metric_certify", account, metric_id)
        _save_form(account, metric_id, **change)
        assert _row(metric_id)["certified_at"]

    def test_deprecating_it(self, account):
        import store

        metric_id = _metric(account)
        _post("metric_certify", account, metric_id)
        store.deprecate_metric(metric_id, account)
        assert _row(metric_id)["certified_at"] == ""


class TestTheRegistry:

    def _render(self, account):
        import store
        from admin import routes

        class _Url:
            path = f"/admin/clients/{account}/metrics"

        class _FakeRequest:
            url = _Url()
            query_params: dict = {}
            session = {"admin_id": "admin_user_1"}
            scope = {"type": "http"}
            cookies: dict = {}
            headers: dict = {}

        context = {
            "client": {"account_id": account, "client_name": "Test Ltd", "state": "READY"},
            "metrics": store.list_metrics(account, active_only=False), "proposals": [],
            "saved": None, "error": None, "db_type": "azure_sql",
        }
        return routes.templates.get_template("client_metrics.html").render(
            request=_FakeRequest(), **context)

    def test_it_shows_which_are_certified_and_offers_the_rest(self, account):
        certified, open_, broken = _metric(account), _metric(account, "Gross Margin", "SUM(GRS_MRG_AMT)"), \
            _metric(account, "Broken", "DELETE FROM MART.SLS_TRX_FCT")
        _post("metric_certify", account, certified)
        html = self._render(account)
        assert html.count(">Certified</span>") == 1
        assert f"/metrics/{certified}/uncertify" in html
        assert f"/metrics/{open_}/certify" in html
        assert f"/metrics/{broken}/certify" not in html
