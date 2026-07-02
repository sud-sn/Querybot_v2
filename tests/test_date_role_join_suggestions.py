"""
tests/test_date_role_join_suggestions.py

Unit tests for the row-calculated metric builder's "suggest joins from schema"
feature:
  GET /admin/clients/{account_id}/metrics/api/date-role-joins

This endpoint exposes date roles the KB build already detected
(core/semantic_model.py) so admins don't have to hand-type required_joins
JSON for row-level calculated metrics.

Strategy: direct unit tests against the async route handler using mocked
dependencies — no FastAPI TestClient, no SQLite, no network.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch


def _arun(coro):
    return asyncio.run(coro)


def _make_request():
    req = MagicMock()
    req.query_params = {}
    return req


_MOCK_MODEL = {
    "date_roles": [
        {
            "name": "Due Date",
            "business_role": "due_date",
            "fact_table": "PROFITABILITY.CUS_ORD_IVC_FCT",
            "fact_column": "DUE_DT_DMS_KEY",
            "dimension_table": "PROFITABILITY.DT_DMS",
            "dimension_key": "DT_DMS_KEY",
            "synonyms": ("due date",),
            "status": "generated",
            "confidence": 90,
        },
        {
            "name": "Payment Date",
            "business_role": "payment_date",
            "fact_table": "PROFITABILITY.CUS_ORD_IVC_FCT",
            "fact_column": "PAY_DT_DMS_KEY",
            "dimension_table": "PROFITABILITY.DT_DMS",
            "dimension_key": "DT_DMS_KEY",
            "synonyms": ("payment date",),
            "status": "generated",
            "confidence": 90,
        },
        {
            "name": "Order Date",
            "business_role": "order_date",
            "fact_table": "PROFITABILITY.CUS_ORD_RCT_FCT",
            "fact_column": "ORD_DT_DMS_KEY",
            "dimension_table": "PROFITABILITY.DT_DMS",
            "dimension_key": "DT_DMS_KEY",
            "synonyms": ("order date",),
            "status": "generated",
            "confidence": 90,
        },
    ],
}


class AliasForDateRoleTests(unittest.TestCase):
    def test_strips_date_suffix_and_shortens(self):
        from admin.routes import _alias_for_date_role
        self.assertEqual(_alias_for_date_role("due_date", set()), "due_dt")
        self.assertEqual(_alias_for_date_role("payment_date", set()), "pay_dt")

    def test_collision_appends_numeric_suffix(self):
        from admin.routes import _alias_for_date_role
        used = {"due_dt"}
        self.assertEqual(_alias_for_date_role("due_date", used), "due2_dt")

    def test_empty_role_key_falls_back(self):
        from admin.routes import _alias_for_date_role
        self.assertEqual(_alias_for_date_role("", set()), "dat_dt")


class MetricDateRoleJoinsRouteTests(unittest.TestCase):
    def _call(self, table="", account_id="acct1"):
        from admin.routes import metric_date_role_joins
        req = _make_request()
        return _arun(metric_date_role_joins(req, account_id, table=table))

    def test_unauthed_raises_401(self):
        with patch("admin.routes._is_auth", return_value=False):
            with self.assertRaises(Exception):
                self._call(table="CUS_ORD_IVC_FCT")

    def test_empty_table_returns_empty_joins(self):
        with patch("admin.routes._is_auth", return_value=True):
            result = self._call(table="")
        self.assertEqual(result.status_code, 200)

    def test_no_kb_dir_returns_empty_joins(self):
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client_state", return_value={}),
        ):
            result = self._call(table="CUS_ORD_IVC_FCT")
        self.assertEqual(result.status_code, 200)

    def test_matches_fact_table_and_returns_two_joins(self):
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client_state", return_value={"kb_dir": "/fake/kb"}),
            patch("core.semantic_model.load_semantic_model", return_value=_MOCK_MODEL),
        ):
            result = self._call(table="CUS_ORD_IVC_FCT")
        self.assertEqual(result.status_code, 200)
        import json
        body = json.loads(result.body)
        joins = body["joins"]
        self.assertEqual(len(joins), 2)
        aliases = {j["alias"] for j in joins}
        self.assertEqual(aliases, {"due_dt", "pay_dt"})
        for j in joins:
            self.assertEqual(j["table"], "DT_DMS")
            self.assertEqual(j["to_column"], "DT_DMS_KEY")
        due = next(j for j in joins if j["alias"] == "due_dt")
        self.assertEqual(due["from_column"], "DUE_DT_DMS_KEY")
        self.assertEqual(due["role"], "due date")

    def test_qualified_table_name_matches_bare_fact_table(self):
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client_state", return_value={"kb_dir": "/fake/kb"}),
            patch("core.semantic_model.load_semantic_model", return_value=_MOCK_MODEL),
        ):
            result = self._call(table="PROFITABILITY.CUS_ORD_RCT_FCT")
        import json
        body = json.loads(result.body)
        self.assertEqual(len(body["joins"]), 1)
        self.assertEqual(body["joins"][0]["alias"], "ord_dt")

    def test_unmatched_table_returns_empty_joins(self):
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client_state", return_value={"kb_dir": "/fake/kb"}),
            patch("core.semantic_model.load_semantic_model", return_value=_MOCK_MODEL),
        ):
            result = self._call(table="SOME_OTHER_TABLE")
        import json
        body = json.loads(result.body)
        self.assertEqual(body["joins"], [])

    def test_model_load_failure_returns_empty_joins_not_500(self):
        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("admin.routes.store.get_client_state", return_value={"kb_dir": "/fake/kb"}),
            patch("core.semantic_model.load_semantic_model", side_effect=RuntimeError("boom")),
        ):
            result = self._call(table="CUS_ORD_IVC_FCT")
        self.assertEqual(result.status_code, 200)
        import json
        self.assertEqual(json.loads(result.body)["joins"], [])


class MetricsTemplateDateRoleUiTests(unittest.TestCase):
    def _template(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        return (root / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")

    def test_button_label_reflects_live_lookup(self):
        template = self._template()
        self.assertIn("Suggest joins from schema", template)
        self.assertNotIn("Use date-role template", template)

    def test_js_calls_date_role_joins_endpoint(self):
        template = self._template()
        self.assertIn("/metrics/api/date-role-joins?table=", template)

    def test_js_warns_when_base_table_missing(self):
        template = self._template()
        self.assertIn("Set the Base table field", template)


if __name__ == "__main__":
    unittest.main()
