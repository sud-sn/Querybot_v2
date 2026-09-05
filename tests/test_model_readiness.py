"""The modelling backlog and the freshness contract — core/model_readiness.py.

Four readiness scores already existed and none answered the question an admin
actually has: what do I do next, and what will it fix? These tests assert the
two properties that make an answer to that worth acting on:

  * ordering is by MEASURED outcome — how many failing question shapes each
    remedy resolves, taken from the metric coverage reports — not by a
    weighting somebody invented; and
  * one remedy is one item, so binding a single date role that unblocks
    eleven shapes reads as one afternoon's work rather than eleven.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_readiness import (  # noqa: E402
    MAX_CURATION_ITEMS,
    BacklogItem,
    build_report,
    curation_backlog,
    is_stale,
    metadata_version,
    metric_backlog,
    rank_backlog,
)

DESCRIPTION = "One row per invoice line at the grain the finance team reports."


class ReadinessCase(unittest.TestCase):

    def setUp(self):
        import store

        from core.curation_weight import invalidate
        self._dir = tempfile.mkdtemp(prefix="qb-ready-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "r.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-ready-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        invalidate()

    def tearDown(self):
        from core.curation_weight import invalidate
        invalidate()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _metric(self, **overrides):
        import store
        fields = {
            "name": "Net Revenue", "synonyms": "revenue",
            "sql_template": "SUM(NET_AMT)", "description": "",
            "allowed_dimensions": "Region", "grain": "month",
        }
        fields.update(overrides)
        store.save_metric(self.account_id, fields)


class TestRanking(unittest.TestCase):

    def test_the_most_unblocking_remedy_comes_first(self):
        ranked = rank_backlog([
            BacklogItem(kind="described", subject="S.RAW", remedy="describe"),
            BacklogItem(kind="date_role", subject="Net Revenue",
                        remedy="bind a date", unblocks=11),
            BacklogItem(kind="dimensions", subject="EBITDA",
                        remedy="add dimensions", unblocks=5),
        ])
        self.assertEqual([i.subject for i in ranked],
                         ["Net Revenue", "EBITDA", "S.RAW"])

    def test_kind_weight_only_breaks_ties(self):
        # A measured outcome always beats a weighting. An item unblocking one
        # shape outranks a heavier kind that unblocks none.
        ranked = rank_backlog([
            BacklogItem(kind="date_role", subject="A", remedy="", unblocks=0),
            BacklogItem(kind="column_synonyms", subject="B", remedy="", unblocks=1),
        ])
        self.assertEqual(ranked[0].subject, "B")

    def test_equal_items_rank_stably(self):
        items = [
            BacklogItem(kind="described", subject="B", remedy=""),
            BacklogItem(kind="described", subject="A", remedy=""),
        ]
        self.assertEqual([i.subject for i in rank_backlog(items)],
                         [i.subject for i in rank_backlog(list(reversed(items)))])


class TestTheMetricBacklog(ReadinessCase):

    def test_a_metric_with_no_date_role_produces_one_item_not_eleven(self):
        # Binding one date role unblocks every period shape at once. Listing
        # them separately makes an afternoon look like a week, and a backlog
        # nobody believes is a backlog nobody starts.
        self._metric()
        items = metric_backlog(self.account_id)
        date_items = [i for i in items if i.kind == "date_role"]
        self.assertEqual(len(date_items), 1)
        self.assertGreater(date_items[0].unblocks, 1)

    def test_the_item_says_which_metric_and_carries_an_example(self):
        self._metric(name="EBITDA")
        item = next(i for i in metric_backlog(self.account_id)
                    if i.kind == "date_role")
        self.assertEqual(item.subject, "EBITDA")
        self.assertTrue(item.remedy)
        self.assertIn("EBITDA", item.detail)

    def test_a_metric_with_no_dimensions_is_also_reported(self):
        self._metric(allowed_dimensions="")
        kinds = {i.kind for i in metric_backlog(self.account_id)}
        self.assertIn("dimensions", kinds)

    def test_a_workspace_with_no_metrics_has_no_metric_backlog(self):
        self.assertEqual(metric_backlog(self.account_id), [])

    def test_a_store_failure_costs_the_backlog_not_the_page(self):
        import store
        self._metric()
        with patch.object(store, "list_metrics", side_effect=RuntimeError("db")):
            self.assertEqual(metric_backlog(self.account_id), [])


class TestTheCurationBacklog(ReadinessCase):

    def test_an_undescribed_table_is_reported(self):
        import store

        from core.curation_weight import invalidate
        store.save_table_description(
            self.account_id, "SALES.ORDERS", synonyms="sales")
        invalidate(self.account_id)
        items = curation_backlog(self.account_id)
        self.assertEqual([i.kind for i in items], ["described"])
        self.assertEqual(items[0].subject, "SALES.ORDERS")

    def test_a_described_table_missing_synonyms_is_reported_next(self):
        import store

        from core.curation_weight import invalidate
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        invalidate(self.account_id)
        items = curation_backlog(self.account_id)
        self.assertEqual([i.kind for i in items], ["synonym"])

    def test_a_fully_curated_table_is_not_in_the_backlog(self):
        import store

        from core.curation_weight import invalidate
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales, bookings")
        invalidate(self.account_id)
        self.assertEqual(curation_backlog(self.account_id), [])

    def test_the_list_is_capped_so_it_can_be_started(self):
        import store

        from core.curation_weight import invalidate
        for i in range(MAX_CURATION_ITEMS + 10):
            store.save_table_description(
                self.account_id, f"SALES.T{i:03d}", synonyms="x")
        invalidate(self.account_id)
        self.assertEqual(len(curation_backlog(self.account_id)),
                         MAX_CURATION_ITEMS)

    def test_an_undescribed_table_never_claims_to_unblock_a_count(self):
        # It degrades every answer touching it, which is real but not
        # countable. Claiming a number would put it above measured items.
        import store

        from core.curation_weight import invalidate
        store.save_table_description(self.account_id, "SALES.ORDERS", synonyms="x")
        invalidate(self.account_id)
        self.assertEqual(curation_backlog(self.account_id)[0].unblocks, 0)


class TestTheWholeReport(ReadinessCase):

    def test_the_report_counts_what_is_done_as_well_as_what_is_left(self):
        import store

        from core.curation_weight import invalidate
        self._metric()
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        invalidate(self.account_id)
        report = build_report(self.account_id)
        self.assertEqual(report.tables_total, 1)
        self.assertEqual(report.tables_described, 1)
        self.assertEqual(report.metrics_total, 1)
        self.assertTrue(report.items)
        self.assertGreater(report.total_unblocked, 0)

    def test_binding_a_date_role_moves_a_metric_into_the_complete_column(self):
        # Write API to read API: the binding goes in through the store the
        # admin UI uses, and the report has to find it.
        import store
        self._metric()
        metric_id = int(store.list_metrics(self.account_id)[0]["id"])
        self.assertEqual(build_report(self.account_id).metrics_complete, 0)

        store.save_metric_date_context(self.account_id, {
            "metric_id": metric_id, "context_name": "invoice",
            "date_role": "invoice_date", "fact_table": "SALES.ORDERS",
            "fact_column": "INVOICE_DATE_KEY",
            "dimension_table": "SALES.D_DATE", "dimension_key": "DATE_KEY",
            "date_value_column": "CALENDAR_DATE", "date_key_type": "surrogate_fk",
            "is_default": 1,
        })
        report = build_report(self.account_id)
        self.assertEqual(report.metrics_total, 1)
        self.assertEqual(report.metrics_complete, 1)
        # And the date-role item leaves the backlog.
        self.assertNotIn("date_role", {i.kind for i in report.items})

    def test_the_report_carries_the_metadata_version(self):
        self._metric()
        report = build_report(self.account_id)
        self.assertTrue(report.metadata_version)
        self.assertEqual(report.metadata_version,
                         metadata_version(self.account_id))

    def test_an_empty_workspace_reports_nothing_to_do_rather_than_erroring(self):
        report = build_report(self.account_id)
        self.assertEqual(report.items, ())
        self.assertEqual(report.tables_total, 0)


class TestTheFreshnessContract(ReadinessCase):

    def test_the_same_model_gives_the_same_version(self):
        self._metric()
        self.assertEqual(metadata_version(self.account_id),
                         metadata_version(self.account_id))

    def test_changing_a_metric_changes_the_version(self):
        import store
        self._metric()
        before = metadata_version(self.account_id)
        self._metric(name="Gross Revenue")
        self.assertNotEqual(before, metadata_version(self.account_id))
        self.assertTrue(store.list_metrics(self.account_id))

    def test_changing_a_metrics_formula_changes_the_version(self):
        import store
        self._metric()
        before = metadata_version(self.account_id)
        metric_id = int(store.list_metrics(self.account_id)[0]["id"])
        store.update_metric(metric_id, {"sql_template": "SUM(GROSS_AMT)"},
                            account_id=self.account_id)
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_changing_a_metrics_dimensions_changes_the_version(self):
        # The version has to cover everything an answer depends on, and
        # approved dimensions decide what SQL can group by.
        import store
        self._metric()
        before = metadata_version(self.account_id)
        metric_id = int(store.list_metrics(self.account_id)[0]["id"])
        store.update_metric(metric_id, {"allowed_dimensions": "Region, Customer"},
                            account_id=self.account_id)
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_changing_a_table_description_changes_the_version(self):
        import store
        self._metric()
        before = metadata_version(self.account_id)
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_adding_an_entity_changes_the_version(self):
        import store
        before = metadata_version(self.account_id)
        store.save_entity(
            self.account_id, entity_name="Order", table_name="ORDERS",
            schema_name="SALES", pk_column="ORDER_ID", display_name="Order",
            description="", entity_type="fact",
        )
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_rewriting_an_existing_description_changes_the_version(self):
        # Adding a description creates a row, so a version built from table
        # NAMES alone would still change and look correct. Rewriting one
        # changes nothing but the prose -- which is exactly what the model
        # depends on, and exactly what a name-only fingerprint misses.
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        before = metadata_version(self.account_id)
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description="One row per shipment, not per invoice line.")
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_changing_an_existing_tables_synonyms_changes_the_version(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales")
        before = metadata_version(self.account_id)
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales, bookings, orders")
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_repointing_an_existing_entity_changes_the_version(self):
        # Same entity name, different table. A fingerprint built from entity
        # names alone reports no change while every join path moved.
        import store
        store.save_entity(
            self.account_id, entity_name="Order", table_name="ORDERS",
            schema_name="SALES", pk_column="ORDER_ID", display_name="Order",
            description="", entity_type="fact",
        )
        before = metadata_version(self.account_id)
        store.save_entity(
            self.account_id, entity_name="Order", table_name="ORDER_HEADER",
            schema_name="SALES", pk_column="ORDER_ID", display_name="Order",
            description="", entity_type="fact",
        )
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_deactivating_an_entity_changes_the_version(self):
        import store
        store.save_entity(
            self.account_id, entity_name="Order", table_name="ORDERS",
            schema_name="SALES", pk_column="ORDER_ID", display_name="Order",
            description="", entity_type="fact",
        )
        before = metadata_version(self.account_id)
        store.delete_entity(self.account_id, "Order")
        self.assertNotEqual(before, metadata_version(self.account_id))

    def test_two_workspaces_have_different_versions(self):
        import store
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        self._metric()
        self.assertNotEqual(metadata_version(self.account_id),
                            metadata_version(other))

    def test_an_answer_stamped_before_a_change_is_stale_after_it(self):
        self._metric()
        stamped = metadata_version(self.account_id)
        self.assertFalse(is_stale(stamped, self.account_id))
        self._metric(name="Gross Revenue")
        self.assertTrue(is_stale(stamped, self.account_id))

    def test_nothing_is_claimed_when_the_version_is_unknown(self):
        # A freshness warning nobody can verify is noise, and noise in a
        # governance surface is worse than silence.
        self.assertFalse(is_stale("", self.account_id))
        with patch("core.model_readiness.metadata_version", return_value=""):
            self.assertFalse(is_stale("anything", self.account_id))

    def test_a_store_failure_yields_no_version_rather_than_a_made_up_one(self):
        import store
        with patch.object(store, "list_entities", side_effect=RuntimeError("db")):
            self.assertEqual(metadata_version(self.account_id), "")


class TestTheAdminApi(ReadinessCase):

    def _get(self):
        import asyncio
        import json
        from unittest.mock import MagicMock

        import admin.routes as routes

        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(
                routes.model_readiness_api(req, self.account_id))
        return json.loads(response.body)

    def test_the_endpoint_reports_the_backlog_it_was_asked_for(self):
        self._metric()
        body = self._get()
        self.assertTrue(body["items"])
        self.assertEqual(body["metrics_total"], 1)
        self.assertTrue(body["metadata_version"])
        self.assertGreater(body["total_unblocked"], 0)

    def test_the_items_are_ordered_most_unblocking_first(self):
        self._metric()
        self._metric(name="EBITDA", allowed_dimensions="")
        unblocks = [i["unblocks"] for i in self._get()["items"]]
        self.assertEqual(unblocks, sorted(unblocks, reverse=True))

    def test_an_unauthenticated_request_gets_nothing(self):
        import asyncio
        from unittest.mock import MagicMock

        import admin.routes as routes
        from fastapi import HTTPException

        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(HTTPException):
                asyncio.run(routes.model_readiness_api(req, self.account_id))


class TestThePipelineStampsIt(unittest.TestCase):

    def test_the_version_is_computed_and_recorded_on_the_trace(self):
        import inspect

        import core.query_pipeline as qp

        source = inspect.getsource(qp._handle_query_impl)
        self.assertIn("from core.model_readiness import metadata_version", source)
        computed_at = source.index("_metadata_version = metadata_version(account_id)")
        traced_at = source.index('"metadata_version",', computed_at)
        self.assertLess(computed_at, traced_at)
        # Guarded: an answer must not be lost to a readiness lookup.
        self.assertIn("log.warning(\"metadata_version not stamped", source)


if __name__ == "__main__":
    unittest.main()
