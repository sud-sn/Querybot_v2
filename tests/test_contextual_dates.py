import json
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from core.contextual_dates import (
    build_contextual_date_plan,
    build_contextual_date_plan_many,
    detect_temporal_window,
    find_explicit_date_roles,
    resolve_contextual_date_binding,
)
from core.semantic_model import (
    build_semantic_model,
    load_semantic_model,
    patch_date_role,
    write_semantic_model,
)
from core.pipeline_context import _merge_semantic_plans
from core.graph_resolver import infer_connected_default_date_fact
from core.query_pipeline import _graph_with_exact_date_edges, _resolved_fact_tables
from core.validator import validate_sql_detailed


def _binding(context, role, fact_column, *, default=False, metric_id=1):
    return {
        "id": metric_id,
        "metric_id": metric_id,
        "metric_name": "Revenue",
        "context_name": context,
        "aliases": context.lower(),
        "date_role": role,
        "fact_table": "SALES.FACT_REVENUE",
        "fact_column": fact_column,
        "dimension_table": "SALES.DIM_DATE",
        "dimension_key": "DATE_KEY",
        "date_value_column": "FULL_DATE",
        "date_key_type": "surrogate_fk",
        "is_default": 1 if default else 0,
        "priority": 50,
    }


def _default_date_graph(*, include_claim=False):
    entities = [
        {
            "entity_name": "Prescription Fill", "entity_type": "fact",
            "schema_name": "PHARMACY", "table_name": "F_RX_FILL",
            "status": "confirmed",
        },
        {
            "entity_name": "Patient", "entity_type": "dimension",
            "schema_name": "PHARMACY", "table_name": "D_PATIENT",
            "status": "confirmed",
        },
        {
            "entity_name": "Date", "entity_type": "dimension",
            "schema_name": "PHARMACY", "table_name": "D_DATE",
            "status": "confirmed",
        },
    ]
    relationships = [
        {
            "id": 1, "from_entity": "Prescription Fill", "to_entity": "Patient",
            "from_column": "PATIENT_ID", "to_column": "PATIENT_ID",
            "join_type": "LEFT", "generated_by": "db_fk", "status": "confirmed",
            "validation_status": "valid", "confidence_score": 100,
        },
        {
            "id": 2, "from_entity": "Prescription Fill", "to_entity": "Date",
            "from_column": "FILL_DATE_ID", "to_column": "DATE_ID",
            "join_type": "LEFT", "generated_by": "date_role", "status": "confirmed",
            "validation_status": "valid", "confidence_score": 100,
        },
    ]
    if include_claim:
        entities.append({
            "entity_name": "Claim", "entity_type": "fact",
            "schema_name": "PHARMACY", "table_name": "F_CLAIM",
            "status": "confirmed",
        })
        relationships.extend([
            {
                "id": 3, "from_entity": "Claim", "to_entity": "Patient",
                "from_column": "PATIENT_ID", "to_column": "PATIENT_ID",
                "join_type": "LEFT", "generated_by": "db_fk", "status": "confirmed",
                "validation_status": "valid", "confidence_score": 100,
            },
            {
                "id": 4, "from_entity": "Claim", "to_entity": "Date",
                "from_column": "CLAIM_DATE_ID", "to_column": "DATE_ID",
                "join_type": "LEFT", "generated_by": "date_role", "status": "confirmed",
                "validation_status": "valid", "confidence_score": 100,
            },
        ])
    return {"entities": entities, "relationships": relationships, "properties": []}


class ContextualDateResolutionTests(unittest.TestCase):
    def setUp(self):
        self.metric = {"id": 1, "name": "Revenue", "base_table": "SALES.FACT_REVENUE"}
        self.bindings = [
            _binding("Sales", "invoice_date", "INVOICE_DATE_KEY", default=True),
            _binding("Inventory Sales", "accounting_date", "INVENTORY_DATE_KEY"),
        ]

    def test_context_selects_inventory_date(self):
        result = resolve_contextual_date_binding(
            "show inventory sales revenue yesterday",
            matched_metrics=[self.metric],
            bindings=self.bindings,
            date_roles=[],
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "INVENTORY_DATE_KEY")
        self.assertEqual(result["binding"]["resolution_source"], "business_context")

    def test_generic_temporal_question_uses_one_default(self):
        result = resolve_contextual_date_binding(
            "what was revenue yesterday",
            matched_metrics=[self.metric],
            bindings=self.bindings,
            date_roles=[],
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "INVOICE_DATE_KEY")

    def test_generic_question_without_default_is_ambiguous(self):
        bindings = [{**item, "is_default": 0} for item in self.bindings]
        result = resolve_contextual_date_binding(
            "what was revenue yesterday",
            matched_metrics=[self.metric],
            bindings=bindings,
            date_roles=[],
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["options"]), 2)

    def test_explicit_approved_role_overrides_default(self):
        roles = [{
            "name": "Delivery Date",
            "business_role": "delivery_date",
            "synonyms": ["shipped date"],
            "fact_table": "SALES.FACT_REVENUE",
            "fact_column": "DELIVERY_DATE_KEY",
            "dimension_table": "SALES.DIM_DATE",
            "dimension_key": "DATE_KEY",
            "date_value_column": "FULL_DATE",
            "status": "approved",
        }]
        result = resolve_contextual_date_binding(
            "show revenue by delivery date",
            matched_metrics=[self.metric],
            bindings=self.bindings,
            date_roles=roles,
        )
        self.assertEqual(result["binding"]["fact_column"], "DELIVERY_DATE_KEY")
        self.assertEqual(result["binding"]["resolution_source"], "explicit_date_role")

    def test_explicit_booked_month_overrides_default_invoice_date(self):
        roles = [{
            "name": "Booked Date",
            "business_role": "booked_date",
            "synonyms": ["booking date"],
            "fact_table": "SALES.FACT_REVENUE",
            "fact_column": "BOOKED_DT_ID",
            "dimension_table": "SALES.DIM_DATE",
            "dimension_key": "DATE_KEY",
            "date_value_column": "FULL_DATE",
            "date_key_type": "surrogate_fk",
            "status": "approved",
        }]
        result = resolve_contextual_date_binding(
            "show net revenue by booked month",
            matched_metrics=[self.metric],
            bindings=self.bindings,
            date_roles=roles,
            required_fact_tables={"SALES.FACT_REVENUE"},
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "BOOKED_DT_ID")
        self.assertEqual(result["binding"]["resolution_source"], "explicit_date_role")

    def test_explicit_high_confidence_generated_role_is_governed_fallback(self):
        roles = [{
            "name": "Order Date",
            "business_role": "order_date",
            "fact_table": "PHARMA_LAB.F_RX_ORDER",
            "fact_column": "ORDER_DATE_ID",
            "dimension_table": "PHARMA_LAB.D_DATE",
            "dimension_key": "DATE_ID",
            "date_value_column": "CALENDAR_DATE",
            "date_key_type": "surrogate_fk",
            "status": "generated",
            "confidence": 99,
        }]
        matches = find_explicit_date_roles(
            "ordered revenue by pharmacy for the last 7 days", roles
        )
        self.assertEqual(len(matches), 1)
        result = resolve_contextual_date_binding(
            "ordered revenue by pharmacy for the last 7 days",
            matched_metrics=[],
            bindings=[],
            date_roles=roles,
            required_fact_tables={"PHARMA_LAB.F_RX_ORDER"},
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "ORDER_DATE_ID")
        self.assertEqual(
            result["binding"]["resolution_source"],
            "explicit_generated_date_role",
        )
        plan = build_contextual_date_plan(result["binding"], "last 7 days")
        self.assertEqual(
            plan["joins"][0]["conditions"], [("ORDER_DATE_ID", "DATE_ID")]
        )
        self.assertEqual(plan["temporal_policies"][0]["date_column"], "CALENDAR_DATE")

    def test_generated_role_never_overrides_approved_explicit_role(self):
        roles = [
            {
                "name": "Order Date", "business_role": "order_date",
                "fact_table": "SALES.FACT_APPROVED", "fact_column": "ORDER_DT_ID",
                "dimension_table": "SALES.DIM_DATE", "dimension_key": "DATE_ID",
                "date_value_column": "CALENDAR_DATE", "date_key_type": "surrogate_fk",
                "status": "approved", "confidence": 100,
            },
            {
                "name": "Order Date", "business_role": "order_date",
                "fact_table": "SALES.FACT_GENERATED", "fact_column": "ORDER_DATE_ID",
                "dimension_table": "SALES.DIM_DATE", "dimension_key": "DATE_ID",
                "date_value_column": "CALENDAR_DATE", "date_key_type": "surrogate_fk",
                "status": "generated", "confidence": 99,
            },
        ]
        matches = find_explicit_date_roles("revenue by order date", roles)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["fact_table"], "SALES.FACT_APPROVED")

    def test_incomplete_or_low_confidence_generated_role_is_not_selected(self):
        roles = [{
            "name": "Order Date", "business_role": "order_date",
            "fact_table": "SALES.FACT_REVENUE", "fact_column": "ORDER_DATE_ID",
            "dimension_table": "SALES.DIM_DATE", "dimension_key": "DATE_ID",
            "date_value_column": "CALENDAR_DATE", "date_key_type": "surrogate_fk",
            "status": "generated", "confidence": 80,
        }]
        self.assertEqual(find_explicit_date_roles("revenue by order date", roles), [])

    def test_fact_default_is_scoped_to_resolved_fact(self):
        roles = [
            {
                "name": "Invoice Date", "business_role": "invoice_date",
                "fact_table": "SALES.FACT_REVENUE", "fact_column": "INVOICE_DATE_KEY",
                "dimension_table": "SALES.DIM_DATE", "dimension_key": "DATE_KEY",
                "date_value_column": "FULL_DATE", "date_key_type": "surrogate_fk",
                "status": "approved", "is_default": True,
            },
            {
                "name": "Fill Date", "business_role": "fill_date",
                "fact_table": "PHARMACY.FACT_PRESCRIPTION", "fact_column": "FILL_DATE",
                "dimension_table": "", "dimension_key": "",
                "date_value_column": "FILL_DATE", "date_key_type": "native_date",
                "status": "approved", "is_default": True,
            },
        ]
        result = resolve_contextual_date_binding(
            "how many patients did we fill yesterday",
            matched_metrics=[],
            bindings=[],
            date_roles=roles,
            required_fact_tables={"PHARMACY.FACT_PRESCRIPTION"},
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "FILL_DATE")
        self.assertEqual(result["binding"]["resolution_source"], "explicit_date_role")

    def test_empty_fact_scope_does_not_use_unrelated_global_default(self):
        roles = [{
            "name": "Fill Date", "business_role": "fill_date",
            "fact_table": "PHARMACY.F_RX_FILL", "fact_column": "FILL_DATE_ID",
            "dimension_table": "PHARMACY.D_DATE", "dimension_key": "DATE_ID",
            "date_value_column": "CALENDAR_DATE", "date_key_type": "surrogate_fk",
            "status": "approved", "is_default": True,
        }]
        result = resolve_contextual_date_binding(
            "show supplier count today",
            matched_metrics=[],
            bindings=[],
            date_roles=roles,
            required_fact_tables=set(),
        )
        self.assertEqual(result["status"], "none")

    def test_semantic_dimension_is_not_misclassified_as_resolved_fact(self):
        graph = _default_date_graph()
        facts = _resolved_fact_tables(
            {"detected": ["Patient"], "anchor": "Patient"},
            graph,
            semantic_plan={
                "fields": [{"table": "PHARMACY.D_PATIENT", "column": "STATE"}],
            },
        )
        self.assertEqual(facts, set())

    def test_graph_infers_unique_default_date_fact_for_dimension(self):
        result = infer_connected_default_date_fact(
            _default_date_graph(),
            requested_entities={"Patient"},
            requested_tables={"PHARMACY.D_PATIENT"},
            candidate_fact_tables={"PHARMACY.F_RX_FILL"},
            excluded_tables={"PHARMACY.D_DATE"},
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["fact_table"], "PHARMACY.F_RX_FILL")
        self.assertEqual(result["target_entities"], ["Patient"])

    def test_graph_refuses_ambiguous_default_date_facts(self):
        graph = _default_date_graph(include_claim=True)
        result = infer_connected_default_date_fact(
            graph,
            requested_entities={"Patient"},
            requested_tables=set(),
            candidate_fact_tables={"PHARMACY.F_RX_FILL", "PHARMACY.F_CLAIM"},
            excluded_tables={"PHARMACY.D_DATE"},
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(
            {item["fact_table"] for item in result["candidates"]},
            {"PHARMACY.F_RX_FILL", "PHARMACY.F_CLAIM"},
        )

    def test_plan_requires_business_date_value_and_join(self):
        plan = build_contextual_date_plan(self.bindings[1])
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["fields"][0]["column"], "FULL_DATE")
        self.assertEqual(plan["fields"][0]["enforcement"], "required")
        self.assertEqual(plan["joins"][0]["conditions"], [("INVENTORY_DATE_KEY", "DATE_KEY")])

    def test_native_date_plan_needs_no_dimension_join(self):
        binding = {
            **_binding("Fill Date", "fill_date", "FILL_DATE"),
            "fact_table": "PHARMACY.FACT_PRESCRIPTION",
            "dimension_table": "",
            "dimension_key": "",
            "date_value_column": "FILL_DATE",
            "date_key_type": "native_date",
        }
        plan = build_contextual_date_plan(binding, "show patient count yesterday")
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["joins"], [])
        self.assertEqual(plan["fields"][0]["table"], "PHARMACY.FACT_PRESCRIPTION")
        self.assertEqual(plan["fields"][0]["column"], "FILL_DATE")
        self.assertEqual(plan["temporal_policies"][0]["anchor_policy"], "latest_available")

        combined = build_contextual_date_plan_many([binding], "show patient count yesterday")
        self.assertTrue(combined["enabled"])
        self.assertEqual(combined["joins"], [])

    def test_relative_window_detection_uses_latest_available_anchor(self):
        self.assertEqual(detect_temporal_window("show fills today")["kind"], "today")
        self.assertEqual(detect_temporal_window("show fills yesterday")["kind"], "yesterday")
        window = detect_temporal_window("show fills in the last 7 days")
        self.assertEqual(window["amount"], 7)
        self.assertEqual(window["unit"], "day")
        self.assertEqual(window["anchor_policy"], "latest_available")

    def test_two_explicit_roles_select_two_role_playing_joins(self):
        roles = [
            {
                "name": "Booked Date",
                "business_role": "booked_date",
                "synonyms": ["booking date"],
                "fact_table": "SALES.FACT_REVENUE",
                "fact_column": "BOOKED_DT_ID",
                "dimension_table": "SALES.DIM_DATE",
                "dimension_key": "DATE_KEY",
                "date_value_column": "FULL_DATE",
                "date_key_type": "surrogate_fk",
                "status": "approved",
            },
            {
                "name": "Order Date",
                "business_role": "order_date",
                "synonyms": ["ordered date"],
                "fact_table": "SALES.FACT_REVENUE",
                "fact_column": "ORDER_DT_ID",
                "dimension_table": "SALES.DIM_DATE",
                "dimension_key": "DATE_KEY",
                "date_value_column": "FULL_DATE",
                "date_key_type": "surrogate_fk",
                "status": "approved",
            },
        ]
        result = resolve_contextual_date_binding(
            "compare revenue by booked date and order date",
            matched_metrics=[self.metric],
            bindings=[],
            date_roles=roles,
        )
        self.assertEqual(result["status"], "selected_many")
        plan = build_contextual_date_plan_many(result["bindings"])
        self.assertEqual(len(plan["joins"]), 2)
        self.assertEqual(
            {tuple(edge["conditions"][0]) for edge in plan["joins"]},
            {("BOOKED_DT_ID", "DATE_KEY"), ("ORDER_DT_ID", "DATE_KEY")},
        )
        self.assertEqual(
            {edge["role_alias"] for edge in plan["joins"]},
            {"booked_date", "order_date"},
        )
        merged = _merge_semantic_plans(plan)
        self.assertEqual(len(merged["joins"]), 2)
        self.assertEqual(len(merged["date_key_policies"]), 2)

    def test_validator_accepts_same_date_dimension_joined_twice(self):
        plan = build_contextual_date_plan_many([
            _binding("Booked", "booked_date", "BOOKED_DT_ID"),
            _binding("Order", "order_date", "ORDER_DT_ID"),
        ])
        columns = {
            "SALES.FACT_REVENUE": {
                "BOOKED_DT_ID": "int",
                "ORDER_DT_ID": "int",
                "AMOUNT": "decimal",
            },
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT booked.FULL_DATE AS BookedDate, ordered.FULL_DATE AS OrderDate, "
            "SUM(f.AMOUNT) AS Revenue FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE booked ON f.BOOKED_DT_ID=booked.DATE_KEY "
            "LEFT JOIN SALES.DIM_DATE ordered ON f.ORDER_DT_ID=ordered.DATE_KEY "
            "GROUP BY booked.FULL_DATE, ordered.FULL_DATE"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_validator_rejects_parsing_surrogate_date_id(self):
        plan = build_contextual_date_plan(_binding("Booked", "booked_date", "BOOKED_DT_ID"))
        columns = {
            "SALES.FACT_REVENUE": {"BOOKED_DT_ID": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT YEAR(f.BOOKED_DT_ID), SUM(f.AMOUNT) FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE d ON f.BOOKED_DT_ID=d.DATE_KEY "
            "GROUP BY YEAR(f.BOOKED_DT_ID)"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "surrogate_date_conversion")

    def test_validator_rejects_dateadd_range_on_surrogate_date_id(self):
        plan = build_contextual_date_plan(
            _binding("Order Date", "order_date", "ORDER_DATE_ID"),
            "last 7 days",
        )
        columns = {
            "SALES.FACT_REVENUE": {"ORDER_DATE_ID": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT SUM(f.AMOUNT) AS Revenue FROM SALES.FACT_REVENUE f "
            "WHERE f.ORDER_DATE_ID >= DATEADD(day, -7, "
            "(SELECT MAX(x.ORDER_DATE_ID) FROM SALES.FACT_REVENUE x))"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertIn(
            result.code,
            {"temporal_anchor_missing", "surrogate_date_conversion"},
        )

    def test_validator_accepts_relative_window_on_dimension_calendar_date(self):
        plan = build_contextual_date_plan(
            _binding("Order Date", "order_date", "ORDER_DATE_ID"),
            "last 7 days",
        )
        columns = {
            "SALES.FACT_REVENUE": {"ORDER_DATE_ID": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT SUM(f.AMOUNT) AS Revenue FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE order_date "
            "ON f.ORDER_DATE_ID=order_date.DATE_KEY "
            "WHERE order_date.FULL_DATE >= DATEADD(day, -7, "
            "(SELECT MAX(d2.FULL_DATE) FROM SALES.DIM_DATE d2))"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_validator_accepts_selected_context_join(self):
        plan = build_contextual_date_plan(self.bindings[1])
        columns = {
            "SALES.FACT_REVENUE": {
                "INVENTORY_DATE_KEY": "int",
                "AMOUNT": "decimal",
            },
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT d.FULL_DATE, SUM(f.AMOUNT) AS Revenue "
            "FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE d ON f.INVENTORY_DATE_KEY=d.DATE_KEY "
            "GROUP BY d.FULL_DATE"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_validator_rejects_wrong_date_role_join(self):
        plan = build_contextual_date_plan(self.bindings[1])
        columns = {
            "SALES.FACT_REVENUE": {
                "INVOICE_DATE_KEY": "int",
                "INVENTORY_DATE_KEY": "int",
                "AMOUNT": "decimal",
            },
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT d.FULL_DATE, SUM(f.AMOUNT) AS Revenue "
            "FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE d ON f.INVOICE_DATE_KEY=d.DATE_KEY "
            "GROUP BY d.FULL_DATE"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("INVENTORY_DATE_KEY", result.reason)

    def test_validator_rejects_temporal_aggregate_without_date_predicate(self):
        plan = build_contextual_date_plan(self.bindings[1])
        columns = {
            "SALES.FACT_REVENUE": {"INVENTORY_DATE_KEY": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT SUM(f.AMOUNT) AS Revenue FROM SALES.FACT_REVENUE f "
            "LEFT JOIN SALES.DIM_DATE d ON f.INVENTORY_DATE_KEY=d.DATE_KEY"
        )
        result = validate_sql_detailed(
            sql,
            set(columns),
            "azure_sql",
            table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("FULL_DATE", result.reason)

    def test_validator_rejects_server_clock_for_relative_business_date(self):
        plan = build_contextual_date_plan(self.bindings[0], "what was revenue yesterday")
        columns = {
            "SALES.FACT_REVENUE": {"INVOICE_DATE_KEY": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT SUM(f.AMOUNT) AS Revenue FROM SALES.FACT_REVENUE f "
            "JOIN SALES.DIM_DATE d ON f.INVOICE_DATE_KEY=d.DATE_KEY "
            "WHERE d.FULL_DATE=DATEADD(day,-1,CAST(GETDATE() AS date))"
        )
        result = validate_sql_detailed(
            sql, set(columns), "azure_sql", table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "temporal_anchor_mismatch")

    def test_validator_accepts_latest_data_relative_anchor(self):
        plan = build_contextual_date_plan(self.bindings[0], "what was revenue yesterday")
        columns = {
            "SALES.FACT_REVENUE": {"INVOICE_DATE_KEY": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT COUNT(*) AS MatchedRows, COUNT(f.AMOUNT) AS NonNullRevenueRows, "
            "COALESCE(SUM(f.AMOUNT), 0) AS Revenue FROM SALES.FACT_REVENUE f "
            "JOIN SALES.DIM_DATE d ON f.INVOICE_DATE_KEY=d.DATE_KEY "
            "WHERE d.FULL_DATE=DATEADD(day,-1,("
            "SELECT MAX(d2.FULL_DATE) FROM SALES.FACT_REVENUE f2 "
            "JOIN SALES.DIM_DATE d2 ON f2.INVOICE_DATE_KEY=d2.DATE_KEY))"
        )
        result = validate_sql_detailed(
            sql, set(columns), "azure_sql", table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_validator_rejects_relative_window_without_latest_data_anchor(self):
        plan = build_contextual_date_plan(self.bindings[0], "revenue for the last 7 days")
        columns = {
            "SALES.FACT_REVENUE": {"INVOICE_DATE_KEY": "int", "AMOUNT": "decimal"},
            "SALES.DIM_DATE": {"DATE_KEY": "int", "FULL_DATE": "date"},
        }
        sql = (
            "SELECT COUNT(*) AS MatchedRows, COUNT(f.AMOUNT) AS NonNullRevenueRows, "
            "COALESCE(SUM(f.AMOUNT), 0) AS Revenue FROM SALES.FACT_REVENUE f "
            "JOIN SALES.DIM_DATE d ON f.INVOICE_DATE_KEY=d.DATE_KEY "
            "WHERE d.FULL_DATE >= '2025-01-01'"
        )
        result = validate_sql_detailed(
            sql, set(columns), "azure_sql", table_columns=columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "temporal_anchor_missing")

    def test_approved_date_role_replaces_competing_graph_edges_for_query(self):
        graph = {
            "entities": [
                {
                    "entity_name": "Revenue Fact", "entity_type": "fact",
                    "schema_name": "SALES", "table_name": "FACT_REVENUE",
                },
                {
                    "entity_name": "Date", "entity_type": "dimension",
                    "schema_name": "SALES", "table_name": "DIM_DATE",
                },
            ],
            "relationships": [
                {
                    "from_entity": "Revenue Fact", "to_entity": "Date",
                    "from_column": "INVOICE_DATE_KEY", "to_column": "DATE_KEY",
                    "join_type": "INNER",
                },
                {
                    "from_entity": "Revenue Fact", "to_entity": "Date",
                    "from_column": "ORDER_DATE_KEY", "to_column": "DATE_KEY",
                    "join_type": "INNER",
                },
            ],
        }
        binding = {
            **self.bindings[0],
            "fact_column": "BOOKED_DATE_KEY",
            "date_role": "booked_date",
        }
        scoped = _graph_with_exact_date_edges(graph, [binding])
        edges = scoped["relationships"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["from_column"], "BOOKED_DATE_KEY")
        self.assertEqual(edges[0]["to_column"], "DATE_KEY")
        self.assertEqual(edges[0]["generated_by"], "date_role")
        self.assertEqual(edges[0]["status"], "confirmed")


class DateRoleDiscoveryTests(unittest.TestCase):
    def _schema(self):
        return {
            "SALES.FACT_REVENUE": {
                "schema": "SALES",
                "table": "FACT_REVENUE",
                "columns": [
                    {"name": "CUS_IVC_DT_DMS_KEY", "type": "int"},
                    {"name": "AMOUNT", "type": "decimal"},
                ],
            },
            "SALES.DIM_DATE": {
                "schema": "SALES",
                "table": "DIM_DATE",
                "columns": [
                    {"name": "DATE_KEY", "type": "int"},
                    {"name": "FULL_DATE", "type": "date"},
                ],
            },
            "__db_fk_constraints__": [{
                "source": "azure_sql",
                "constraint_name": "FK_REVENUE_INVOICE_DATE",
                "parent_schema": "SALES",
                "parent_table": "FACT_REVENUE",
                "parent_col": "CUS_IVC_DT_DMS_KEY",
                "ref_schema": "SALES",
                "ref_table": "DIM_DATE",
                "ref_col": "DATE_KEY",
                "ordinal": 1,
                "enforced": True,
            }],
        }

    def test_declared_fk_discovers_date_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(self._schema()), encoding="utf-8")
            model = build_semantic_model(tmp)
        role = model["date_roles"][0]
        self.assertEqual(role["dimension_key"], "DATE_KEY")
        self.assertEqual(role["date_value_column"], "FULL_DATE")
        self.assertEqual(role["confidence"], 99)
        self.assertEqual(role["status"], "generated")
        self.assertEqual(role["date_key_type"], "surrogate_fk")

    def test_two_fact_date_keys_can_reference_same_dimension_key(self):
        schema = self._schema()
        schema["SALES.FACT_REVENUE"]["columns"] = [
            {"name": "BOOKED_DT_ID", "type": "int"},
            {"name": "ORDER_DT_ID", "type": "int"},
            {"name": "AMOUNT", "type": "decimal"},
        ]
        schema["__db_fk_constraints__"] = [
            {
                "source": "azure_sql", "constraint_name": "FK_BOOKED_DATE",
                "parent_schema": "SALES", "parent_table": "FACT_REVENUE",
                "parent_col": "BOOKED_DT_ID", "ref_schema": "SALES",
                "ref_table": "DIM_DATE", "ref_col": "DATE_KEY", "ordinal": 1,
            },
            {
                "source": "azure_sql", "constraint_name": "FK_ORDER_DATE",
                "parent_schema": "SALES", "parent_table": "FACT_REVENUE",
                "parent_col": "ORDER_DT_ID", "ref_schema": "SALES",
                "ref_table": "DIM_DATE", "ref_col": "DATE_KEY", "ordinal": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            model = build_semantic_model(tmp)
        roles = {role["fact_column"]: role for role in model["date_roles"]}
        self.assertEqual(set(roles), {"BOOKED_DT_ID", "ORDER_DT_ID"})
        self.assertEqual(roles["BOOKED_DT_ID"]["dimension_key"], "DATE_KEY")
        self.assertEqual(roles["ORDER_DT_ID"]["dimension_key"], "DATE_KEY")
        self.assertEqual(roles["BOOKED_DT_ID"]["business_role"], "booked_date")
        self.assertEqual(roles["ORDER_DT_ID"]["business_role"], "order_date")

    def test_manual_role_survives_rebuild(self):
        schema = self._schema()
        schema["SALES.FACT_REVENUE"]["columns"].append(
            {"name": "INVENTORY_PERIOD_KEY", "type": "int"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            schema_dir = Path(tmp, "schema")
            kb_dir = Path(tmp, "kb")
            schema_dir.mkdir()
            schema_dir.joinpath("_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            changed = patch_date_role(
                kb_dir=str(kb_dir),
                fact_table="SALES.FACT_REVENUE",
                fact_column="INVENTORY_PERIOD_KEY",
                dimension_table="SALES.DIM_DATE",
                dimension_key="DATE_KEY",
                date_value_column="FULL_DATE",
                business_role="inventory_sales_date",
                name="Inventory Sales Date",
                status="approved",
                create_if_missing=True,
            )
            self.assertTrue(changed)
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            model = load_semantic_model(str(kb_dir))
        role = next(r for r in model["date_roles"] if r["fact_column"] == "INVENTORY_PERIOD_KEY")
        self.assertEqual(role["business_role"], "inventory_sales_date")
        self.assertEqual(role["status"], "approved")


@contextmanager
def _memory_db(conn):
    yield conn


class DateContextStoreTests(unittest.TestCase):
    def test_store_is_tenant_scoped(self):
        from store.db import _SCHEMA
        import store.date_context_store as date_store

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO client(account_id, client_name, platform_type) VALUES ('a', 'A', 'web')")
        conn.execute("INSERT INTO client(account_id, client_name, platform_type) VALUES ('b', 'B', 'web')")
        conn.execute("INSERT INTO metric_registry(account_id, name, sql_template) VALUES ('a', 'Revenue', 'SUM(x)')")
        metric_id = conn.execute("SELECT id FROM metric_registry WHERE account_id='a'").fetchone()[0]
        factory = lambda: _memory_db(conn)
        with patch.object(date_store, "get_db", side_effect=factory):
            saved = date_store.save_metric_date_context(
                "a", _binding("Sales", "invoice_date", "INVOICE_DATE_KEY", default=True, metric_id=metric_id)
            )
            self.assertGreater(saved, 0)
            self.assertEqual(len(date_store.list_metric_date_contexts("a")), 1)
            self.assertEqual(date_store.list_metric_date_contexts("b"), [])
        conn.close()


if __name__ == "__main__":
    unittest.main()
