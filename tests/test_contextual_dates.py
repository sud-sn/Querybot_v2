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
    format_required_anchor,
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

    def test_business_term_field_on_unrelated_graph_anchor_still_resolves_fact(self):
        # Live-bug reproduction: "revenue" is mapped as a Business Term field
        # to F_RX_FILL.NET_REVENUE_AMT (not a Metric), but the graph anchored
        # on an unrelated table (F_CLAIM, from KB-retrieval scoring on the
        # phrase "revenue trend" -- unrelated to the business-term mapping).
        # F_RX_FILL never appeared in the graph's "detected" set for THIS
        # resolution pass, so the field-loop's add_if_fact(source) (without
        # preserve_unknown=True) silently dropped it -- fact_scope ended up
        # {F_CLAIM} only, the admin-approved default role on F_RX_FILL never
        # got a chance to apply, and the whole query ran fully ungoverned.
        graph = _default_date_graph(include_claim=True)
        facts = _resolved_fact_tables(
            {"detected": ["Claim"], "anchor": "Claim"},
            graph,
            semantic_plan={
                "fields": [{
                    "term": "revenue", "table": "PHARMACY.F_RX_FILL",
                    "column": "NET_REVENUE_AMT", "source_table": "PHARMACY.F_RX_FILL",
                }],
            },
        )
        self.assertIn("PHARMACY.F_RX_FILL", facts)

    def test_business_term_field_fact_default_selected_despite_unrelated_graph_anchor(self):
        # End-to-end version of the same live bug: with the fact-scope fix,
        # resolve_contextual_date_binding must select the business-term
        # field's fact's admin-approved default role, not fall through to
        # "no governed date context" just because the graph anchored
        # elsewhere and no metric matched at all.
        roles = [{
            "name": "Fill Date", "business_role": "fill_date",
            "fact_table": "PHARMACY.F_RX_FILL", "fact_column": "FILL_DATE",
            "dimension_table": "", "dimension_key": "",
            "date_value_column": "FILL_DATE", "date_key_type": "native_date",
            "status": "approved", "is_default": True,
        }]
        graph = _default_date_graph(include_claim=True)
        fact_scope = _resolved_fact_tables(
            {"detected": ["Claim"], "anchor": "Claim"},
            graph,
            semantic_plan={
                "fields": [{
                    "term": "revenue", "table": "PHARMACY.F_RX_FILL",
                    "column": "NET_REVENUE_AMT", "source_table": "PHARMACY.F_RX_FILL",
                }],
            },
        )
        result = resolve_contextual_date_binding(
            "show my revenue trend for the last 7 days",
            matched_metrics=[],
            bindings=[],
            date_roles=roles,
            required_fact_tables=fact_scope,
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["binding"]["fact_column"], "FILL_DATE")
        self.assertEqual(result["reason"], "default date role for resolved fact")

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

    def test_rolling_window_recognizes_latest_as_synonym_for_last(self):
        # Regression: "in the latest 7 days" (a natural, common phrasing)
        # was silently NOT recognized as relative-date wording -- only
        # "last/past/previous N days" was. That meant temporal_policies
        # never got populated for a question phrased with "latest", which
        # in turn meant the temporal_anchor_missing/temporal_role_mismatch
        # governance checks in core/validator.py never ran at all for that
        # question, letting SQL that used the wrong date-role column slip
        # past to a vague field_plan_mismatch instead of the specific,
        # actionable temporal check.
        window = detect_temporal_window(
            "Which pharmacies had the most prescription fills in the latest 7 days?"
        )
        self.assertEqual(window["kind"], "last_n")
        self.assertEqual(window["amount"], 7)
        self.assertEqual(window["unit"], "day")
        self.assertEqual(window["anchor_policy"], "latest_available")

    def test_rolling_window_recognizes_latest_across_units(self):
        for unit in ("day", "week", "month", "quarter", "year"):
            with self.subTest(unit=unit):
                window = detect_temporal_window(f"show revenue for the latest 3 {unit}s")
                self.assertEqual(window["kind"], "last_n")
                self.assertEqual(window["amount"], 3)
                self.assertEqual(window["unit"], unit)

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

    def test_validator_rejects_relative_window_anchored_on_unjoined_dimension(self):
        # A calendar dimension commonly carries future rows with no matching
        # fact data -- MAX() over SALES.DIM_DATE alone (no join back to the
        # fact table) can anchor to a date with zero fact rows. This is the
        # exact anti-pattern _temporal_anchor_scope_errors now rejects.
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
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "temporal_anchor_unscoped")

    def test_validator_accepts_relative_window_anchored_on_joined_dimension(self):
        # Corrected version of the query above: the anchor subquery now
        # joins back to the fact table (mirroring what
        # format_required_anchor produces for a surrogate_fk role), so the
        # calendar value is real but scoped to rows with an actual match.
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
            "(SELECT MAX(d2.FULL_DATE) FROM SALES.DIM_DATE d2 "
            "JOIN SALES.FACT_REVENUE f2 ON f2.ORDER_DATE_ID = d2.DATE_KEY))"
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


class TemporalGovernanceHardeningTests(unittest.TestCase):
    """Live-failure hardening: the LLM substituted DISPENSE_DATE_ID +
    MAX(CALENDAR_DATE)-over-unrestricted-D_DATE for the governed FILL_DATE
    policy; the repaired query then failed at execution (DB paused) and
    that second failure was hidden behind the stale validation message."""

    PHARMA_COLUMNS = {
        "PHARMA_LAB.F_RX_FILL": {
            "DISPENSE_DATE_ID": "int", "FILL_DATE": "date",
            "NET_REVENUE_AMT": "decimal", "PRESCRIBER_ID": "int",
        },
        "PHARMA_LAB.D_PRESCRIBER": {"PRESCRIBER_ID": "int", "SPECIALTY_NAME": "varchar"},
        "PHARMA_LAB.D_DATE": {"DATE_ID": "int", "CALENDAR_DATE": "date"},
    }
    FILL_DATE_PLAN = {
        "enabled": True,
        # format_semantic_field_plan early-returns on an empty fields list,
        # so carry the governed date field the real plan would carry.
        "fields": [{
            "term": "Fill Date",
            "table": "PHARMA_LAB.F_RX_FILL",
            "column": "FILL_DATE",
            "role": "date_dimension",
            "enforcement": "optional",
        }],
        "joins": [],
        "required_tables": [],
        "temporal_policies": [{
            "kind": "this_month",
            "anchor_policy": "latest_available",
            "fact_table": "PHARMA_LAB.F_RX_FILL",
            "fact_column": "FILL_DATE",
            "date_table": "PHARMA_LAB.F_RX_FILL",
            "date_column": "FILL_DATE",
            "dimension_table": "",
            "dimension_key": "",
            "date_key_type": "native_date",
            "business_role": "fill_date",
        }],
    }

    def test_live_bug_sql_flags_alternate_date_role(self):
        sql = (
            "SELECT dpr.SPECIALTY_NAME, SUM(frx.NET_REVENUE_AMT) AS TOTAL "
            "FROM PHARMA_LAB.F_RX_FILL frx "
            "JOIN PHARMA_LAB.D_PRESCRIBER dpr ON frx.PRESCRIBER_ID = dpr.PRESCRIBER_ID "
            "WHERE frx.DISPENSE_DATE_ID IN ("
            "  SELECT DATE_ID FROM PHARMA_LAB.D_DATE "
            "  WHERE MONTH(CALENDAR_DATE) = MONTH((SELECT MAX(CALENDAR_DATE) FROM PHARMA_LAB.D_DATE))"
            ") GROUP BY dpr.SPECIALTY_NAME"
        )
        result = validate_sql_detailed(
            sql, set(self.PHARMA_COLUMNS), "azure_sql",
            table_columns=self.PHARMA_COLUMNS,
            semantic_context={"semantic_plan": self.FILL_DATE_PLAN},
        )
        self.assertFalse(result.ok)
        codes = {e.get("code") for e in (result.errors or [])}
        self.assertIn("temporal_role_mismatch", codes)
        # Both the substituted fact key and the un-governed dimension PK get
        # flagged — the substituted role key must be among them.
        role_columns = {
            e.get("column") for e in result.errors
            if e.get("code") == "temporal_role_mismatch"
        }
        self.assertIn("DISPENSE_DATE_ID", role_columns)

    def test_live_bug_reproduced_end_to_end_via_latest_n_days_phrasing(self):
        """
        Full regression for the actual live failure: "...in the latest 7
        days?" was never recognized as relative-date wording by
        detect_temporal_window() (only "last/past/previous N days" was),
        so temporal_policies never got compiled for this question and the
        specific temporal_anchor_missing/temporal_role_mismatch governance
        check in core/validator.py never ran -- letting SQL anchored on
        DISPENSE_DATE_ID/D_DATE instead of the approved FILL_DATE fall
        through to a vague, unhelpful field_plan_mismatch instead.

        Builds the plan through the REAL pipeline (detect_temporal_window +
        build_contextual_date_plan from the actual question text), unlike
        FILL_DATE_PLAN above which hand-constructs temporal_policies and so
        never exercised the question-parsing step that actually broke.
        """
        question = "Which pharmacies had the most prescription fills in the latest 7 days?"
        binding = {
            "fact_table": "PHARMA_LAB.F_RX_FILL",
            "fact_column": "FILL_DATE",
            "dimension_table": "",
            "dimension_key": "",
            "date_value_column": "",
            "date_key_type": "native_date",
            "context_name": "Fill Date",
            "date_role": "fill_date",
            "is_default": 1,
            "resolution_source": "metric_default",
            "governance_status": "",
        }
        plan = build_contextual_date_plan(binding, question)
        self.assertTrue(plan.get("temporal_policies"), "latest N days must compile a temporal policy")

        bad_sql = (
            "WITH latest_date AS ("
            "  SELECT MAX(dd.CALENDAR_DATE) AS MAX_DATE FROM PHARMA_LAB.D_DATE dd"
            "), recent_fills AS ("
            "  SELECT frx.PHARMACY_ID, COUNT(frx.RX_FILL_ID) AS TOTAL_FILLS"
            "  FROM PHARMA_LAB.F_RX_FILL frx"
            "  INNER JOIN PHARMA_LAB.D_DATE dd ON frx.DISPENSE_DATE_ID = dd.DATE_ID"
            "  WHERE dd.CALENDAR_DATE >= DATEADD(DAY, -7, (SELECT MAX_DATE FROM latest_date))"
            "    AND dd.CALENDAR_DATE <= (SELECT MAX_DATE FROM latest_date)"
            "  GROUP BY frx.PHARMACY_ID"
            ") SELECT dp.PHARMACY_NAME, rf.TOTAL_FILLS FROM recent_fills rf "
            "JOIN PHARMA_LAB.D_PHARMACY dp ON rf.PHARMACY_ID = dp.PHARMACY_ID "
            "ORDER BY rf.TOTAL_FILLS DESC"
        )
        table_columns = dict(self.PHARMA_COLUMNS)
        table_columns["PHARMA_LAB.D_PHARMACY"] = {"PHARMACY_ID": "int", "PHARMACY_NAME": "varchar"}
        table_columns["PHARMA_LAB.F_RX_FILL"] = {
            **self.PHARMA_COLUMNS["PHARMA_LAB.F_RX_FILL"],
            "PHARMACY_ID": "int", "RX_FILL_ID": "int",
        }

        result = validate_sql_detailed(
            bad_sql, set(table_columns), "azure_sql",
            table_columns=table_columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        # Must be caught by the specific temporal check, not silently pass
        # and not fall through to the generic field_plan_mismatch path.
        self.assertIn(result.code, ("temporal_anchor_missing", "temporal_role_mismatch"))

        from core.semantic_model import build_field_plan_repair_note
        note = build_field_plan_repair_note(plan)
        self.assertIn("FILL_DATE", note)
        self.assertIn("PHARMA_LAB.F_RX_FILL", note)

    def test_approved_native_date_anchor_passes(self):
        sql = (
            "SELECT dpr.SPECIALTY_NAME, SUM(frx.NET_REVENUE_AMT) AS TOTAL "
            "FROM PHARMA_LAB.F_RX_FILL frx "
            "JOIN PHARMA_LAB.D_PRESCRIBER dpr ON frx.PRESCRIBER_ID = dpr.PRESCRIBER_ID "
            "WHERE frx.FILL_DATE >= DATEADD(day, 1, EOMONTH((SELECT MAX(FILL_DATE) FROM PHARMA_LAB.F_RX_FILL), -1)) "
            "GROUP BY dpr.SPECIALTY_NAME"
        )
        result = validate_sql_detailed(
            sql, set(self.PHARMA_COLUMNS), "azure_sql",
            table_columns=self.PHARMA_COLUMNS,
            semantic_context={"semantic_plan": self.FILL_DATE_PLAN},
        )
        self.assertTrue(result.ok, result.reason)

    def test_prompt_block_carries_required_anchor_subquery(self):
        from core.semantic_planner import format_semantic_field_plan
        text = format_semantic_field_plan(self.FILL_DATE_PLAN, "azure_sql")
        self.assertIn("REQUIRED ANCHOR", text)
        self.assertIn("(SELECT MAX(FILL_DATE) FROM PHARMA_LAB.F_RX_FILL)", text)

    def test_temporal_codes_have_business_failure_messages(self):
        from core.failure_messages import translate_failure
        generic = "The generated query did not pass QueryBot's safety and accuracy checks."
        for code in (
            "temporal_anchor_missing", "temporal_anchor_mismatch",
            "temporal_role_mismatch", "temporal_anchor_unscoped",
        ):
            with self.subTest(code=code):
                rca = translate_failure(kind="validation", code=code, reason="x")
                self.assertNotEqual(rca["most_likely_reason"], generic)

    def test_stale_surrogate_date_example_is_dropped_from_prompt(self):
        from core.examples import _is_stale_surrogate_date_example, format_examples_for_prompt
        poisoned = (
            "SELECT SUM(NET_REVENUE_AMT) FROM PHARMA_LAB.F_RX_FILL "
            "WHERE YEAR(TRY_CONVERT(date, CONVERT(varchar(8), DISPENSE_DATE_ID), 112)) = 2026"
        )
        clean = (
            "SELECT SUM(NET_REVENUE_AMT) FROM PHARMA_LAB.F_RX_FILL "
            "WHERE FILL_DATE >= '2026-01-01'"
        )
        self.assertTrue(_is_stale_surrogate_date_example(poisoned))
        self.assertFalse(_is_stale_surrogate_date_example(clean))
        rendered = format_examples_for_prompt([
            {"question": "poisoned", "sql": poisoned},
            {"question": "clean", "sql": clean},
        ])
        self.assertNotIn("TRY_CONVERT", rendered)
        self.assertIn("FILL_DATE", rendered)

    def test_bare_max_surrogate_anchor_example_is_also_dropped_from_prompt(self):
        # Live-bug reproduction: a harvested/approved example carrying the
        # "SNAPSHOT_DATE_ID = (SELECT MAX(SNAPSHOT_DATE_ID) FROM fact)"
        # anti-pattern -- no YEAR/CONVERT/CAST/DATEADD/DATEDIFF wrapping at
        # all, so _surrogate_date_misuse_columns (which only looks inside
        # those function calls) never caught it. Two independent
        # repair-note fixes for this exact live case had zero effect on
        # the regenerated SQL because a poisoned few-shot example was
        # still being injected into both the first-pass and retry prompts,
        # outweighing the text guidance every time.
        from core.examples import _is_stale_surrogate_date_example, format_examples_for_prompt
        poisoned = (
            "SELECT dsu.SUPPLIER_NAME, SUM(fin.AVAILABLE_QUANTITY) AS TOTAL_AVAILABLE "
            "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin "
            "INNER JOIN PHARMA_LAB.D_SUPPLIER dsu ON fin.SUPPLIER_ID = dsu.SUPPLIER_ID "
            "WHERE fin.SNAPSHOT_DATE_ID = (SELECT MAX(SNAPSHOT_DATE_ID) FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT) "
            "GROUP BY dsu.SUPPLIER_NAME"
        )
        correct = (
            "SELECT dsu.SUPPLIER_NAME, SUM(fin.AVAILABLE_QUANTITY) AS TOTAL_AVAILABLE "
            "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin "
            "INNER JOIN PHARMA_LAB.D_SUPPLIER dsu ON fin.SUPPLIER_ID = dsu.SUPPLIER_ID "
            "INNER JOIN PHARMA_LAB.D_DATE dt ON fin.SNAPSHOT_DATE_ID = dt.DATE_ID "
            "WHERE dt.CALENDAR_DATE = (SELECT MAX(d2.CALENDAR_DATE) FROM PHARMA_LAB.D_DATE d2 "
            "JOIN PHARMA_LAB.F_INVENTORY_SNAPSHOT f2 ON f2.SNAPSHOT_DATE_ID = d2.DATE_ID) "
            "GROUP BY dsu.SUPPLIER_NAME"
        )
        self.assertTrue(_is_stale_surrogate_date_example(poisoned))
        self.assertFalse(_is_stale_surrogate_date_example(correct))
        rendered = format_examples_for_prompt([
            {"question": "poisoned bare-max", "sql": poisoned},
            {"question": "correct join-scoped", "sql": correct},
        ])
        self.assertNotIn("poisoned bare-max", rendered)
        self.assertIn("correct join-scoped", rendered)

    def test_bare_max_on_native_date_column_is_not_flagged(self):
        # A plain native calendar column (no _ID/_KEY suffix) anchored via
        # MAX() with no JOIN is the CORRECT pattern for that shape (mirrors
        # core/report_engine.py's own _apply_latest_date_filter) -- must
        # not be caught by the new surrogate-only check.
        from core.examples import _is_stale_surrogate_date_example
        native = (
            "SELECT SUM(NET_REVENUE_AMT) FROM PHARMA_LAB.F_RX_FILL "
            "WHERE FILL_DATE = (SELECT MAX(FILL_DATE) FROM PHARMA_LAB.F_RX_FILL)"
        )
        self.assertFalse(_is_stale_surrogate_date_example(native))

    def test_pipeline_reports_retry_execution_failure_honestly(self):
        # Wiring guards for the hidden-second-failure fix: the validation
        # terminal branch must yield to a real execution error from the
        # repaired query, and the history note must be present.
        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("if not ok and exec_error is None:", src)
        self.assertIn("A repaired query passed validation but failed to execute.", src)
        self.assertIn('"temporal_role_mismatch"', src)

    def test_governed_store_drops_stale_when_fresh_fill_request(self):
        src = (Path(__file__).resolve().parents[1] / "core" / "governed_store.py").read_text(encoding="utf-8")
        self.assertIn("parsed = fresh + stale if len(fresh) < n else fresh", src)


class SurrogateFkAnchorSourceTests(unittest.TestCase):
    """
    The actual root cause of "revenue yesterday" returning nothing for a
    role-playing (surrogate-FK) date: build_contextual_date_plan compiled
    the anchor against the raw, unrestricted date DIMENSION table (e.g.
    D_DATE) instead of the governed FACT table (e.g. F_RX_FILL) -- a
    calendar dimension commonly carries future rows with no matching fact
    data, so anchoring there silently yields a date with zero rows. Native
    dates were never affected (no dimension involved); every existing
    TemporalGovernanceHardeningTests fixture is native, which is why this
    shipped unnoticed.
    """

    SURROGATE_BINDING = {
        "fact_table": "PHARMA_LAB.F_RX_FILL",
        "fact_column": "DISPENSE_DATE_ID",
        "dimension_table": "PHARMA_LAB.D_DATE",
        "dimension_key": "DATE_ID",
        "date_value_column": "CALENDAR_DATE",
        "date_key_type": "surrogate_fk",
        "context_name": "Dispense Date",
        "date_role": "dispense_date",
        "is_default": 1,
    }

    def test_surrogate_fk_plan_carries_fact_scoped_anchor_fields(self):
        plan = build_contextual_date_plan(self.SURROGATE_BINDING, "what were fills yesterday")
        policy = plan["temporal_policies"][0]
        self.assertEqual(policy["anchor_table"], "PHARMA_LAB.F_RX_FILL")
        self.assertEqual(policy["anchor_column"], "DISPENSE_DATE_ID")
        # date_table/date_column stay dimension-side -- still needed for the
        # filter clause, which needs a real calendar value -- only the
        # anchor derivation changes.
        self.assertEqual(policy["date_table"], "PHARMA_LAB.D_DATE")

    def test_format_required_anchor_joins_fact_to_dimension_for_surrogate_fk(self):
        plan = build_contextual_date_plan(self.SURROGATE_BINDING, "what were fills yesterday")
        anchor = format_required_anchor(plan["temporal_policies"][0])
        self.assertIn("MAX(PHARMA_LAB.D_DATE.CALENDAR_DATE)", anchor)
        self.assertIn("JOIN PHARMA_LAB.F_RX_FILL", anchor)
        self.assertIn(
            "PHARMA_LAB.F_RX_FILL.DISPENSE_DATE_ID = PHARMA_LAB.D_DATE.DATE_ID", anchor,
        )
        # The literal reported anti-pattern must never appear.
        self.assertNotIn("FROM PHARMA_LAB.D_DATE)", anchor)

    def test_format_required_anchor_native_date_unchanged(self):
        native_plan = build_contextual_date_plan({
            "fact_table": "PHARMA_LAB.F_RX_FILL", "fact_column": "FILL_DATE",
            "dimension_table": "", "dimension_key": "", "date_value_column": "",
            "date_key_type": "native_date", "context_name": "Fill Date",
            "date_role": "fill_date", "is_default": 1,
        }, "what was revenue yesterday")
        anchor = format_required_anchor(native_plan["temporal_policies"][0])
        self.assertEqual(anchor, "(SELECT MAX(FILL_DATE) FROM PHARMA_LAB.F_RX_FILL)")

    def test_prompt_uses_fact_scoped_anchor_for_surrogate_fk(self):
        from core.semantic_planner import format_semantic_field_plan

        plan = build_contextual_date_plan(self.SURROGATE_BINDING, "what were fills yesterday")
        plan["fields"] = [{
            "term": "Dispense Date", "table": "PHARMA_LAB.D_DATE", "column": "CALENDAR_DATE",
            "role": "date_dimension", "enforcement": "optional",
        }]
        text = format_semantic_field_plan(plan, "azure_sql")
        self.assertIn("JOIN PHARMA_LAB.F_RX_FILL", text)
        self.assertNotIn("FROM PHARMA_LAB.D_DATE)", text)


class TemporalAnchorScopeValidatorTests(unittest.TestCase):
    """core.validator._temporal_anchor_scope_errors -- closes the gap where
    _temporal_anchor_errors only checked that the approved column NAME
    appeared under some MAX() node anywhere in the tree, never which table
    that MAX was scoped to."""

    POLICIES = [{
        "anchor_policy": "latest_available",
        "date_column": "CALENDAR_DATE",
        "anchor_table": "PHARMA_LAB.F_RX_FILL",
        "fact_table": "PHARMA_LAB.F_RX_FILL",
    }]

    def _errors(self, sql: str):
        import sqlglot
        from core.validator import _temporal_anchor_scope_errors
        tree = sqlglot.parse_one(sql, read="tsql")
        return _temporal_anchor_scope_errors(tree, self.POLICIES)

    def test_unscoped_dimension_anchor_is_rejected(self):
        errors = self._errors(
            "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_FILL f "
            "JOIN PHARMA_LAB.D_DATE d ON f.DISPENSE_DATE_ID = d.DATE_ID "
            "WHERE d.CALENDAR_DATE = (SELECT MAX(CALENDAR_DATE) FROM PHARMA_LAB.D_DATE)"
        )
        codes = {e["code"] for e in errors}
        self.assertIn("temporal_anchor_unscoped", codes)

    def test_join_scoped_anchor_passes(self):
        errors = self._errors(
            "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_FILL f WHERE f.FILL_DATE = "
            "(SELECT MAX(d2.CALENDAR_DATE) FROM PHARMA_LAB.D_DATE d2 "
            "JOIN PHARMA_LAB.F_RX_FILL f2 ON f2.DISPENSE_DATE_ID = d2.DATE_ID)"
        )
        self.assertEqual(errors, [])

    def test_cte_governed_anchor_shape_resolves_through_alias(self):
        # Matches the user's own reported-correct query shape: a `governed`
        # CTE filters fact rows, an `anchor` CTE takes MAX over `governed`
        # (a CTE alias, not a literal table) -- must resolve back to the
        # real fact table, not be flagged as unscoped.
        errors = self._errors(
            "WITH governed AS ("
            "  SELECT CALENDAR_DATE FROM PHARMA_LAB.F_RX_FILL WHERE CALENDAR_DATE IS NOT NULL"
            "), anchor AS ("
            "  SELECT MAX(CALENDAR_DATE) AS as_of_date FROM governed"
            ") SELECT * FROM governed CROSS JOIN anchor"
        )
        self.assertEqual(errors, [])

    def test_no_governed_policy_returns_no_errors(self):
        from core.validator import _temporal_anchor_scope_errors
        import sqlglot
        tree = sqlglot.parse_one("SELECT MAX(CALENDAR_DATE) FROM PHARMA_LAB.D_DATE", read="tsql")
        self.assertEqual(_temporal_anchor_scope_errors(tree, []), [])

    def test_wired_into_validate_sql_detailed(self):
        sql = (
            "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_FILL f "
            "JOIN PHARMA_LAB.D_DATE d ON f.DISPENSE_DATE_ID = d.DATE_ID "
            "WHERE d.CALENDAR_DATE = (SELECT MAX(CALENDAR_DATE) FROM PHARMA_LAB.D_DATE)"
        )
        plan = {
            "enabled": True,
            "fields": [{"term": "Dispense Date", "table": "PHARMA_LAB.D_DATE",
                        "column": "CALENDAR_DATE", "role": "date_dimension",
                        "enforcement": "optional"}],
            "joins": [], "required_tables": [],
            "temporal_policies": self.POLICIES,
        }
        table_columns = {
            "PHARMA_LAB.F_RX_FILL": {"DISPENSE_DATE_ID": "int"},
            "PHARMA_LAB.D_DATE": {"DATE_ID": "int", "CALENDAR_DATE": "date"},
        }
        result = validate_sql_detailed(
            sql, set(table_columns), "azure_sql",
            table_columns=table_columns,
            semantic_context={"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "temporal_anchor_unscoped")

    def test_temporal_anchor_unscoped_gets_a_repair_retry(self):
        # A query rejected by this new check must get the same one repair
        # attempt every other temporal_anchor_* code already gets -- found
        # missing from query_pipeline.py's retryable-codes set while
        # investigating a live "why no answer" report (the query would
        # otherwise go straight to a terminal failure with zero retries).
        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        retryable_line = next(line for line in src.splitlines() if line.strip().startswith("retryable ="))
        self.assertIn('"temporal_anchor_unscoped"', retryable_line)


class GraphPlanMismatchBeatsFieldPlanMismatchTests(unittest.TestCase):
    """Live-bug reproduction: for a surrogate-FK date role that is ALSO an
    entity-graph edge, validate_sql_detailed's check order means
    _graph_plan_errors fires (code=graph_plan_mismatch) before the
    field_plan_mismatch check ever runs. The temporal_anchor_* and
    field_plan_mismatch repair notes built in query_pipeline.py were never
    reached for this exact shape -- the LLM only got the generic
    "copy the FROM/JOIN skeleton" entity-graph note, regressed to the
    fact-only surrogate-key anti-pattern on retry, and then failed a SECOND,
    non-retryable-again check. Confirmed by reproducing the user's reported
    SQL/scenario directly (not a synthetic simplification)."""

    def _semantic_context(self):
        temporal_policy = {
            "fact_table": "PHARMA_LAB.F_INVENTORY_SNAPSHOT",
            "fact_column": "SNAPSHOT_DATE_ID",
            "dimension_table": "PHARMA_LAB.D_DATE",
            "dimension_key": "DATE_ID",
            "date_column": "CALENDAR_DATE",
            "date_key_type": "surrogate_fk",
            "anchor_table": "PHARMA_LAB.F_INVENTORY_SNAPSHOT",
            "anchor_column": "SNAPSHOT_DATE_ID",
        }
        semantic_plan = {
            "enabled": True,
            "fields": [{
                "term": "Snapshot Date", "table": "PHARMA_LAB.D_DATE", "column": "CALENDAR_DATE",
                "role": "contextual_date", "display_required": False,
                "source_table": "PHARMA_LAB.F_INVENTORY_SNAPSHOT", "source_key_column": "SNAPSHOT_DATE_ID",
                "enforcement": "required", "date_key_type": "surrogate_fk", "role_alias": "snapshot_date",
            }],
            "joins": [{
                "from": "PHARMA_LAB.F_INVENTORY_SNAPSHOT", "to": "PHARMA_LAB.D_DATE",
                "conditions": [("SNAPSHOT_DATE_ID", "DATE_ID")], "role_alias": "snapshot_date",
            }],
            "temporal_policies": [temporal_policy],
        }
        graph_context = {
            "enabled": True,
            "resolved_edges": [{
                "from_schema": "PHARMA_LAB", "from_table": "F_INVENTORY_SNAPSHOT",
                "to_schema": "PHARMA_LAB", "to_table": "D_DATE",
                "conditions": [("SNAPSHOT_DATE_ID", "DATE_ID")],
                "join_type": "INNER", "id": "edge1", "relationship_key": "snap-date",
            }],
        }
        return {"semantic_plan": semantic_plan, "graph_context": graph_context}, semantic_plan

    def test_naive_anchor_fails_as_graph_plan_mismatch_not_field_plan_mismatch(self):
        sql = (
            "SELECT SUM(INVENTORY_VALUE_AMT) AS TOTAL_INVENTORY_VALUE\n"
            "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT\n"
            "WHERE SNAPSHOT_DATE_ID = (SELECT MAX(SNAPSHOT_DATE_ID) FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT);"
        )
        table_columns = {
            "PHARMA_LAB.F_INVENTORY_SNAPSHOT": {"SNAPSHOT_DATE_ID": "int", "INVENTORY_VALUE_AMT": "decimal"},
            "PHARMA_LAB.D_DATE": {"DATE_ID": "int", "CALENDAR_DATE": "date"},
        }
        semantic_context, _ = self._semantic_context()
        result = validate_sql_detailed(
            sql, set(table_columns), "azure_sql", None, table_columns, semantic_context,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")
        self.assertIn("PHARMA_LAB.D_DATE", result.reason)

    def test_graph_plan_mismatch_branch_appends_date_anchor_when_edge_is_the_date_role(self):
        # query_pipeline.py's graph_plan_mismatch repair note must detect that
        # a missing edge names the governed date dimension and append the
        # same copy-pasteable REQUIRED ANCHOR guidance the temporal_anchor_*
        # path uses -- otherwise this exact case gets zero anchor guidance
        # on its only retry attempt.
        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        graph_branch_start = src.index('elif last_code == "graph_plan_mismatch":')
        next_branch_start = src.index("elif last_code ==", graph_branch_start + 10)
        graph_branch = src[graph_branch_start:next_branch_start]
        self.assertIn("_governed_date_anchor_repair_lines", graph_branch)
        self.assertIn("REQUIRED ANCHOR", str(
            __import__("core.query_pipeline", fromlist=["_governed_date_anchor_repair_lines"])
            ._governed_date_anchor_repair_lines(self._semantic_context()[1])
        ))


class GovernedDateAnchorRepairLinesTests(unittest.TestCase):
    """core.query_pipeline._governed_date_anchor_repair_lines -- the shared
    anchor-guidance builder now reused by both the temporal_anchor_* repair
    path and the graph_plan_mismatch repair path."""

    def test_surrogate_fk_policy_produces_join_scoped_anchor(self):
        from core.query_pipeline import _governed_date_anchor_repair_lines

        plan = {"temporal_policies": [{
            "fact_table": "PHARMA_LAB.F_INVENTORY_SNAPSHOT",
            "fact_column": "SNAPSHOT_DATE_ID",
            "dimension_table": "PHARMA_LAB.D_DATE",
            "dimension_key": "DATE_ID",
            "date_column": "CALENDAR_DATE",
            "date_key_type": "surrogate_fk",
        }]}
        lines = _governed_date_anchor_repair_lines(plan)
        self.assertIn("REQUIRED ANCHOR", lines)
        self.assertIn(
            "(SELECT MAX(PHARMA_LAB.D_DATE.CALENDAR_DATE) FROM PHARMA_LAB.D_DATE "
            "JOIN PHARMA_LAB.F_INVENTORY_SNAPSHOT ON "
            "PHARMA_LAB.F_INVENTORY_SNAPSHOT.SNAPSHOT_DATE_ID = PHARMA_LAB.D_DATE.DATE_ID)",
            lines,
        )

    def test_no_temporal_policies_returns_empty_string(self):
        from core.query_pipeline import _governed_date_anchor_repair_lines
        self.assertEqual(_governed_date_anchor_repair_lines({}), "")
        self.assertEqual(_governed_date_anchor_repair_lines({"temporal_policies": []}), "")


if __name__ == "__main__":
    unittest.main()
