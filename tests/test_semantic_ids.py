"""core/semantic_ids.py — canonical semantic object identifiers.

This is the prerequisite Sprint 1 was missing before Sprint 2 (conflict
detectors) can start: a stable way to say "this exact metric / field / join"
that survives across compiles and can never accidentally merge two distinct
objects that happen to share a display name.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_ids import (  # noqa: E402
    conflict_key, date_role_id, entity_id, field_id, join_id, metric_id,
    parse_semantic_id, resolve_entity_table_fqn, table_id, term_id,
)


class MintingTests(unittest.TestCase):
    def test_ids_are_typed_and_stable(self):
        self.assertEqual(metric_id(42), "metric:42")
        self.assertEqual(metric_id("42"), "metric:42")  # DB rows hand back str or int
        self.assertEqual(term_id(7), "term:7")
        self.assertEqual(join_id(99), "join:99")

    def test_entity_and_field_ids_normalize_case_and_whitespace(self):
        # Table/column casing is inconsistent across discovery, admin edits,
        # and legacy data in this codebase — the id must not care.
        self.assertEqual(entity_id("fact_sales"), entity_id("FACT_SALES"))
        self.assertEqual(entity_id(" Fact_Sales "), entity_id("FACT_SALES"))
        self.assertEqual(
            field_id("erp.fact_sales", "net_amt"),
            field_id("ERP.FACT_SALES", "NET_AMT"),
        )

    def test_table_and_field_and_date_role_ids(self):
        self.assertEqual(table_id("ERP.FACT_SALES"), "table:ERP.FACT_SALES")
        self.assertEqual(
            field_id("ERP.FACT_SALES", "NET_AMT"), "field:ERP.FACT_SALES.NET_AMT",
        )
        self.assertEqual(
            date_role_id("ERP.FACT_SALES", "ORDER_DT_DMS_KEY"),
            "date_role:ERP.FACT_SALES.ORDER_DT_DMS_KEY",
        )


class DuplicateNameDistinctionTests(unittest.TestCase):
    """The one property the whole scheme exists for: two DB rows that share
    a display name (metric_registry has no UNIQUE constraint on name — see
    core/semantic_ids.py's module docstring) must never mint the same id,
    or a "duplicate metric name" conflict detector could never tell the two
    apart to report the collision in the first place."""

    def test_same_named_metrics_get_distinct_ids(self):
        self.assertNotEqual(metric_id(3), metric_id(9))

    def test_same_named_terms_get_distinct_ids(self):
        self.assertNotEqual(term_id(1), term_id(2))


class EntityTableFqnResolutionTests(unittest.TestCase):
    def test_resolves_schema_dot_table(self):
        entity = {"entity_name": "FACT_SALES", "schema_name": "ERP", "table_name": "FACT_SALES"}
        self.assertEqual(resolve_entity_table_fqn(entity), "ERP.FACT_SALES")

    def test_falls_back_to_bare_table_without_schema(self):
        entity = {"entity_name": "FACT_SALES", "schema_name": "", "table_name": "FACT_SALES"}
        self.assertEqual(resolve_entity_table_fqn(entity), "FACT_SALES")

    def test_missing_entity_fields_do_not_raise(self):
        self.assertEqual(resolve_entity_table_fqn({}), "")

    def test_graph_property_and_model_field_converge_on_same_id(self):
        # The actual reason this bridge exists: the entity graph
        # (entity_properties, keyed by entity_name) and the schema-derived
        # semantic model (keyed by table fqn) describe the SAME physical
        # column through two different admin surfaces.
        entity = {"entity_name": "FACT_SALES", "schema_name": "ERP", "table_name": "FACT_SALES"}
        from_graph_property = field_id(resolve_entity_table_fqn(entity), "NET_AMT")
        from_model_field = field_id("ERP.FACT_SALES", "NET_AMT")
        self.assertEqual(from_graph_property, from_model_field)


class ParseSemanticIdTests(unittest.TestCase):
    def test_round_trips_simple_ids(self):
        self.assertEqual(parse_semantic_id("metric:42"), ("metric", "42"))
        self.assertEqual(parse_semantic_id("entity:FACT_SALES"), ("entity", "FACT_SALES"))

    def test_field_id_key_keeps_its_internal_dot(self):
        # field/date_role keys are themselves "TABLE.COLUMN" — only the
        # FIRST colon (type separator) may split.
        object_type, key = parse_semantic_id("field:ERP.FACT_SALES.NET_AMT")
        self.assertEqual(object_type, "field")
        self.assertEqual(key, "ERP.FACT_SALES.NET_AMT")

    def test_malformed_ids_raise(self):
        for bad in ("", "no-colon-here", "unknowntype:123", ":empty-type", "metric:"):
            with self.assertRaises(ValueError, msg=bad):
                parse_semantic_id(bad)


class ConflictKeyTests(unittest.TestCase):
    def test_order_independent_for_reconciliation(self):
        # store.reconcile_semantic_conflicts() matches on this string across
        # compiles — discovery order of the two colliding objects can differ
        # run to run, so the key must not.
        a = conflict_key("duplicate_metric_name", "metric:9", "metric:3")
        b = conflict_key("duplicate_metric_name", "metric:3", "metric:9")
        self.assertEqual(a, b)

    def test_different_codes_or_participants_differ(self):
        base = conflict_key("duplicate_metric_name", "metric:3", "metric:9")
        self.assertNotEqual(base, conflict_key("other_code", "metric:3", "metric:9"))
        self.assertNotEqual(base, conflict_key("duplicate_metric_name", "metric:3", "metric:7"))

    def test_falsy_participants_are_dropped_not_stringified(self):
        # A detector that accidentally passes an unset id (None/"") must not
        # get a key containing the literal text "None".
        key = conflict_key("code", "metric:3", "", None)
        self.assertNotIn("None", key)
        self.assertEqual(key, "code::metric:3")

    def test_single_object_conflict(self):
        self.assertEqual(conflict_key("stale_reference", "field:X.Y"), "stale_reference::field:X.Y")


if __name__ == "__main__":
    unittest.main()
