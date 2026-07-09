"""
tests/test_value_index.py

Per-client value index for literal grounding:
  1. Filterable-column selection (display/code on dims; masked/PII skipped)
  2. Build: normalization, cap/truncation, value-level PII drop, hygiene
     rejection (newlines/length), atomic write, meta stats
  3. Lookups: exact, normalized, fuzzy thresholds, allowed-table scoping
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.value_index import (
    build_value_index, index_exists, load_index_stats,
    lookup_exact, lookup_fuzzy, normalize_value,
    select_filterable_columns, value_index_enabled,
)


def _schema():
    return {
        "EMCODW.EMDW_DMART.CUS_DMS": {
            "columns": [
                {"name": "CUS_DMS_KEY", "type": "int"},
                {"name": "CUS_NM", "type": "varchar(100)"},
                {"name": "CUS_EMAIL", "type": "varchar(100)"},   # PII by name
            ],
            "schema": "EMDW_DMART", "database": "EMCODW",
            "masked_fields": [], "mask_mode": "partial",
        },
        "EMCODW.EMDW_DMART.ITM_DMS": {
            "columns": [
                {"name": "ITM_DSC", "type": "varchar(200)"},
                {"name": "ITM_SECRET", "type": "varchar(50)"},
            ],
            "schema": "EMDW_DMART", "database": "EMCODW",
            "masked_fields": ["ITM_SECRET"], "mask_mode": "partial",
        },
        "EMCODW.EMDW_DMART.HR_DMS": {
            "columns": [{"name": "EMP_NM", "type": "varchar(100)"}],
            "schema": "EMDW_DMART", "database": "EMCODW",
            "masked_fields": [], "mask_mode": "all",              # whole table masked
        },
        "EMCODW.EMDW_DMART.FNN_FCT": {
            "columns": [{"name": "PAY_AMT", "type": "decimal"}],
            "schema": "EMDW_DMART", "database": "EMCODW",
            "masked_fields": [], "mask_mode": "partial",
        },
    }


def _fake_run_query(creds, db_type, sql, max_rows=200):
    if "CUS_NM" in sql:
        return [{"CUS_NM": v} for v in [
            "EMCO Corporation", "EMCO Corp EU", "EMCO Corp USA",
            "Acme Industries", "Beta Traders",
            "bad\nvalue", "x" * 300,
        ]]
    if "ITM_DSC" in sql:
        return [{"ITM_DSC": v} for v in ["STEEL ROD 10MM", "Copper Wire"]]
    return []


class SelectFilterableColumnsTests(unittest.TestCase):
    def test_display_columns_on_dims_selected_pii_and_masked_skipped(self):
        cols = select_filterable_columns(_schema())
        selected = {(c["table_fqn"].split(".")[-1], c["column"]) for c in cols}
        self.assertIn(("CUS_DMS", "CUS_NM"), selected)
        self.assertIn(("ITM_DMS", "ITM_DSC"), selected)
        # PII name pattern, masked field, mask_mode=all table, numeric fact col
        self.assertNotIn(("CUS_DMS", "CUS_EMAIL"), selected)
        self.assertNotIn(("ITM_DMS", "ITM_SECRET"), selected)
        self.assertNotIn(("HR_DMS", "EMP_NM"), selected)

    def test_status_type_group_columns_indexed_even_on_fact_tables(self):
        # Regression: _STS/_TYP/_GRP columns carry the values users filter by
        # ("how many orders are cancelled") and live on FACT tables as often
        # as dimensions — but _FILTERABLE_ROLES only covered display/code on
        # dimension tables, so status values were never indexed and got no
        # grounding and no zero-row explanation.
        schema = {
            "EMCODW.EMDW_DMART.ORD_FCT": {
                "columns": [
                    {"name": "ORD_STS", "type": "varchar(20)"},
                    {"name": "ORD_TYP", "type": "varchar(20)"},
                    {"name": "ITM_GRP", "type": "varchar(20)"},
                    {"name": "ORD_NM", "type": "varchar(100)"},  # display on a FACT: excluded
                ],
                "schema": "EMDW_DMART", "database": "EMCODW",
                "masked_fields": [], "mask_mode": "partial",
            },
        }
        selected = {c["column"] for c in select_filterable_columns(schema)}
        self.assertIn("ORD_STS", selected)
        self.assertIn("ORD_TYP", selected)
        self.assertIn("ITM_GRP", selected)
        self.assertNotIn("ORD_NM", selected)
        self.assertNotIn(("FNN_FCT", "PAY_AMT"), selected)

    def test_business_name_from_enrichment(self):
        cols = {c["column"]: c for c in select_filterable_columns(_schema())}
        self.assertEqual(cols["CUS_NM"]["business_name"], "customer name")


class BuildValueIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.schema_dir = Path(self.tmp) / "schema"
        self.schema_dir.mkdir()
        (self.schema_dir / "_schema.json").write_text(json.dumps(_schema()), encoding="utf-8")
        self.base = str(Path(self.tmp) / "clients")

    def _build(self, **kw):
        return build_value_index(
            "acct", {}, "azure_sql", str(self.schema_dir),
            run_query_fn=_fake_run_query, base_dir=self.base, **kw,
        )

    def test_build_writes_atomic_index_with_stats(self):
        stats = self._build()
        self.assertTrue(index_exists("acct", base_dir=self.base))
        self.assertFalse((Path(self.base) / "acct" / "value_index.sqlite.tmp").exists())
        self.assertEqual(stats["columns_indexed"], 2)
        # 5 clean CUS_NM values (newline + oversized rejected) + 2 ITM_DSC
        self.assertEqual(stats["values_indexed"], 7)
        persisted = load_index_stats("acct", base_dir=self.base)
        self.assertEqual(persisted["values_indexed"], 7)

    def test_hygiene_rejects_newlines_and_long_values(self):
        self._build()
        self.assertEqual(lookup_exact("acct", "bad\nvalue", base_dir=self.base), [])
        self.assertEqual(lookup_exact("acct", "x" * 300, base_dir=self.base), [])

    def test_value_level_pii_drops_whole_column(self):
        def rq(creds, db_type, sql, max_rows=200):
            if "CUS_NM" in sql:
                return [{"CUS_NM": f"user{i}@example.com"} for i in range(10)]
            return _fake_run_query(creds, db_type, sql, max_rows)
        stats = build_value_index(
            "acct2", {}, "azure_sql", str(self.schema_dir),
            run_query_fn=rq, base_dir=self.base,
        )
        self.assertEqual(stats["columns_skipped_pii"], 1)
        self.assertEqual(lookup_exact("acct2", "user1@example.com", base_dir=self.base), [])

    def test_truncation_recorded_at_cap(self):
        def rq(creds, db_type, sql, max_rows=200):
            if "CUS_NM" in sql:
                return [{"CUS_NM": f"Customer {i}"} for i in range(12)]
            return []
        stats = build_value_index(
            "acct3", {}, "azure_sql", str(self.schema_dir),
            run_query_fn=rq, base_dir=self.base, per_column_cap=10,
        )
        self.assertIn("EMCODW.EMDW_DMART.CUS_DMS.CUS_NM", stats["truncated_columns"])

    def test_lookup_exact_case_insensitive_then_normalized(self):
        self._build()
        hits = lookup_exact("acct", "emco corporation", base_dir=self.base)
        self.assertEqual(hits[0]["value"], "EMCO Corporation")
        self.assertEqual(hits[0]["method"], "exact")
        hits2 = lookup_exact("acct", "  EMCO, Corporation.  ", base_dir=self.base)
        self.assertEqual(hits2[0]["value"], "EMCO Corporation")
        self.assertEqual(hits2[0]["method"], "normalized")

    def test_lookup_fuzzy_typo_and_threshold(self):
        self._build()
        vals = [m["value"] for m in lookup_fuzzy("acct", "acme industry", base_dir=self.base)]
        self.assertIn("Acme Industries", vals)
        # first-syllable typo still reaches the scorer via the 2-char prefix probe
        loose = lookup_fuzzy("acct", "emko corpp", base_dir=self.base, min_score=0.55)
        self.assertTrue(any("EMCO" in m["value"] for m in loose))
        # default threshold keeps typo matches out of the injection path
        strict = lookup_fuzzy("acct", "zzz qqq", base_dir=self.base)
        self.assertEqual(strict, [])

    def test_allowed_tables_scoping(self):
        self._build()
        hits = lookup_exact("acct", "STEEL ROD 10MM",
                            allowed_tables={"CUS_DMS"}, base_dir=self.base)
        self.assertEqual(hits, [])
        hits2 = lookup_exact("acct", "STEEL ROD 10MM",
                             allowed_tables={"ITM_DMS"}, base_dir=self.base)
        self.assertEqual(hits2[0]["value"], "STEEL ROD 10MM")

    def test_missing_index_returns_empty(self):
        self.assertFalse(index_exists("nobody", base_dir=self.base))
        self.assertEqual(lookup_exact("nobody", "x", base_dir=self.base), [])
        self.assertEqual(lookup_fuzzy("nobody", "xyz", base_dir=self.base), [])


class FlagAndNormalizeTests(unittest.TestCase):
    def test_normalize_value(self):
        self.assertEqual(normalize_value("  EMCO, Corp.  "), "emco corp")
        self.assertEqual(normalize_value("A-B_C  D"), "a b c d")

    def test_value_index_enabled_default_on_with_opt_out(self):
        self.assertTrue(value_index_enabled(None))
        self.assertTrue(value_index_enabled({}))
        self.assertTrue(value_index_enabled({"value_index_enabled": True}))
        self.assertFalse(value_index_enabled({"value_index_enabled": False}))
        self.assertFalse(value_index_enabled({"value_index_enabled": "0"}))


if __name__ == "__main__":
    unittest.main()
