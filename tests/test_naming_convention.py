"""
Tests for core/naming_convention.py

Verifies that structural naming patterns are correctly classified and that
the hint + doc generation produces expected output.
"""

import unittest
from core.naming_convention import (
    match_column_suffix,
    match_table_suffix,
    match_audit_prefix,
    match_entity_prefix,
    get_naming_hints,
    build_naming_convention_doc,
)


class SuffixMatchingTests(unittest.TestCase):

    def test_dms_key_matches_surrogate_fk(self):
        rule = match_column_suffix("WHS_DMS_KEY")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "surrogate_fk")
        self.assertEqual(rule.aggregation, "identifier")

    def test_dms_key_wins_over_plain_key(self):
        # _DMS_KEY is longer and more specific than _KEY — must match first
        rule = match_column_suffix("ITM_GRP_DMS_KEY")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.suffix, "_DMS_KEY")

    def test_dt_dms_key_matches_date_fk(self):
        rule = match_column_suffix("CUS_IVC_DT_DMS_KEY")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "date_fk")

    def test_amt_is_additive_measure(self):
        rule = match_column_suffix("SOP_CUS_IVC_LIN_AMT")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "measure")
        self.assertEqual(rule.aggregation, "additive")
        self.assertEqual(rule.format_hint, "currency")

    def test_pct_is_non_additive(self):
        rule = match_column_suffix("GROSS_MARGIN_PCT")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "ratio")
        self.assertEqual(rule.aggregation, "non_additive")
        self.assertIn("NEVER SUM", rule.sql_guidance)
        self.assertIn("anti_pattern", rule.__dataclass_fields__)

    def test_rate_is_non_additive(self):
        rule = match_column_suffix("FILL_RATE")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.aggregation, "non_additive")

    def test_bal_is_semi_additive(self):
        rule = match_column_suffix("ITM_INV_BAL")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.aggregation, "semi_additive")
        self.assertIn("time", rule.sql_guidance.lower())

    def test_ts_is_timestamp_not_business_date(self):
        rule = match_column_suffix("AZ_LST_UPD_TS")
        # AZ_ will match audit prefix first in get_naming_hints,
        # but suffix match itself should still resolve to timestamp
        rule2 = match_column_suffix("LAST_MODIFIED_TS")
        self.assertIsNotNone(rule2)
        self.assertEqual(rule2.role, "timestamp")
        self.assertIn("ETL", rule2.sql_guidance)

    def test_dsc_is_display_field(self):
        rule = match_column_suffix("WHS_DSC")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "display")
        self.assertEqual(rule.aggregation, "dimension")

    def test_nm_is_display_field(self):
        rule = match_column_suffix("CUS_NM")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "display")

    def test_cd_is_code(self):
        rule = match_column_suffix("WHS_CD")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "code")

    def test_sts_is_status(self):
        rule = match_column_suffix("ITM_STS_CD")
        # _CD wins (longer than _STS from the end) — but _STS should still match
        rule2 = match_column_suffix("ORD_STS")
        self.assertIsNotNone(rule2)
        self.assertEqual(rule2.role, "status")

    def test_flg_is_flag(self):
        rule = match_column_suffix("ACTIVE_FLG")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.role, "flag")
        self.assertIn("NEVER SUM", rule.anti_pattern.upper() if rule.anti_pattern else "")

    def test_qty_is_additive(self):
        rule = match_column_suffix("SHP_QTY")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.aggregation, "additive")

    def test_unknown_column_returns_none(self):
        rule = match_column_suffix("CONO")   # ERP code — no suffix match
        self.assertIsNone(rule)


class TableSuffixTests(unittest.TestCase):

    def test_fct_is_fact_table(self):
        rule = match_table_suffix("CUS_ORD_IVC_FCT")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.table_type, "fact_table")

    def test_dms_is_dimension_table(self):
        rule = match_table_suffix("WHS_DMS")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.table_type, "dimension_table")

    def test_schema_qualified_table_name(self):
        rule = match_table_suffix("PROFITABILITY.CUS_ORD_IVC_FCT")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.table_type, "fact_table")

    def test_stg_is_staging(self):
        rule = match_table_suffix("CUS_ORD_STG")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.table_type, "staging_table")

    def test_unknown_suffix_returns_none(self):
        rule = match_table_suffix("OOLINE")
        self.assertIsNone(rule)


class AuditPrefixTests(unittest.TestCase):

    def test_az_prefix_is_audit(self):
        rule = match_audit_prefix("AZ_LST_UPD_TS")
        self.assertIsNotNone(rule)
        self.assertIn("Azure", rule.meaning)

    def test_etl_prefix_is_audit(self):
        rule = match_audit_prefix("ETL_BATCH_ID")
        self.assertIsNotNone(rule)

    def test_dw_prefix_is_audit(self):
        rule = match_audit_prefix("DW_INSERT_DT")
        self.assertIsNotNone(rule)

    def test_business_column_not_audit(self):
        self.assertIsNone(match_audit_prefix("WHS_DMS_KEY"))
        self.assertIsNone(match_audit_prefix("SOP_CUS_IVC_LIN_AMT"))


class EntityPrefixTests(unittest.TestCase):

    def test_whs_prefix(self):
        entity = match_entity_prefix("WHS_DMS_KEY")
        self.assertEqual(entity, "Warehouse")

    def test_cus_prefix(self):
        entity = match_entity_prefix("CUS_NM")
        self.assertEqual(entity, "Customer")

    def test_sop_prefix(self):
        entity = match_entity_prefix("SOP_CUS_LIN_GRS_PFT_AMT")
        self.assertEqual(entity, "Sales Order Processing (calculated / derived measure)")

    def test_unknown_prefix_returns_none(self):
        self.assertIsNone(match_entity_prefix("ORNO"))


class HintGenerationTests(unittest.TestCase):

    def test_hints_cover_dms_key_columns(self):
        hints = get_naming_hints(["WHS_DMS_KEY", "ITM_DMS_KEY"], "CUS_ORD_IVC_FCT")
        hints_lower = hints.lower()
        self.assertIn("whs_dms_key", hints_lower)
        self.assertIn("surrogate_fk", hints_lower)
        self.assertIn("never select", hints_lower)

    def test_hints_flag_audit_columns(self):
        hints = get_naming_hints(["AZ_LST_UPD_TS", "AZ_EXT_ID"], "CUS_ORD_IVC_FCT")
        self.assertIn("AUDIT/ETL", hints)
        self.assertIn("AZ_LST_UPD_TS", hints)

    def test_hints_flag_non_additive_pct(self):
        hints = get_naming_hints(["GROSS_MARGIN_PCT"], "CUS_ORD_IVC_FCT")
        self.assertIn("non_additive", hints.lower())
        self.assertIn("NEVER SUM", hints)

    def test_hints_include_table_type(self):
        hints = get_naming_hints(["WHS_DMS_KEY"], "CUS_ORD_IVC_FCT")
        self.assertIn("fact_table", hints.lower())

    def test_empty_columns_returns_empty(self):
        self.assertEqual(get_naming_hints([]), "")

    def test_mixed_columns_all_classified(self):
        cols = ["WHS_DMS_KEY", "WHS_DSC", "SOP_CUS_IVC_LIN_AMT",
                "GROSS_MARGIN_PCT", "AZ_LST_UPD_TS", "ORD_STS"]
        hints = get_naming_hints(cols, "CUS_ORD_IVC_FCT")
        hints_lower = hints.lower()
        self.assertIn("surrogate_fk", hints_lower)   # WHS_DMS_KEY
        self.assertIn("display", hints_lower)         # WHS_DSC
        self.assertIn("additive", hints_lower)        # AMT
        self.assertIn("non_additive", hints_lower)    # PCT
        self.assertIn("AUDIT/ETL", hints)             # AZ_


class NamingConventionDocTests(unittest.TestCase):

    def test_doc_contains_all_major_sections(self):
        doc = build_naming_convention_doc()
        self.assertIn("## Table Types", doc)
        self.assertIn("## Column Suffix Rules", doc)
        self.assertIn("## Audit / ETL Prefixes", doc)
        self.assertIn("## Entity Prefix Vocabulary", doc)
        self.assertIn("## Key Rules Summary", doc)

    def test_doc_contains_critical_rules(self):
        doc = build_naming_convention_doc()
        self.assertIn("_DMS_KEY", doc)
        self.assertIn("NEVER SELECT", doc)
        self.assertIn("_PCT", doc)
        self.assertIn("NEVER SUM", doc)
        self.assertIn("AZ_", doc)
        self.assertIn("_FCT", doc)

    def test_doc_is_valid_markdown(self):
        doc = build_naming_convention_doc()
        self.assertIn("# Data Warehouse Naming Convention", doc)
        # All section headers should be present
        for section in ["Table Types", "Column Suffix Rules", "Audit", "Entity Prefix", "Key Rules"]:
            self.assertIn(section, doc)


if __name__ == "__main__":
    unittest.main()
