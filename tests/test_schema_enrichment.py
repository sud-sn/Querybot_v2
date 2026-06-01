import unittest

from core.schema_enrichment import enrich_columns, format_schema_intelligence, parse_schema_markdown


class SchemaEnrichmentTests(unittest.TestCase):
    def test_expands_erp_short_codes_from_sales_tables(self):
        cols = ["ORNO", "PONR", "POSX", "DIVI", "WHLO", "ITDS", "PCLA", "YEA4"]

        enriched = {c.column: c for c in enrich_columns(cols)}

        self.assertEqual(enriched["ORNO"].expanded_name, "order number")
        self.assertEqual(enriched["PONR"].expanded_name, "order line number")
        self.assertEqual(enriched["POSX"].expanded_name, "order line suffix")
        self.assertEqual(enriched["DIVI"].expanded_name, "division")
        self.assertEqual(enriched["WHLO"].expanded_name, "warehouse")
        self.assertEqual(enriched["ITDS"].expanded_name, "item description")
        self.assertEqual(enriched["PCLA"].role, "measure")
        self.assertGreaterEqual(enriched["ORNO"].confidence, 90)
        self.assertGreaterEqual(enriched["DIVI"].confidence, 90)

    def test_classifies_production_fact_fields(self):
        cols = [
            "CUS_ORD_IVC_FCT_KEY",
            "CUS_ORD_NUM",
            "CUS_ORD_LIN_NUM",
            "CUS_ORD_LIN_SFX",
            "CUS_IVC_DT_DMS_KEY",
            "CUS_DMS_KEY",
            "CUS_IVC_LIN_AMT",
            "SOP_CUS_IVC_LIN_CST_AMT",
            "DEL_REC_IND",
        ]

        enriched = {c.column: c for c in enrich_columns(cols)}

        self.assertEqual(enriched["CUS_ORD_IVC_FCT_KEY"].role, "surrogate_key")
        self.assertEqual(enriched["CUS_IVC_DT_DMS_KEY"].role, "date_key")
        self.assertIn("YYYYMMDD", " ".join(enriched["CUS_IVC_DT_DMS_KEY"].warnings))
        self.assertEqual(enriched["CUS_DMS_KEY"].role, "dimension_key")
        self.assertEqual(enriched["CUS_IVC_LIN_AMT"].role, "measure")
        self.assertIn("customer invoice line amount", enriched["CUS_IVC_LIN_AMT"].business_candidates)
        self.assertEqual(enriched["SOP_CUS_IVC_LIN_CST_AMT"].role, "measure")
        self.assertEqual(enriched["DEL_REC_IND"].role, "status_filter")
        self.assertEqual(enriched["DEL_REC_IND"].default_filter, "DEL_REC_IND = 0")

    def test_detects_known_join_aliases_for_order_line_tables(self):
        cols = ["CUS_ORD_NUM", "CUS_ORD_LIN_NUM", "CUS_ORD_LIN_SFX", "DLV_NUM"]

        enriched = {c.column: c for c in enrich_columns(cols)}

        self.assertEqual(enriched["CUS_ORD_NUM"].join_equivalents, ["ORNO"])
        self.assertEqual(enriched["CUS_ORD_LIN_NUM"].join_equivalents, ["PONR"])
        self.assertEqual(enriched["CUS_ORD_LIN_SFX"].join_equivalents, ["POSX"])
        self.assertEqual(enriched["DLV_NUM"].join_equivalents, ["DLIX"])

    def test_parses_schema_markdown_and_preserves_metadata(self):
        schema_md = """
| Column | Type | Nullable | Distinct Values |
| --- | --- | --- | --- |
| `CUS_DMS_KEY` | bigint | True | 1055930, 1035573 |
| `CUS_IVC_LIN_AMT` | decimal | True | 128.46, 52.00 |
"""

        parsed = parse_schema_markdown(schema_md)
        enriched = {c.column: c for c in enrich_columns(["CUS_DMS_KEY", "CUS_IVC_LIN_AMT"], schema_md)}

        self.assertEqual(parsed["CUS_DMS_KEY"]["type"], "bigint")
        self.assertEqual(parsed["CUS_DMS_KEY"]["distinct_values"], "1055930, 1035573")
        self.assertEqual(enriched["CUS_DMS_KEY"].data_type, "bigint")
        self.assertEqual(enriched["CUS_IVC_LIN_AMT"].role, "measure")

    def test_format_block_includes_kb_generation_rules(self):
        block = format_schema_intelligence(
            "CUS_ORD_IVC_FCT",
            ["CUS_ORD_NUM", "CUS_DMS_KEY", "CUS_IVC_LIN_AMT", "DEL_REC_IND"],
        )

        self.assertIn("SCHEMA INTELLIGENCE", block)
        self.assertIn("CUS_IVC_LIN_AMT: role=measure", block)
        self.assertIn("DEL_REC_IND = 0", block)
        self.assertIn("known join equivalents=ORNO", block)
        self.assertIn("thousands separators", block)


if __name__ == "__main__":
    unittest.main()
