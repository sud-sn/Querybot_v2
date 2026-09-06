import contextlib
import unittest

from core.schema_enrichment import (
    enrich_columns,
    format_column_reference_for_vocab,
    format_schema_intelligence,
    parse_schema_markdown,
)


@contextlib.contextmanager
def _m3_vocab():
    """Run the block under the Infor M3 pack, as an M3 tenant would.

    Cross-system key equivalences are tenant vocabulary now, so a test that
    asserts on them has to say which tenant it is speaking as.
    """
    from core.vocab_packs import (
        _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack,
    )

    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
    token = activate_vocab(vocab)
    try:
        yield vocab
    finally:
        deactivate_vocab(token)


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
        # No auto default_filter: soft-delete indicators must not be silently
        # applied to every query — only when the user explicitly asks for
        # active/non-deleted records (see commit d06fabc).
        self.assertEqual(enriched["DEL_REC_IND"].default_filter, "")
        self.assertIn("do NOT auto-filter", " ".join(enriched["DEL_REC_IND"].evidence))

    def test_join_aliases_come_from_the_tenants_pack_not_from_everyone(self):
        # These are Infor M3 codes. They used to be in the BUILTIN vocabulary,
        # so every workspace was told CUS_ORD_NUM and ORNO were the same key
        # whether or not it ran M3 — and a customer whose warehouse names its
        # keys differently could not say so without a code change.
        #
        # With the pack, all eight. Without it, none: that is the point.
        cols = ["CUS_ORD_NUM", "CUS_ORD_LIN_NUM", "CUS_ORD_LIN_SFX", "DLV_NUM"]

        with _m3_vocab():
            enriched = {c.column: c for c in enrich_columns(cols)}
            self.assertEqual(enriched["CUS_ORD_NUM"].join_equivalents, ["ORNO"])
            self.assertEqual(enriched["CUS_ORD_LIN_NUM"].join_equivalents, ["PONR"])
            self.assertEqual(enriched["CUS_ORD_LIN_SFX"].join_equivalents, ["POSX"])
            self.assertEqual(enriched["DLV_NUM"].join_equivalents, ["DLIX"])

        plain = {c.column: c for c in enrich_columns(cols)}
        for column in cols:
            self.assertEqual(plain[column].join_equivalents, [], column)

    def test_an_m3_warehouse_gets_the_pack_without_anybody_selecting_it(self):
        # The safety net for the change above: an M3-shaped warehouse must
        # auto-apply the pack, or moving these out of the builtin would take
        # them away from the tenants they were put there for.
        from core.identifier_intelligence import detect_naming_profile

        profile = detect_naming_profile(
            ["CUS_ORD_NUM", "CUS_DMS_KEY", "WHS_DMS_KEY", "ORNO", "CUNO",
             "WHLO", "ITNO", "ORQT", "IVDT"],
            ["ERP_CUS_ORD_FCT", "DMS_CUSTOMER", "OOHEAD", "MITMAS"],
        )
        self.assertIn("infor_m3", profile.get("auto_applied_packs") or [])

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
        with _m3_vocab():
            block = format_schema_intelligence(
                "CUS_ORD_IVC_FCT",
                ["CUS_ORD_NUM", "CUS_DMS_KEY", "CUS_IVC_LIN_AMT", "DEL_REC_IND"],
            )

        self.assertIn("SCHEMA INTELLIGENCE", block)
        self.assertIn("CUS_IVC_LIN_AMT: role=measure", block)
        # Soft-delete indicators are documented but never auto-applied as a
        # mandatory filter (see commit d06fabc).
        self.assertNotIn("DEL_REC_IND = 0", block)
        self.assertIn("do NOT auto-filter", block)
        self.assertIn("known join equivalents=ORNO", block)
        self.assertIn("thousands separators", block)

    def test_common_dimension_display_fields_get_business_meaning(self):
        cols = ["WHS_DMS_KEY", "WHS_CD", "WHS_DSC", "ITM_DSC", "CUS_NM"]

        enriched = {c.column: c for c in enrich_columns(cols)}

        self.assertEqual(enriched["WHS_DMS_KEY"].expanded_name, "warehouse dimension key")
        self.assertEqual(enriched["WHS_DSC"].expanded_name, "warehouse description")
        self.assertIn("warehouse name", enriched["WHS_DSC"].business_candidates)
        self.assertEqual(enriched["ITM_DSC"].expanded_name, "item description")
        self.assertIn("customer name", enriched["CUS_NM"].business_candidates)

    def test_token_expansion_handles_enterprise_warehouse_columns(self):
        cols = [
            "FIFO_BI_SAL_MGP_EXT",
            "Total_Charge_USD",
            "CUR_ON_HND_QTY",
            "WHS_DSC",
        ]

        enriched = {c.column: c for c in enrich_columns(cols)}

        self.assertIn("first in first out", enriched["FIFO_BI_SAL_MGP_EXT"].expanded_name)
        self.assertIn("margin profit", enriched["FIFO_BI_SAL_MGP_EXT"].expanded_name)
        self.assertIn("total charge usd", enriched["Total_Charge_USD"].expanded_name)
        self.assertEqual(enriched["CUR_ON_HND_QTY"].role, "measure")
        self.assertEqual(enriched["WHS_DSC"].expanded_name, "warehouse description")

    def test_business_vocab_reference_preserves_exact_columns_with_hints(self):
        line = format_column_reference_for_vocab(
            "WHS_DMS",
            ["WHS_DMS_KEY", "WHS_CD", "WHS_DSC"],
        )

        self.assertIn("WHS_DSC (role=", line)
        self.assertIn("meaning=warehouse description", line)
        self.assertIn("terms=warehouse description, warehouse, warehouse name", line)
        self.assertIn("WHS_DMS_KEY", line)


if __name__ == "__main__":
    unittest.main()
