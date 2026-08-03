import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.date_roles import detect_date_role, generated_date_role_synonyms
from core.identifier_intelligence import (
    analyze_identifier,
    canonical_identifier,
    detect_naming_profile,
    expand_question_for_retrieval,
    tokenize_identifier,
)
from core.knowledge import _inject_deterministic_synonyms
from core.schema_enrichment import enrich_columns
from core.semantic_model import _date_roles
from core.vector_store import _bm25_tokenize
from core.vocab_packs import (
    _account_cache,
    load_detected_naming_profile,
    save_detected_naming_profile,
    vocab_for_account,
)


class IdentifierTokenizationTests(unittest.TestCase):
    def test_snake_camel_pascal_and_compact_dates_share_one_canonical_form(self):
        self.assertEqual(["ORD", "DT"], tokenize_identifier("ORD_DT"))
        self.assertEqual(["ORDER", "DATE"], tokenize_identifier("orderDate"))
        self.assertEqual(["ORDER", "DATE"], tokenize_identifier("OrderDate"))
        self.assertEqual(["ORDER", "DATE"], tokenize_identifier("ORDERDATE"))
        self.assertEqual(["ORD", "DT"], tokenize_identifier("ORDDT"))
        self.assertEqual("ORDER_DATE", canonical_identifier("OrderDate"))

    def test_abbreviated_date_produces_physical_and_business_aliases(self):
        analysis = analyze_identifier("ORD_DT")
        self.assertEqual("order date", analysis.expanded_name)
        self.assertGreaterEqual(analysis.confidence, 80)
        self.assertIn("ord dt", analysis.aliases)
        self.assertIn("order dt", analysis.aliases)
        self.assertIn("ord date", analysis.aliases)

    def test_unknown_compact_code_abstains(self):
        analysis = analyze_identifier("ZXQ7B")
        self.assertLess(analysis.confidence, 70)
        self.assertIn("no governed expansion found", analysis.evidence)

    def test_readable_and_governed_compact_names_reach_confident_kb_fields(self):
        fields = {field.column: field for field in enrich_columns(["OrderDate", "ORDERDATE"])}
        self.assertGreaterEqual(fields["OrderDate"].confidence, 70)
        self.assertGreaterEqual(fields["ORDERDATE"].confidence, 70)
        self.assertEqual("order date", fields["OrderDate"].expanded_name)

    def test_etl_prefix_is_classified_as_infrastructure(self):
        field = enrich_columns(["ETL_LOAD_TS"])[0]
        self.assertEqual("infrastructure", field.role)
        self.assertIn("data platform field", field.expanded_name)


class NamingPackDetectionTests(unittest.TestCase):
    def test_detects_dynamics_from_pascal_case_schema(self):
        profile = detect_naming_profile(
            ["CustAccount", "SalesId", "ItemId", "InvoiceDate"],
            ["SalesTable"],
        )
        self.assertEqual("pascal_case", profile["primary_style"])
        self.assertEqual(["dynamics"], profile["auto_applied_packs"])
        self.assertEqual("dynamics", profile["pack_recommendations"][0]["pack_id"])
        self.assertEqual(100.0, profile["vocabulary_coverage_pct"])

    def test_detects_netsuite_from_lower_compact_schema(self):
        profile = detect_naming_profile(
            ["tranid", "trandate", "entityid", "netamount"],
            ["transaction"],
        )
        self.assertEqual("lower_compact", profile["primary_style"])
        self.assertEqual(["netsuite"], profile["auto_applied_packs"])
        self.assertEqual(100.0, profile["vocabulary_coverage_pct"])

    def test_detects_m3_short_codes(self):
        profile = detect_naming_profile(
            ["ORNO", "PONR", "WHLO", "ORDT", "ITDS"],
            ["OOLINE"],
        )
        self.assertEqual(["infor_m3"], profile["auto_applied_packs"])

    def test_shared_single_field_never_auto_selects_vendor(self):
        profile = detect_naming_profile(["ItemId"], ["CustomTable"])
        self.assertEqual([], profile["auto_applied_packs"])

    def test_persisted_auto_pack_is_used_at_runtime_when_admin_has_no_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("core.vocab_packs._CLIENTS_DIR", Path(tmp)), \
                 patch("core.vocab_packs._client_pack_ids", return_value=[]):
                _account_cache.clear()
                save_detected_naming_profile("client_a", {"auto_applied_packs": ["dynamics"]})
                self.assertEqual(
                    ["dynamics"],
                    load_detected_naming_profile("client_a")["auto_applied_packs"],
                )
                vocab = vocab_for_account("client_a")
                self.assertEqual("Customer Account", vocab.column_dict["CUSTACCOUNT"][0])
                self.assertIn("dynamics", vocab.source_packs)
                _account_cache.clear()


class DynamicDateRoleSynonymTests(unittest.TestCase):
    def test_multiple_physical_styles_resolve_to_order_date(self):
        for column in ("ORD_DT", "ORDDT", "OrderDate", "orderDate", "ORDERDATE"):
            with self.subTest(column=column):
                role = detect_date_role(column)
                self.assertIsNotNone(role)
                self.assertEqual("order_date", role.key)

    def test_physical_and_human_variants_are_generated(self):
        role = detect_date_role("ORD_DT")
        synonyms = set(generated_date_role_synonyms(role, "ORD_DT"))
        self.assertTrue({"ord dt", "order dt", "ordered date", "date ordered"} <= synonyms)
        self.assertTrue({"order month", "order quarter", "order year"} <= synonyms)

    def test_etl_timestamp_is_not_promoted_to_business_date_role(self):
        meta = {
            "table": "FACT_ORDER",
            "schema": "ERP",
            "columns": [
                {"name": "ETL_LOAD_TS", "type": "datetime2"},
                {"name": "OrderDate", "type": "date"},
            ],
        }
        schema = {"DB.ERP.FACT_ORDER": meta}
        roles = _date_roles(schema, "DB.ERP.FACT_ORDER", meta)
        self.assertEqual(["OrderDate"], [role["fact_column"] for role in roles])


class NamingGovernanceWiringTests(unittest.TestCase):
    def test_model_health_surfaces_detected_style_and_pack(self):
        template = (
            Path(__file__).resolve().parents[1]
            / "admin" / "templates" / "client_model_health.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Naming Intelligence", template)
        self.assertIn("naming.auto_applied_packs", template)
        self.assertIn("naming.pack_recommendations", template)


class SymmetricRetrievalTests(unittest.TestCase):
    def test_question_shorthand_adds_governed_phrase(self):
        expanded = expand_question_for_retrieval("revenue by ord dt monthly")
        self.assertIn("revenue by ord dt monthly", expanded)
        self.assertIn("order date", expanded)

    def test_bm25_keeps_exact_identifier_and_splits_it(self):
        tokens = _bm25_tokenize("Revenue by ORD_DT and NDC_CODE")
        self.assertIn("ord_dt", tokens)
        self.assertIn("ord", tokens)
        self.assertIn("dt", tokens)
        self.assertIn("ndc_code", tokens)
        self.assertIn("ndc", tokens)

    def test_kb_synonym_injection_covers_non_dictionary_abbreviation(self):
        source = """# DB.S.FACT\n\n## Business Synonyms\n| Plain English | Column | Notes |\n|---|---|---|\n"""
        patched = _inject_deterministic_synonyms(source, ["ORD_DT"])
        self.assertIn("| ord dt | `ORD_DT` |", patched)
        self.assertIn("| order dt | `ORD_DT` |", patched)
        self.assertIn("| date ordered | `ORD_DT` |", patched)


if __name__ == "__main__":
    unittest.main()
