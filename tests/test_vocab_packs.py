"""
tests/test_vocab_packs.py

Terminology pack infrastructure:
  1. Pack file parsing (all shipped packs load; stubs validate)
  2. Merge semantics (override / union / prepend / dedupe)
  3. GOLDEN backward-compat gate — builtin vocab == module constants, and
     enrich/plan/date-role outputs are identical three ways:
     no vocab / builtin activated / infor_m3 pack merged.
  4. Drift guard — packs/infor_m3.json stays byte-equivalent to the builtins.
  5. Generic star-schema pack classification.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vocab_packs import (
    MergedVocab, builtin_vocab, load_pack, list_available_packs,
    _merge_pack, _clone_builtin, activate_vocab, deactivate_vocab,
)

_ALL_PACK_IDS = {
    "infor_m3", "generic_star_schema", "sap", "oracle_ebs",
    "dynamics", "netsuite", "jde",
}


class PackLoadingTests(unittest.TestCase):
    def test_all_shipped_packs_parse(self):
        manifests = {m["pack_id"]: m for m in list_available_packs()}
        self.assertEqual(set(manifests), _ALL_PACK_IDS)
        self.assertEqual(manifests["infor_m3"]["status"], "complete")
        self.assertEqual(manifests["generic_star_schema"]["status"], "complete")
        for stub in ("sap", "oracle_ebs", "dynamics", "netsuite", "jde"):
            self.assertEqual(manifests[stub]["status"], "stub", stub)

    def test_unknown_pack_returns_empty(self):
        self.assertEqual(load_pack("no_such_pack"), {})
        self.assertEqual(load_pack("../../etc/passwd"), {})

    def test_bad_regex_is_skipped_not_fatal(self):
        v = MergedVocab()
        _merge_pack(v, {
            "pack_id": "synthetic",
            "table_classification": {"fact_patterns": ["([unclosed"]},
            "date_role_patterns": [{"pattern": "([bad", "role": "order_date"}],
        }, "synthetic")
        self.assertEqual(v.fact_patterns, [])
        self.assertEqual(v.date_role_patterns, [])

    def test_unknown_date_role_key_is_skipped(self):
        v = MergedVocab()
        _merge_pack(v, {
            "date_role_patterns": [{"pattern": "^FOO_DT$", "role": "not_a_real_role"}],
        }, "synthetic")
        self.assertEqual(v.date_role_patterns, [])


class MergeSemanticsTests(unittest.TestCase):
    def test_dict_override_set_union_pattern_prepend(self):
        v = MergedVocab(
            column_dict={"AAA": ("Old", ["old"])},
            raw_measure_codes={"AAA"},
        )
        _merge_pack(v, {
            "column_dict": {"AAA": {"label": "New", "synonyms": ["new"]}, "BBB": {"label": "B", "synonyms": []}},
            "raw_measure_codes": ["CCC"],
            "abbreviations": {"XYZ": "example"},
        }, "p1")
        self.assertEqual(v.column_dict["AAA"], ("New", ["new"]))
        self.assertIn("BBB", v.column_dict)
        self.assertEqual(v.raw_measure_codes, {"AAA", "CCC"})
        # abbreviations feed BOTH the enrichment map and the planner map
        self.assertEqual(v.abbreviations["XYZ"], "example")
        self.assertEqual(v.planner_abbreviations["XYZ"], "example")

    def test_remerging_same_pack_is_idempotent(self):
        a = _clone_builtin()
        _merge_pack(a, load_pack("infor_m3"), "infor_m3")
        b = _clone_builtin()
        _merge_pack(b, load_pack("infor_m3"), "infor_m3")
        _merge_pack(b, load_pack("infor_m3"), "infor_m3")
        self.assertEqual(
            [p.pattern for p, _, _ in a.numbered_series],
            [p.pattern for p, _, _ in b.numbered_series],
        )


class GoldenBackwardCompatTests(unittest.TestCase):
    """The critical regression gate: default behavior must be bit-identical."""

    _COLUMNS = ["CUAM", "ORNO", "DIC3", "AZ_LOAD_TS", "CUS_IVC_LIN_AMT",
                "DUE_DT_DMS_KEY", "TOTALLY_UNKNOWN_COL"]

    def test_builtin_vocab_mirrors_module_constants(self):
        from core.erp_column_dict import ERP_COLUMN_DICT
        from core.schema_enrichment import (
            ABBREVIATIONS, RAW_IDENTIFIER_CODES, RAW_MEASURE_CODES, RAW_DATE_CODES,
        )
        from core.semantic_planner import _DIRECT_ALIASES, _JOIN_SYNONYMS
        from core.semantic_planner import _ABBREVIATIONS as PLANNER_ABBR
        from core.naming_convention import ENTITY_PREFIX_VOCABULARY
        b = builtin_vocab()
        self.assertEqual(b.column_dict, {k: (v[0], list(v[1])) for k, v in ERP_COLUMN_DICT.items()})
        self.assertEqual(b.abbreviations, dict(ABBREVIATIONS))
        self.assertEqual(b.planner_abbreviations, dict(PLANNER_ABBR))
        self.assertEqual(b.direct_aliases, {k: set(v) for k, v in _DIRECT_ALIASES.items()})
        self.assertEqual(b.join_synonyms, {k: set(v) for k, v in _JOIN_SYNONYMS.items()})
        self.assertEqual(b.entity_prefixes, dict(ENTITY_PREFIX_VOCABULARY))
        self.assertEqual(b.raw_identifier_codes, set(RAW_IDENTIFIER_CODES))
        self.assertEqual(b.raw_measure_codes, set(RAW_MEASURE_CODES))
        self.assertEqual(b.raw_date_codes, set(RAW_DATE_CODES))

    def _three_way(self, fn):
        """Run fn(vocab_kwarg) with: no vocab, builtin activated, m3-merged."""
        default = fn(None)
        token = activate_vocab(builtin_vocab())
        try:
            activated = fn(None)
        finally:
            deactivate_vocab(token)
        m3 = _clone_builtin()
        _merge_pack(m3, load_pack("infor_m3"), "infor_m3")
        with_pack = fn(m3)
        return default, activated, with_pack

    def test_enrich_columns_three_way_identical(self):
        from core.schema_enrichment import enrich_columns
        d, a, p = self._three_way(lambda v: enrich_columns(self._COLUMNS, vocab=v))
        self.assertEqual(d, a)
        self.assertEqual(d, p)

    def test_semantic_field_plan_three_way_identical(self):
        from core.semantic_planner import build_semantic_field_plan
        cols = {
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"CUS_IVC_LIN_AMT": "decimal", "CUS_DMS_KEY": "int"},
            "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "int", "CUS_NM": "varchar"},
        }
        q = "total invoice amount by customer"
        d, a, p = self._three_way(lambda v: build_semantic_field_plan(q, cols, None, vocab=v))
        self.assertEqual(d, a)
        self.assertEqual(d, p)

    def test_detect_date_role_three_way_identical(self):
        from core.date_roles import detect_date_role
        for col in ("DUE_DT_DMS_KEY", "PAY_DT_DMS_KEY", "IVDT", "RANDOM_COL"):
            d, a, p = self._three_way(lambda v, c=col: detect_date_role(c, vocab=v))
            self.assertEqual(d, a, col)
            self.assertEqual(d, p, col)

    def test_infor_m3_pack_matches_builtin_column_dict(self):
        """Drift guard: the pack file must stay in sync with the constants."""
        from core.erp_column_dict import ERP_COLUMN_DICT
        pack = load_pack("infor_m3")
        pack_dict = {
            code: (e["label"], list(e["synonyms"]))
            for code, e in pack["column_dict"].items()
        }
        self.assertEqual(pack_dict, {k: (v[0], list(v[1])) for k, v in ERP_COLUMN_DICT.items()})


class GenericStarSchemaPackTests(unittest.TestCase):
    def _vocab(self):
        v = _clone_builtin()
        _merge_pack(v, load_pack("generic_star_schema"), "generic_star_schema")
        return v

    def test_classifies_kimball_tables(self):
        from core.naming_convention import match_table_suffix
        v = self._vocab()
        self.assertEqual(match_table_suffix("FACT_SALES", vocab=v).table_type, "fact_table")
        self.assertEqual(match_table_suffix("D_CUSTOMER", vocab=v).table_type, "dimension_table")
        self.assertEqual(match_table_suffix("DIM_PRODUCT", vocab=v).table_type, "dimension_table")
        # Builtin DMS/FCT conventions keep working with the pack active.
        self.assertEqual(match_table_suffix("FNN_FCT", vocab=v).table_type, "fact_table")
        self.assertEqual(match_table_suffix("CUS_DMS", vocab=v).table_type, "dimension_table")
        # Without the pack, FACT_SALES has no naming-convention rule (legacy).
        self.assertIsNone(match_table_suffix("FACT_SALES", vocab=builtin_vocab()))

    def test_plain_english_date_roles(self):
        from core.date_roles import detect_date_role
        v = self._vocab()
        self.assertEqual(detect_date_role("ORDER_DATE_KEY", vocab=v).key, "order_date")
        self.assertEqual(detect_date_role("DUE_DATE", vocab=v).key, "due_date")
        self.assertEqual(detect_date_role("PAYMENT_DATE_KEY", vocab=v).key, "payment_date")
        self.assertIsNone(detect_date_role("ORDER_DATE_KEY", vocab=builtin_vocab()))


class ActivationPointTests(unittest.TestCase):
    def test_build_kb_activates_client_vocab(self):
        src = (ROOT / "core" / "knowledge.py").read_text(encoding="utf-8")
        self.assertIn("vocab_for_account", src)
        self.assertIn("activate_vocab", src)
        self.assertIn("get_erp_hints(table_cols, vocab=_vocab)", src)
        self.assertIn("build_naming_convention_doc(vocab=_vocab)", src)

    def test_query_pipeline_activates_client_vocab(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("vocab_for_account(account_id)", src)
        self.assertIn("vocab=_vocab", src)

    def test_erp_packs_migration_and_meta_kwarg(self):
        db_src = (ROOT / "store" / "db.py").read_text(encoding="utf-8")
        self.assertIn('("client", "erp_packs"', db_src)
        cfg_src = (ROOT / "store" / "config_store.py").read_text(encoding="utf-8")
        self.assertIn("erp_packs: Optional[str] = None", cfg_src)

    def test_admin_ui_exposes_pack_checkboxes(self):
        tmpl = (ROOT / "admin" / "templates" / "client_detail.html").read_text(encoding="utf-8")
        self.assertIn('name="erp_packs"', tmpl)
        self.assertIn("erp_packs_available", tmpl)


if __name__ == "__main__":
    unittest.main()
