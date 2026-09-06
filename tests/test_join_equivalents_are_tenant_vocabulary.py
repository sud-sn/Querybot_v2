"""
tests/test_join_equivalents_are_tenant_vocabulary.py

J4 — cross-system key equivalences belong to a tenant, not to the product.

"CUS_ORD_NUM and ORNO are the same key" is a fact about one warehouse. It was
stated in three places at once:

  - core.schema_enrichment.KNOWN_JOIN_EQUIVALENTS — eight Infor M3 codes, read
    by schema discovery's alias-join pass, with no per-tenant override at all
  - core.semantic_planner._JOIN_SYNONYMS — three of the same codes, in the
    BUILTIN vocabulary, so every workspace carried them whether or not it ran
    M3
  - packs/infor_m3.json's join_synonyms — three entries, read by the planner

Three copies of one idea, already drifted to three entries against eight. Now
there is one: the packs. The builtin holds none, discovery reads the tenant's
vocabulary, and a warehouse with M3-shaped identifiers auto-applies the pack
so an M3 tenant gains the five it was missing rather than losing the three it
had.

Everything here executes the real lookup under a real merged vocabulary.
"""

from __future__ import annotations

import contextlib
import json
import unittest
from pathlib import Path

from core.schema_enrichment import enrich_columns, join_equivalents_for
from core.vocab_packs import (
    _clone_builtin,
    _merge_pack,
    activate_vocab,
    builtin_vocab,
    deactivate_vocab,
    load_pack,
)

ROOT = Path(__file__).resolve().parents[1]

M3_KEYS = {
    "CUS_ORD_NUM": ["ORNO"],
    "CUS_ORD_LIN_NUM": ["PONR"],
    "CUS_ORD_LIN_SFX": ["POSX"],
    "DLV_NUM": ["DLIX"],
    "CUS_IVC_NUM": ["IVNO"],
    "CUS_DMS_KEY": ["CUNO", "PYNO"],
    "WHS_DMS_KEY": ["WHLO"],
    "FCY_DMS_KEY": ["FACI"],
}


@contextlib.contextmanager
def pack(pack_id: str):
    vocab = _clone_builtin()
    loaded = load_pack(pack_id)
    assert loaded, f"pack {pack_id} did not load"
    _merge_pack(vocab, loaded, pack_id)
    token = activate_vocab(vocab)
    try:
        yield vocab
    finally:
        deactivate_vocab(token)


class TestOnlyThePackCarriesThem(unittest.TestCase):

    def test_a_workspace_with_no_pack_has_no_equivalences(self):
        # The bug: three M3 codes in the builtin meant a plumbing wholesaler
        # running Dynamics was told CUS_ORD_NUM and ORNO were the same key.
        self.assertEqual(builtin_vocab().join_synonyms, {})
        for column in M3_KEYS:
            self.assertEqual(join_equivalents_for(column), [], column)

    def test_an_m3_workspace_gets_all_eight(self):
        with pack("infor_m3"):
            for column, codes in M3_KEYS.items():
                self.assertEqual(join_equivalents_for(column), codes, column)

    def test_the_five_that_only_existed_in_the_hardcoded_map_survived(self):
        # These lived in KNOWN_JOIN_EQUIVALENTS and nowhere else, so deleting
        # that constant without moving them would have quietly removed five
        # join paths from every M3 tenant.
        with pack("infor_m3"):
            for column in ("DLV_NUM", "CUS_IVC_NUM", "CUS_DMS_KEY",
                           "WHS_DMS_KEY", "FCY_DMS_KEY"):
                self.assertTrue(join_equivalents_for(column), column)

    def test_the_order_is_sorted_not_whatever_the_set_iterates(self):
        # The vocabulary stores these as a SET, and Python randomises string
        # hashing per process -- so an unsorted read renders "CUNO, PYNO" on
        # one build and "PYNO, CUNO" on the next: a gratuitous diff in every
        # regenerated knowledge base and a cache miss in every prompt.
        #
        # Five codes rather than the two a real column has. With two, an
        # unsorted implementation comes out right by chance about half the
        # time and the test only fails on some runs; with five the odds are
        # one in a hundred and twenty.
        vocab = _clone_builtin()
        _merge_pack(vocab, {"join_synonyms": {
            "WIDE_KEY": ["ECHO", "ALPHA", "DELTA", "BRAVO", "CHARLIE"]}}, "test")
        token = activate_vocab(vocab)
        try:
            self.assertEqual(
                join_equivalents_for("WIDE_KEY"),
                ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"])
        finally:
            deactivate_vocab(token)

    def test_another_erps_pack_does_not_bring_m3_keys(self):
        with pack("dynamics"):
            self.assertEqual(join_equivalents_for("CUS_ORD_NUM"), [])

    def test_the_lookup_is_case_insensitive_and_survives_junk(self):
        with pack("infor_m3"):
            self.assertEqual(join_equivalents_for("cus_ord_num"), ["ORNO"])
            self.assertEqual(join_equivalents_for("  CUS_ORD_NUM  ".strip()), ["ORNO"])
        for junk in ("", None, "NOT_A_COLUMN"):
            with self.subTest(junk=junk):
                self.assertEqual(join_equivalents_for(junk), [])

    def test_enrichment_carries_the_equivalence_through(self):
        with pack("infor_m3"):
            enriched = {c.column: c for c in enrich_columns(["WHS_DMS_KEY"])}
            self.assertEqual(enriched["WHS_DMS_KEY"].join_equivalents, ["WHLO"])
        plain = {c.column: c for c in enrich_columns(["WHS_DMS_KEY"])}
        self.assertEqual(plain["WHS_DMS_KEY"].join_equivalents, [])


class TestThereIsOnlyOneCopyLeft(unittest.TestCase):

    def test_the_hardcoded_map_is_gone(self):
        # A source check, and the right kind: the defect was that the constant
        # EXISTED, so its absence is the invariant.
        for module in ("core/schema_enrichment.py", "core/schema.py"):
            source = (ROOT / module).read_text(encoding="utf-8")
            self.assertNotIn("KNOWN_JOIN_EQUIVALENTS", source, module)

    def test_the_builtin_declares_itself_empty_on_purpose(self):
        from core.semantic_planner import _JOIN_SYNONYMS

        self.assertEqual(_JOIN_SYNONYMS, {})

    def test_the_pack_file_holds_every_equivalence_the_constant_had(self):
        # The five below existed ONLY in KNOWN_JOIN_EQUIVALENTS. Deleting that
        # constant without moving them would have quietly removed five join
        # paths from every M3 tenant, and nothing else would have noticed.
        stored = json.loads(
            (ROOT / "packs" / "infor_m3.json").read_text(encoding="utf-8"))
        for column in ("DLV_NUM", "CUS_IVC_NUM", "CUS_DMS_KEY",
                       "WHS_DMS_KEY", "FCY_DMS_KEY"):
            self.assertIn(column, stored["join_synonyms"], column)

    def test_the_pack_file_holds_all_eight(self):
        stored = json.loads(
            (ROOT / "packs" / "infor_m3.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {k: sorted(v) for k, v in stored["join_synonyms"].items()},
            {k: sorted(v) for k, v in M3_KEYS.items()})


class TestAnM3WarehouseFindsItsPackByItself(unittest.TestCase):
    """
    The safety net. Moving these out of the builtin only stays safe if an
    M3-shaped warehouse applies the pack without anybody selecting it.
    """

    M3_COLUMNS = ["CUS_ORD_NUM", "CUS_DMS_KEY", "WHS_DMS_KEY", "ORNO",
                  "CUNO", "WHLO", "ITNO", "ORQT", "IVDT"]
    M3_TABLES = ["ERP_CUS_ORD_FCT", "DMS_CUSTOMER", "OOHEAD", "MITMAS"]

    def test_it_is_detected(self):
        from core.identifier_intelligence import detect_naming_profile

        profile = detect_naming_profile(self.M3_COLUMNS, self.M3_TABLES)
        self.assertIn("infor_m3", profile.get("auto_applied_packs") or [])

    def test_a_warehouse_that_is_not_m3_does_not_get_it(self):
        from core.identifier_intelligence import detect_naming_profile

        profile = detect_naming_profile(
            ["customer_id", "order_id", "net_amount", "created_at"],
            ["dim_customer", "fact_orders"])
        self.assertNotIn("infor_m3", profile.get("auto_applied_packs") or [])


class TestDiscoveryRunsUnderTheTenantsVocabulary(unittest.TestCase):
    """
    The join map and its date-role detection read the vocabulary through a
    ContextVar, and nothing was setting it during discovery — so a workspace
    with the M3 pack selected had its schema discovered as though it had none,
    and the pack only started applying at question time.
    """

    def test_the_alias_join_pass_uses_the_active_vocabulary(self):
        from core.schema import _build_join_map

        master = {
            "DBO.ERP_CUS_ORD_FCT": {"columns": [
                {"name": "CUS_ORD_NUM", "type": "varchar"},
                {"name": "CUS_IVC_LIN_AMT", "type": "decimal"}]},
            "DBO.OOLINE": {"columns": [
                {"name": "ORNO", "type": "varchar"},
                {"name": "ORQT", "type": "decimal"}]},
        }
        with pack("infor_m3"):
            found = _build_join_map(master)
        self.assertIn("ORNO", found)
        self.assertIn("CUS_ORD_NUM", found)

        # And without it, the alias pass has nothing to offer: the two tables
        # share no column name, so only the pack could have linked them.
        plain = _build_join_map(master)
        self.assertNotIn("CUS_ORD_NUM = ORNO", plain.replace("`", ""))

    def test_the_discovery_route_activates_it(self):
        # Placement inside an 11k-line module: the activation has to wrap the
        # discover_and_write call, and a try/finally has to give it back.
        import inspect

        import admin.routes as routes

        source = inspect.getsource(routes)
        block = source[source.index("_discovery_vocab_token = activate_vocab("):]
        block = block[:block.index("deactivate_vocab(_discovery_vocab_token)")]
        self.assertIn("discover_and_write(", block)
        self.assertIn("finally:", block)
        # THIS tenant's vocabulary, not the builtin. Activating the builtin
        # would satisfy every other assertion here and change nothing at all.
        self.assertIn("vocab_for_account(account_id)", block)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
