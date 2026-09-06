# -*- coding: utf-8 -*-
"""tests/test_industry_vocabulary_packs.py

A client's business is not the same thing as their ERP.

The terminology packs shipped so far all answered "how does this ERP spell its
columns" — Infor M3, Dynamics, NetSuite. Nothing answered "what trade is this
customer in", so a distributor whose warehouse says SHOWRM, PCTR, BKLG and
WTRWKS got no help with any of it, and the only way to teach the product those
words was to add them to the built-in lexicon, where every other tenant would
then carry them too.

Two industry packs, selectable alongside whichever ERP pack the naming detects:
wholesale distribution, and the construction-products trade on top of it.

Both are deliberately built from `abbreviations` and `direct_aliases` only. A
pack earns its detection score from `column_dict` and `table_dict`, and an
industry pack has no business competing there — nothing in a column *name*
tells you the company sells plumbing supplies. The tests below prove that by
running the real detector, not by reading the files.

They also hold the line the other way: the product's own vocabulary must not
name a customer. Two entries did — one abbreviation and one entity prefix —
and the reason they were there is that the mechanism for a tenant to say it
themselves was half-wired: the naming-convention KB announced which packs were
in force and then printed the built-in prefixes regardless.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.identifier_intelligence import detect_naming_profile  # noqa: E402
from core.naming_convention import (  # noqa: E402
    ENTITY_PREFIX_VOCABULARY,
    build_naming_convention_doc,
)
from core.schema_enrichment import drop_repeated_unit, enrich_columns  # noqa: E402
from core.vocab_packs import (  # noqa: E402
    _clone_builtin,
    _merge_pack,
    activate_vocab,
    deactivate_vocab,
    list_available_packs,
    load_pack,
)

ROOT = Path(__file__).resolve().parents[1]

DISTRIBUTION = "wholesale_distribution"
CONSTRUCTION = "construction_products"

# An M3-shaped warehouse. The detector must still pick infor_m3 with the new
# packs on disk, or adding them would have cost every M3 tenant their pack.
M3_COLUMNS = ["CUS_ORD_NUM", "CUS_DMS_KEY", "WHS_DMS_KEY", "ORNO", "CUNO",
              "WHLO", "ITNO", "ORQT", "IVDT"]
M3_TABLES = ["ERP_CUS_ORD_FCT", "DMS_CUSTOMER", "OOHEAD", "MITMAS"]


def enriched(columns, *pack_ids):
    """Run the real enrichment under exactly these packs."""
    vocab = _clone_builtin()
    for pack_id in pack_ids:
        _merge_pack(vocab, load_pack(pack_id), pack_id)
    token = activate_vocab(vocab)
    try:
        return {c.column: c for c in enrich_columns(list(columns))}
    finally:
        deactivate_vocab(token)


class TestThePacksExist(unittest.TestCase):

    def test_both_are_offered_to_an_admin(self):
        offered = {m["pack_id"]: m for m in list_available_packs()}
        for pack_id in (DISTRIBUTION, CONSTRUCTION):
            self.assertIn(pack_id, offered)
            self.assertEqual(offered[pack_id]["status"], "complete")
            self.assertTrue(offered[pack_id]["description"])

    def test_they_carry_vocabulary_and_nothing_that_scores(self):
        for pack_id in (DISTRIBUTION, CONSTRUCTION):
            pack = load_pack(pack_id)
            self.assertTrue(pack["abbreviations"], pack_id)
            self.assertNotIn("column_dict", pack, pack_id)
            self.assertNotIn("table_dict", pack, pack_id)
            self.assertNotIn("record_prefixes", pack, pack_id)
            self.assertNotIn("date_role_patterns", pack, pack_id)


class TestTheyNeverCompeteWithTheErpPack(unittest.TestCase):
    """The load-bearing property, checked by running the detector rather than
    by reading the JSON: an industry pack has nothing to say about how a column
    is spelled, so it must not be able to win, tie, or narrow the margin that
    lets an ERP pack be applied automatically."""

    def test_an_m3_warehouse_still_auto_applies_infor_m3(self):
        profile = detect_naming_profile(M3_COLUMNS, M3_TABLES)
        self.assertEqual(profile.get("auto_applied_packs"), ["infor_m3"])

    def test_neither_industry_pack_is_even_recommended(self):
        profile = detect_naming_profile(M3_COLUMNS, M3_TABLES)
        recommended = {r["pack_id"] for r in profile.get("pack_recommendations") or []}
        self.assertNotIn(DISTRIBUTION, recommended)
        self.assertNotIn(CONSTRUCTION, recommended)

    def test_a_distribution_shaped_warehouse_does_not_summon_them_either(self):
        # Every column here is one the distribution pack knows. It still must
        # not be auto-applied: a word in a column name is not consent.
        profile = detect_naming_profile(
            ["SHOWRM_SAL_AMT", "PCTR_CD", "BKLG_QTY", "WTRWKS_MARG_PCT",
             "CNTR_SAL_AMT", "STKOUT_CNT"],
            ["F_BRANCH_SALES"])
        self.assertNotIn(DISTRIBUTION, profile.get("auto_applied_packs") or [])
        self.assertNotIn(CONSTRUCTION, profile.get("auto_applied_packs") or [])


class TestWhatTheDistributionPackTeaches(unittest.TestCase):

    SPELLINGS = {
        "SHOWRM_SAL_AMT": "showroom sales amount",
        "PCTR_CD": "profit centre code",
        "CNTR_SAL_AMT": "counter sales amount",
        "BKLG_QTY": "backlog quantity",
        "STKOUT_CNT": "stock out count",
        "LDT_DAYS": "lead time days",
        "ROP_QTY": "reorder point quantity",
        "WHSE_CD": "warehouse code",
    }

    def test_the_warehouses_spellings_become_words(self):
        got = enriched(self.SPELLINGS, DISTRIBUTION)
        for column, expected in self.SPELLINGS.items():
            self.assertEqual(got[column].expanded_name, expected, column)

    def test_without_the_pack_they_stay_unreadable(self):
        # The other half of the claim: these are NOT builtin, which is why a
        # distributor needed a pack rather than a code change.
        plain = enriched(self.SPELLINGS)
        unreadable = [c for c, e in plain.items()
                      if e.expanded_name != self.SPELLINGS[c]]
        self.assertGreaterEqual(len(unreadable), 6, plain)

    def test_the_term_a_person_would_type_is_offered_too(self):
        # core.semantic_model._runtime_match_score wants every word of a term
        # in the question, so the longest form alone answers only the longest
        # phrasing. "counter sales by branch" has to reach CNTR_SAL_AMT.
        column = enriched(["CNTR_SAL_AMT"], DISTRIBUTION)["CNTR_SAL_AMT"]
        self.assertIn("counter sales", column.business_candidates)

    def test_a_conventional_measure_name_carries_the_words_people_use(self):
        vocab = _clone_builtin()
        _merge_pack(vocab, load_pack(DISTRIBUTION), DISTRIBUTION)
        self.assertIn("on hand", vocab.direct_aliases["ON_HAND_QTY"])
        self.assertIn("profit centre", vocab.direct_aliases["PFT_CTR_CD"])


class TestWhatTheTradePackTeaches(unittest.TestCase):

    def test_the_trades_spellings_become_words(self):
        got = enriched(["WTRWKS_MARG_PCT", "PLBG_SAL_AMT", "FTGS_QTY",
                        "OILFLD_REV_AMT", "BLR_CNT"],
                       CONSTRUCTION)
        self.assertEqual(got["WTRWKS_MARG_PCT"].expanded_name,
                         "waterworks margin percent")
        self.assertEqual(got["PLBG_SAL_AMT"].expanded_name, "plumbing sales amount")
        self.assertEqual(got["FTGS_QTY"].expanded_name, "fittings quantity")
        self.assertEqual(got["OILFLD_REV_AMT"].expanded_name, "oilfield revenue amount")
        self.assertEqual(got["BLR_CNT"].expanded_name, "boiler count")

    def test_an_acronym_people_actually_say_is_left_alone(self):
        # Expanding HVAC gives "heating ventilation and air conditioning
        # product group code", which is longer than anything a person types and
        # therefore matches nothing. The word in the column IS the word people
        # use, so the pack must not touch it.
        column = enriched(["HVAC_PDC_GRP_CD"], CONSTRUCTION)["HVAC_PDC_GRP_CD"]
        self.assertEqual(column.expanded_name, "hvac product group code")
        self.assertIn("hvac product group", column.business_candidates)

    def test_the_same_holds_for_the_distribution_pack(self):
        for column, expected in (("SKU_CNT", "sku count"),
                                 ("RMA_CNT", "rma count")):
            got = enriched([column], DISTRIBUTION)[column]
            self.assertEqual(got.expanded_name, expected)

    def test_no_shipped_pack_expands_an_acronym_people_say(self):
        say_it = {"HVAC", "PVF", "SKU", "RMA", "BTU", "PSI", "CFM", "GPM",
                  "SEER", "PVC", "PEX", "DSO", "MSRP", "ASN", "OTIF"}
        for pack_id in (DISTRIBUTION, CONSTRUCTION):
            expanded = set(load_pack(pack_id).get("abbreviations") or {}) & say_it
            self.assertEqual(expanded, set(), f"{pack_id} expands {expanded}")


class TestTheyLayerOverTheErpPack(unittest.TestCase):

    def test_an_m3_code_and_a_trade_word_both_resolve_together(self):
        got = enriched(["ORNO", "WHLO", "SHOWRM_SAL_AMT", "WTRWKS_MARG_PCT"],
                       "infor_m3", DISTRIBUTION, CONSTRUCTION)
        self.assertEqual(got["ORNO"].expanded_name, "order number")
        self.assertEqual(got["WHLO"].expanded_name, "warehouse")
        self.assertEqual(got["SHOWRM_SAL_AMT"].expanded_name, "showroom sales amount")
        self.assertEqual(got["WTRWKS_MARG_PCT"].expanded_name,
                         "waterworks margin percent")

    def test_the_erp_pack_still_wins_where_the_two_disagree(self):
        # Precedence is merge order, and the ERP pack is what the database
        # actually is. Nothing in either industry pack may quietly redefine an
        # M3 field code.
        m3 = set(load_pack("infor_m3").get("abbreviations") or {})
        for pack_id in (DISTRIBUTION, CONSTRUCTION):
            clash = m3 & set(load_pack(pack_id).get("abbreviations") or {})
            self.assertEqual(clash, set(), f"{pack_id} redefines M3 codes: {clash}")


class TestTheStutter(unittest.TestCase):
    """QOH means "quantity on hand" to whoever writes it down, and QOH_QTY then
    reads "quantity on hand quantity". Easy to write, and it matches worse than
    either half."""

    def test_the_repeat_is_dropped(self):
        self.assertEqual(drop_repeated_unit("quantity on hand quantity"),
                         "quantity on hand")

    def test_a_phrase_that_means_both_amounts_is_left_alone(self):
        self.assertEqual(drop_repeated_unit("tax amount percent of total amount"),
                         "tax amount percent of total amount")

    def test_an_ordinary_name_is_untouched(self):
        for phrase in ("net sales amount", "balance value amount",
                       "counter sales amount", "on hand quantity"):
            self.assertEqual(drop_repeated_unit(phrase), phrase)

    def test_a_repeat_that_is_not_a_unit_noun_is_untouched(self):
        # CST_CTR_CST is a real column shape — the cost booked to a cost
        # centre. Dropping its last word loses the measure and leaves the
        # dimension, which reads fine and means something else entirely.
        for phrase in ("cost centre cost", "order to order", "sales of sales"):
            self.assertEqual(drop_repeated_unit(phrase), phrase)

    def test_it_runs_inside_the_real_enrichment(self):
        # Through enrich_columns, under a vocabulary that stutters — which is
        # what a tenant overlay does the first time somebody writes one.
        vocab = _clone_builtin()
        _merge_pack(vocab, {"abbreviations": {"QOH": "quantity on hand"}}, "test")
        token = activate_vocab(vocab)
        try:
            column = {c.column: c for c in enrich_columns(["QOH_QTY"])}["QOH_QTY"]
        finally:
            deactivate_vocab(token)
        self.assertEqual(column.expanded_name, "quantity on hand")
        self.assertEqual(column.business_candidates[0], "quantity on hand")


class TestTheProductDoesNotNameACustomer(unittest.TestCase):

    CUSTOMER = "EMCO"

    def test_no_shipped_pack_carries_a_customers_name(self):
        for path in sorted((ROOT / "packs").glob("*.json")):
            pack = json.loads(path.read_text(encoding="utf-8"))
            for key in ("abbreviations", "direct_aliases", "column_dict",
                        "table_dict", "entity_prefixes", "record_prefixes"):
                self.assertNotIn(self.CUSTOMER, {str(k).upper()
                                                 for k in (pack.get(key) or {})},
                                 f"{path.name}:{key}")

    def test_the_builtin_vocabulary_does_not_either(self):
        vocab = _clone_builtin()
        self.assertNotIn(self.CUSTOMER, vocab.abbreviations)
        self.assertNotIn(self.CUSTOMER, vocab.planner_abbreviations)
        self.assertNotIn(f"{self.CUSTOMER}_RGN", ENTITY_PREFIX_VOCABULARY)
        self.assertNotIn(self.CUSTOMER, ENTITY_PREFIX_VOCABULARY)

    def test_a_tenant_can_say_it_themselves_and_it_reaches_the_model(self):
        # The reason the two entries above were in the product: this did not
        # work. build_naming_convention_doc announced which packs were in force
        # and then printed the built-in prefixes regardless, so a tenant could
        # declare their own prefix and never see it.
        vocab = _clone_builtin()
        _merge_pack(vocab, {"entity_prefixes": {"ACME": "Company"}}, "overlay")
        token = activate_vocab(vocab)
        try:
            doc = build_naming_convention_doc()
        finally:
            deactivate_vocab(token)
        self.assertIn("| `ACME_` | Company |", doc)
        self.assertIn("| `CUS_` | Customer |", doc)      # builtin still there
        self.assertNotIn("ACME", build_naming_convention_doc())


class TestATenantSelectsThemForReal(unittest.TestCase):
    """Through store and vocab_for_account, because the selection an admin
    makes is what has to arrive — not a hand-merged vocabulary."""

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-packs-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-pack-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        from core.vocab_packs import forget_account_vocab

        forget_account_vocab(self.account_id)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def select(self, *pack_ids):
        import store

        from core.vocab_packs import forget_account_vocab, vocab_for_account

        store.update_client_meta(self.account_id, erp_packs=json.dumps(list(pack_ids)))
        forget_account_vocab(self.account_id)
        return vocab_for_account(self.account_id)

    def test_selecting_all_three_merges_all_three(self):
        vocab = self.select("infor_m3", DISTRIBUTION, CONSTRUCTION)
        self.assertEqual(vocab.abbreviations["SHOWRM"], "showroom")
        self.assertEqual(vocab.abbreviations["WTRWKS"], "waterworks")
        self.assertIn("ORNO", vocab.column_dict)

    def test_a_workspace_that_selected_none_gets_none_of_it(self):
        vocab = self.select()
        self.assertNotIn("SHOWRM", vocab.abbreviations)
        self.assertNotIn("WTRWKS", vocab.abbreviations)

    def test_the_selection_reaches_the_enrichment_the_pipeline_runs(self):
        from core.vocab_packs import activate_vocab, deactivate_vocab

        token = activate_vocab(self.select("infor_m3", DISTRIBUTION))
        try:
            column = {c.column: c for c in enrich_columns(["SHOWRM_SAL_AMT"])}
        finally:
            deactivate_vocab(token)
        self.assertEqual(column["SHOWRM_SAL_AMT"].expanded_name,
                         "showroom sales amount")


class TestTheAdminSurfacesKeepTheTwoKindsApart(unittest.TestCase):
    """The setup wizard picks ONE source system and writes it as the whole
    pack list. Offering an industry pack there would let an admin replace their
    ERP vocabulary with it, and saving a source system afterwards would drop
    whatever they had added in Client Settings."""

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-packui-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-packui-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _packs(self):
        import store

        return json.loads(store.get_client(self.account_id).get("erp_packs") or "[]")

    def _save_source(self, pack_id):
        import asyncio
        from unittest.mock import MagicMock, patch

        import admin.routes as routes

        request = MagicMock()
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.admin_setup_source_system(
                request, self.account_id, source_pack=pack_id))

    def test_the_source_system_list_offers_no_industry_pack(self):
        from jinja2 import Environment, FileSystemLoader

        import admin.routes as routes

        env = Environment(loader=FileSystemLoader(str(ROOT / "admin" / "templates")))
        rendered = env.from_string(
            (ROOT / "admin" / "templates" / "client_setup.html").read_text()
            .split('<select id="setup_source_pack"')[1].split("</select>")[0]
        ).render(erp_packs_available=routes._list_erp_packs(),
                 client_source_pack="other")
        self.assertIn("Infor M3", rendered)
        self.assertNotIn(DISTRIBUTION, rendered)
        self.assertNotIn(CONSTRUCTION, rendered)

    def test_saving_a_source_system_keeps_the_industry_packs(self):
        import store

        store.update_client_meta(
            self.account_id, erp_packs=json.dumps([DISTRIBUTION, CONSTRUCTION]))
        self._save_source("infor_m3")
        self.assertEqual(sorted(self._packs()),
                         sorted(["infor_m3", DISTRIBUTION, CONSTRUCTION]))

    def test_changing_the_source_system_replaces_only_that(self):
        import store

        store.update_client_meta(
            self.account_id, erp_packs=json.dumps(["infor_m3", DISTRIBUTION]))
        self._save_source("netsuite")
        self.assertEqual(sorted(self._packs()), sorted(["netsuite", DISTRIBUTION]))

    def test_choosing_other_still_leaves_the_industry_vocabulary(self):
        import store

        store.update_client_meta(
            self.account_id, erp_packs=json.dumps(["infor_m3", DISTRIBUTION]))
        self._save_source("other")
        self.assertEqual(self._packs(), [DISTRIBUTION])

    def test_the_wizard_does_not_read_an_industry_pack_as_a_second_source(self):
        # _setup_source_pack_value returns "multiple" for more than one source
        # pack, and the page then refuses to save. A workspace with one ERP
        # pack and one industry pack has exactly one source system.
        import store

        import admin.routes as routes

        store.update_client_meta(
            self.account_id, erp_packs=json.dumps(["infor_m3", DISTRIBUTION]))
        client = store.get_client(self.account_id)
        self.assertEqual(routes._setup_source_pack_value(client), "infor_m3")

    def test_two_real_source_packs_still_read_as_multiple(self):
        import store

        import admin.routes as routes

        store.update_client_meta(
            self.account_id, erp_packs=json.dumps(["infor_m3", "netsuite"]))
        self.assertEqual(
            routes._setup_source_pack_value(store.get_client(self.account_id)),
            "multiple")


if __name__ == "__main__":
    unittest.main()
