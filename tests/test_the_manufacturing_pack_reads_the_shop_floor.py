# -*- coding: utf-8 -*-
"""A manufacturer's warehouse, read by a product that knew order-to-cash.

The Infor M3 pack covered orders, invoices, inventory, purchasing and the
ledgers, and the product had no manufacturing vocabulary at all. Measured on
the real enrichment before this pack, with the M3 and star-schema packs active:

    SCRP_QTY   -> "scrp quantity"      MO_NO   -> "mo no"  (confidence 35)
    YLD_PCT    -> "yld percent"        WC_CD   -> "wc code"
    DWNTM_HRS  -> "dwntm hrs" (35)     PLNT_CD -> "plnt code"
    VHMFNO     -> "vhmfno" (35)        VOPLGR  -> "voplgr" (35)
    NET_SLS_AMT -> "net sls amount"    CAL_YR  -> "cal yr" (35)
    MO_STRT_DT -> no business date     RPT_DT  -> no business date

and the product had no production date roles: a manufacturing order has four
dates a plant lives by -- planned and actual start and finish -- plus release
and reporting, and "output last month" is a different number under each.

Two packs now carry it. packs/manufacturing.json is an industry pack: the
shop floor's spellings, the words people type for conventional mart columns,
and the production date roles under the names a mart gives them. It carries
no table or column dictionary, so it can never be auto-applied and never
displaces the ERP pack. packs/infor_m3.json learns M3's own manufacturing
files (MWOHED, MWOOPE, MWOMAT, product structures, routings, work centres,
planned orders), their record prefixes and field codes, and the mart tokens
an M3 data mart spells its measures and calendar with.

The six role KEYS are builtin, because vocab_packs validates every pack
pattern against DATE_ROLES and drops one it cannot find; the spellings are
not, because START_DT on a subscription table is not a production start.
Every test runs the real enrichment, the real date-role detector, the real
naming-profile detector or the real planner.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.date_roles import DATE_ROLES, detect_date_role, generated_date_role_synonyms  # noqa: E402
from core.identifier_intelligence import analyze_identifier, detect_naming_profile  # noqa: E402
from core.naming_convention import match_table_suffix  # noqa: E402
from core.schema_enrichment import enrich_columns  # noqa: E402
from core.semantic_planner import build_semantic_field_plan  # noqa: E402
from core.vocab_packs import (  # noqa: E402
    _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, list_available_packs, load_pack,
)

MANUFACTURING = "manufacturing"
M3 = "infor_m3"
ROLE_KEYS = {role.key: role for role in DATE_ROLES}

PRODUCTION = {
    "planned_start_date": ["PLN_STRT_DT", "PLANNED_START_DATE", "SCHED_START_DT", "MO_STRT_DT", "WO_START_DATE"],
    "actual_start_date": ["ACT_STRT_DT", "ACTUAL_START_DATE", "STARTED_DT"],
    "planned_finish_date": ["PLN_FIN_DT", "PLANNED_FINISH_DATE", "SCHED_END_DT", "MO_FIN_DT", "PLN_CMPL_DT"],
    "actual_finish_date": ["ACT_FIN_DT", "ACTUAL_FINISH_DATE", "COMPLETED_DT", "COMPLETION_DATE", "FINISHED_DATE"],
    "release_date": ["RLS_DT", "RELEASE_DATE", "MO_RLSE_DT"],
    "reported_date": ["RPT_DT", "REPORTED_DATE", "REPORTING_DT", "PRDN_DT", "PRODUCTION_DATE"],
}


def with_pack(*pack_ids):
    vocab = _clone_builtin()
    for pack_id in pack_ids:
        _merge_pack(vocab, load_pack(pack_id), pack_id)
    return vocab


def role_of(column, vocab=None):
    found = detect_date_role(column, vocab=vocab if vocab is not None else _clone_builtin())
    return found.key if found else None


def enriched(columns, *pack_ids):
    token = activate_vocab(with_pack(*pack_ids))
    try:
        return {c.column: c for c in enrich_columns(list(columns))}
    finally:
        deactivate_vocab(token)


class TheRolesExistForThePacksToPointAt(unittest.TestCase):

    def test_all_six_are_registered(self):
        for key in PRODUCTION:
            self.assertIn(key, ROLE_KEYS, key)

    def test_each_has_a_business_label_in_both_languages(self):
        from core import i18n
        for key in PRODUCTION:
            self.assertTrue(ROLE_KEYS[key].label.endswith("Date"), key)
            for lang in i18n.SUPPORTED_LANGUAGES:
                self.assertNotEqual(i18n.lookup(f"date_role.{key}", lang), f"date_role.{key}", (key, lang))
            self.assertNotEqual(i18n.lookup(f"date_role.{key}", "en"), i18n.lookup(f"date_role.{key}", "fr"), key)

    def test_a_question_can_name_each_one_in_business_words(self):
        expected = {
            "planned_start_date": "scheduled start",
            "actual_start_date": "when started",
            "planned_finish_date": "planned completion",
            "actual_finish_date": "when finished",
            "release_date": "when released",
            "reported_date": "production date",
        }
        for key, phrase in expected.items():
            self.assertIn(phrase, generated_date_role_synonyms(ROLE_KEYS[key]), key)

    def test_sales_facts_still_prefer_their_own_dates(self):
        """Priority decides the default when several roles could answer. A
        production date must never outrank an invoice or order date, or an
        order-to-cash mart that also carries a reporting date would start
        answering revenue questions by production date."""
        for key in PRODUCTION:
            self.assertLess(ROLE_KEYS[key].priority, ROLE_KEYS["invoice_date"].priority, key)
            self.assertLess(ROLE_KEYS[key].priority, ROLE_KEYS["order_date"].priority, key)


class TheBuiltinVocabularyReadsNoneOfIt(unittest.TestCase):
    """The load-bearing property: no pack, no production reading."""

    def test_not_one_mart_spelling_resolves_without_the_pack(self):
        for key, columns in PRODUCTION.items():
            for column in columns:
                with self.subTest(column=column):
                    self.assertIsNone(role_of(column), column)

    def test_every_one_resolves_with_it(self):
        vocab = with_pack(MANUFACTURING)
        for key, columns in PRODUCTION.items():
            for column in columns:
                with self.subTest(column=column):
                    self.assertEqual(role_of(column, vocab), key, column)

    def test_a_date_dimension_key_carries_the_role_too(self):
        vocab = with_pack(M3, MANUFACTURING)
        self.assertEqual(role_of("MO_STRT_DT_DMS_KEY", vocab), "planned_start_date")
        self.assertEqual(role_of("RPT_DT_DMS_KEY", vocab), "reported_date")

    def test_a_bare_start_or_end_date_is_left_for_the_admin(self):
        """Ambiguous by design: START_DT could be a contract, a subscription
        or a shift. The pack maps the spellings that say which."""
        vocab = with_pack(MANUFACTURING)
        self.assertIsNone(role_of("START_DT", vocab))
        self.assertIsNone(role_of("END_DATE", vocab))

    def test_m3s_own_codes_resolve_with_the_erp_pack(self):
        vocab = with_pack(M3)
        self.assertEqual(role_of("VHSTDT", vocab), "planned_start_date")
        self.assertEqual(role_of("VHFIDT", vocab), "planned_finish_date")
        self.assertEqual(role_of("VORPDT", vocab), "reported_date")
        self.assertIsNone(role_of("VHSTDT"))


class TheShopFloorsSpellingsBecomeWords(unittest.TestCase):

    SPELLINGS = {
        "SCRP_QTY": "scrap quantity",
        "YLD_PCT": "yield percent",
        "WC_CD": "work centre code",
        "WRKCTR_NM": "work centre name",
        "PLNT_CD": "plant code",
        "SHFT_CD": "shift code",
        "DWNTM_HRS": "downtime hours",
        "CHGOVR_MINS": "changeover minutes",
        "RWK_QTY": "rework quantity",
        "FG_QTY": "finished goods quantity",
        "RM_QTY": "raw material quantity",
        "PRDN_QTY": "production quantity",
        "CYC_TM": "cycle time",
    }

    def test_the_spellings_become_words(self):
        got = enriched(self.SPELLINGS, MANUFACTURING)
        for column, expected in self.SPELLINGS.items():
            self.assertEqual(got[column].expanded_name, expected, column)

    def test_without_the_pack_they_stay_unreadable(self):
        plain = enriched(self.SPELLINGS)
        unreadable = [c for c, e in plain.items() if e.expanded_name != self.SPELLINGS[c]]
        self.assertGreaterEqual(len(unreadable), 10, plain)

    def test_the_term_a_person_would_type_is_offered_too(self):
        """Two routes to the reader's word. The expansion gives the enrichment
        its candidates ("scrap percent"); the pack's aliases give the planner
        the phrases a person types instead ("scrap rate", "yield")."""
        column = enriched(["SCRP_PCT"], MANUFACTURING)["SCRP_PCT"]
        self.assertIn("scrap percent", column.business_candidates)
        vocab = with_pack(MANUFACTURING)
        self.assertIn("scrap rate", vocab.direct_aliases["SCRP_PCT"])
        self.assertIn("yield", vocab.direct_aliases["YLD_PCT"])
        self.assertIn("work centre", vocab.direct_aliases["WC_CD"])
        self.assertIn("return rate", vocab.direct_aliases["RTN_PCT"])

    def test_an_acronym_people_say_is_left_alone(self):
        got = enriched(["OEE_PCT", "MRP_QTY", "COGS_AMT"], MANUFACTURING)
        self.assertEqual(got["OEE_PCT"].expanded_name, "oee percent")
        self.assertEqual(got["MRP_QTY"].expanded_name, "mrp quantity")
        self.assertEqual(got["COGS_AMT"].expanded_name, "cogs amount")


class AnM3DataMartReadsItsOwnMeasuresAndCalendar(unittest.TestCase):
    """The tokens an M3 mart spells its measures and calendar with, learned
    by the M3 pack itself, because they are how that ERP's marts are built."""

    MART = {
        "NET_SLS_AMT": "net sales amount",
        "GRS_SLS_AMT": "gross sales amount",
        "RTN_AMT": "return amount",
        "RTN_QTY": "return quantity",
        "CUS_NO": "customer number",
        "IVC_NO": "invoice number",
        "CAL_YR": "calendar year",
        "MTH_NO": "month number",
        "QTR_NO": "quarter number",
        "WK_OF_YR": "week of year",
        "FAC_CD": "facility code",
        "DIV_CD": "division code",
    }

    def test_the_mart_columns_read_as_business_words(self):
        got = enriched(self.MART, M3)
        for column, expected in self.MART.items():
            self.assertEqual(got[column].expanded_name, expected, column)

    def test_with_the_manufacturing_pack_on_top_the_shop_floor_reads_too(self):
        got = enriched(["NET_SLS_AMT", "MO_NO", "SCRP_QTY", "WC_CD"], M3, MANUFACTURING)
        self.assertEqual(got["NET_SLS_AMT"].expanded_name, "net sales amount")
        self.assertEqual(got["MO_NO"].expanded_name, "manufacturing order number")
        self.assertEqual(got["SCRP_QTY"].expanded_name, "scrap quantity")
        self.assertEqual(got["WC_CD"].expanded_name, "work centre code")


class TheM3PackKnowsTheManufacturingFiles(unittest.TestCase):

    def test_record_prefixed_manufacturing_codes_resolve(self):
        vocab = with_pack(M3)
        expected = {
            "VHMFNO": "manufacturing order number",
            "VHPRNO": "product number",
            "VHMAQT": "manufactured quantity",
            "VHSCQT": "scrapped quantity",
            "VOOPNO": "operation number",
            "VOPLGR": "work center",
            "VMMTNO": "component item number",
            "PMCNQT": "quantity per unit",
            "PPPLNM": "work center name",
            "ROPLPN": "planned order number",
        }
        for physical, meaning in expected.items():
            analysis = analyze_identifier(physical, vocab=vocab)
            self.assertEqual(analysis.expanded_name, meaning, physical)
            self.assertEqual(analysis.confidence, 95, physical)

    def test_without_the_pack_they_are_noise(self):
        self.assertLess(analyze_identifier("VHMFNO", vocab=_clone_builtin()).confidence, 70)

    def test_roles_follow_the_codes(self):
        vocab = with_pack(M3)
        got = {c.column: c for c in enrich_columns(
            ["VHMFNO", "VHMAQT", "VHWHST", "VHSTDT", "VOPLGR", "VOSETI"], vocab=vocab)}
        self.assertEqual(got["VHMFNO"].role, "identifier")
        self.assertEqual(got["VHMAQT"].role, "measure")
        self.assertEqual(got["VOSETI"].role, "measure")
        self.assertEqual(got["VHWHST"].role, "status_filter")
        self.assertEqual(got["VHSTDT"].role, "date_key")
        self.assertEqual(got["VOPLGR"].role, "identifier")

    def test_the_files_are_classified_from_the_pack_alone(self):
        vocab = with_pack(M3)
        for table in ("MWOHED", "MWOOPE", "MWOMAT", "MMOPLP"):
            self.assertEqual(match_table_suffix(table, vocab=vocab).table_type, "fact_table", table)
        for table in ("MPDHED", "MPDMAT", "MPDOPE", "MPDWCT"):
            self.assertEqual(match_table_suffix(table, vocab=vocab).table_type, "dimension_table", table)
        self.assertIsNone(match_table_suffix("MWOHED", vocab=_clone_builtin()))

    def test_a_manufacturing_schema_auto_applies_the_m3_pack(self):
        profile = detect_naming_profile(
            ["VHCONO", "VHFACI", "VHMFNO", "VHPRNO", "VHMAQT", "VOOPNO", "VOPLGR", "VMMTNO"],
            ["MWOHED", "MWOOPE", "MWOMAT"],
        )
        self.assertEqual(profile["auto_applied_packs"], [M3])
        self.assertNotIn(MANUFACTURING, {r["pack_id"] for r in profile["pack_recommendations"]})

    def test_the_planner_reads_manufactured_and_scrapped_quantities(self):
        vocab = with_pack(M3)
        columns = {"M3.MWOHED": {"VHMFNO": "varchar", "VHPRNO": "varchar", "VHMAQT": "decimal",
                                 "VHSCQT": "decimal", "VHSTDT": "int"}}
        plan = build_semantic_field_plan("total manufactured quantity and scrapped quantity by product number",
                                         columns, None, vocab=vocab)
        selected = {field["column"]: field["role"] for field in plan["fields"]}
        self.assertEqual(selected.get("VHMAQT"), "measure")
        self.assertEqual(selected.get("VHSCQT"), "measure")
        self.assertEqual(selected.get("VHPRNO"), "dimension")


class ThePackKeepsTheContract(unittest.TestCase):

    def test_it_is_an_industry_pack_that_carries_nothing_that_scores(self):
        offered = {m["pack_id"]: m for m in list_available_packs()}
        self.assertEqual(offered[MANUFACTURING]["pack_kind"], "industry")
        self.assertEqual(offered[MANUFACTURING]["status"], "complete")
        pack = load_pack(MANUFACTURING)
        for key in ("column_dict", "table_dict", "record_prefixes"):
            self.assertNotIn(key, pack)

    def test_it_never_displaces_the_erp_pack_on_an_m3_warehouse(self):
        profile = detect_naming_profile(
            ["CUS_ORD_NUM", "CUS_DMS_KEY", "WHS_DMS_KEY", "ORNO", "CUNO", "WHLO", "ITNO", "ORQT", "IVDT",
             "SCRP_QTY", "PLN_STRT_DT", "RPT_DT"],
            ["ERP_CUS_ORD_FCT", "DMS_CUSTOMER", "OOHEAD", "MITMAS", "PRDN_FCT"],
        )
        self.assertEqual(profile.get("auto_applied_packs"), [M3])

    def test_the_erp_pack_wins_where_the_two_could_disagree(self):
        m3 = set(load_pack(M3).get("abbreviations") or {})
        mine = set(load_pack(MANUFACTURING).get("abbreviations") or {})
        self.assertEqual(m3 & mine, set())
        self.assertEqual(set(load_pack(M3)["direct_aliases"]) & set(load_pack(MANUFACTURING)["direct_aliases"]), set())

    def test_every_date_pattern_names_a_registered_role(self):
        pack = json.loads((ROOT / "packs" / "manufacturing.json").read_text(encoding="utf-8"))
        for entry in pack["date_role_patterns"]:
            self.assertIn(entry["role"], ROLE_KEYS, entry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
