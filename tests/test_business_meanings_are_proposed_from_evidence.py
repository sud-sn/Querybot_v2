"""
tests/test_business_meanings_are_proposed_from_evidence.py

A warehouse's codes that the built-in dictionary cannot read -- because they
mean one thing in one column and another in the next, or because no
vocabulary knows them -- are read from what the warehouse itself carries:
where the code stands, what stands beside it, what its column holds, and who
the party it names is to the business. Each reading is a proposal with its
evidence, for an admin to confirm; a code nothing reads is listed for the
admin to write.

On a real inventory warehouse's schema and sampled values, read offline, the
engine proposes: SLR, read "seller", as supplier (its table carries a payee
and the stock facts point at it); a profit center's CO, read "company", as a
country (its only value is CA); DLY as daily in the daily balance table's
name; QR as a quarter in the calendar; CTY and PRV as a city and a province
(from their values); HND as "hand" after ON; FR as French beside its English
twin; and lists the four codes nothing reads.

Synthetic names in that naming convention; no customer data.
"""

from __future__ import annotations

import unittest

from core.business_meaning import CODE, COLUMN, TABLE, propose_meanings, value_kinds
from core.vocab_packs import _clone_builtin, _merge_pack, builtin_vocab, load_pack


def _tables(spec: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    return {table: {column: "nvarchar" for column in columns} for table, columns in spec.items()}


STAR = _tables({
    "MART.DT_DMS": ["DT_DMS_KEY", "DMS_DT", "YR", "QR", "MTH", "WK_OF_YR", "DAY_OF_QR", "HDY_FLG"],
    "MART.ITM_BAL_DLY_FCT": [
        "ITM_BAL_DLY_FCT_KEY", "ITM_DMS_KEY", "SLR_DMS_KEY", "WHS_DMS_KEY", "PFT_CTR_DMS_KEY",
        "ITM_STK_ZON_DMS_KEY", "ON_HND_QTY", "ALC_ON_HND_QTY", "AZ_LST_UPD_TS",
    ],
    "MART.ITM_BAL_PRD_FCT": ["ITM_BAL_PRD_FCT_KEY", "ITM_DMS_KEY", "PSV_RE_CLS_QTY", "NUM_OF_PHY_INV"],
    "MART.SLR_DMS": ["SLR_DMS_KEY", "SLR_CD", "SLR_NM", "PYE_CD", "CO_OP_SLR_FLG", "SLR_STT_PRV_DMS_KEY"],
    "MART.PFT_CTR_DMS": [
        "PFT_CTR_DMS_KEY", "PFT_CTR_NM", "PC_ADR_LIN_1", "PC_CTY", "PC_PRV", "PC_PSL_CD", "PC_CO",
    ],
    "MART.WHS_DMS": ["WHS_DMS_KEY", "WHS_DSC", "SQ_FT", "SHOW_RM_FLG"],
    "MART.ITM_DMS": ["ITM_DMS_KEY", "ITM_NM", "ITM_FR_NM", "QXR_IND", "ACME_GRP_CD"],
})

VALUES = {
    "MART.PFT_CTR_DMS.PC_CO": ["CA", "CA", "", "NO_VALUE"],
    "PFT_CTR_DMS.PC_PRV": ["ON", "QC", "AB", "BC", "NO_VALUE", None],
    "PFT_CTR_DMS.PC_CTY": ["LONDON", "CALGARY", "OTTAWA", "LAVAL", "NULL value provided"],
}


def _m3():
    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
    return vocab


def _proposals(tables=STAR, vocab=None, values=None, ignore=("ACME",)):
    found = propose_meanings(tables, vocab=vocab or builtin_vocab(), values=values, ignore=ignore)
    return {(p.scope, p.subject): p for p in found}


class TheValuesSettleWhatANameMeans(unittest.TestCase):

    def test_what_sampled_values_are(self):
        cases = (
            (["CA"], {"country", "state"}),
            (["ON", "QC", "AB"], {"province"}),
            (["NY", "TX", "CA"], {"state"}),
            (["ON", "NY"], {"state or province"}),
            (["H3B 2Y5", "M5V 3L9", "90210"], {"postal code"}),
            (["LONDON", "CALGARY", "OTTAWA"], {"city"}),
            (["ORANGE COUNTY", "KING COUNTY", "COOK COUNTY"], {"county"}),
            (["CAN", "USA"], {"country"}),
            (["100", "200"], set()),
        )
        for values, kinds in cases:
            with self.subTest(values=values):
                self.assertEqual(value_kinds(values)[0], kinds)

    def test_placeholder_rows_and_blanks_are_not_values(self):
        kinds, real = value_kinds(["CA", "", None, "NO_VALUE", "NULL value provided", "NO_MATCH"])
        self.assertEqual(real, ["CA"])
        self.assertIn("country", kinds)

    def test_a_country_code_column_read_as_a_company(self):
        proposal = _proposals(values=VALUES)[(COLUMN, "PC_CO")]
        self.assertEqual(proposal.reading, "profit center country")
        self.assertEqual((proposal.rule, proposal.confidence), ("values", 90))
        self.assertIn("(CA)", " ".join(proposal.evidence))
        self.assertIn('CO reads "company" elsewhere', proposal.evidence)

    def test_values_that_are_not_countries_veto_the_address(self):
        values = dict(VALUES, **{"PFT_CTR_DMS.PC_CO": ["100", "200", "300"]})
        self.assertNotIn((COLUMN, "PC_CO"), _proposals(values=values))

    def test_without_values_the_address_only_suggests_it(self):
        proposal = _proposals()[(COLUMN, "PC_CO")]
        self.assertEqual((proposal.reading, proposal.rule), ("profit center country", "address"))
        self.assertLess(proposal.confidence, 70)

    def test_province_and_city_from_their_values(self):
        found = _proposals(values=VALUES)
        self.assertEqual(found[(CODE, "PRV")].reading, "province")
        self.assertIn("province code (AB, BC, ON, QC)", " ".join(found[(CODE, "PRV")].evidence))
        self.assertEqual(found[(CODE, "CTY")].reading, "city")


class WhereACodeStands(unittest.TestCase):

    def test_a_quarter_in_a_calendar(self):
        proposal = _proposals()[(CODE, "QR")]
        self.assertEqual((proposal.reading, proposal.rule), ("quarter", "calendar"))
        self.assertEqual(proposal.where, ["DT_DMS.DAY_OF_QR", "DT_DMS.QR"])

    def test_a_qr_outside_the_calendar_is_not_a_quarter(self):
        tables = dict(STAR, **_tables({"MART.ITM_DMS": ["ITM_DMS_KEY", "ITM_NM", "ITM_QR_CD"]}))
        found = _proposals(tables)
        self.assertEqual(found[(COLUMN, "QR")].reading, "quarter")
        self.assertEqual(found[(COLUMN, "DAY_OF_QR")].reading, "day of quarter")
        self.assertEqual(found[(CODE, "QR")].reading, "")
        self.assertEqual(found[(CODE, "QR")].where, ["ITM_DMS.ITM_QR_CD"])

    def test_daily_is_the_grain_of_the_table(self):
        proposal = _proposals()[(CODE, "DLY")]
        self.assertEqual((proposal.reading, proposal.rule), ("daily", "grain"))
        self.assertEqual(proposal.where, ["ITM_BAL_DLY_FCT", "ITM_BAL_DLY_FCT.ITM_BAL_DLY_FCT_KEY"])
        self.assertIn("ITM_BAL_PRD_FCT", proposal.evidence[0])

    def test_daily_in_a_table_and_delivery_in_a_date(self):
        tables = dict(STAR, **_tables({
            "MART.CUS_ORD_FCT": ["CUS_ORD_FCT_KEY", "CFM_DLY_DT", "RQS_DLY_DT"],
            "MART.CUS_SHP_FCT": ["CUS_SHP_FCT_KEY", "CFM_DLY_DT"],
        }))
        found = _proposals(tables)
        self.assertNotIn((CODE, "DLY"), found)
        self.assertEqual(found[(TABLE, "ITM_BAL_DLY_FCT")].reading, "item balance daily fact")
        self.assertEqual(found[(COLUMN, "ITM_BAL_DLY_FCT_KEY")].reading, "item balance daily fact key")
        self.assertEqual(found[(COLUMN, "CFM_DLY_DT")].reading, "confirmed delivery date")
        self.assertEqual(found[(COLUMN, "CFM_DLY_DT")].rule, "event_date")
        self.assertEqual(found[(COLUMN, "CFM_DLY_DT")].where, [
            "CUS_ORD_FCT.CFM_DLY_DT", "CUS_SHP_FCT.CFM_DLY_DT"])

    def test_a_code_read_by_its_neighbour(self):
        found = _proposals()
        for code, reading in (
            ("HND", "hand"), ("SQ", "square"), ("FT", "feet"), ("RM", "room"),
            ("CTR", "center"), ("INV", "inventory"), ("STT", "state"),
        ):
            with self.subTest(code=code):
                self.assertEqual(found[(CODE, code)].reading, reading)
        self.assertEqual(found[(CODE, "HND")].where, [
            "ITM_BAL_DLY_FCT.ALC_ON_HND_QTY", "ITM_BAL_DLY_FCT.ON_HND_QTY"])

    def test_the_same_code_read_two_ways_by_its_neighbours(self):
        tables = _tables({
            "MART.ORD_LIN_FCT": ["LST_UPD_DT", "LST_PRC_AMT", "EXT_SYS_ID", "EXT_PRC_AMT",
                                 "PRV_YR_NET_AMT", "SLY_CHN_CD", "SLS_CHN_CD"],
        })
        found = _proposals(tables)
        for column, reading in (
            ("LST_UPD_DT", "last updated date"), ("LST_PRC_AMT", "list price amount"),
            ("EXT_SYS_ID", "external sys id"), ("EXT_PRC_AMT", "extended price amount"),
            ("SLY_CHN_CD", "supply chain code"), ("SLS_CHN_CD", "sls channel code"),
        ):
            with self.subTest(column=column):
                self.assertEqual(found[(COLUMN, column)].reading, reading)
        self.assertEqual(found[(CODE, "PRV")].reading, "previous")
        for code in ("LST", "EXT", "CHN"):
            self.assertNotIn((CODE, code), found)

    def test_two_codes_read_as_one_word_belong_to_the_column(self):
        found = _proposals()
        self.assertEqual(found[(COLUMN, "CO_OP_SLR_FLG")].reading, "co-op seller flag")
        self.assertEqual(found[(COLUMN, "PSV_RE_CLS_QTY")].reading, "positive reclass quantity")
        for code in ("CO", "OP", "RE", "CLS"):
            self.assertNotIn((CODE, code), found)


class AReadingTheVocabularyAlreadyHas(unittest.TestCase):

    def test_a_pack_that_reads_daily_as_delivery_is_corrected_where_it_is_wrong(self):
        found = _proposals(vocab=_m3())
        self.assertNotIn((CODE, "DLY"), found)
        table = found[(TABLE, "ITM_BAL_DLY_FCT")]
        self.assertEqual(table.reading, "item balance daily fact")
        self.assertIn('DLY reads "delivery" elsewhere', table.evidence)
        self.assertEqual(found[(COLUMN, "ITM_BAL_DLY_FCT_KEY")].reading, "item balance daily fact key")

    def test_another_spelling_of_the_same_word_is_not_a_correction(self):
        found = _proposals(vocab=_m3())
        self.assertFalse([key for key in found if "PFT_CTR" in key[1]])
        self.assertNotIn((CODE, "CTR"), found)

    def test_a_code_the_vocabulary_reads_the_same_way_is_not_proposed(self):
        found = _proposals(vocab=_m3())
        self.assertNotIn((CODE, "HND"), found)


class WhoAPartyIsToTheBusiness(unittest.TestCase):

    def test_a_seller_the_business_pays_is_its_supplier(self):
        proposal = _proposals()[(CODE, "SLR")]
        self.assertEqual((proposal.reading, proposal.rule), ("supplier", "party"))
        self.assertEqual(proposal.synonyms, ["vendor", "seller"])
        self.assertIn("SLR_DMS carries a payee (PYE_CD)", proposal.evidence[0])
        self.assertIn("SLR_DMS.SLR_NM", proposal.where)

    def test_a_seller_whose_items_it_stocks_is_its_supplier(self):
        tables = dict(STAR, **_tables({"MART.SLR_DMS": ["SLR_DMS_KEY", "SLR_CD", "SLR_NM"]}))
        proposal = _proposals(tables)[(CODE, "SLR")]
        self.assertEqual(proposal.reading, "supplier")
        self.assertEqual(proposal.evidence, [
            "ITM_BAL_DLY_FCT holds stock (ON_HND_QTY, ALC_ON_HND_QTY) by this party"])

    def test_a_seller_with_neither_is_left_as_read(self):
        tables = _tables({
            "MART.SLR_DMS": ["SLR_DMS_KEY", "SLR_CD", "SLR_NM"],
            "MART.CUS_ORD_FCT": ["CUS_ORD_FCT_KEY", "SLR_DMS_KEY", "NET_AMT"],
        })
        self.assertNotIn((CODE, "SLR"), _proposals(tables))


class WhatIsLeftForTheAdmin(unittest.TestCase):

    def test_a_code_nothing_reads_is_listed_where_it_appears(self):
        proposal = _proposals()[(CODE, "QXR")]
        self.assertEqual((proposal.reading, proposal.rule, proposal.confidence), ("", "unread", 0))
        self.assertEqual(proposal.where, ["ITM_DMS.QXR_IND"])

    def test_words_numbers_names_and_load_columns_are_not_codes(self):
        found = _proposals()
        for token in ("OF", "DAY", "ACME", "AZ", "LST", "UPD", "TS", "1"):
            with self.subTest(token=token):
                self.assertNotIn((CODE, token), found)
        self.assertIn((CODE, "ACME"), _proposals(ignore=()))


class FrenchTwins(unittest.TestCase):

    def test_a_column_repeating_another_in_french(self):
        proposal = _proposals()[(CODE, "FR")]
        self.assertEqual((proposal.reading, proposal.rule), ("french", "twin"))
        self.assertEqual(proposal.evidence, ["ITM_FR_NM repeats ITM_NM in french"])

    def test_fr_without_a_twin_is_not_french(self):
        tables = _tables({"MART.PRC_LST_DMS": ["PRC_LST_DMS_KEY", "PRC_FR_DT", "PRC_TO_DT"]})
        proposal = _proposals(tables)[(CODE, "FR")]
        self.assertEqual(proposal.reading, "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
