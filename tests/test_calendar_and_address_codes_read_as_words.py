"""
tests/test_calendar_and_address_codes_read_as_words.py

A date dimension spells its own columns WK_OF_YR, DAY_OF_MTH, BUS_DAY_OF_WK,
HDY_FLG; a party's address ADR_LIN_1 and PSL_CD; an item its GRS_WT. Only two
packs knew YR, MTH and WK, and no vocabulary knew HDY, ADR, PSL or WT, so on a
warehouse without those packs:

- the reader saw "Wk Of Yr" and "Hdy Flg" where a column was named;
- the knowledge base documented WK_OF_YR as 'wk of yr', confidence 35,
  needs_admin_context -- asking the admin to write what the name already says;
- "average item gross weight" matched no column, "postal code" none either.

The built-in dictionary now reads fourteen such codes. It applies to every
column of every warehouse, so it holds only codes with one meaning. DLY is
daily in a table's name and delivery in a date's; QR is a quarter or a QR
code; those, and the others like them, are not guessed.

Column names are synthetic, in the naming convention of a real inventory
warehouse's schema read offline. No customer data.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.identifier_intelligence import tokenize_identifier
from core.schema_enrichment import (
    display_label,
    format_column_reference_for_vocab,
    format_schema_intelligence,
)
from core.semantic_planner import build_semantic_field_plan
from core.vocab_packs import (
    _account_cache,
    _clone_builtin,
    _merge_pack,
    activate_vocab,
    builtin_vocab,
    deactivate_vocab,
    load_pack,
    vocab_for_account,
)

NEW_CODES = {
    "YR", "MTH", "WK", "HDY", "FLG", "EFC", "FRQ", "UPD",
    "ADR", "PSL", "ZON", "WT", "MST", "PCY",
}


class _BuiltinVocabulary(unittest.TestCase):
    """Read through the built-in vocabulary alone: no pack, no tenant terms."""

    def setUp(self):
        self._token = activate_vocab(builtin_vocab())

    def tearDown(self):
        deactivate_vocab(self._token)


class TheReaderSeesWords(_BuiltinVocabulary):

    READINGS = {
        "WK_OF_YR": "Week Of Year",
        "DAY_OF_MTH": "Day Of Month",
        "BUS_DAY_OF_WK": "Business Day Of Week",
        "MTH_NM": "Month Name",
        "IVC_YR": "Invoice Year",
        "HDY_FLG": "Holiday Flag",
        "HDY_NM": "Holiday Name",
        "WK_DAY_FLG": "Week Day Flag",
        "PC_ADR_LIN_1": "Profit Center Address Line 1",
        "PC_PSL_CD": "Profit Center Postal Code",
        "ITM_GRS_WT": "Item Gross Weight",
        "ITM_STK_ZON_DMS_KEY": "Item Stock Zone Dimension Key",
        "ITM_BAL_EFC_DT_DMS_KEY": "Item Balance Effective Date Dimension Key",
        "ABC_CLS_FRQ_DMS_KEY": "Abc Class Frequency Dimension Key",
        "ITM_MST_CRN_DT_DMS_KEY": "Item Master Creation Date Dimension Key",
        "RPL_PCY_CD": "Replacement Policy Code",
        "UPD_TS": "Updated Timestamp",
    }

    def test_each_code_reads_as_its_word(self):
        for column, expected in self.READINGS.items():
            with self.subTest(column=column):
                self.assertEqual(display_label(column), expected)

    def test_every_new_code_is_read_somewhere_above(self):
        """Each code has a reading here, so dropping one fails a subtest."""
        read = {token for column in self.READINGS for token in column.split("_")}
        self.assertEqual(NEW_CODES - read, set())


class ACodeWithTwoMeaningsIsNotGuessed(_BuiltinVocabulary):
    """The dictionary reads every column the same way. A code that means one
    thing in one column and another thing in the next is left as spelled --
    in both, since a global reading would be wrong in one of them."""

    AMBIGUOUS = (
        ("DLY", ("ITM_BAL_DLY_FCT_KEY", "CFM_DLY_DT"), ("daily", "delivery")),
        ("QR", ("DAY_OF_QR", "ITM_QR_CD"), ("quarter",)),
        ("CTY", ("PC_CTY", "CUS_CTY_NM"), ("city", "county", "country")),
        ("PRV", ("PC_PRV", "PRV_YR_NET_AMT"), ("province", "previous")),
        ("STT", ("SLR_STT_PRV_DMS_KEY", "CTR_STT_DT"), ("state", "start", "status")),
        ("INV", ("NUM_OF_PHY_INV", "INV_NUM"), ("inventory", "invoice")),
        ("LST", ("LST_UPD_DT", "LST_PRC_AMT"), ("last", "list")),
        ("EXT", ("EXT_SYS_ID", "EXT_PRC_AMT"), ("external", "extended")),
        ("CTR", ("PFT_CTR_CD", "CLK_CTR_PCT"), ("centre", "center", "counter")),
        ("RM", ("SHOW_RM_FLG", "RM_ISS_QTY"), ("room", "raw material")),
        ("CHN", ("SLY_CHN_PCY_CD", "SLS_CHN_CD"), ("chain", "channel")),
    )

    def test_the_code_is_left_as_spelled(self):
        for code, columns, meanings in self.AMBIGUOUS:
            for column in columns:
                with self.subTest(column=column):
                    label = display_label(column).lower()
                    self.assertIn(code.lower(), label.split())
                    for meaning in meanings:
                        self.assertNotIn(meaning, label)


class CodesAndWordsAreNotSplitOnTheNewCodes(unittest.TestCase):
    """The segmenter splits a compact code only into dictionary entries, and a
    two-letter entry is a new way to split one: COUNT was once read as
    CO + UNT, "company unit"."""

    def test_an_erp_field_code_is_never_read_through_them(self):
        m3 = _clone_builtin()
        _merge_pack(m3, load_pack("infor_m3"), "infor_m3")
        codes = [code for code in m3.column_dict if code.isalnum()]
        self.assertGreater(len(codes), 100)
        for code in codes:
            with self.subTest(code=code):
                self.assertEqual(set(tokenize_identifier(code, vocab=m3)) & NEW_CODES, set())

    def test_ordinary_words_stay_whole(self):
        vocab = builtin_vocab()
        for word in (
            "WEEKLY", "YEARLY", "MONTHLY", "WEIGHT", "ADDRESS", "POSTAL", "ZONE",
            "MASTER", "POLICY", "HOLIDAY", "FLAG", "UPDATED", "EFFECTIVE",
            "FREQUENCY", "COUNT", "WITH", "WORKSHOP", "MYSTERY",
        ):
            with self.subTest(word=word):
                self.assertEqual(tokenize_identifier(word, vocab=vocab), [word])


class TheQuestionReachesTheColumn(_BuiltinVocabulary):
    """The planner matches a question to a column through the column's words.
    It trusts an expansion only when the name is read with confidence, and a
    single unread code kept the whole name below that bar."""

    TABLE_COLUMNS = {
        "MART.ITM_DMS": {c: "" for c in (
            "ITM_DMS_KEY", "ITM_CD", "ITM_NM", "ITM_GRS_WT", "ITM_NET_WT")},
        "MART.PC_DVN_DMS": {c: "" for c in (
            "PC_DVN_DMS_KEY", "PC_ADR_LIN_1", "PC_PSL_CD")},
        "MART.DT_DMS": {c: "" for c in (
            "DT_DMS_KEY", "DMS_DT", "WK_OF_YR", "HDY_FLG", "HDY_NM")},
    }

    def _fields(self, question: str) -> set[tuple[str, str]]:
        plan = build_semantic_field_plan(question, self.TABLE_COLUMNS)
        return {(f["table"], f["column"]) for f in plan.get("fields", [])}

    def test_a_weight(self):
        self.assertIn(("MART.ITM_DMS", "ITM_GRS_WT"),
                      self._fields("average item gross weight by item"))
        self.assertIn(("MART.ITM_DMS", "ITM_NET_WT"),
                      self._fields("list items with their item net weight"))

    def test_a_postal_code(self):
        self.assertIn(("MART.PC_DVN_DMS", "PC_PSL_CD"),
                      self._fields("profit center postal code for each division"))

    def test_a_holiday(self):
        self.assertIn(("MART.DT_DMS", "HDY_NM"),
                      self._fields("stock on hand by holiday name"))


class TheKnowledgeBaseIsToldTheMeaning(_BuiltinVocabulary):
    """The KB build reads these blocks; a field below confidence 70 is written
    up as a candidate needing the admin's context."""

    MEANINGS = (
        ("WK_OF_YR", "week of year"),
        ("HDY_FLG", "holiday flag"),
        ("PC_PSL_CD", "profit center postal code"),
        ("ITM_GRS_WT", "item gross weight"),
    )

    def test_each_field_is_documented_as_its_meaning(self):
        text = format_schema_intelligence("DT_DMS", [c for c, _ in self.MEANINGS])
        lines = {line.split(":", 1)[0][2:]: line for line in text.splitlines() if line.startswith("- ")}
        for column, meaning in self.MEANINGS:
            with self.subTest(column=column):
                self.assertIn(f"expanded='{meaning}'", lines[column])
                confidence = int(re.search(r"confidence=(\d+)", lines[column]).group(1))
                self.assertGreaterEqual(confidence, 70)

    def test_the_admin_is_not_asked_to_write_it(self):
        text = format_column_reference_for_vocab("DT_DMS", [c for c, _ in self.MEANINGS])
        for column, meaning in self.MEANINGS:
            with self.subTest(column=column):
                entry = re.search(rf"{column} \(([^)]*)\)", text).group(1)
                self.assertIn(f"meaning={meaning}", entry)
                self.assertNotIn("needs_admin_context", entry)


class ATenantStillDecides(unittest.TestCase):
    """The built-in reading is a default. A tenant's own vocabulary wins."""

    def test_a_tenant_overlay_reads_a_code_its_own_way(self):
        account = "acct_calendar_codes"
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / account).mkdir()
            (Path(tmp) / account / "vocab.json").write_text(
                json.dumps({"abbreviations": {"WT": "wait time"}}), encoding="utf-8")
            with patch("core.vocab_packs._CLIENTS_DIR", Path(tmp)), \
                 patch("core.vocab_packs._client_pack_ids", return_value=[]):
                _account_cache.pop(account, None)
                token = activate_vocab(vocab_for_account(account))
                try:
                    self.assertEqual(display_label("ORD_WT_HRS"), "Order Wait Time Hrs")
                    self.assertEqual(display_label("WK_OF_YR"), "Week Of Year")
                finally:
                    deactivate_vocab(token)
                    _account_cache.pop(account, None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
