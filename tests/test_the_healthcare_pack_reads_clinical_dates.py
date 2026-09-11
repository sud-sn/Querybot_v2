# -*- coding: utf-8 -*-
"""tests/test_the_healthcare_pack_reads_clinical_dates.py

A claim for a 1 Jan service, submitted on the 15th, paid in February.
"January revenue" is a different number under each of those three dates, and
the product could read none of them.

Executed against the builtin vocabulary, every one of these returned None:

    SERVICE_DATE  DATE_OF_SERVICE  DOS  ENCOUNTER_DATE  VISIT_DATE
    DISPENSE_DATE  FILL_DATE  RX_FILL_DATE
    ADMIT_DATE  ADMISSION_DATE  DISCHARGE_DATE
    CLAIM_DATE  CLAIM_SUBMISSION_DATE  PRESCRIPTION_DATE  WRITTEN_DATE

derive_date_role rescued the star-schema spellings -- a physical FK into a date
dimension is evidence a name is not, so SERVICE_DATE_KEY was already reviewable
-- and left every flat claims or pharmacy table with no discoverable business
date at all. Since an aggregating fact with no settled date is now REFUSED
rather than answered over all time, that was a provider or payer tenant whose
period questions the product declined and could not explain.

The spellings live in packs/healthcare.json and NOT in the builtin patterns,
which is the load-bearing decision here and is tested both ways below.
SERVICE_DATE is a field-service visit in distribution; DISCHARGE_DATE is
effluent in utilities and a mortgage discharge in lending; ADMISSION_DATE is a
university or a turnstile. Reading any of them as a governed clinical date by
default would filter an answer confidently, disclose it as correct, and be
about the wrong event. A pack is how a tenant says "we are a hospital".

The six role KEYS do have to be builtin, because vocab_packs validates every
pack pattern against DATE_ROLES and skips one it cannot find.

One seam had to be fixed for the pack to be safe to ship. A pack's detection
score comes from column_dict, table_dict and record_prefixes -- how a warehouse
SPELLS things -- and an industry pack carries none of that, so its
unique_evidence is 0 and it can never be auto-applied. But date_role_patterns DO
add to the score, and the auto-apply decision read recommendations[0] before
testing eligibility: an industry pack that out-scored the ERP pack became the
top, failed the unique_evidence gate, and left the warehouse with NO pack at
all. A hospital group running Dynamics lost its Dynamics pack to a pack that
could never have been applied. Invisible until now only because the two industry
packs shipped so far score exactly zero on every warehouse.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.date_roles import DATE_ROLES, detect_date_role  # noqa: E402
from core.identifier_intelligence import detect_naming_profile  # noqa: E402
from core.schema_enrichment import enrich_columns  # noqa: E402
from core.vocab_packs import (  # noqa: E402
    _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab,
    list_available_packs, load_pack,
)

HEALTHCARE = "healthcare"
ROLE_KEYS = {role.key: role for role in DATE_ROLES}

# The six clinical roles, and the spellings a provider, payer or PBM warehouse
# actually uses for each.
CLINICAL = {
    "service_date": [
        "SERVICE_DATE", "SVC_DATE", "SRVC_DATE", "SVC_DT", "DOS",
        "DATE_OF_SERVICE", "ENCOUNTER_DATE", "ENCTR_DT", "VISIT_DATE",
        "TREATMENT_DATE", "PROCEDURE_DATE", "PROC_DT", "SURGERY_DATE",
        "SERVICE_FROM_DATE", "FROM_SERVICE_DATE", "SERVICE_DATE_KEY",
    ],
    "dispense_date": [
        "DISPENSE_DATE", "DISPENSED_DATE", "DISPENSING_DATE", "FILL_DATE",
        "FILLED_DATE", "FILLED_DT", "RX_FILL_DATE", "DISPENSE_DATE_SK",
    ],
    "admission_date": [
        "ADMIT_DATE", "ADMISSION_DATE", "ADMITTED_DATE", "ADM_DT",
        "IP_ADMIT_DATE", "INPATIENT_ADMIT_DATE", "DATE_OF_ADMISSION",
        "ADMIT_DATE_KEY",
    ],
    "discharge_date": [
        "DISCHARGE_DATE", "DISCHARGED_DATE", "DSCH_DATE", "DSCH_DT",
        "DISCH_DATE", "DATE_OF_DISCHARGE", "DISCHARGE_DATE_ID",
    ],
    "claim_date": [
        "CLAIM_DATE", "CLM_DATE", "CLM_DT", "CLAIM_SUBMISSION_DATE",
        "CLAIM_SUBMIT_DATE", "SUBMISSION_DATE", "CLAIM_RECEIVED_DATE",
        "CLAIM_RECEIPT_DATE", "CLAIM_ENTRY_DATE",
    ],
    "prescription_date": [
        "PRESCRIPTION_DATE", "PRESCRIBED_DATE", "WRITTEN_DATE",
        "RX_WRITTEN_DATE", "RX_DT", "ORDER_WRITTEN_DATE",
    ],
}

# Revenue-cycle dates that are the SAME business event the product already
# knows, under the words healthcare writes them in. A second role key for one
# of these would be worse than none: the resolver would then see two candidates
# and ask a reader to choose between two names for one thing.
ONTO_EXISTING = {
    "payment_date": ["PAID_DATE", "ADJUDICATION_DATE", "ADJUDICATED_DATE",
                     "REMIT_DATE", "REMITTANCE_DATE", "CHECK_DATE", "EOB_DATE"],
    "invoice_date": ["STATEMENT_DATE", "BILL_DATE", "BILLED_DATE",
                     "BILLING_DATE", "INVOICE_SENT_DATE"],
    "accounting_date": ["POSTED_DATE", "POSTING_DATE", "GL_POST_DATE"],
}

# Never a business date, pack or no pack. A period filter landing on one of
# these would answer "revenue last month" by date of birth.
NEVER = [
    "BIRTH_DATE", "DATE_OF_BIRTH", "DOB", "BIRTHDATE",
    "DEATH_DATE", "DATE_OF_DEATH", "DECEASED_DATE",
    "ENROLLMENT_DATE", "ENROLMENT_DATE", "ELIGIBILITY_DATE", "TERM_DATE",
    "EFFECTIVE_DATE", "COVERAGE_START_DATE",
    # Dropped from the pack deliberately: each names a different event than the
    # role it would have landed on.
    "DC_DATE",          # a distribution centre in a health system's supply mart
    "ADJ_DATE",         # an adjustment is not an adjudication
    "SOLD_DATE",        # picked up at the counter, not filled
    "ARRIVAL_DATE",     # ED arrival is not an inpatient admission
    "SERVICE_THRU_DATE",  # the end of a service span; DOS is the from-date
]


def with_pack(*pack_ids):
    vocab = _clone_builtin()
    for pack_id in pack_ids:
        _merge_pack(vocab, load_pack(pack_id), pack_id)
    return vocab


def role_of(column, vocab=None):
    found = detect_date_role(column, vocab=vocab if vocab is not None else _clone_builtin())
    return found.key if found else None


class TheRolesExistForThePackToPointAt(unittest.TestCase):

    def test_all_six_are_registered(self):
        for key in CLINICAL:
            self.assertIn(key, ROLE_KEYS, key)

    def test_a_pack_pattern_naming_an_unregistered_role_would_be_dropped(self):
        """Which is why the keys are builtin even though the spellings are not.
        Executed rather than asserted: this is the mechanism the pack depends
        on."""
        vocab = _clone_builtin()
        _merge_pack(vocab, {"date_role_patterns": [
            {"pattern": "^TRIAGE_DATE$", "role": "triage_date"},
        ]}, "synthetic")
        self.assertEqual(vocab.date_role_patterns, [])
        self.assertIsNone(role_of("TRIAGE_DATE", vocab))

    def test_each_has_a_business_label_and_both_languages(self):
        from core import i18n

        for key in CLINICAL:
            role = ROLE_KEYS[key]
            self.assertTrue(role.label.endswith("Date"), role.label)
            for lang in i18n.SUPPORTED_LANGUAGES:
                rendered = i18n.lookup(f"date_role.{key}", lang)
                self.assertNotEqual(rendered, f"date_role.{key}", (key, lang))
            self.assertNotEqual(
                i18n.lookup(f"date_role.{key}", "en"),
                i18n.lookup(f"date_role.{key}", "fr"), key)

    def test_a_question_can_name_each_one_in_business_words(self):
        """generated_date_role_synonyms is what matches a reader's phrase to a
        role. "date of service" and "when dispensed" have to reach one."""
        from core.date_roles import generated_date_role_synonyms

        expected = {
            "service_date": "date of service",
            "dispense_date": "when dispensed",
            "admission_date": "date admitted",
            "discharge_date": "when discharged",
            "claim_date": "date submitted",
            "prescription_date": "date prescribed",
        }
        for key, phrase in expected.items():
            terms = generated_date_role_synonyms(ROLE_KEYS[key])
            self.assertIn(phrase, terms, (key, terms))


class TheBuiltinVocabularyStillReadsNoneOfIt(unittest.TestCase):
    """The load-bearing property. A distribution tenant's SERVICE_DATE is a
    field-service visit, and no pack means no clinical reading."""

    # Two spellings are not unreadable without the pack -- they are readable as
    # something GENERIC. "claim received date" contains the builtin word
    # "received", so it lands on receipt_date, which is a defensible reading of
    # the name and the wrong business date for a claim: a payer's month-end is
    # about when the claim arrived as a claim, and receipt_date is the word the
    # rest of the product uses for goods arriving. The pack specialises them.
    SPECIALISED = {"CLAIM_RECEIVED_DATE": "receipt_date",
                   "CLAIM_RECEIPT_DATE": "receipt_date"}

    def test_not_one_clinical_spelling_resolves_without_the_pack(self):
        for key, columns in CLINICAL.items():
            for column in columns:
                if column.endswith(("_KEY", "_ID", "_SK", "_FK")):
                    continue   # a date-dimension FK is evidence a name is not
                if column in self.SPECIALISED:
                    continue
                with self.subTest(column=column):
                    self.assertIsNone(role_of(column), column)

    def test_the_two_generic_readings_are_specialised_not_replaced(self):
        """Pack patterns run BEFORE the builtin ones for exactly this: the
        builtin keeps a sane generic answer for a tenant with no pack, and the
        pack narrows it for a tenant who has said what business they are in."""
        for column, generic in self.SPECIALISED.items():
            self.assertEqual(role_of(column), generic, column)
            self.assertEqual(role_of(column, with_pack(HEALTHCARE)),
                             "claim_date", column)

    def test_a_date_dimension_key_is_still_reviewable_without_the_pack(self):
        """derive_date_role, unchanged: a physical FK into a date dimension is
        hard evidence whatever the column is called."""
        for column in ("SERVICE_DATE_KEY", "ADMIT_DATE_KEY",
                       "DISCHARGE_DATE_ID", "DISPENSE_DATE_SK"):
            self.assertIsNotNone(role_of(column), column)

    def test_no_builtin_pattern_mentions_a_clinical_word(self):
        """Read as a compiled pattern list, not as source text: whatever the
        spelling, no builtin regex may claim one of these columns."""
        import core.date_roles as dr

        for pattern, role_key in dr._COLUMN_PATTERNS:
            for column in ("SERVICE_DT", "ADMIT_DT", "DISCHARGE_DT",
                           "DISPENSE_DT", "CLAIM_DT", "PRESCRIPTION_DT"):
                self.assertIsNone(pattern.search(column), (column, role_key))


class TheClinicalDatesReadCorrectlyWithThePack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vocab = with_pack(HEALTHCARE)

    def test_every_clinical_spelling_reaches_its_own_role(self):
        for key, columns in CLINICAL.items():
            for column in columns:
                with self.subTest(column=column):
                    self.assertEqual(role_of(column, self.vocab), key, column)

    def test_the_revenue_cycle_dates_reuse_the_roles_that_exist(self):
        for key, columns in ONTO_EXISTING.items():
            for column in columns:
                with self.subTest(column=column):
                    self.assertEqual(role_of(column, self.vocab), key, column)

    def test_no_patient_attribute_ever_becomes_a_business_date(self):
        for column in NEVER:
            with self.subTest(column=column):
                self.assertIsNone(role_of(column, self.vocab), column)
                self.assertIsNone(role_of(column), column)

    def test_the_three_dates_a_claim_carries_stay_three_dates(self):
        """The whole reason this pack exists. Service, submission and payment
        are different months, and collapsing any two onto one role would make
        "January revenue" silently mean whichever the resolver picked."""
        keys = {role_of(column, self.vocab) for column in
                ("DATE_OF_SERVICE", "CLAIM_SUBMISSION_DATE", "PAID_DATE")}
        self.assertEqual(keys, {"service_date", "claim_date", "payment_date"})

    def test_the_pharmacy_dates_stay_two_dates(self):
        """Written and filled are different events, and days-to-fill is a
        question a PBM asks."""
        self.assertNotEqual(role_of("WRITTEN_DATE", self.vocab),
                            role_of("FILL_DATE", self.vocab))

    def test_an_inpatient_stay_has_both_ends(self):
        self.assertEqual(role_of("ADMIT_DATE", self.vocab), "admission_date")
        self.assertEqual(role_of("DISCHARGE_DATE", self.vocab), "discharge_date")


class ThePackIsOfferedAndShapedLikeAnIndustryPack(unittest.TestCase):

    def test_an_admin_is_offered_it(self):
        offered = {m["pack_id"]: m for m in list_available_packs()}
        self.assertIn(HEALTHCARE, offered)
        self.assertEqual(offered[HEALTHCARE]["status"], "complete")
        self.assertTrue(offered[HEALTHCARE]["description"])

    def test_it_says_nothing_about_how_a_column_is_spelled(self):
        """No column_dict or table_dict: an industry pack must not compete with
        the ERP pack the naming detects."""
        pack = load_pack(HEALTHCARE)
        self.assertNotIn("column_dict", pack)
        self.assertNotIn("table_dict", pack)
        self.assertNotIn("record_prefixes", pack)
        self.assertTrue(pack["abbreviations"])
        self.assertTrue(pack["date_role_patterns"])

    def test_every_pattern_compiles_and_names_a_real_role(self):
        pack = load_pack(HEALTHCARE)
        vocab = with_pack(HEALTHCARE)
        self.assertEqual(len(vocab.date_role_patterns),
                         len(pack["date_role_patterns"]),
                         "a pattern was dropped: bad regex or unknown role")

    COLUMNS = ["SVC_DT", "LOS_DAYS", "RX_CNT", "DRG_CD", "IP_ADMIT_DATE",
               "PAT_ID"]

    @staticmethod
    def _expanded(vocab, columns):
        token = activate_vocab(vocab)
        try:
            return {c.column: (c.expanded_name, c.date_role)
                    for c in enrich_columns(list(columns))}
        finally:
            deactivate_vocab(token)

    def test_it_teaches_the_words_a_clinician_types(self):
        """Run through the real enrichment, not read out of the JSON. These are
        the words that end up in the prompt, the answer card and the retrieval
        index, so an unexpanded LOS_DAYS is a column no question can reach."""
        out = self._expanded(with_pack(HEALTHCARE), self.COLUMNS)
        self.assertEqual(out["LOS_DAYS"][0], "length of stay days")
        self.assertEqual(out["RX_CNT"][0], "prescription count")
        self.assertEqual(out["DRG_CD"][0], "diagnosis related group code")
        self.assertEqual(out["PAT_ID"][0], "patient id")

    def test_the_clinical_date_role_arrives_on_the_enriched_column(self):
        """The date role has to survive enrichment, not just detection -- this
        is the field the semantic model is built from."""
        out = self._expanded(with_pack(HEALTHCARE), self.COLUMNS)
        self.assertEqual(out["SVC_DT"][1], "Service Date")
        self.assertEqual(out["IP_ADMIT_DATE"][1], "Admission Date")

    def test_a_tenant_without_it_learns_none_of_those_words(self):
        out = self._expanded(_clone_builtin(), self.COLUMNS)
        self.assertEqual(out["LOS_DAYS"][0], "los days")
        self.assertEqual(out["RX_CNT"][0], "rx count")
        self.assertEqual(out["DRG_CD"][0], "drg code")
        self.assertEqual(out["SVC_DT"][1], "")


class ItNeverCostsATenantTheirErpPack(unittest.TestCase):
    """The seam. An industry pack scores through date_role_patterns, so it can
    top the recommendation list; it can never be auto-applied, so topping the
    list must not stop the ERP pack underneath it from being."""

    M3_COLUMNS = ["CUS_ORD_NUM", "CUS_DMS_KEY", "WHS_DMS_KEY", "ORNO", "CUNO",
                  "WHLO", "ITNO", "ORQT", "IVDT"]
    M3_TABLES = ["ERP_CUS_ORD_FCT", "DMS_CUSTOMER", "OOHEAD", "MITMAS"]

    # A hospital group running Dynamics finance beside a claims mart. Healthcare
    # out-scores Dynamics here on date patterns alone.
    DYNAMICS_COLUMNS = ["CUSTACCOUNT", "SALESID", "LINEAMOUNT", "RECID",
                        "DATAAREAID", "INVENTSITEID", "PURCHID"]
    CLAIMS_COLUMNS = [
        "SERVICE_DATE", "ADMIT_DATE", "DISCHARGE_DATE", "CLAIM_DATE",
        "FILL_DATE", "PRESCRIPTION_DATE", "PAID_DATE", "STATEMENT_DATE",
        "DATE_OF_SERVICE", "ENCOUNTER_DATE", "VISIT_DATE", "DISPENSE_DATE",
        "ADMISSION_DATE", "PROCEDURE_DATE", "WRITTEN_DATE",
        "CLAIM_SUBMISSION_DATE", "REMIT_DATE", "DATE_OF_ADMISSION",
        "DATE_OF_DISCHARGE", "POSTED_DATE", "ADJUDICATION_DATE",
        "TREATMENT_DATE",
    ]

    def test_an_m3_warehouse_still_auto_applies_infor_m3(self):
        profile = detect_naming_profile(self.M3_COLUMNS, self.M3_TABLES)
        self.assertEqual(profile.get("auto_applied_packs"), ["infor_m3"])

    def test_healthcare_is_not_even_recommended_for_an_m3_warehouse(self):
        profile = detect_naming_profile(self.M3_COLUMNS, self.M3_TABLES)
        recommended = {r["pack_id"] for r in profile["pack_recommendations"]}
        self.assertNotIn(HEALTHCARE, recommended)

    def test_a_hospital_on_dynamics_keeps_dynamics(self):
        profile = detect_naming_profile(
            self.DYNAMICS_COLUMNS + self.CLAIMS_COLUMNS, ["F_CLAIM_LINE"])
        ranked = [r["pack_id"] for r in profile["pack_recommendations"]]
        self.assertEqual(ranked[0], HEALTHCARE,
                         "the fixture no longer tests the case it was built for")
        self.assertEqual(profile.get("auto_applied_packs"), ["dynamics"])

    def test_the_industry_pack_is_still_never_applied_by_itself(self):
        """A word in a column name is not consent. A pure claims warehouse gets
        a recommendation and no automatic decision."""
        profile = detect_naming_profile(
            self.CLAIMS_COLUMNS, ["F_CLAIM_LINE", "D_MEMBER"])
        self.assertEqual(profile.get("auto_applied_packs"), [])
        top = profile["pack_recommendations"][0]
        self.assertEqual(top["pack_id"], HEALTHCARE)
        self.assertEqual(top["unique_evidence"], 0)

    # Five columns unique to Dynamics and five unique to NetSuite: both clear the
    # evidence gate, both clear 82%, and they land within a couple of points of
    # each other. This is the case the MARGIN is for, and the fix must not have
    # spent it -- auto-applying NetSuite here by two points would give half a
    # Dynamics warehouse the wrong ERP vocabulary with no one asked.
    TWO_ERPS = ["CUSTACCOUNT", "SALESID", "LINEAMOUNT", "RECID", "DATAAREAID",
                "ACCOUNTNUMBER", "COMPANYNAME", "CLOSEDATE", "CREATEDDATE",
                "ENTITYID"]

    def test_two_erp_packs_disagreeing_still_reach_no_decision(self):
        profile = detect_naming_profile(self.TWO_ERPS, [])
        eligible = [r for r in profile["pack_recommendations"]
                    if r["unique_evidence"] >= 4]
        self.assertGreaterEqual(len(eligible), 2, eligible)
        self.assertGreaterEqual(eligible[0]["confidence"], 82.0, eligible)
        self.assertLess(eligible[0]["confidence"] - eligible[1]["confidence"],
                        10.0, "the fixture no longer tests a close call")
        self.assertEqual(profile.get("auto_applied_packs"), [],
                         "a two-point lead decided which ERP a tenant runs")


class ThePackReachesTheReviewQueue(unittest.TestCase):
    """A detected role changes nothing until it arrives as a draft an admin can
    approve, carrying the vocabulary's synonyms rather than the column's."""

    def test_a_claims_fact_proposes_its_three_dates(self):
        from core.model_drafts import date_role_drafts

        token = activate_vocab(with_pack(HEALTHCARE))
        try:
            drafts = date_role_drafts([
                {"entity": "ClaimLine", "column": "DATE_OF_SERVICE"},
                {"entity": "ClaimLine", "column": "CLAIM_SUBMISSION_DATE"},
                {"entity": "ClaimLine", "column": "PAID_DATE"},
                {"entity": "ClaimLine", "column": "DOB"},
            ], properties=[])
        finally:
            deactivate_vocab(token)
        by_column = {d.column: d for d in drafts}
        self.assertEqual(set(by_column), {"DATE_OF_SERVICE",
                                         "CLAIM_SUBMISSION_DATE", "PAID_DATE"})
        self.assertEqual(by_column["DATE_OF_SERVICE"].payload["display_name"],
                         "Service Date")
        self.assertIn("date of service",
                      by_column["DATE_OF_SERVICE"].payload["synonyms"])

    def test_the_same_columns_propose_nothing_without_the_pack(self):
        from core.model_drafts import date_role_drafts

        token = activate_vocab(_clone_builtin())
        try:
            drafts = date_role_drafts([
                {"entity": "ClaimLine", "column": "DATE_OF_SERVICE"},
                {"entity": "ClaimLine", "column": "CLAIM_SUBMISSION_DATE"},
            ], properties=[])
        finally:
            deactivate_vocab(token)
        self.assertEqual(drafts, [])

    def test_a_pharmacy_fact_proposes_written_and_filled_separately(self):
        from core.model_drafts import date_role_drafts

        token = activate_vocab(with_pack(HEALTHCARE))
        try:
            drafts = date_role_drafts([
                {"entity": "RxClaim", "column": "WRITTEN_DATE"},
                {"entity": "RxClaim", "column": "FILL_DATE"},
            ], properties=[])
        finally:
            deactivate_vocab(token)
        labels = {d.column: d.payload["display_name"] for d in drafts}
        self.assertEqual(labels, {"WRITTEN_DATE": "Prescription Date",
                                 "FILL_DATE": "Dispense Date"})


if __name__ == "__main__":
    unittest.main()
