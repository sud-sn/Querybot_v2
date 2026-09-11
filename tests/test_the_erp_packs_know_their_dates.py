"""
tests/test_the_erp_packs_know_their_dates.py

The SAP pack knew nothing about SAP dates.

core/date_roles.py detects a column's business date role, and packs under
packs/ extend it per ERP -- core.vocab_packs merges their date_role_patterns
ahead of the builtin regexes so a pack can name a column the generic English
rules cannot see. The product ships packs for SAP, Oracle EBS, JD Edwards,
Dynamics, NetSuite and Infor M3.

Three of them declared ZERO date_role_patterns. Measured by executing
detect_date_role with each pack active:

    sap.json           0/12   AUDAT, FKDAT, ERDAT, BUDAT, VDATU, LFDAT, ...
    jde.json           0/ 8   TRDJ, DRQJ, RPDIVJ, DGL, UPMJ, ...
    oracle_ebs.json    7/12   the 7 came from the builtin English patterns;
                              the pack contributed nothing

SAP and JDE name their date fields in five-character codes -- AUDAT, FKDAT,
TRDJ -- with no DT or DATE token anywhere in them, so not one builtin pattern
can match, and a pack is the ONLY place that vocabulary can live. A SAP tenant
therefore had no discoverable business date on any fact: every period question
either refused or was answered on a date the model chose.

Two roles were missing from the builtin set for those packs to point at.
Document date is not the posting date -- a document dated the 28th can post in
the following period, which is the entire reason finance keeps both -- and
transaction date is what an AR or GL question means when it says "transaction".
Without them, the Dynamics pack had mapped DOCUMENTDATE, TRANSDATE and
ACCOUNTINGDATE all to accounting_date: three different business dates arriving
as three roles under ONE key, which the resolver then picks between on an
alphabetical tie-break.

And the `_at` suffix -- created_at, shipped_at, invoiced_at, the dominant
convention in anything built with dbt or on a Postgres-shaped mart -- matched
nothing at all, because every pattern requires a DT or DATE token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.date_roles import DATE_ROLES, detect_date_role
from core.vocab_packs import _clone_builtin, _merge_pack

PACKS = Path(__file__).resolve().parents[1] / "packs"
_ROLE_KEYS = {role.key for role in DATE_ROLES}


def vocab_for(pack_name):
    vocab = _clone_builtin()
    _merge_pack(vocab, json.loads((PACKS / pack_name).read_text(encoding="utf-8")),
                pack_name)
    return vocab


# (pack, column, expected role) — real field names from each ERP.
ERP_COLUMNS = [
    # ── SAP ECC / S4 ────────────────────────────────────────────────────────
    ("sap.json", "AUDAT", "order_date"),
    ("sap.json", "VDATU", "requested_delivery_date"),
    ("sap.json", "LFDAT", "delivery_date"),
    ("sap.json", "WADAT", "planned_delivery_date"),
    ("sap.json", "WADAT_IST", "delivery_date"),
    ("sap.json", "FKDAT", "invoice_date"),
    ("sap.json", "BUDAT", "accounting_date"),
    ("sap.json", "BLDAT", "document_date"),
    ("sap.json", "NETDT", "due_date"),
    ("sap.json", "ERDAT", "creation_date"),
    ("sap.json", "AEDAT", "modified_date"),
    # ── JD Edwards, with and without the table alias ────────────────────────
    ("jde.json", "TRDJ", "order_date"),
    ("jde.json", "SDTRDJ", "order_date"),
    ("jde.json", "DRQJ", "requested_delivery_date"),
    ("jde.json", "SDDRQJ", "requested_delivery_date"),
    ("jde.json", "PDDJ", "planned_delivery_date"),
    ("jde.json", "ADDJ", "delivery_date"),
    ("jde.json", "RPDIVJ", "invoice_date"),
    ("jde.json", "DGL", "accounting_date"),
    ("jde.json", "UPMJ", "modified_date"),
    # ── Oracle EBS ──────────────────────────────────────────────────────────
    ("oracle_ebs.json", "ORDERED_DATE", "order_date"),
    ("oracle_ebs.json", "REQUEST_DATE", "requested_delivery_date"),
    ("oracle_ebs.json", "PROMISE_DATE", "planned_delivery_date"),
    ("oracle_ebs.json", "SCHEDULE_SHIP_DATE", "planned_delivery_date"),
    ("oracle_ebs.json", "ACTUAL_SHIPMENT_DATE", "delivery_date"),
    ("oracle_ebs.json", "TRX_DATE", "transaction_date"),
    ("oracle_ebs.json", "INVOICE_DATE", "invoice_date"),
    ("oracle_ebs.json", "GL_DATE", "accounting_date"),
    # ── Dynamics 365 F&O ────────────────────────────────────────────────────
    ("dynamics.json", "SALESDATE", "order_date"),
    ("dynamics.json", "INVOICEDATE", "invoice_date"),
    ("dynamics.json", "TRANSDATE", "transaction_date"),
    ("dynamics.json", "DOCUMENTDATE", "document_date"),
    ("dynamics.json", "ACCOUNTINGDATE", "accounting_date"),
    # ── the two that already worked, so a regression here is visible ───────
    # NetSuite's TRANDATE stays accounting_date, deliberately: it is the date
    # that drives the posting period on a transaction record, and unlike
    # Dynamics there is no second column collapsing onto it. Renaming it for
    # symmetry with the new transaction_date role would churn a working pack
    # for no gain.
    ("netsuite.json", "TRANDATE", "accounting_date"),
    ("infor_m3.json", "IVDT", "invoice_date"),
    ("infor_m3.json", "ORDT", "order_date"),
]


class TestEveryShippedPackNamesItsDates:

    @pytest.mark.parametrize("pack,column,expected", ERP_COLUMNS,
                             ids=[f"{p.split('.')[0]}:{c}" for p, c, _ in ERP_COLUMNS])
    def test_the_column_resolves_to_the_right_role(self, pack, column, expected):
        role = detect_date_role(column, vocab=vocab_for(pack))
        assert role is not None, (
            f"{column} carries no business date role with {pack} active, so a "
            "fact keyed on it has no governed date at all")
        assert role.key == expected, (column, role.key)

    @pytest.mark.parametrize("pack", [
        "sap.json", "jde.json", "oracle_ebs.json", "dynamics.json",
        "netsuite.json", "infor_m3.json",
    ])
    def test_the_pack_declares_some(self, pack):
        """The defect stated directly. A pack with an empty list looks
        installed and contributes nothing."""
        data = json.loads((PACKS / pack).read_text(encoding="utf-8"))
        assert data.get("date_role_patterns"), (
            f"{pack} ships with no date vocabulary, so every date column on "
            "that ERP falls through to the generic English rules")


class TestThePatternsAreWellFormed:

    ALL = sorted(p.name for p in PACKS.glob("*.json"))

    @pytest.mark.parametrize("pack", ALL)
    def test_every_role_named_is_one_that_exists(self, pack):
        """core.vocab_packs skips an unknown role with a log line nobody reads,
        so a typo silently removes the pattern."""
        data = json.loads((PACKS / pack).read_text(encoding="utf-8"))
        unknown = [x.get("role") for x in (data.get("date_role_patterns") or [])
                   if x.get("role") not in _ROLE_KEYS]
        assert not unknown, (pack, unknown)

    @pytest.mark.parametrize("pack", ALL)
    def test_every_pattern_compiles(self, pack):
        import re

        data = json.loads((PACKS / pack).read_text(encoding="utf-8"))
        for entry in data.get("date_role_patterns") or []:
            re.compile(entry["pattern"])

    @pytest.mark.parametrize("pack", ALL)
    def test_no_pattern_is_declared_twice(self, pack):
        data = json.loads((PACKS / pack).read_text(encoding="utf-8"))
        patterns = [x["pattern"] for x in data.get("date_role_patterns") or []]
        assert len(patterns) == len(set(patterns)), pack


class TestDistinctDatesStayDistinct:
    """Two columns under one role key are collapsed by the resolver on an
    alphabetical tie-break -- not a decision anybody made."""

    def test_dynamics_keeps_its_three_apart(self):
        vocab = vocab_for("dynamics.json")
        keys = {detect_date_role(c, vocab=vocab).key
                for c in ("DOCUMENTDATE", "TRANSDATE", "ACCOUNTINGDATE")}
        assert len(keys) == 3, (
            "three different business dates share one role key, so a fact "
            f"carrying all three offers the reader one of them: {keys}")

    def test_sap_keeps_the_document_date_off_the_posting_date(self):
        """BLDAT and BUDAT differ by design: a document dated the 28th can post
        in the next period, and finance reports on both."""
        vocab = vocab_for("sap.json")
        assert detect_date_role("BLDAT", vocab=vocab).key == "document_date"
        assert detect_date_role("BUDAT", vocab=vocab).key == "accounting_date"

    def test_sap_keeps_planned_goods_issue_off_actual(self):
        vocab = vocab_for("sap.json")
        assert detect_date_role("WADAT", vocab=vocab).key == "planned_delivery_date"
        assert detect_date_role("WADAT_IST", vocab=vocab).key == "delivery_date"

    @pytest.mark.parametrize("pack,columns", [
        ("sap.json", ["AUDAT", "FKDAT", "VDATU", "LFDAT", "BUDAT", "BLDAT"]),
        ("oracle_ebs.json", ["ORDERED_DATE", "INVOICE_DATE", "REQUEST_DATE",
                             "PROMISE_DATE", "GL_DATE", "TRX_DATE"]),
    ])
    def test_a_fact_with_many_dates_offers_many_choices(self, pack, columns):
        vocab = vocab_for(pack)
        keys = {detect_date_role(c, vocab=vocab).key for c in columns}
        assert len(keys) == len(columns), (pack, keys)


class TestTheAtSuffixConvention:
    """created_at / shipped_at / invoiced_at -- dbt, Rails, any
    Postgres-shaped mart. Not one pattern recognised it, because every one
    requires a DT or DATE token. Builtin, not a pack: it belongs to no vendor."""

    @pytest.mark.parametrize("column,expected", [
        ("created_at", "creation_date"),
        ("updated_at", "modified_date"),
        ("ordered_at", "order_date"),
        ("invoiced_at", "invoice_date"),
        ("shipped_at", "delivery_date"),
        ("delivered_at", "delivery_date"),
        ("paid_at", "payment_date"),
        ("received_at", "receipt_date"),
        ("booked_at", "booked_date"),
    ])
    def test_it_resolves_without_any_pack(self, column, expected):
        role = detect_date_role(column)
        assert role is not None, column
        assert role.key == expected, (column, role.key)

    @pytest.mark.parametrize("column", [
        "ORDERED_AT", "OrderedAt", "shipped_at", "SHIPPED_AT",
    ])
    def test_however_it_is_cased(self, column):
        assert detect_date_role(column) is not None, column

    @pytest.mark.parametrize("column", [
        "amount_at", "quantity_at", "rate_at",
    ])
    def test_a_non_date_word_does_not_become_a_date(self, column):
        """The rule rewrites the SUFFIX; it must not make every _at column a
        date, or an amount column acquires a business date role."""
        assert detect_date_role(column) is None, column

    @pytest.mark.parametrize("column", [
        "ORDER_STATUS", "INVOICE_NUMBER", "ORDER_QTY", "INVOICE_AMOUNT",
        "DELIVERY_NOTE", "ORDER_TYPE", "PAYMENT_TERMS", "RECEIPT_NUMBER",
        "ORDER_STAT", "INVOICE_NO",
    ])
    def test_a_date_word_with_a_non_date_suffix_is_not_a_date(self, column):
        """The one that matters, and the one a loose rule breaks first.

        Every column here BEGINS with a word that names a date role, so a
        suffix rule that rewrites more than it should turns ORDER_STATUS into
        order_date and INVOICE_NUMBER into invoice_date -- and then the product
        puts a governed date filter on a status column and reports the result
        as a period. Widening _AT_SUFFIX_RE to `(^|_)([A-Z0-9]+)_[A-Z]+$` does
        exactly that, and passes every other assertion in this file.
        """
        assert detect_date_role(column) is None, (
            f"{column} resolved to a business date role")


class TestTheNewRolesAreCompleteRoles:
    """A role key with no label and no translation reaches a reader as a
    machine token."""

    @pytest.mark.parametrize("key", ["document_date", "transaction_date"])
    def test_it_is_in_the_builtin_set(self, key):
        assert key in _ROLE_KEYS

    @pytest.mark.parametrize("key", ["document_date", "transaction_date"])
    def test_it_has_a_label_in_both_languages(self, key):
        from core.i18n import t

        english = t(f"date_role.{key}", lang="en")
        french = t(f"date_role.{key}", lang="fr")
        assert english and english != f"date_role.{key}"
        assert french and french != english

    @pytest.mark.parametrize("key", ["document_date", "transaction_date"])
    def test_it_has_searchable_terms(self, key):
        """date_role_terms feeds the synonyms an admin sees and the vocabulary
        a question is matched against. A role with only its own key is
        unfindable by anyone who does not already know it."""
        from core.date_roles import date_role_terms

        role = next(r for r in DATE_ROLES if r.key == key)
        assert len(date_role_terms(role)) >= 2, role


class TestNothingThatResolvedStoppedResolving:
    """The builtin patterns are ordered specific-before-general, and a new
    alternation in the wrong one silently re-points a whole convention."""

    @pytest.mark.parametrize("column,expected", [
        ("CUS_ORD_IVC_DT", "invoice_date"),
        ("CUS_ORD_DT", "order_date"),
        ("RQD_DLV_DT", "requested_delivery_date"),
        ("CFM_DLV_DT", "confirmed_delivery_date"),
        ("PLD_DLV_DT", "planned_delivery_date"),
        ("DLV_DT", "delivery_date"),
        ("INVOICE_DATE", "invoice_date"),
        ("ORDER_DATE", "order_date"),
        ("REQUESTED_DELIVERY_DATE", "requested_delivery_date"),
        ("DUE_DATE", "due_date"),
        ("POSTING_DATE", "accounting_date"),
        ("RECEIPT_DATE", "receipt_date"),
        ("ORDER_DATE_KEY", "order_date"),
        ("INVOICE_DATE_SK", "invoice_date"),
    ])
    def test_the_existing_vocabulary_is_unchanged(self, column, expected):
        role = detect_date_role(column)
        assert role is not None, column
        assert role.key == expected, (column, role.key)

    @pytest.mark.parametrize("column", [
        "CUSTOMER_ID", "NET_AMT", "QUANTITY", "REGION_CD", "STATUS",
        "WAREHOUSE", "PRODUCT_KEY",
    ])
    def test_a_column_that_is_not_a_date_still_is_not(self, column):
        assert detect_date_role(column) is None, column
