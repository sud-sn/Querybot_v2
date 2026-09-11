"""
tests/test_the_builtin_date_words_are_the_ones_people_type.py

BUSINESS_DATE resolved to no business date.

core/date_roles.py holds the builtin date vocabulary, and that vocabulary is
what a tenant gets when no ERP pack is active -- store.db defaults
client.erp_packs to '[]', and vocab_packs.get_active_vocab() then falls back to
builtin_vocab(). So the builtin set is not a baseline nobody runs; it is the
default, and every tenant onboarded before an admin picks a pack is on it.

Measured by executing detect_date_role with no pack active, every one of these
returned None:

    BUSINESS_DATE  BUSINESS_DT  BIZ_DT      the explicit declaration
    SNAPSHOT_DATE  AS_OF_DATE   ASOF_DT     the date of a balance snapshot
    SALE_DATE      SALES_DATE   SLS_DT      the revenue date in a sales mart
    TXN_DATE                                transaction, unabbreviated nowhere
    POST_DATE      JOURNAL_DATE JRNL_DT     the posting date, in ledger words
    SETTLEMENT_DATE SETTLE_DT               when the money moved
    MODIFIED_DATE  LAST_MODIFIED_DATE       the role is CALLED modified_date

The last one is the shape of the whole defect: a role named `modified_date`,
labelled "Last Modified Date", that the column MODIFIED_DATE could not reach.
The dictionary knew the abbreviations (LMDT, LST_UPD) and the American spelling
(UPDATED_DATE) and not its own name.

This matters more than it used to. An aggregating fact with no settled business
date is now REFUSED rather than answered over all time, so a date the
dictionary cannot read is no longer a quiet default -- it is a question the
product declines, and the reader has nothing to do about it.

Two roles had to be added for the first two rows. A snapshot's own date had no
name even though the product already reasons about snapshots (semi-additive
measures, latest_snapshot windows, question_has_snapshot_intent), and
`business_date` existed only as the string derive_date_role falls back to when
a label sanitizes to nothing.

What is deliberately still absent is in the last class below: spellings that
cannot be read without knowing the table. The builtin set claims unambiguous
English and nothing else -- a confident wrong default date is worse than none,
because it is governed, disclosed, and wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.date_roles import DATE_ROLES, detect_date_role, derive_date_role
from core.vocab_packs import _clone_builtin, _merge_pack

PACKS = Path(__file__).resolve().parents[1] / "packs"
_ROLE_KEYS = {role.key for role in DATE_ROLES}


def role_of(column, vocab=None):
    found = detect_date_role(column, vocab=vocab if vocab is not None else _clone_builtin())
    return found.key if found else None


class TestTheExplicitDeclarationIsReadable:
    """A column named BUSINESS_DATE is a modeller saying so in words. It was
    the one case the product could not read."""

    @pytest.mark.parametrize("column", [
        "BUSINESS_DATE", "BUSINESS_DT", "BIZ_DATE", "BIZ_DT",
        "BUSINESS_DATE_KEY", "BUSINESS_DT_DMS_KEY", "business_date",
        "FACT_BUSINESS_DATE",
    ])
    def test_it_resolves_to_the_business_date_role(self, column):
        assert role_of(column) == "business_date", column

    def test_the_role_is_registered_not_derived(self):
        """derive_date_role already produced the STRING "business_date" as its
        fallback key, with one synonym and a confidence of 60. A registered role
        carries the vocabulary's own synonyms, so a question that says "business
        date" matches it."""
        assert "business_date" in _ROLE_KEYS
        registered = next(r for r in DATE_ROLES if r.key == "business_date")
        derived = derive_date_role("SOME_BUSINESS_DATE_KEY")
        assert len(registered.synonyms) > len(derived.synonyms)
        assert registered.priority > derived.priority

    def test_a_specific_business_word_beside_the_date_still_wins(self):
        """Every builtin pattern anchors its word directly against _DT -- the
        file's own convention -- so the token next to the date decides.
        BUSINESS_INVOICE_DATE is an invoice date; INVOICE_BUSINESS_DATE reads as
        the invoice's business date and lands on business_date. Recorded as the
        rule rather than engineered around: a special case here would be the one
        pattern in the file that does not follow it."""
        assert role_of("BUSINESS_INVOICE_DATE") == "invoice_date"
        assert role_of("INVOICE_BUSINESS_DATE") == "business_date"
        assert role_of("ORDER_DATE") == "order_date"


class TestASnapshotsOwnDateHasAName:

    @pytest.mark.parametrize("column,expected", [
        ("SNAPSHOT_DATE", "snapshot_date"),
        ("SNAPSHOT_DT", "snapshot_date"),
        ("AS_OF_DATE", "snapshot_date"),
        ("ASOF_DATE", "snapshot_date"),
        ("AS_OF_DT", "snapshot_date"),
        ("INVENTORY_SNAPSHOT_DATE", "snapshot_date"),
    ])
    def test_it_resolves(self, column, expected):
        assert role_of(column) == expected, column

    def test_the_label_is_a_business_phrase(self):
        role = next(r for r in DATE_ROLES if r.key == "snapshot_date")
        assert role.label == "Snapshot Date"
        assert "as of date" in role.synonyms

    def test_no_synonym_is_a_bare_word_that_matches_any_question(self):
        """"as of" and "snapshot" alone appear in questions that are not about
        this column at all ("revenue as of last month"), and a synonym that
        broad would attach this role to every date question."""
        for role in DATE_ROLES:
            for synonym in role.synonyms:
                assert len(synonym.split()) >= 2, (role.key, synonym)


class TestTheRolesKnowTheirOwnNames:
    """Each of these is a spelling of a role that already existed."""

    @pytest.mark.parametrize("column,expected", [
        ("MODIFIED_DATE", "modified_date"),
        ("LAST_MODIFIED_DATE", "modified_date"),
        ("MODIFICATION_DATE", "modified_date"),
        ("TXN_DATE", "transaction_date"),
        ("POST_DATE", "accounting_date"),
        ("JOURNAL_DATE", "accounting_date"),
        ("JRNL_DT", "accounting_date"),
        ("SETTLEMENT_DATE", "payment_date"),
        ("SETTLE_DT", "payment_date"),
        ("SALE_DATE", "order_date"),
        ("SALES_DATE", "order_date"),
        ("SLS_DT", "order_date"),
    ])
    def test_it_resolves_to_the_role_that_already_existed(self, column, expected):
        assert role_of(column) == expected, column

    def test_no_new_key_was_invented_for_a_spelling(self):
        """A second key for the same business date is worse than no key: the
        resolver then sees two candidates and asks the reader to choose between
        two names for one thing."""
        for column in ("MODIFIED_DATE", "TXN_DATE", "POST_DATE",
                       "SETTLEMENT_DATE", "SALES_DATE"):
            assert role_of(column) in _ROLE_KEYS, column


class TestNothingThatWorkedStoppedWorking:

    @pytest.mark.parametrize("column,expected", [
        ("INVOICE_DATE", "invoice_date"),
        ("CUS_IVC_DT", "invoice_date"),
        ("ORDER_DATE", "order_date"),
        ("SALES_ORDER_DATE", "order_date"),
        ("POSTING_DATE", "accounting_date"),
        ("GL_DATE", "accounting_date"),
        ("UPDATED_DATE", "modified_date"),
        ("TRANSACTION_DATE", "transaction_date"),
        ("TRX_DATE", "transaction_date"),
        ("DOCUMENT_DATE", "document_date"),
        ("SHIP_DATE", "delivery_date"),
        ("DUE_DATE", "due_date"),
        ("CREATED_AT", "creation_date"),
        ("DT_DMS_KEY", "date"),
    ])
    def test_it_still_resolves_the_same_way(self, column, expected):
        assert role_of(column) == expected, column

    @pytest.mark.parametrize("column", [
        # Saved by pattern ORDER -- delivery_date is matched before
        # accounting_date -- which is why these are not the case that tests the
        # anchoring, and why the ones below are.
        "POST_SHIP_DATE", "POST_DELIVERY_DATE",
    ])
    def test_a_post_prefix_before_a_known_word_reaches_that_word(self, column):
        assert role_of(column) == "delivery_date"

    @pytest.mark.parametrize("column", [
        "POST_AUDIT_DATE", "POST_CLOSE_DATE", "POST_REVIEW_DATE",
        "JOURNAL_APPROVAL_DATE", "GL_CLOSE_DATE",
    ])
    def test_a_post_prefix_before_an_unknown_word_claims_nothing(self, column):
        """POST_DT is a posting date; POST_AUDIT_DT is "after the audit", and
        nothing else in the file claims it. So this is the case that actually
        tests the anchoring: every builtin alternation puts its word directly
        against _DT, and loosening POST to POST.*_DT would turn each of these
        into a confident accounting date on a column that is not one."""
        assert role_of(column) is None, column


class TestTheAmbiguousOnesAreStillLeftAlone:
    """Not an oversight. Each of these needs the table to be read before it
    means anything, and the builtin set claims unambiguous English only."""

    @pytest.mark.parametrize("column", [
        # Bare TRANS/TRAN: transaction, transfer, translation. Documented in
        # core/date_roles.py and left to the pack that knows which its ERP means.
        "TRANS_DATE", "TRAN_DATE",
        # BUS: business, bus, bushel.
        "BUS_DT",
        # Operational dates whose subject is the table, not the column.
        "START_DATE", "END_DATE", "COMPLETION_DATE", "EFFECTIVE_DATE",
        "ACTIVITY_DATE", "PERIOD_END_DATE", "RUN_DATE", "REPORT_DATE",
        # A voucher is an AP document in JDE and a discount coupon in retail.
        "VOUCHER_DATE",
        # ETL plumbing, and not a business date in any tenant.
        "LOAD_DATE", "ETL_LOAD_DATE", "DW_LOAD_DT",
        # Vertical vocabulary: these need roles with real clinical meaning, and
        # a healthcare pack, not a generic English guess.
        "SERVICE_DATE", "FILL_DATE", "DISPENSE_DATE", "ADMIT_DATE",
    ])
    def test_it_resolves_to_no_role_rather_than_a_wrong_one(self, column):
        assert role_of(column) is None, (
            f"{column} now claims a business date role; a governed default date "
            "that is confidently wrong is worse than none, because it filters "
            "the answer and is disclosed as correct")

    def test_but_a_key_suffix_still_makes_any_of_them_reviewable(self):
        """A physical FK into a date dimension is hard evidence regardless of
        the name, which is what derive_date_role is for. The ambiguity above is
        only about reading a bare column name."""
        for column in ("SERVICE_DATE_KEY", "ACTIVITY_DATE_ID",
                       "EFFECTIVE_DATE_SK"):
            assert role_of(column) is not None, column


class TestThePackAndTheBuiltinAgreeOnTransactionDate:
    """generic_star_schema mapped TRANSACTION|TXN|TRANS_DATE to
    accounting_date, written before a transaction_date role existed. Packs run
    ahead of the builtin patterns, so a tenant with that pack active got a
    different business role for the same column than a tenant without it -- and
    a question about a transaction date matched the Accounting Date synonyms."""

    def test_the_generic_pack_now_names_the_transaction_role(self):
        vocab = _clone_builtin()
        pack = json.loads(
            (PACKS / "generic_star_schema.json").read_text(encoding="utf-8"))
        _merge_pack(vocab, pack, "generic_star_schema")
        for column in ("TRANSACTION_DATE", "TXN_DATE", "TRANS_DATE"):
            assert role_of(column, vocab) == "transaction_date", column

    def test_the_pack_still_wins_where_its_erp_genuinely_differs(self):
        """NetSuite's TRANDATE really is the posting date -- it drives the
        accounting period -- so that pack keeps its own mapping. The point is
        that a pack disagrees deliberately, not by having been written first."""
        vocab = _clone_builtin()
        pack = json.loads((PACKS / "netsuite.json").read_text(encoding="utf-8"))
        _merge_pack(vocab, pack, "netsuite")
        assert role_of("TRANDATE", vocab) == "accounting_date"

    def test_every_pack_pattern_still_names_a_role_that_exists(self):
        for path in sorted(PACKS.glob("*.json")):
            pack = json.loads(path.read_text(encoding="utf-8"))
            for item in pack.get("date_role_patterns") or []:
                assert item.get("role") in _ROLE_KEYS, (path.name, item)


class TestTheDraftProposerCarriesTheNewRolesThrough:
    """detect_date_role finding a role changes nothing on its own: the role has
    to arrive as a reviewable draft with the vocabulary's synonyms on it, or the
    resolver still reads an unclassified column."""

    def test_a_business_date_column_becomes_a_reviewable_draft(self):
        from core.model_drafts import date_role_drafts

        drafts = date_role_drafts(
            [{"entity": "SalesFact", "column": "BUSINESS_DATE"}],
            properties=[],
        )
        assert len(drafts) == 1, drafts
        draft = drafts[0]
        assert draft.kind == "date_role"
        assert draft.payload["display_name"] == "Business Date"
        assert "business date" in draft.payload["synonyms"]
        assert draft.confidence >= 90, draft.confidence

    def test_the_synonyms_come_from_the_registered_role_not_the_column(self):
        """This is what registering the role bought: a derived role answers with
        exactly one term -- its own label -- so a reader asking about "business
        day" matched nothing."""
        from core.model_drafts import date_role_drafts

        draft = date_role_drafts(
            [{"entity": "SalesFact", "column": "BUSINESS_DATE_KEY"}],
            properties=[],
        )[0]
        terms = draft.payload["synonyms"]
        assert "business day" in terms, terms

    @pytest.mark.parametrize("column,expected_label", [
        ("SNAPSHOT_DATE", "Snapshot Date"),
        ("MODIFIED_DATE", "Last Modified Date"),
        ("SALES_DATE", "Order Date"),
        ("TXN_DATE", "Transaction Date"),
    ])
    def test_each_new_spelling_reaches_a_draft(self, column, expected_label):
        from core.model_drafts import date_role_drafts

        drafts = date_role_drafts(
            [{"entity": "SalesFact", "column": column}], properties=[])
        assert len(drafts) == 1, (column, drafts)
        assert drafts[0].payload["display_name"] == expected_label

    def test_an_ambiguous_column_still_produces_no_draft(self):
        from core.model_drafts import date_role_drafts

        assert date_role_drafts(
            [{"entity": "SalesFact", "column": "EFFECTIVE_DATE"},
             {"entity": "SalesFact", "column": "LOAD_DATE"}],
            properties=[],
        ) == []
