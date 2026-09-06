"""
tests/test_date_role_spelled_out.py

The spelled-out English form of a date role, beside the ERP abbreviation.

`IVC_DT` was recognised as the invoice date and `INVOICE_DATE` was not — so
the most ordinary column name in a warehouse carried no business date role at
all. On a fact table holding an invoice date and an order date, a question
about invoicing then had nothing to tell the two apart, and the temporal
governance the whole module exists for did not apply to either.

The same hole covered DELIVERY_DATE, POSTING_DATE, RECEIVED_DATE and
BILLING_DATE. CANCELLED_ORDER_DATE was worse than missing: it resolved to
`order_date`, so a question about cancellations would have filtered on the
date the order was placed.

Every assertion here calls the real `detect_date_role`.
"""

from __future__ import annotations

import pytest

from core.date_roles import DATE_ROLES, date_role_terms, detect_date_role
from core.vocab_packs import builtin_vocab


def role_of(column: str) -> str | None:
    found = detect_date_role(column, vocab=builtin_vocab())
    return found.key if found else None


class TestTheSpelledOutNameResolves:

    @pytest.mark.parametrize("column,expected", [
        ("INVOICE_DATE", "invoice_date"),
        ("BILLING_DATE", "invoice_date"),
        ("BILLED_DATE", "invoice_date"),
        ("CUS_INVOICE_DATE", "invoice_date"),
        ("DELIVERY_DATE", "delivery_date"),
        ("DELIVERED_DATE", "delivery_date"),
        ("SHIPPED_DATE", "delivery_date"),
        ("SHIPMENT_DATE", "delivery_date"),
        ("SHIPPING_DATE", "delivery_date"),
        ("RECEIPT_DATE", "receipt_date"),
        ("RECEIVED_DATE", "receipt_date"),
        ("POSTING_DATE", "accounting_date"),
        ("POSTED_DATE", "accounting_date"),
        ("LEDGER_DATE", "accounting_date"),
        ("GL_DATE", "accounting_date"),
        ("CANCELLED_ORDER_DATE", "cancelled_order_date"),
        ("CANCELED_ORDER_DATE", "cancelled_order_date"),
    ])
    def test_it_resolves_to_the_role_the_words_name(self, column, expected):
        assert role_of(column) == expected

    @pytest.mark.parametrize("column,expected", [
        ("IVC_DT", "invoice_date"),
        ("CUS_IVC_DT", "invoice_date"),
        ("SLR_IVC_DT", "invoice_date"),
        ("DLV_DT", "delivery_date"),
        ("SHIP_DT", "delivery_date"),
        ("RCT_DT", "receipt_date"),
        ("RCV_DT", "receipt_date"),
        ("ACC_DT", "accounting_date"),
        ("ACCT_DT", "accounting_date"),
        ("CCL_ORD_DT", "cancelled_order_date"),
        ("REQ_DLV_DT", "requested_delivery_date"),
        ("CFM_DLV_DT", "confirmed_delivery_date"),
        ("PLD_DLV_DT", "planned_delivery_date"),
        ("VLD_DLV_DT", "valid_delivery_date"),
        ("ORD_DT", "order_date"),
        ("PAY_DT", "payment_date"),
        ("DUE_DT", "due_date"),
        ("BKD_DT", "booked_date"),
    ])
    def test_the_abbreviation_still_resolves(self, column, expected):
        # The alternates were added beside these, never in place of them.
        assert role_of(column) == expected


class TestTheSpecificRoleStillBeatsTheGeneralOne:
    """
    A qualified delivery date is not the delivery date, and resolving it to
    one is worse than resolving nothing: no role leaves the question
    unanswered, while the wrong role silently filters on the wrong column and
    returns a confident number.
    """

    @pytest.mark.parametrize("column,expected", [
        ("REQUESTED_DELIVERY_DATE", "requested_delivery_date"),
        ("REQUESTED_SHIP_DATE", "requested_delivery_date"),
        ("CONFIRMED_DELIVERY_DATE", "confirmed_delivery_date"),
        ("PLANNED_DELIVERY_DATE", "planned_delivery_date"),
        ("PLANED_SHIP_DATE", "planned_delivery_date"),
        ("VALID_DELIVERY_DATE", "valid_delivery_date"),
    ])
    def test_a_leading_qualifier_wins(self, column, expected):
        assert role_of(column) == expected

    @pytest.mark.parametrize("column,expected", [
        ("ShippingDateRequested", "requested_delivery_date"),
        ("DELIVERY_DATE_CONFIRMED", "confirmed_delivery_date"),
        ("SHIP_DATE_PLANNED", "planned_delivery_date"),
        ("DELIVERY_DATE_VALIDATED", "valid_delivery_date"),
    ])
    def test_a_trailing_qualifier_wins_too(self, column, expected):
        # A qualifier can follow the date as well as lead it, and widening the
        # general pattern to catch SHIPPING_DATE is exactly what made this
        # case resolvable-and-wrong rather than merely unresolved.
        assert role_of(column) == expected

    def test_a_plain_order_date_is_not_read_as_a_cancellation(self):
        assert role_of("ORDER_DATE") == "order_date"
        assert role_of("CUS_ORDER_DATE") == "order_date"


class TestNothingUnrelatedStartedResolving:
    """A date role fires on a date. Widening a pattern must not widen it here."""

    @pytest.mark.parametrize("column", [
        "CUSTOMER_ID", "NET_AMOUNT", "SHIP_TO_COUNTRY", "INVOICE_NUMBER",
        "INVOICE_AMOUNT", "DELIVERY_ADDRESS", "GL_ACCOUNT", "LEDGER_BALANCE",
        "ORDER_STATUS", "RECEIPT_QUANTITY", "SHIPPING_COST", "POSTING_USER",
    ])
    def test_it_stays_unresolved(self, column):
        assert role_of(column) is None

    def test_a_dynamics_only_spelling_still_needs_its_pack(self):
        # The builtins reaching further must not reach so far that a
        # terminology pack has nothing left to add.
        assert role_of("TransDate") is None


class TestTheResolvedRoleCarriesUsableTerms:

    @pytest.mark.parametrize("column", ["INVOICE_DATE", "DELIVERY_DATE",
                                        "POSTING_DATE", "CANCELLED_ORDER_DATE"])
    def test_the_dictionary_entry_is_reached_not_a_derived_stub(self, column):
        # Resolving through the dictionary rather than deriving a role from the
        # column name is what carries the synonyms a reader would actually
        # type. A derived role answers with one term: its own label.
        role = detect_date_role(column, vocab=builtin_vocab())
        assert role in DATE_ROLES, f"{column} derived a role instead of matching one"
        assert len(date_role_terms(role)) >= 2, column
        assert role.priority >= 70, column
