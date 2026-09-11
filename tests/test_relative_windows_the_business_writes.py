"""
tests/test_relative_windows_the_business_writes.py

"Revenue YTD" was not a temporal question, as far as the governance was
concerned.

core.contextual_dates.detect_temporal_window is the gate on the entire
data-relative regime. Everything downstream hangs off its return value: the
fail-closed check in core/query_pipeline.py that refuses a period question when
the fact has no governed business date fires only on
`_effective_temporal_window and not temporal_policies`; the temporal policy that
reaches the compiler and the validator comes from it; the freshness disclosure
is keyed on its `kind`; and the business-date anchor is resolved for the policy
it produces.

A question it does not recognise carries no window at all, so none of that
happens and generation falls through to the dialect's own date recipes -- which
read the SERVER CLOCK. That is precisely what the anchor regime exists to
prevent, and it fails in the safe-looking direction: a confident answer over a
window nobody governed, with no caveat, no disclosure and no trace step.

Executed, at HEAD~1, the detector returned {} for all of these:

    revenue ytd, revenue mtd, revenue year-to-date, last week, past month,
    trailing 30 days, rolling 12 months

while "revenue last 7 days" and "revenue this month" worked. And
core.date_roles.question_has_temporal_intent ALREADY listed "ytd" and "mtd" as
temporal terms, so the product recognised the words in one place and dropped
them in the one that governs.

Every spelling here maps onto a kind that already existed, deliberately: a
to-date window IS the current period anchored on the data, so the compiler,
the banner and the disclosure set need no new cases. The one new kind,
previous_week, is not compilable -- for the same documented reason this_week is
not, that the dialects disagree about where a week starts -- and is still worth
detecting, because a detected window is governed and gated even when the SQL
falls back.
"""

from __future__ import annotations

import pytest

from core.contextual_dates import detect_temporal_window
from core.date_roles import question_has_temporal_intent
from core.pipeline_helpers import _COMPILABLE_WINDOW_KINDS
from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS


class TestTheWordsFinanceActuallyWrites:

    @pytest.mark.parametrize("question,kind", [
        ("revenue ytd", "this_year"),
        ("YTD revenue by region", "this_year"),
        ("revenue year-to-date", "this_year"),
        ("revenue year to date", "this_year"),
        ("revenue qtd", "this_quarter"),
        ("revenue quarter to date", "this_quarter"),
        ("revenue mtd", "this_month"),
        ("MTD revenue", "this_month"),
        ("revenue month to date", "this_month"),
        ("revenue wtd", "this_week"),
        ("revenue week to date", "this_week"),
    ])
    def test_to_date_wording_is_the_current_period(self, question, kind):
        window = detect_temporal_window(question)
        assert window.get("kind") == kind, (question, window)
        assert window.get("anchor_policy") == "latest_available", question
        assert window.get("amount") == 0, question

    @pytest.mark.parametrize("question,kind", [
        ("revenue last week", "previous_week"),
        ("revenue past week", "previous_week"),
        ("revenue prior week", "previous_week"),
        ("revenue past month", "previous_month"),
        ("revenue past quarter", "previous_quarter"),
        ("revenue past year", "previous_year"),
    ])
    def test_the_period_before_this_one(self, question, kind):
        window = detect_temporal_window(question)
        assert window.get("kind") == kind, (question, window)
        assert window.get("amount") == 1, question

    @pytest.mark.parametrize("question,amount,unit", [
        ("revenue for the trailing 30 days", 30, "day"),
        ("revenue trailing 12 months", 12, "month"),
        ("revenue rolling 12 months", 12, "month"),
        ("rolling 4 quarters of revenue", 4, "quarter"),
    ])
    def test_trailing_and_rolling_are_the_same_window_as_last_n(
            self, question, amount, unit):
        window = detect_temporal_window(question)
        assert window.get("kind") == "last_n", (question, window)
        assert window.get("amount") == amount
        assert window.get("unit") == unit


class TestNothingThatWorkedStoppedWorking:

    @pytest.mark.parametrize("question,kind", [
        ("revenue today", "today"),
        ("revenue yesterday", "yesterday"),
        ("revenue this month", "this_month"),
        ("revenue current quarter", "this_quarter"),
        ("revenue last month", "previous_month"),
        ("revenue last quarter", "previous_quarter"),
        ("revenue last year", "previous_year"),
        ("revenue last 7 days", "last_n"),
        ("revenue past 3 months", "last_n"),
        ("revenue for the last 6 available months", "latest_n_observed"),
    ])
    def test_the_existing_vocabulary_is_unchanged(self, question, kind):
        assert detect_temporal_window(question).get("kind") == kind, question

    def test_a_numbered_window_still_beats_the_bare_period(self):
        """"last 3 months" is a rolling window, not the previous month. The
        numbered branch runs after the bare ones, so this is the ordering that
        matters most and the one a new pattern is likeliest to break."""
        window = detect_temporal_window("revenue last 3 months")
        assert window["kind"] == "last_n"
        assert window["amount"] == 3


class TestItStillSaysNoWhenThereIsNoRelativeWindow:
    """A detector that matches everything is worse than one that matches too
    little: it would put a data-relative window on questions that named their
    own period."""

    @pytest.mark.parametrize("question", [
        "total revenue",
        "revenue by month",
        "revenue by region",
        "how many customers are there",
        "revenue in March 2026",
        "revenue in 2026",
        "which invoices are past due",
        "revenue for the past",
        "show me the date column",
    ])
    def test_no_window_is_detected(self, question):
        assert detect_temporal_window(question) == {}, question


class TestTheKindsAreOnesTheRestOfTheProductKnows:
    """A new kind that nothing downstream handles is a silent fallback, and
    looks identical to a working one."""

    ALL = [
        "revenue ytd", "revenue qtd", "revenue mtd", "revenue wtd",
        "revenue last week", "revenue past month", "revenue past quarter",
        "revenue past year", "revenue trailing 30 days",
        "revenue rolling 12 months",
    ]

    def test_every_kind_is_compilable_or_deliberately_not(self):
        """this_week and previous_week are the documented exceptions: the
        dialects disagree about where a week starts, so the boundary has to be
        governed rather than guessed. Everything else must compile."""
        deliberate = {"this_week", "previous_week"}
        for question in self.ALL:
            kind = detect_temporal_window(question)["kind"]
            assert kind in _COMPILABLE_WINDOW_KINDS or kind in deliberate, (
                f"{question!r} produces {kind!r}, which the compiler does not "
                "handle and which is not one of the documented exceptions -- "
                "the SQL will silently fall back to free-form generation")

    def test_the_current_period_kinds_are_disclosed_to_the_reader(self):
        """A "this period" answer is anchored on the newest date in the data,
        not on today, and the reader is told so. The to-date spellings must
        land on kinds that set carries, or "revenue ytd" answers a window
        ending days ago with no note."""
        for question in ("revenue ytd", "revenue qtd", "revenue mtd",
                         "revenue wtd"):
            assert detect_temporal_window(question)["kind"] in \
                BUSINESS_DATE_WINDOW_KINDS, question


class TestTheTwoHalvesOfTheProductAgree:
    """question_has_temporal_intent listed ytd and mtd while the window
    detector did not. One half of the product knowing a word the governing
    half does not is how the gap arrived, so it is worth a test."""

    @pytest.mark.parametrize("question", [
        "revenue ytd", "revenue mtd", "revenue last week",
        "revenue trailing 30 days", "revenue past month",
    ])
    def test_a_question_with_temporal_intent_and_relative_wording_has_a_window(
            self, question):
        assert question_has_temporal_intent(question), question
        assert detect_temporal_window(question), (
            f"{question!r} reads as temporal but carries no window, so the "
            "fail-closed check for a fact with no governed business date "
            "never runs and generation uses the server clock")
