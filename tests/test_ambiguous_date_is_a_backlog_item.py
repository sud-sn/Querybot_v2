"""
tests/test_ambiguous_date_is_a_backlog_item.py

Whether a measure can answer a question about a period is decided by
core.contextual_dates.resolve_contextual_date_binding, and by nothing else.
The readiness backlog has to report the answer that function gives.

The first version of this file restated the rule instead. It had a private
`_dates_are_ambiguous` that said "two or more bound dates and not exactly one
default means the resolver will ask", applied to the measure's own date
contexts -- and it was wrong in both directions, because the resolver prefers
the FACT's approved default Date Role over the measure's contexts:

  * A measure with two non-default contexts, on a fact with one approved
    default role, resolves cleanly (fact_default_date_role). The admin was
    handed fifteen backlog items for a measure that never interrupts anybody.
  * A measure with NO contexts at all, on that same fact, resolves just as
    cleanly -- and every temporal shape was reported as missing a date role.
    That is the ordinary way to govern dates, so this was most workspaces.

The agreement test could not catch either, because it passed `date_roles=[]`
to the resolver: precisely the input that makes the fact-default branch
unreachable. A fixture that omits what production always sets does not test
agreement, it tests two functions on inputs production never sends.

So there is no restated rule any more. `_date_gap` asks the resolver and maps
its status to a remedy, and the matrix below drives BOTH sources of dates
through both halves at once. Absence and ambiguity stay separate items,
because "bind a business date" and "pick which of these is the default" are
different afternoons.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.contextual_dates import resolve_contextual_date_binding
from core.metric_coverage import (
    _date_gap, _resolver_verdict, check_variation, coverage_report,
    variations_for,
)

METRIC = {
    "id": 7,
    "name": "Net Revenue",
    "base_table": "DBO.F_SALES",
    "allowed_dimensions": "Region, Product",
    "grain": "day",
}

# ── the measure's own date contexts ──────────────────────────────────────────
NO_CONTEXT: list[dict] = []
ONE_DEFAULT_CONTEXT = [
    {"id": 1, "metric_id": 7, "context_name": "Invoice Date", "is_default": 1},
    {"id": 2, "metric_id": 7, "context_name": "Order Date", "is_default": 0},
]
NO_DEFAULT_CONTEXT = [
    {"id": 1, "metric_id": 7, "context_name": "Invoice Date", "is_default": 0},
    {"id": 2, "metric_id": 7, "context_name": "Order Date", "is_default": 0},
]
TWO_DEFAULT_CONTEXTS = [
    {"id": 1, "metric_id": 7, "context_name": "Invoice Date", "is_default": 1},
    {"id": 2, "metric_id": 7, "context_name": "Order Date", "is_default": 1},
]


# ── the fact's Date Roles, which the resolver prefers ────────────────────────
def _role(column, key, label, **kw):
    role = {
        "fact_table": "DBO.F_SALES", "fact_column": column,
        "role_key": key, "label": label,
        "status": "approved", "is_default": 0,
        "date_key_type": "native_date",
    }
    role.update(kw)
    return role


NO_FACT_ROLE: list[dict] = []
ONE_APPROVED_DEFAULT = [_role("INVOICE_DT", "invoice_date", "Invoice Date",
                              is_default=1)]
TWO_APPROVED_NO_DEFAULT = [
    _role("INVOICE_DT", "invoice_date", "Invoice Date"),
    _role("SHIP_DT", "ship_date", "Ship Date"),
]
TWO_APPROVED_TWO_DEFAULTS = [
    _role("INVOICE_DT", "invoice_date", "Invoice Date", is_default=1),
    _role("SHIP_DT", "ship_date", "Ship Date", is_default=1),
]
# Generated, never approved. The resolver does not treat these as a default.
ONE_UNAPPROVED_DEFAULT = [_role("INVOICE_DT", "invoice_date", "Invoice Date",
                                is_default=1, status="generated")]

QUESTION = "Net Revenue by month"

# (label, contexts, fact roles). Every combination the two halves can disagree
# on, including the two that shipped wrong.
STATES = [
    ("nothing bound anywhere", NO_CONTEXT, NO_FACT_ROLE),
    ("no context, fact has one approved default",
     NO_CONTEXT, ONE_APPROVED_DEFAULT),
    ("no context, fact has two approved and no default",
     NO_CONTEXT, TWO_APPROVED_NO_DEFAULT),
    ("no context, fact has two defaults", NO_CONTEXT, TWO_APPROVED_TWO_DEFAULTS),
    ("no context, fact default is only generated",
     NO_CONTEXT, ONE_UNAPPROVED_DEFAULT),
    ("two non-default contexts, no fact role",
     NO_DEFAULT_CONTEXT, NO_FACT_ROLE),
    ("two non-default contexts, fact has one approved default",
     NO_DEFAULT_CONTEXT, ONE_APPROVED_DEFAULT),
    ("two non-default contexts, fact has two and no default",
     NO_DEFAULT_CONTEXT, TWO_APPROVED_NO_DEFAULT),
    ("one default context, no fact role", ONE_DEFAULT_CONTEXT, NO_FACT_ROLE),
    ("one default context, fact disagrees",
     ONE_DEFAULT_CONTEXT, TWO_APPROVED_NO_DEFAULT),
    ("two default contexts, no fact role", TWO_DEFAULT_CONTEXTS, NO_FACT_ROLE),
    ("two default contexts, fact has one approved default",
     TWO_DEFAULT_CONTEXTS, ONE_APPROVED_DEFAULT),
]


def _resolver_status(contexts, fact_roles, question=QUESTION):
    """The runtime's own answer, called the way the pipeline calls it."""
    return resolve_contextual_date_binding(
        question,
        matched_metrics=[dict(METRIC)],
        bindings=[dict(row) for row in contexts],
        date_roles=[dict(role) for role in fact_roles],
        required_fact_tables={"DBO.F_SALES"},
    )["status"]


class TestTheBacklogAsksTheResolver:
    """Not a copy of its rule -- the function itself."""

    @pytest.mark.parametrize("label,contexts,fact_roles", STATES,
                             ids=[s[0] for s in STATES])
    def test_the_verdict_is_the_resolvers_own(self, label, contexts, fact_roles):
        assert _resolver_verdict(
            METRIC, QUESTION,
            metric_date_contexts=contexts,
            fact_date_roles=fact_roles,
        ) == _resolver_status(contexts, fact_roles), label


class TestWhichStatusEarnsWhichRemedy:
    """A status the reader feels, mapped to the one thing that ends it."""

    @staticmethod
    def _period_variation():
        variation = next(v for v in variations_for(dict(METRIC))
                         if v.kind == "grain")
        return variation

    @pytest.mark.parametrize("label,contexts,fact_roles", STATES,
                             ids=[s[0] for s in STATES])
    def test_the_remedy_matches_what_the_runtime_does(
            self, label, contexts, fact_roles):
        variation = self._period_variation()
        gap = _date_gap(variation, dict(METRIC),
                        metric_date_contexts=contexts,
                        fact_date_roles=fact_roles, lang="en")
        status = _resolver_status(contexts, fact_roles, variation.question)
        if status == "none":
            assert gap is not None and gap.reason_id == "coverage.gap.no_date_role", label
        elif status == "ambiguous":
            assert gap is not None, label
            assert gap.reason_id == "coverage.gap.no_default_date_role", label
        else:
            assert gap is None, (label, status)

    def test_absence_and_ambiguity_are_different_items(self):
        """One says bind a date; the other says pick between the ones you
        have. An admin sent to the wrong screen loses the afternoon."""
        variation = self._period_variation()
        absent = _date_gap(variation, dict(METRIC),
                           metric_date_contexts=NO_CONTEXT,
                           fact_date_roles=NO_FACT_ROLE, lang="en")
        undecided = _date_gap(variation, dict(METRIC),
                              metric_date_contexts=NO_DEFAULT_CONTEXT,
                              fact_date_roles=NO_FACT_ROLE, lang="en")
        assert absent.reason_id != undecided.reason_id
        assert absent.reason != undecided.reason
        assert absent.missing == undecided.missing == "date_role"


class TestTheWholeReportAgrees:
    """check_variation and coverage_report must PASS BOTH sources through.

    The defect was not only in the rule: production readiness loaded the
    measure's contexts and never the fact's roles, so the fact half was
    always empty however good the rule was.
    """

    @pytest.mark.parametrize("label,contexts,fact_roles", STATES,
                             ids=[s[0] for s in STATES])
    def test_the_report_reports_exactly_what_the_runtime_does(
            self, label, contexts, fact_roles):
        report = coverage_report(dict(METRIC),
                                 metric_date_contexts=contexts,
                                 fact_date_roles=fact_roles)
        date_reasons = {gap.reason_id for gap in report.gaps
                        if gap.missing == "date_role"}
        blocked = _resolver_status(contexts, fact_roles) in {"none", "ambiguous"}
        assert bool(date_reasons) is blocked, (label, date_reasons)

    def test_the_fact_role_alone_makes_a_measure_complete(self):
        """The state that was reported as fifteen gaps. It is zero."""
        report = coverage_report(dict(METRIC),
                                 metric_date_contexts=NO_CONTEXT,
                                 fact_date_roles=ONE_APPROVED_DEFAULT)
        assert [g.reason_id for g in report.gaps if g.missing == "date_role"] == []
        assert _resolver_status(NO_CONTEXT, ONE_APPROVED_DEFAULT) == "selected"

    def test_dropping_the_fact_roles_brings_the_false_gaps_back(self):
        """Proves the fact half is actually read, rather than accepted and
        discarded -- the shape of every swallowed-argument bug on this repo."""
        without = coverage_report(dict(METRIC),
                                  metric_date_contexts=NO_CONTEXT,
                                  fact_date_roles=[])
        assert [g.reason_id for g in without.gaps if g.missing == "date_role"]

    def test_a_dimension_question_naming_a_period_needs_the_date_too(self):
        """The second date branch, which had its own copy of the rule."""
        variation = next(v for v in variations_for(dict(METRIC))
                         if v.kind == "dimension" and v.grain)
        settled = check_variation(variation, dict(METRIC),
                                  metric_date_contexts=NO_CONTEXT,
                                  fact_date_roles=ONE_APPROVED_DEFAULT)
        assert settled is None
        undecided = check_variation(variation, dict(METRIC),
                                    metric_date_contexts=NO_DEFAULT_CONTEXT,
                                    fact_date_roles=NO_FACT_ROLE)
        assert undecided is not None
        assert undecided.reason_id == "coverage.gap.no_default_date_role"


class TestTheRemedyIsReadableInBothLanguages:

    def test_both_reasons_are_in_the_catalogue(self):
        for message_id in ("coverage.gap.no_date_role",
                           "coverage.gap.no_default_date_role"):
            for lang in ("en", "fr"):
                assert i18n.t(message_id, lang=lang).strip(), (message_id, lang)

    def test_the_french_report_says_it_in_french(self):
        english = coverage_report(dict(METRIC),
                                  metric_date_contexts=NO_DEFAULT_CONTEXT,
                                  fact_date_roles=NO_FACT_ROLE, lang="en")
        french = coverage_report(dict(METRIC),
                                 metric_date_contexts=NO_DEFAULT_CONTEXT,
                                 fact_date_roles=NO_FACT_ROLE, lang="fr")
        assert english.gaps and french.gaps
        assert english.gaps[0].reason != french.gaps[0].reason


class TestNeverRaises:
    """Coverage is advice. Advice that breaks the readiness page is worse
    than none, and the resolver is now inside the call."""

    @pytest.mark.parametrize("junk", [
        None, [], [None], ["not a dict"], [{"id": 1}, None],
    ])
    def test_junk_in_either_list_is_survived(self, junk):
        report = coverage_report(dict(METRIC),
                                 metric_date_contexts=junk,
                                 fact_date_roles=junk)
        assert report.total > 0

    def test_a_metric_with_no_table_still_reports(self):
        """`required_fact_tables` is derived from base_table, which a draft
        defined in chat may not have yet."""
        report = coverage_report({**METRIC, "base_table": ""},
                                 metric_date_contexts=NO_CONTEXT,
                                 fact_date_roles=ONE_APPROVED_DEFAULT)
        assert report.total > 0
