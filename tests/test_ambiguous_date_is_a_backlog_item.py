"""
tests/test_ambiguous_date_is_a_backlog_item.py

A measure that interrupts every temporal question looked finished.

core.contextual_dates.resolve_contextual_date_binding returns "ambiguous" when
a measure has several bound business dates and none of them is the default: it
cannot choose, so it asks the reader which date to use. Correct behaviour --
guessing between invoice date and delivery date is how a number comes back
wrong.

But it asks EVERY TIME. There is no learning step: the answer is remembered for
the thread, not for the model, so the next reader and the next conversation are
asked again. A measure in that state is not a working measure with a small
rough edge; it is a measure that cannot answer a question about a period
without a round trip through a human.

core.metric_coverage treated "has date roles" as covered, so that measure
appeared complete in readiness while producing a clarification for every
period question anyone asked of it. The admin was never told, and the remedy is
one click: mark one of the roles as the default.

Absence was already reported (coverage.gap.no_date_role). This is the other
half -- presence without a decision.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.metric_coverage import (
    _dates_are_ambiguous, check_variation, coverage_report, variations_for,
)

METRIC = {
    "id": 7,
    "name": "Net Revenue",
    "base_table": "DBO.F_SALES",
    "allowed_dimensions": "Region, Product",
    "grain": "day",
}

ONE_DEFAULT = [
    {"id": 1, "context_name": "Invoice Date", "is_default": 1},
    {"id": 2, "context_name": "Order Date", "is_default": 0},
]
NO_DEFAULT = [
    {"id": 1, "context_name": "Invoice Date", "is_default": 0},
    {"id": 2, "context_name": "Order Date", "is_default": 0},
]
TWO_DEFAULTS = [
    {"id": 1, "context_name": "Invoice Date", "is_default": 1},
    {"id": 2, "context_name": "Order Date", "is_default": 1},
]


class TestWhichShapesCountAsAmbiguous:
    """The same rule the resolver applies, so the backlog and the runtime
    cannot disagree about whether a measure is finished."""

    def test_no_dates_at_all_is_absence_not_ambiguity(self):
        assert _dates_are_ambiguous([]) is False
        assert _dates_are_ambiguous(None) is False

    def test_one_date_is_never_ambiguous(self):
        """However it is marked. There is nothing to choose between."""
        assert _dates_are_ambiguous([{"id": 1}]) is False
        assert _dates_are_ambiguous([{"id": 1, "is_default": 1}]) is False

    def test_several_with_one_default_is_settled(self):
        assert _dates_are_ambiguous(ONE_DEFAULT) is False

    def test_several_with_none_default_is_ambiguous(self):
        assert _dates_are_ambiguous(NO_DEFAULT) is True

    def test_several_defaults_is_ambiguous_too(self):
        """The resolver cannot choose between two defaults either -- it returns
        "multiple metric defaults"."""
        assert _dates_are_ambiguous(TWO_DEFAULTS) is True

    def test_junk_in_the_list_does_not_make_it_ambiguous(self):
        assert _dates_are_ambiguous([{"id": 1, "is_default": 1}, None, "x"]) is False


class TestTheRuntimeAndTheBacklogAgree:
    """If these two ever diverge, an admin is told a measure is finished while
    readers are being asked about it."""

    def _resolver_says(self, contexts):
        from core.contextual_dates import resolve_contextual_date_binding

        return resolve_contextual_date_binding(
            "net revenue by month",
            matched_metrics=[dict(METRIC)],
            bindings=[dict(c, metric_id=7) for c in contexts],
            date_roles=[],
        )["status"]

    @pytest.mark.parametrize("contexts,ambiguous", [
        (NO_DEFAULT, True),
        (TWO_DEFAULTS, True),
        (ONE_DEFAULT, False),
    ])
    def test_the_backlog_flags_exactly_what_the_resolver_asks_about(
            self, contexts, ambiguous):
        assert _dates_are_ambiguous(contexts) is ambiguous
        status = self._resolver_says(contexts)
        assert (status == "ambiguous") is ambiguous, (contexts, status)


class TestTheGapIsReported:

    def _gap_for(self, kind, contexts, lang="en"):
        variations = [v for v in variations_for(dict(METRIC), lang=lang)
                      if v.kind == kind]
        assert variations, kind
        return check_variation(variations[0], dict(METRIC),
                               date_roles=contexts, lang=lang)

    @pytest.mark.parametrize("kind", ["grain", "comparison", "trend"])
    def test_a_period_question_reports_it(self, kind):
        gap = self._gap_for(kind, NO_DEFAULT)
        assert gap is not None, kind
        assert gap.missing == "date_role", gap
        assert "none is the default" in gap.reason, gap.reason

    @pytest.mark.parametrize("kind", ["grain", "comparison", "trend"])
    def test_a_settled_measure_reports_nothing(self, kind):
        assert self._gap_for(kind, ONE_DEFAULT) is None, kind

    def test_absence_is_still_reported_as_absence(self):
        """The two gaps are different remedies -- add a role, or pick one --
        and must not collapse into each other."""
        gap = self._gap_for("grain", [])
        assert gap is not None
        assert "No business date is bound" in gap.reason, gap.reason

    def test_the_remedy_is_the_one_click_that_ends_it(self):
        gap = self._gap_for("grain", NO_DEFAULT)
        assert "Mark one as the default" in gap.reason, gap.reason


class TestItReachesTheAdminsBacklog:

    def test_the_report_carries_the_gap(self):
        report = coverage_report(dict(METRIC), date_roles=NO_DEFAULT)
        reasons = {gap.reason for gap in report.gaps}
        assert any("none is the default" in reason for reason in reasons), reasons

    def test_a_settled_measure_has_no_date_gap(self):
        report = coverage_report(dict(METRIC), date_roles=ONE_DEFAULT)
        assert not any("date" in gap.missing for gap in report.gaps), \
            [(g.missing, g.reason) for g in report.gaps]

    def test_it_is_one_backlog_item_not_one_per_question(self):
        """An admin who marks one default unblocks every period shape at once.
        Listing them separately makes a one-click fix look like a week."""
        from core.model_readiness import _kind_for

        report = coverage_report(dict(METRIC), date_roles=NO_DEFAULT)
        date_gaps = [g for g in report.gaps if g.missing == "date_role"]
        assert len(date_gaps) >= 2, "expected several shapes to be blocked"
        by_remedy = {(g.missing, g.reason) for g in date_gaps}
        assert len(by_remedy) == 1, by_remedy
        assert _kind_for("date_role") == "date_role"

    def test_it_carries_the_weight_of_a_date_gap(self):
        """date_role is the heaviest kind in the backlog: it blocks more
        question shapes than anything else."""
        from core.model_readiness import BacklogItem

        assert BacklogItem("date_role", "Net Revenue", "x").weight == 5


class TestItIsInTheAdminsLanguage:

    def test_the_gap_reads_in_french(self):
        variation = next(v for v in variations_for(dict(METRIC), lang="fr")
                         if v.kind == "grain")
        gap = check_variation(variation, dict(METRIC),
                              date_roles=NO_DEFAULT, lang="fr")
        assert gap is not None
        assert "aucune n'est la date par défaut" in gap.reason, gap.reason

    def test_both_date_gaps_exist_in_both_languages(self):
        for key in ("coverage.gap.no_date_role",
                    "coverage.gap.no_default_date_role"):
            for lang in ("en", "fr"):
                text = i18n.t(key, lang=lang)
                assert text and text != key, (key, lang)
        assert i18n.t("coverage.gap.no_default_date_role", lang="en") != \
            i18n.t("coverage.gap.no_default_date_role", lang="fr")
