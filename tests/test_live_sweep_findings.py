"""
tests/test_live_sweep_findings.py

Found by running thirteen realistic questions against the live test warehouse,
not by reading code. Each one produced a plausible answer, which is why none of
them had been reported.

  B1  Every calendar-period question left the governed compiler.
  B2  A filter value could not pull in the table it belongs to.
  B5  Two trend classifiers disagreed inside one answer.
  B6  A warehouse was rendered as "redacted segment" beside its own name.
"""

from __future__ import annotations

import pytest


# ── B1: calendar periods are compilable ──────────────────────────────────────


class TestCalendarPeriodsReachTheGovernedCompiler:
    """"today" compiled deterministically; "this month", "last month", "this
    year" and friends were rejected by the window-kind gate and answered by
    free-form SQL instead — the shape most likely to get a period boundary
    subtly wrong, and the reason those answers needed repair retries that then
    capped their confidence at 75."""

    ANCHOR = "CAST('2026-06-30' AS date)"

    def test_the_gate_accepts_calendar_periods(self):
        from core.pipeline_helpers import _COMPILABLE_WINDOW_KINDS

        for kind in ("this_month", "previous_month", "this_quarter",
                     "previous_quarter", "this_year", "previous_year"):
            assert kind in _COMPILABLE_WINDOW_KINDS
        # Unchanged.
        for kind in ("today", "yesterday", "last_n", "latest_n_observed"):
            assert kind in _COMPILABLE_WINDOW_KINDS

    @pytest.mark.parametrize("kind,start_frag,ends_at_anchor", [
        ("this_month",   "DATEDIFF(month, 0,",   True),
        ("this_quarter", "DATEDIFF(quarter, 0,", True),
        ("this_year",    "DATEDIFF(year, 0,",    True),
    ])
    def test_a_current_period_ends_at_the_anchor_not_the_calendar_end(
        self, kind, start_frag, ends_at_anchor,
    ):
        """The current period is only partly loaded. Running to its calendar
        end would claim coverage the warehouse does not have."""
        from core.pipeline_helpers import calendar_period_bounds

        start, end = calendar_period_bounds("azure_sql", self.ANCHOR, kind)
        assert start_frag in start
        assert end == self.ANCHOR

    @pytest.mark.parametrize("kind", [
        "previous_month", "previous_quarter", "previous_year",
    ])
    def test_a_previous_period_ends_the_day_before_this_one_starts(self, kind):
        """Which handles February and leap years without a special case."""
        from core.pipeline_helpers import calendar_period_bounds

        start, end = calendar_period_bounds("azure_sql", self.ANCHOR, kind)
        assert start and end
        assert end.startswith("DATEADD(day, -1,")
        assert self.ANCHOR in start

    def test_oracle_uses_its_own_truncation(self):
        from core.pipeline_helpers import calendar_period_bounds

        start, end = calendar_period_bounds("oracle", self.ANCHOR, "previous_month")
        assert "TRUNC(" in start and "ADD_MONTHS(" in start
        assert end.endswith("- 1")

    def test_this_week_is_deliberately_not_compiled(self):
        """SQL Server's DATEDIFF(week, …) boundary and Oracle's TRUNC(d,'IW')
        are not the same day. A silently different first-day-of-week moves the
        answer, so it keeps falling back rather than guessing per dialect."""
        from core.pipeline_helpers import (
            _CALENDAR_PERIOD_KINDS, calendar_period_bounds,
        )

        assert "this_week" not in _CALENDAR_PERIOD_KINDS
        assert calendar_period_bounds("azure_sql", self.ANCHOR, "this_week") == ("", "")

    def test_an_unknown_dialect_declines_rather_than_emitting_wrong_sql(self):
        from core.pipeline_helpers import calendar_period_bounds

        assert calendar_period_bounds("mysql", self.ANCHOR, "this_month") == ("", "")


# ── B2: a value brings in its own table ──────────────────────────────────────


class TestANearMissValueStillIdentifiesItsTable:
    """Refusing to substitute "Calgary Distribution Centre East" for "Calgary
    Distribution Centre" is right. Dropping the warehouse table along with it
    was not: the question resolved to the fact alone and came back "I couldn't
    find the right tables or columns", which is not what went wrong."""

    GRAPH = {
        "entities": [
            {"entity_name": "Warehouse", "table_name": "WHS_DMS", "schema_name": "EMDW_DMART"},
            {"entity_name": "Sales", "table_name": "CUS_ORD_IVC_FCT", "schema_name": "EMDW_DMART"},
        ],
    }

    def _entities(self, resolved):
        from core.query_pipeline import _graph_entities_for_verified_values

        return _graph_entities_for_verified_values(resolved, self.GRAPH)

    def test_a_narrowed_value_contributes_its_table(self):
        assert self._entities({
            "verified": [], "in_lists": [], "clarify": [],
            "narrowed": [{
                "table_fqn": "EMDW_DMART.WHS_DMS", "column": "WHS_NM",
                "value": "Calgary Distribution Centre", "dropped": ["east"],
            }],
        }) == {"Warehouse"}

    def test_verified_and_in_list_values_still_do(self):
        assert self._entities({
            "verified": [{"table_fqn": "EMDW_DMART.WHS_DMS", "column": "WHS_NM"}],
            "in_lists": [], "clarify": [], "narrowed": [],
        }) == {"Warehouse"}

    def test_nothing_resolved_contributes_nothing(self):
        assert self._entities(
            {"verified": [], "in_lists": [], "clarify": [], "narrowed": []}
        ) == set()

    def test_the_value_is_still_not_substituted(self):
        """Identifying the table and accepting the value are separate
        decisions — this fix must not undo the one that stopped the wrong
        customer's number being returned."""
        from core.value_resolver import uncovered_phrase_tokens

        assert uncovered_phrase_tokens(
            "Calgary Distribution Centre East", "Calgary Distribution Centre",
        ) == ["east"]


# ── B5: one series, one verdict ──────────────────────────────────────────────


class TestAFlatSeriesIsNotGivenADirection:
    """A 0.7% drift was reported as "the trend in Revenue is downward" in a
    follow-up chip, directly beside a narrative reading "holding steady — no
    urgent action indicated". Two classifiers, one series, opposite verdicts."""

    @staticmethod
    def _series(step):
        return [{"MONTH": f"2026-0{i}", "REVENUE": 1000 + i * step} for i in range(1, 7)]

    def test_a_barely_moving_series_is_reported_as_flat(self):
        from core.stat_signals import compute_signals

        kinds = {s["type"] for s in compute_signals(self._series(2))}
        assert "flat_trend" in kinds
        assert "temporal" not in kinds

    def test_a_real_move_still_gets_its_direction(self):
        from core.stat_signals import compute_signals

        signals = compute_signals(self._series(-100))
        temporal = [s for s in signals if s["type"] == "temporal"]
        assert temporal and temporal[0]["direction"] == "downward"

    def test_the_flat_case_asks_something_that_is_actually_true(self):
        from core.stat_signals import compute_signals, template_suggestions

        questions = template_suggestions(
            compute_signals(self._series(2)), ["MONTH", "REVENUE"],
        )
        assert not any("downward" in q or "upward" in q for q in questions)
        assert any("barely moved" in q for q in questions)

    def test_the_band_matches_the_narrative_layers_intent(self):
        from core.stat_signals import _FLAT_TREND_PCT

        assert 0 < _FLAT_TREND_PCT <= 10


# ── B6: a warehouse is not a person ──────────────────────────────────────────


class TestBusinessEntityLabelsAreNotRedacted:
    """The insight line read "redacted segment leads at $3,874,769.20" while
    the answer directly above it named the warehouse. Worse, it turned on the
    ALIAS the generated SQL happened to pick — "warehouse" rendered, "warehouse
    name" redacted — so the same question could redact or not between runs."""

    @pytest.mark.parametrize("column", [
        "warehouse name", "warehouse", "WHS_NM", "supplier name", "item name",
        "division name", "profit center name", "profit centre name",
        "branch name", "site name", "product name", "month name",
    ])
    def test_a_business_entity_label_survives(self, column):
        from core.insight import _is_sensitive_field

        assert _is_sensitive_field(column) is False

    @pytest.mark.parametrize("column", [
        "customer name", "employee name", "patient name", "contact name",
        "first_name", "email", "phone_number", "member_id",
    ])
    def test_anything_that_could_name_a_person_still_redacts(self, column):
        """The list fails toward masking on purpose. A customer may well be a
        person, so it stays redacted — that behaviour is deliberate and
        separately covered by tests/test_pii_hardening.py."""
        from core.insight import _is_sensitive_field

        assert _is_sensitive_field(column) is True

    def test_the_display_label_follows_the_same_rule(self):
        from core.insight import _display_label

        assert _display_label("Toronto DC", "warehouse name") == "Toronto DC"
        assert _display_label("A Person", "customer name") == "redacted segment"


# ── B4: a low verdict states its reason ──────────────────────────────────────


def test_a_low_confidence_answer_shows_its_warning_inline():
    """Reasons and warnings live inside the "How this answer was produced"
    disclosure so they do not compete with a good answer. When the verdict is
    that the answer may be wrong, that is precisely the thing the reader needs,
    and a 49/100 result looked no different from any other."""
    from pathlib import Path

    source = Path("portal/templates/portal_chat.html").read_text(encoding="utf-8")
    assert "const lowWarn" in source
    assert "confidenceLevel === 'low'" in source
    assert "trust-warn" in source
