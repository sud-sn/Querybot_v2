"""
tests/test_metric_registry_time_scope.py

The metric registry is a deterministic fast path: match a synonym, run the
stored SQL, present the answer as a trusted metric. It matched on synonyms
alone, so every qualifier the user wrapped around that synonym was dropped.

Grouping qualifiers were already guarded ("revenue by region" falls through to
the planner, because a fixed template has no GROUP BY). Time qualifiers were
not. "revenue", "revenue today", "revenue yesterday" and "revenue last month"
all matched a metric named "revenue" and all received the identical SQL — the
same number for four different questions, presented at high confidence, and the
same number again the next day. That is the reported stale-answer symptom
arriving by a route that never consults the business-date anchor at all, so no
amount of anchor hardening reaches it.

These tests drive store.match_metric — the exact function core/query_pipeline.py
calls to make this routing decision — rather than reading the source, because a
guard that is present in the file and never reached is the failure mode this
repository keeps producing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import store.config_store as config_store

# A plain lifetime-total metric: no window in the name, none in the SQL.
TOTAL = {
    "name": "Revenue",
    "synonyms": "revenue, sales value",
    "formula_type": "query",
    "sql_template": "SELECT SUM(AMOUNT) AS REVENUE FROM FACT",
}
# A metric an admin deliberately authored FOR one window. The matched wording
# carries the window, so the template was written to express it.
MONTHLY = {
    "name": "Revenue This Month",
    "synonyms": "revenue this month",
    "formula_type": "query",
    "sql_template": "SELECT SUM(AMOUNT) FROM FACT WHERE PERIOD = CURRENT_PERIOD",
}


def _match(question: str, metrics=(TOTAL, MONTHLY)):
    with patch.object(config_store, "list_metrics", return_value=list(metrics)):
        matched = config_store.match_metric("acct", question)
    return (matched or {}).get("name")


@pytest.mark.parametrize("question", [
    "revenue",
    "what is my revenue",
    "total revenue",
    "show me sales value",
])
def test_an_unscoped_question_still_takes_the_deterministic_route(question):
    """The fast path is the point of the registry. Nothing here states a
    window, so the stored template answers exactly the question asked."""
    assert _match(question) == "Revenue"


@pytest.mark.parametrize("question", [
    "revenue today",
    "revenue yesterday",
    "revenue last month",
    "revenue this year",
    "revenue for the last 7 days",
    "sales value this quarter",
])
def test_a_time_scoped_question_is_planned_instead_of_templated(question):
    """Each of these previously returned the lifetime total, badged as a
    trusted metric. The number was wrong, it did not differ between the
    questions, and it did not change when the warehouse was reloaded."""
    assert _match(question) is None, (
        f"{question!r} was answered from a fixed template that carries no "
        f"time scope"
    )


def test_all_four_phrasings_no_longer_collapse_to_one_answer():
    """The symptom as reported: ask about three different periods, get one
    number three times. Whatever each of these resolves to now, they must not
    all resolve to the same stored template."""
    routed = {q: _match(q) for q in (
        "revenue today", "revenue yesterday", "revenue last month",
    )}
    assert set(routed.values()) == {None}, (
        f"still collapsing to a single template: {routed}"
    )


@pytest.mark.parametrize("question", [
    "revenue in March 2026",
    "revenue for 2025",
    "revenue in February 2026",
])
def test_an_absolute_date_never_qualifies_for_a_template(question):
    """A synonym does not name a specific month, so a match on a dated
    question always means the date was silently dropped."""
    assert _match(question) is None


def test_a_metric_authored_for_a_window_still_serves_that_window():
    """The guard must not cost admins the period metrics they defined on
    purpose. The matched wording carries the same relative window the question
    does, so the template was written for it."""
    assert _match("revenue this month") == "Revenue This Month"


def test_the_window_must_match_not_merely_be_present():
    """'Revenue This Month' is not an answer to 'revenue last month'."""
    assert _match("revenue last month") is None


def test_a_clarification_label_does_not_read_as_the_user_scoping_a_window():
    """Clarification wrappers carry administrative text, often column names
    like 'order month'. Treating that as the user asking about a month would
    withhold the fast path from a question that never scoped anything."""
    question = ("revenue Clarification for the same request: "
                "Synonyms: order month, ship month")
    assert _match(question) == "Revenue"


def test_a_followup_that_adds_a_window_is_caught():
    """'and for today?' against an earlier result scopes the question just as
    much as typing it out, and reaches this function as combined text."""
    assert _match("revenue\nFollow-up request: and for today?") is None


def test_the_guard_declines_rather_than_falls_open_when_it_cannot_check():
    """Failing open here would restore the exact defect: an unscoped-looking
    question answered from a fixed template. Planning the question instead is
    slower and correct."""
    with patch("core.contextual_dates.detect_temporal_window",
               side_effect=RuntimeError("detector unavailable")):
        assert _match("revenue") is None, (
            "match_metric answered from a stored template while unable to tell "
            "whether the question scoped a time window"
        )


def test_the_pipeline_routes_through_this_function():
    """If the pipeline stopped calling match_metric, every test above would
    keep passing while the fast path went unguarded."""
    import inspect

    import core.query_pipeline as pipeline

    source = inspect.getsource(pipeline._handle_query_impl)
    assert "store.match_metric(" in source
