from pathlib import Path

from gateway.base import PlatformEvent
from core.analytical_intent import plan_analytical_intent
from core.clarification import (
    attach_clarification_resolution,
    can_request_clarification,
    clear_pending,
    get_pending,
    prepare_clarification_meta,
    save_pending,
)


METRICS = [
    {"name": "Revenue", "category": "Sales", "synonyms": "sales value"},
    {"name": "Inventory Value", "category": "Inventory", "synonyms": "stock value"},
]


def _event(text="question"):
    return PlatformEvent("acct", "user", "channel", text, "web", raw={})


def test_daily_snapshot_requests_a_governed_subject():
    plan = plan_analytical_intent("what was my today's data", metrics=METRICS)

    assert plan.intent == "daily_snapshot"
    assert plan.time_range == "today"
    assert plan.needs_clarification
    assert plan.clarification.slot == "subject"
    assert [item["label"] for item in plan.clarification.options] == ["Sales", "Inventory"]
    assert "latest complete governed business date" in " ".join(plan.assumptions)


def test_snapshot_reply_is_replanned_as_the_same_request():
    question = (
        "what was my today's data\n\n"
        "Clarification for the same request: Sales.\n"
        "Use this clarification to interpret the original request; "
        "do not treat it as a separate question."
    )
    plan = plan_analytical_intent(question, metrics=METRICS)

    assert plan.intent == "daily_snapshot"
    assert plan.metrics == ("Revenue",)
    assert not plan.needs_clarification
    assert any("User clarification" in value for value in plan.assumptions)


def test_complete_trend_plan_preserves_grain_and_time():
    plan = plan_analytical_intent(
        "show revenue trend by customer this month",
        metrics=METRICS,
    )

    assert plan.intent == "trend"
    assert plan.metrics == ("Revenue",)
    assert plan.dimensions == ("customer",)
    assert plan.entity_grain == "customer"
    assert plan.time_range == "this month"
    assert not plan.needs_clarification
    assert "ANALYTICAL INTENT PLAN" in plan.prompt_context()


def test_unknown_churn_requires_a_business_definition():
    plan = plan_analytical_intent("find the persons who have churned", metrics=METRICS)

    assert plan.intent == "entity_lookup"
    assert plan.clarification.slot == "business_definition"
    assert "threshold" in plan.clarification.question.lower()


def test_governed_churn_term_removes_definition_question():
    terms = [{"term": "Churned customer", "aliases": "churn,churned"}]
    plan = plan_analytical_intent(
        "find the customers who churned",
        metrics=METRICS,
        terms=terms,
    )

    assert "Churned customer" in plan.business_concepts
    assert not plan.needs_clarification


def test_bare_named_quarter_requires_calendar_basis():
    plan = plan_analytical_intent(
        "compare revenue in Q1 and Q2",
        metrics=METRICS,
    )

    assert plan.intent == "comparison"
    assert plan.time_range == "Q1"
    assert plan.calendar_basis == "unresolved"
    assert plan.clarification.slot == "calendar_basis"
    assert [option["label"] for option in plan.clarification.options] == [
        "Calendar quarters",
        "Fiscal quarters",
    ]


def test_explicit_calendar_quarter_does_not_clarify():
    plan = plan_analytical_intent(
        "compare revenue in calendar Q1 and calendar Q2",
        metrics=METRICS,
    )

    assert plan.calendar_basis == "calendar"
    assert plan.calendar_basis_source == "question"
    assert not plan.needs_clarification


def test_fiscal_quarter_requires_start_month_when_not_configured():
    plan = plan_analytical_intent(
        "compare revenue in fiscal Q1 and fiscal Q2",
        metrics=METRICS,
    )

    assert plan.calendar_basis == "fiscal"
    assert plan.clarification.slot == "fiscal_year_start_month"
    assert len(plan.clarification.options) == 12
    assert plan.clarification.options[3]["label"] == "April"


def test_fiscal_quarter_uses_approved_or_thread_profile():
    plan = plan_analytical_intent(
        "compare revenue in Q1 and Q2",
        metrics=METRICS,
        calendar_profile={
            "basis": "fiscal",
            "fiscal_year_start_month": 4,
            "source": "user_confirmed",
        },
    )

    assert plan.calendar_basis == "fiscal"
    assert plan.fiscal_year_start_month == 4
    assert not plan.needs_clarification
    assert "starting in April" in " ".join(plan.assumptions)


def test_fiscal_clarification_sequence_resolves_without_january_default():
    after_basis = (
        "compare revenue in Q1 and Q2\n\n"
        "Clarification for the same request: Use fiscal quarters for this request.\n"
        "Use this clarification to interpret the original request; "
        "do not treat it as a separate question."
    )
    first_plan = plan_analytical_intent(after_basis, metrics=METRICS)

    assert first_plan.clarification.slot == "fiscal_year_start_month"
    assert first_plan.fiscal_year_start_month is None

    after_month = (
        after_basis
        + "\n\nClarification for the same request: The fiscal year starts in April.\n"
        + "Use this clarification to interpret the original request; "
        + "do not treat it as a separate question."
    )
    final_plan = plan_analytical_intent(after_month, metrics=METRICS)

    assert final_plan.calendar_basis == "fiscal"
    assert final_plan.fiscal_year_start_month == 4
    assert not final_plan.needs_clarification


def test_bounded_loop_allows_only_new_sources():
    first_event = _event()
    first_meta = prepare_clarification_meta(
        first_event,
        {"question": "Which metric?"},
        source="analytical_plan:metric",
    )
    assert first_meta["round_count"] == 1

    second_event = _event("Revenue")
    attach_clarification_resolution(
        second_event,
        {"clarification_meta": first_meta},
    )
    assert not can_request_clarification(second_event, "analytical_plan:metric")
    assert can_request_clarification(second_event, "metric_date_context")

    second_meta = prepare_clarification_meta(
        second_event,
        {"question": "Which date?"},
        source="metric_date_context",
    )
    assert second_meta["round_count"] == 2

    third_event = _event("Invoice date")
    attach_clarification_resolution(
        third_event,
        {"clarification_meta": second_meta},
    )
    third_meta = prepare_clarification_meta(
        third_event,
        {"question": "Which customer status?"},
        source="value_resolver",
    )
    final_event = _event("Active")
    attach_clarification_resolution(
        final_event,
        {"clarification_meta": third_meta},
    )
    assert not can_request_clarification(final_event, "another_source")


def test_pending_clarifications_are_isolated_by_thread(tmp_path, monkeypatch):
    import store.database as database

    monkeypatch.setattr(database, "DATABASE_URL", "")
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "clarifications.db")

    save_pending(
        "acct",
        "user",
        "revenue today",
        clarification_meta={"source": "metric_scope"},
        session_id="acct:user:thread:a",
    )
    save_pending(
        "acct",
        "user",
        "inventory today",
        clarification_meta={"source": "metric_date_context"},
        session_id="acct:user:thread:b",
    )

    thread_a = get_pending("acct", "user", session_id="acct:user:thread:a")
    thread_b = get_pending("acct", "user", session_id="acct:user:thread:b")
    assert thread_a["original_q"] == "revenue today"
    assert thread_b["original_q"] == "inventory today"

    clear_pending("acct", "user", session_id="acct:user:thread:a")
    assert get_pending("acct", "user", session_id="acct:user:thread:a") is None
    assert get_pending("acct", "user", session_id="acct:user:thread:b") is not None
