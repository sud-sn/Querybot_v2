"""
tests/test_frozen_windows_and_planning_failure.py

The last two findings on the shortlist. Both are the same shape as the ones
before them: something stops working and nothing says so.

ALERTS PINNED TO THE DAY THEY WERE CREATED. An alert stores the SQL that
answered the question and re-runs it unchanged forever. When the question named
a relative window, "today" was already resolved to a literal date before that
SQL was stored — so the alert monitors the day it was created on, permanently.
"I'll monitor NET_REVENUE (baseline: 1,240,553) and flag it when the value
changes by more than 10%" and then it never flags anything, because the number
it re-reads is the same number every time. With an above/below condition the
opposite happens: it fires on every check and never stops.

FIELD PLANNING FAILING TO {} AT DEBUG LEVEL. The deterministic field plan
carries four separate guarantees — term-to-column bindings, the required join
path, the superseded-column list that forbids a retired column, and the
temporal policy that scopes the window. An exception replaced all four with an
empty dict, logged below the default level, and the answer was written from raw
KB prose while looking exactly like a planned one.
"""

from __future__ import annotations

import pytest

import core.alert_engine as alert_engine
import core.date_anchor as date_anchor

POLICY = {
    "kind": "today", "anchor_policy": "latest_available",
    "fact_table": "DW.F_SALES", "fact_column": "INV_DT", "date_column": "INV_DT",
}
PINNED_SQL = "SELECT SUM(AMT) AS TOTAL FROM DW.F_SALES WHERE INV_DT = '2025-04-17'"


def _alert(**overrides):
    base = {
        "id": "a1", "question": "what is my revenue today", "sql": PINNED_SQL,
        "account_id": "acct", "db_type": "azure_sql", "status": "active",
        "anchor_policy": POLICY, "anchor_value": "2025-04-17",
    }
    base.update(overrides)
    return base


@pytest.fixture
def anchor(monkeypatch):
    """Control what the warehouse reports as its newest business date."""
    state = {"value": "2025-04-17"}

    def _resolve(*_args, **_kwargs):
        return dict(state) if state.get("value") else {}

    monkeypatch.setattr(date_anchor, "resolve_business_anchor", _resolve)
    return state


def _refresh(alert):
    return alert_engine._refresh_relative_window(alert, {"db_type": "azure_sql"})


# ── The alert follows the data ───────────────────────────────────────────────


def test_the_window_moves_when_the_warehouse_does(anchor):
    """The reported scenario: the dev database had its range moved from
    2025-04-17 back to 2024-08-11. The alert must read the new date."""
    anchor["value"] = "2024-08-11"
    sql, problem = _refresh(_alert())
    assert problem == ""
    assert "'2024-08-11'" in sql
    assert "'2025-04-17'" not in sql, "still querying the day it was created on"


def test_an_unchanged_anchor_leaves_the_sql_alone(anchor):
    sql, problem = _refresh(_alert())
    assert problem == ""
    assert sql == PINNED_SQL


def test_a_fixed_period_alert_is_not_re_anchored(anchor):
    """"Revenue in March 2026" is pinned deliberately. Only a RELATIVE window
    is wrong to freeze, and such an alert carries no latest_available policy."""
    sql, problem = _refresh(_alert(
        question="revenue in March 2026", anchor_policy={}, anchor_value="",
    ))
    assert problem == ""
    assert sql == PINNED_SQL


# ── When it cannot be moved, it is not evaluated ─────────────────────────────


def test_an_alert_predating_the_stored_policy_is_refused(anchor):
    """Alerts already on disk have no policy and no recorded anchor, so which
    literal in their SQL was the anchor is unknowable. Comparing against a
    frozen window is not a check — it is a fixed answer wearing one."""
    _sql, problem = _refresh(_alert(anchor_policy={}, anchor_value=""))
    assert problem == "relative_window_not_re_anchorable"


def test_an_unresolvable_anchor_is_refused(anchor):
    anchor["value"] = ""
    assert _refresh(_alert())[1] == "anchor_unavailable"


def test_an_anchor_literal_missing_from_the_sql_is_refused(anchor):
    """Re-anchoring works by substituting the recorded literal. If it is not
    there, substituting would be guessing at what the query means."""
    assert _refresh(_alert(sql="SELECT SUM(AMT) FROM DW.F_SALES"))[1] == "anchor_literal_absent"


def test_a_probe_failure_is_refused_not_ignored(anchor, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(date_anchor, "resolve_business_anchor", _boom)
    assert _refresh(_alert())[1] == "anchor_probe_failed"


def test_the_check_refuses_rather_than_comparing_a_frozen_window(monkeypatch):
    """The guard has to be wired into check_alert_now, not merely exist."""
    monkeypatch.setattr(
        alert_engine, "get_alert",
        lambda _id: _alert(anchor_policy={}, anchor_value=""),
    )
    result = alert_engine.check_alert_now("a1", {"db_type": "azure_sql"})
    assert result["ok"] is False
    assert result["reason"] == "relative_window_not_re_anchorable"
    assert "Recreate it" in result["detail"]


# ── The policy is captured at creation ───────────────────────────────────────


def test_the_policy_and_the_resolved_date_are_taken_from_the_plan():
    assert alert_engine._relative_window_of({
        "temporal_policies": [POLICY],
        "resolved_date_anchor": {"value": "2025-04-17"},
    }) == (POLICY, "2025-04-17")


def test_a_plan_with_no_relative_window_yields_nothing():
    assert alert_engine._relative_window_of({"temporal_policies": []}) == ({}, "")
    assert alert_engine._relative_window_of(None) == ({}, "")


def test_the_pipeline_puts_the_resolved_anchor_on_the_plan():
    """_relative_window_of reads it off the semantic plan, and the plan is what
    the result cache carries to whatever an answer becomes."""
    import inspect

    import core.query_pipeline as pipeline

    source = inspect.getsource(pipeline._handle_query_impl)
    assert '_semantic_plan["resolved_date_anchor"] = _resolved_anchor' in source


def test_the_alert_caller_passes_the_plan():
    from pathlib import Path

    source = Path("gateway/webhooks.py").read_text(encoding="utf-8")
    assert "semantic_plan = cached.get(\"semantic_plan\")," in source


# ── Field planning failure is loud, and costs confidence ─────────────────────


def test_a_planning_failure_is_recorded_on_the_plan():
    import inspect

    import core.query_pipeline as pipeline

    source = inspect.getsource(pipeline._handle_query_impl)
    assert '_semantic_plan = {"planning_failed": True}' in source
    assert 'log.debug("Semantic field planning skipped' not in source


def test_the_flag_survives_the_plan_merge():
    from core.pipeline_context import _merge_semantic_plans

    merged = _merge_semantic_plans({"planning_failed": True}, {"enabled": False})
    assert merged.get("planning_failed") is True

    clean = _merge_semantic_plans({"enabled": False}, {"enabled": False})
    assert not clean.get("planning_failed")


def test_ambiguous_measures_survive_the_merge_too():
    """They are only ever set on a DISABLED plan — which is exactly the plan
    the merge skips — so collecting them below that check meant they could
    never reach the prompt at all."""
    from core.pipeline_context import _merge_semantic_plans

    merged = _merge_semantic_plans(
        {"enabled": False, "ambiguous_measures": ["DW.F.GROSS_AMT", "DW.F.NET_AMT"]},
        {"enabled": False},
    )
    assert merged.get("ambiguous_measures") == ["DW.F.GROSS_AMT", "DW.F.NET_AMT"]


def test_an_answer_planned_without_the_field_plan_is_marked_down():
    from core.answer_confidence import build_answer_confidence

    failed = build_answer_confidence(
        validation_code="ok", row_count=10, tables_used=["DW.F_SALES"],
        semantic_planning_failed=True,
    )
    nothing_to_plan = build_answer_confidence(
        validation_code="ok", row_count=10, tables_used=["DW.F_SALES"],
    )
    assert failed["score"] < nothing_to_plan["score"]
    assert any("business-term mapping" in w for w in failed["warnings"])


def test_a_successful_plan_is_unaffected():
    from core.answer_confidence import build_answer_confidence

    planned = build_answer_confidence(
        validation_code="ok", row_count=10, tables_used=["DW.F_SALES"],
        has_semantic_plan=True, semantic_planning_failed=True,
    )
    assert not any("business-term mapping" in w for w in planned["warnings"])


def test_the_pipeline_hands_the_failure_to_the_confidence_scorer():
    from pathlib import Path

    pipeline = Path("core/query_pipeline.py").read_text(encoding="utf-8")
    assert '"semantic_planning_failed": bool(' in pipeline

    renderer = Path("core/result_renderer.py").read_text(encoding="utf-8")
    assert 'semantic_planning_failed=bool(confidence_context.get(' in renderer
