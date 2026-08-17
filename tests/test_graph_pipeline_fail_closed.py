"""Wiring guards for governed entity-graph planning in the SQL pipeline."""

from pathlib import Path


PIPELINE_SOURCE = (
    Path(__file__).parents[1] / "core" / "query_pipeline.py"
).read_text(encoding="utf-8")


def test_blocked_graph_plan_stops_before_sql_generation():
    blocked_guard = PIPELINE_SOURCE.index(
        '_graph_ctx.get("planning_status") == "blocked"'
    )
    sql_generation = PIPELINE_SOURCE.index(
        "sql, tok_in, tok_out = await llm_complete(", blocked_guard
    )
    guard_return = PIPELINE_SOURCE.index("            return", blocked_guard)

    assert blocked_guard < guard_return < sql_generation
    assert "I couldn't build a trusted join plan" in PIPELINE_SOURCE[blocked_guard:guard_return]


def test_ambiguous_graph_path_is_persisted_as_resumable_clarification():
    assert 'selected_clarification_option(event, "graph_join_path")' in PIPELINE_SOURCE
    assert '"source": "graph_join_path"' in PIPELINE_SOURCE
    assert 'can_request_clarification(event, "graph_join_path")' in PIPELINE_SOURCE
