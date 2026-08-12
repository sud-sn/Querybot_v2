"""Regression coverage for failures observed in the live Azure acceptance run.

Fixtures deliberately use neutral names: the behavior must derive from each
tenant's semantic model and registry rather than any customer's schema.
"""

from pathlib import Path

from core.analytical_intent import plan_analytical_intent
from core.analytical_request_plan import compile_analytical_request_plan
from core.distribution_analysis import detect_boxplot_intent
from core.pipeline_helpers import attempt_field_plan_repair
from core.result_commands import parse_result_command
from core.source_resolution import resolve_source_scope
from core.semantic_model import _scope_plan_to_single_fact
from core.metric_scope import metric_source_tables
from core.pipeline_trace import _learning_result_is_eligible
from core.semantic_planner import build_semantic_field_plan
from core.validator import validate_sql_detailed
from core.vocab_packs import MergedVocab


def _source_model():
    return {
        "tables": [
            {
                "qualified_name": "TENANT.F_SALES",
                "schema": "TENANT",
                "type": "fact",
                "entity": "sales invoice",
                "grain": "one row per invoice line",
                "fields": [
                    {"column": "NET_REVENUE", "role": "attribute", "expanded_name": "Revenue"},
                ],
            },
            {
                "qualified_name": "TENANT.F_INVENTORY_MONTH",
                "schema": "TENANT",
                "type": "fact",
                "entity": "inventory snapshot",
                "grain": "one row per warehouse month",
                "fields": [
                    {"column": "INVENTORY_VALUE", "role": "measure", "expanded_name": "Inventory Value"},
                ],
            },
            {
                "qualified_name": "TENANT.F_INVENTORY_DAY",
                "schema": "TENANT",
                "type": "fact",
                "entity": "inventory snapshot",
                "grain": "one row per warehouse day",
                "fields": [
                    {"column": "INVENTORY_VALUE", "role": "measure", "expanded_name": "Inventory Value"},
                ],
            },
            {
                "qualified_name": "TENANT.ERP_BALANCE",
                "schema": "TENANT",
                "type": "fact",
                "entity": "item balance",
                "grain": "one row per item warehouse period",
                "fields": [
                    {"column": "BALANCE_VALUE", "role": "measure", "expanded_name": "Inventory Value"},
                ],
            },
        ]
    }


def test_approved_metric_source_beats_unrelated_monthly_grain():
    scope = resolve_source_scope(
        "show revenue by invoice month",
        _source_model(),
        authoritative_fact_tables={"TENANT.F_SALES"},
    )
    assert scope["status"] == "selected"
    assert scope["selected_fact"] == "TENANT.F_SALES"
    assert scope["reason"] == "approved metric source binding"


def test_metric_source_tables_collapse_bare_two_and_three_part_aliases():
    tables = metric_source_tables(
        {
            "base_table": "CUS_ORD_IVC_FCT",
            "required_columns": "SOP_CUS_IVC_LIN_AMT",
        },
        {
            "CUS_ORD_IVC_FCT": {"SOP_CUS_IVC_LIN_AMT": "decimal"},
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"SOP_CUS_IVC_LIN_AMT": "decimal"},
            "EMCODW_DEV.EMDW_DMART.CUS_ORD_IVC_FCT": {"SOP_CUS_IVC_LIN_AMT": "decimal"},
        },
    )
    assert tables == {"EMDW_DMART.CUS_ORD_IVC_FCT"}


def test_calendar_day_column_is_never_compiled_as_a_measure():
    plan = build_semantic_field_plan(
        "show revenue trend by day",
        {
            "OPS.F_SALES": {"NET_REVENUE": "decimal", "DATE_SK": "int"},
            "OPS.D_DATE": {"DATE_SK": "int", "DAY": "int", "FULL_DATE": "date"},
        },
        fact_tables={"OPS.F_SALES"},
        preferred_fact_tables={"OPS.F_SALES"},
    )
    day = next(field for field in plan["fields"] if field["column"] == "DAY")
    assert day["role"] == "date_dimension"
    assert day["enforcement"] == "optional"


def test_empty_metric_result_is_not_eligible_for_positive_learning():
    assert not _learning_result_is_eligible(0, {"status": "empty"})
    assert not _learning_result_is_eligible(
        1,
        {
            "status": "fail",
            "numeric_columns": [],
            "resolved_metrics": ["Revenue"],
        },
    )
    assert _learning_result_is_eligible(
        1,
        {
            "status": "pass",
            "numeric_columns": ["Revenue"],
            "resolved_metrics": ["Revenue"],
        },
    )


def test_explicit_erp_source_beats_metric_or_grain_inference():
    vocab = MergedVocab(table_dict={
        "ERP_BALANCE": {
            "label": "ERP Item Balance",
            "synonyms": ["ERP balance", "item balance"],
            "type": "fact",
        }
    })
    scope = resolve_source_scope(
        "From ERP balance show the latest inventory value by warehouse",
        _source_model(),
        vocab=vocab,
        authoritative_fact_tables={"TENANT.F_INVENTORY_DAY"},
    )
    assert scope["status"] == "selected"
    assert scope["selected_fact"] == "TENANT.ERP_BALANCE"
    assert scope["reason"] == "explicit tenant source terminology"


def test_explicit_source_demotes_a_lone_rival_fact_binding():
    fields = [{
        "term": "inventory", "table": "TENANT.F_INVENTORY_DAY",
        "column": "INVENTORY_VALUE", "role": "measure", "enforcement": "required",
    }]
    tables = _source_model()["tables"]
    anchor = _scope_plan_to_single_fact(
        fields, [], tables, preferred_fact_tables={"TENANT.ERP_BALANCE"},
    )
    assert anchor == "TENANT.ERP_BALANCE"
    assert fields[0]["enforcement"] == "optional"


def test_multi_grain_comparison_compiles_isolated_subplans():
    scope = resolve_source_scope(
        "compare monthly inventory with latest daily inventory",
        _source_model(),
    )
    assert scope["status"] == "compound"
    assert set(scope["selected_facts"]) == {
        "TENANT.F_INVENTORY_MONTH", "TENANT.F_INVENTORY_DAY",
    }
    semantic = {
        "enabled": True,
        "source_scope": scope,
        "fields": [
            {"term": "monthly inventory", "table": "TENANT.F_INVENTORY_MONTH", "column": "INVENTORY_VALUE", "role": "measure"},
            {"term": "daily inventory", "table": "TENANT.F_INVENTORY_DAY", "column": "INVENTORY_VALUE", "role": "measure"},
        ],
    }
    plan = compile_analytical_request_plan(
        "compare monthly inventory with latest daily inventory", semantic,
    )
    assert len(plan["subrequests"]) == 2
    assert all(item["aggregate_before_join"] for item in plan["subrequests"])
    assert plan["combination"]["join_inputs"] == "aggregated_subplans_only"


def test_measure_repair_never_changes_return_date_join_key():
    fact = "TENANT.F_RETURNS"
    date = "TENANT.D_DATE"
    columns = {
        fact: {
            "RETURN_DATE_SK": "int", "RETURN_QUANTITY": "int", "REFUND_AMOUNT": "decimal",
        },
        date: {"DATE_SK": "int", "FULL_DATE": "date"},
    }
    sql = (
        "SELECT d.FULL_DATE, SUM(r.REFUND_AMOUNT) AS REFUNDS "
        "FROM TENANT.F_RETURNS r JOIN TENANT.D_DATE d "
        "ON r.RETURN_DATE_SK = d.DATE_SK GROUP BY d.FULL_DATE"
    )
    context = {"semantic_plan": {
        "enabled": True,
        "fields": [{
            "term": "returns", "table": fact, "column": "RETURN_QUANTITY",
            "role": "measure", "enforcement": "required",
        }],
        "joins": [],
    }}
    repaired = attempt_field_plan_repair(
        sql, "azure_sql", {fact, date}, None, columns, context,
    )
    assert repaired
    repaired_u = repaired.upper()
    assert "R.RETURN_DATE_SK = D.DATE_SK" in repaired_u
    assert "SUM(R.RETURN_QUANTITY)" in repaired_u


def test_identifier_only_antijoin_is_not_rewritten_as_a_measure_join():
    orders = "TENANT.F_ORDERS"
    shipments = "TENANT.F_SHIPMENTS"
    columns = {
        orders: {"ORDER_SK": "int", "ORDER_NUMBER": "varchar"},
        shipments: {"ORDER_SK": "int", "SHIPPED_AMOUNT": "decimal"},
    }
    sql = (
        "SELECT o.ORDER_NUMBER FROM TENANT.F_ORDERS o "
        "LEFT JOIN TENANT.F_SHIPMENTS s ON o.ORDER_SK = s.ORDER_SK "
        "WHERE s.ORDER_SK IS NULL"
    )
    context = {"semantic_plan": {
        "enabled": True,
        "fields": [{
            "term": "shipped", "table": shipments, "column": "SHIPPED_AMOUNT",
            "role": "measure", "enforcement": "required",
        }],
        "joins": [],
    }}
    repaired = attempt_field_plan_repair(
        sql, "azure_sql", {orders, shipments}, None, columns, context,
    )
    assert repaired == ""
    assert "o.ORDER_SK = s.ORDER_SK" in sql


def test_temporal_q1_q3_do_not_trigger_boxplot():
    assert not detect_boxplot_intent("show revenue for calendar Q1")
    assert not detect_boxplot_intent("show the revenue trend from Q1 to Q3")
    assert not detect_boxplot_intent("compare fiscal Q1 and fiscal Q3")


def test_statistical_q1_q3_still_trigger_boxplot():
    assert detect_boxplot_intent("show Q1, median and Q3 quartiles for invoice value")
    assert detect_boxplot_intent("box plot with Q1 and Q3 for order amount")


def test_quarter_range_preserves_every_intermediate_period():
    plan = plan_analytical_intent(
        "show the revenue trend from calendar Q1 to Q3",
        metrics=[{"name": "Revenue"}],
    )
    assert plan.intent == "trend"
    assert plan.calendar_basis == "calendar"
    assert plan.quarter_periods == ("Q1", "Q2", "Q3")
    assert not plan.needs_clarification


def test_elliptical_highest_and_lowest_are_local_result_operations():
    highest = parse_result_command("Which one is highest?")
    lowest = parse_result_command("Which one is lowest?")
    assert highest and highest.action == "keep_top" and highest.limit == 1
    assert highest.direction == "desc"
    assert lowest and lowest.action == "keep_top" and lowest.direction == "asc"


def test_clarification_submit_is_not_blocked_by_stale_processing_state():
    source = (Path(__file__).parents[1] / "portal" / "templates" / "portal_chat.html").read_text(
        encoding="utf-8"
    )
    start = source.index("function _submitClarification")
    function_source = source[start:start + 2200]
    assert "processingActive && agentRunState" in function_source
    assert "setAgentRunState('waiting_for_user');" in function_source
    assert "QueryBot is still finishing the current step; applying" in function_source
    assert "pending_id: data.pending_id" in function_source
