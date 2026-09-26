# -*- coding: utf-8 -*-
"""An investigation calls named tools on the results it already has.

core/investigation_tools.py: the planner names a tool and gives its inputs.
A tool that works on a result takes a step of this same run, reads that
step's cached rows and computes on them -- no SQL, no second warehouse
query -- and keeps what it computed as a result of its own, which a later
step can read in turn. `query` and `drill` ask through the governed
pipeline, as a reader's own question does.

The warehouse is replaced at its boundary, core.dispatcher.dispatch, by one
that sends each answer through the real result renderer
(core.result_renderer._send_results), so a result is cached -- and marked cut
at the row limit or not -- exactly as a real answer is. Everything after
that is real: the tools, the analyses they call, the forecast's policy,
series and fit gates, the planner's prompt and parsing, the synthesis check
and the socket's trail. The data is synthetic.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import tempfile
from contextlib import ExitStack
from unittest.mock import patch

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_investigations_call_tools.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

import core.dispatcher as dispatcher  # noqa: E402
import core.investigation_planner as investigation_planner  # noqa: E402
import gateway.webhooks as wh  # noqa: E402
from core.analysis_narrative import numbers_in  # noqa: E402
from core.investigation import ToolResult, investigation_session_id, run_query_tool  # noqa: E402
from core.investigation_planner import (  # noqa: E402
    InvestigationOutcome,
    InvestigationStep,
    build_planner_prompt,
    parse_planner_decision,
    run_investigation,
)
from core.investigation_tools import REGISTRY, describe_tools, run_tool  # noqa: E402
from core.result_cache import result_cache  # noqa: E402

MONTHS = [f"2024-{month:02d}" for month in range(1, 13)]
TREND = [1000.0 + 50.0 * i + (7.0 if i % 2 else -6.0) for i in range(12)]

ANSWERS: dict[str, tuple[list[dict], str]] = {
    "revenue by month": (
        [{"MONTH": m, "REVENUE": v} for m, v in zip(MONTHS, TREND)],
        "SELECT MONTH, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY MONTH ORDER BY MONTH",
    ),
    "revenue by region": (
        [{"REGION": "North", "REVENUE": 500.0}, {"REGION": "South", "REVENUE": 300.0},
         {"REGION": "East", "REVENUE": 150.0}, {"REGION": "West", "REVENUE": 50.0}],
        "SELECT REGION, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY REGION",
    ),
    "revenue by region last year": (
        [{"REGION": "North", "REVENUE": 400.0}, {"REGION": "South", "REVENUE": 350.0},
         {"REGION": "East", "REVENUE": 150.0}, {"REGION": "West", "REVENUE": 100.0}],
        "SELECT REGION, SUM(REVENUE) AS REVENUE FROM SALES_LY GROUP BY REGION",
    ),
    "revenue by region by product line": (
        [{"REGION": "North", "PRODUCT_LINE": "Pumps", "REVENUE": 320.0},
         {"REGION": "North", "PRODUCT_LINE": "Valves", "REVENUE": 180.0}],
        "SELECT REGION, PRODUCT_LINE, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY REGION, PRODUCT_LINE",
    ),
    "revenue by region and month": (
        [{"REGION": r, "MONTH": m, "REVENUE": v}
         for r, base in (("North", 100.0), ("South", 80.0)) for m, v in zip(MONTHS[:3], (base, base + 5, base + 9))],
        "SELECT REGION, MONTH, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY REGION, MONTH",
    ),
    "large sales by region": (
        [{"REGION": "North", "REVENUE": 1234.5}, {"REGION": "South", "REVENUE": 765.5}],
        "SELECT REGION, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY REGION",
    ),
    "stock balance by warehouse and month": (
        [{"WAREHOUSE": w, "MONTH": m, "STOCK_BALANCE": v}
         for w, base in (("Lyon", 900.0), ("Nantes", 400.0)) for m, v in zip(MONTHS[:3], (base, base - 50, base + 20))],
        "SELECT WAREHOUSE, MONTH, SUM(STOCK_BALANCE) AS STOCK_BALANCE FROM STOCK GROUP BY WAREHOUSE, MONTH",
    ),
    "scrap by plant": (
        [{"PLANT": f"Plant {name}", "SCRAP_QTY": qty}
         for name, qty in zip("ABCDEFGHIJ", (100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 100.0, 1000.0, 100.0))],
        "SELECT PLANT, SUM(SCRAP_QTY) AS SCRAP_QTY FROM SCRAP GROUP BY PLANT",
    ),
    "scrap by plant and month": (
        [{"PLANT": f"Plant {name}", "MONTH": month, "SCRAP_QTY": qty}
         for (name, month), qty in zip([(n, m) for n in "ABCDE" for m in MONTHS[:2]],
                                       (100.0, 102.0, 98.0, 101.0, 99.0, 1000.0, 97.0, 100.0, 103.0, 100.0))],
        "SELECT PLANT, MONTH, SUM(SCRAP_QTY) AS SCRAP_QTY FROM SCRAP GROUP BY PLANT, MONTH",
    ),
    "scrap at three plants": (
        [{"PLANT": "Plant A", "SCRAP_QTY": 100.0}, {"PLANT": "Plant B", "SCRAP_QTY": 900.0},
         {"PLANT": "Plant C", "SCRAP_QTY": 110.0}],
        "SELECT PLANT, SUM(SCRAP_QTY) AS SCRAP_QTY FROM SCRAP GROUP BY PLANT",
    ),
    "units sold by price": (
        [{"ITEM": f"I{i}", "PRICE": p, "UNITS": u}
         for i, (p, u) in enumerate(zip((10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0),
                                        (100.0, 97.0, 86.0, 84.0, 74.0, 70.0, 66.0, 55.0)))],
        "SELECT ITEM, PRICE, SUM(UNITS) AS UNITS FROM SALES GROUP BY ITEM, PRICE",
    ),
    "order values": (
        [{"ORDER_ID": f"O{i}", "ORDER_VALUE": float(v)}
         for i, v in enumerate([12, 15, 18, 22, 25, 27, 30, 31, 33, 35, 36, 38, 40, 41, 43,
                                45, 47, 50, 52, 55, 58, 60, 64, 68, 72, 77, 81, 90, 96, 120])],
        "SELECT ORDER_ID, SUM(ORDER_VALUE) AS ORDER_VALUE FROM ORDERS GROUP BY ORDER_ID",
    ),
    "customers by cohort": (
        [{"COHORT_MONTH": "2024-01", "PERIOD_NUMBER": p, "CUSTOMERS": c}
         for p, c in enumerate((100.0, 80.0, 60.0, 50.0))]
        + [{"COHORT_MONTH": "2024-02", "PERIOD_NUMBER": p, "CUSTOMERS": c}
           for p, c in enumerate((200.0, 150.0, 100.0))],
        "SELECT COHORT_MONTH, PERIOD_NUMBER, COUNT(*) AS CUSTOMERS FROM CUSTOMERS GROUP BY COHORT_MONTH, PERIOD_NUMBER",
    ),
    "visitors by stage": (
        [{"STAGE": "Visited", "VISITORS": 1000.0}, {"STAGE": "Added to cart", "VISITORS": 400.0},
         {"STAGE": "Checked out", "VISITORS": 100.0}],
        "SELECT STAGE, COUNT(*) AS VISITORS FROM FUNNEL GROUP BY STAGE",
    ),
    "flat revenue by month": (
        [{"MONTH": m, "REVENUE": 500.0} for m in MONTHS],
        "SELECT MONTH, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY MONTH ORDER BY MONTH",
    ),
    "erratic revenue by month": (
        [{"MONTH": m, "REVENUE": v}
         for m, v in zip(MONTHS, (100.0, 900.0, 150.0, 800.0, 120.0, 950.0, 90.0, 870.0, 130.0, 910.0, 100.0, 880.0))],
        "SELECT MONTH, SUM(REVENUE) AS REVENUE FROM SALES GROUP BY MONTH ORDER BY MONTH",
    ),
    "raw revenue by month": (
        [{"MONTH": m, "REVENUE": v} for m, v in zip(MONTHS, TREND)],
        "SELECT MONTH, REVENUE FROM SALES ORDER BY MONTH",
    ),
}


class _Warehouse:
    """dispatch, replaced at its boundary: each known question's rows go out
    through the real renderer, which caches them the way a real answer is
    cached; anything else is answered the way an empty result is."""

    def __init__(self, cut: set[str] | None = None):
        self.asked: list[str] = []
        self.cut = set(cut or ())

    async def dispatch(self, account_id, event, adapter, _background, portal_user=None):
        from core.result_renderer import _send_results

        question = event.text
        self.asked.append(question)
        if question not in ANSWERS:
            await adapter.send_message(event, "No records matched the filters.")
            return
        rows, sql = ANSWERS[question]
        await _send_results(
            event, adapter, question, [dict(row) for row in rows], sql, 12, portal_user, account_id,
            {"id": 0, "db_type": "azure_sql"},
            confidence_context={"rows_truncated": question in self.cut},
        )


def _account() -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _user(**extra) -> dict:
    return {"id": 7, "name": "Ada", "role": "viewer", "group_name": "", **extra}


class _Run:
    """One investigation's run: its account, id, warehouse and steps so far."""

    def __init__(self, cut: set[str] | None = None, user: dict | None = None):
        self.account_id = _account()
        self.run_id = f"run-{os.urandom(3).hex()}"
        self.user = user or _user()
        self.warehouse = _Warehouse(cut)
        self.steps: list[InvestigationStep] = []

    def ask(self, question: str) -> ToolResult:
        with patch.object(dispatcher, "dispatch", self.warehouse.dispatch):
            result = asyncio.run(run_query_tool(
                account_id=self.account_id, portal_user=self.user, run_id=self.run_id, question=question,
            ))
        self.steps.append(InvestigationStep(index=len(self.steps) + 1, question=question, result=result))
        return result

    def tool(self, name: str, inputs: dict | None = None, *, lang: str | None = None) -> ToolResult:
        with patch.object(dispatcher, "dispatch", self.warehouse.dispatch):
            result = asyncio.run(run_tool(
                name, inputs, account_id=self.account_id, portal_user=self.user, run_id=self.run_id,
                steps=self.steps, lang=lang,
            ))
        self.steps.append(InvestigationStep(index=len(self.steps) + 1, question=result.question, result=result))
        return result

    def kept(self, result: ToolResult) -> dict:
        return result_cache.get_snapshot(
            investigation_session_id(self.account_id, self.user, self.run_id), result.result_id,
        )


# ── The catalogue the planner reads, and how its choice is read ──────────────


class TestThePlannerIsOfferedTheTools:

    def test_every_tool_is_described_with_its_inputs(self):
        catalogue = describe_tools("en")
        for name in ("query", "drill", "compare", "forecast", "anomalies", "correlate",
                     "distribution", "cohort", "funnel", "contribution"):
            assert f"- {name}: " in catalogue
        assert "periods? (how many periods ahead, 1 to 12; 3 when left out)" in catalogue
        assert "- sandbox" not in catalogue

    def test_a_french_reader_is_given_the_french_catalogue(self):
        assert "Prolonge la série d'une étape" in describe_tools("fr")

    def test_the_prompt_lists_the_tools_and_each_steps_columns(self):
        run = _Run()
        run.ask("revenue by month")
        system, user = build_planner_prompt("why is revenue up", run.steps, steps_left=3)
        assert "Tools:\n- query: " in system
        assert "Step 1 asked: revenue by month" in user
        assert "Its result's columns: MONTH, REVENUE" in user

    def test_the_budget_counts_steps_not_questions(self):
        system, _user = build_planner_prompt("why is revenue up", [], steps_left=2)
        assert "2 more step(s) may be taken" in system


class TestThePlannersChoiceIsRead:

    def test_a_tool_with_its_inputs(self):
        decision = parse_planner_decision(
            '{"action": "tool", "tool": "Forecast", "inputs": {"step": 1, "periods": 3}, "reason": "project it"}'
        )
        assert (decision.action, decision.tool, decision.inputs) == ("tool", "forecast", {"step": 1, "periods": 3})

    def test_a_tool_there_is_not_is_no_decision(self):
        assert parse_planner_decision('{"action": "tool", "tool": "regress", "inputs": {"step": 1}}').action == ""

    def test_inputs_that_are_not_an_object_are_no_decision(self):
        assert parse_planner_decision('{"action": "tool", "tool": "forecast", "inputs": [1, 3]}').action == ""

    def test_the_query_tool_is_a_question(self):
        decision = parse_planner_decision(
            '{"action": "tool", "tool": "query", "inputs": {"question": "revenue by region"}}'
        )
        assert (decision.action, decision.question) == ("query", "revenue by region")


# ── Each tool computes on a step's own rows ─────────────────────────────────


class TestAToolWorksOnWhatAStepFound:

    def test_a_forecast_projects_the_steps_series_without_asking_again(self):
        run = _Run()
        first = run.ask("revenue by month")
        result = run.tool("forecast", {"step": 1, "periods": 3})
        assert result.ok, result.error
        assert run.warehouse.asked == ["revenue by month"]
        kept = run.kept(result)
        projected = [row for row in kept["rows"] if row.get("is_forecast")]
        assert [row["MONTH"] for row in projected] == ["2025-01", "2025-02", "2025-03"]
        assert kept["parent_result_id"] == first.result_id
        assert kept["operation"] == "investigation:forecast"
        # What the planner reads is what was kept: every projected value, and its range.
        for row in projected:
            for figure in (row["forecast_value"], row["forecast_low"], row["forecast_high"]):
                assert any(abs(figure - found) <= 0.01 for found in numbers_in(result.brief))
        assert result.brief.startswith("REVENUE by MONTH, projected with the exponential smoothing model (R² 1.00): ")
        assert result.question == "Forecast of step 1, 3 period(s) ahead"

    def test_the_values_that_stand_out_are_named(self):
        run = _Run()
        run.ask("scrap by plant")
        result = run.tool("anomalies", {"step": 1})
        assert result.ok, result.error
        assert result.brief == (
            "1 of 10 values of SCRAP_QTY stand out (outside the interquartile range): Plant I: 1,000"
        )
        flagged = [row["PLANT"] for row in run.kept(result)["rows"] if row["anomaly_flag"]]
        assert flagged == ["Plant I"]

    def test_a_value_that_stands_out_is_named_by_every_column_that_names_it(self):
        run = _Run()
        run.ask("scrap by plant and month")
        result = run.tool("anomalies", {"step": 1})
        assert result.brief.endswith(": Plant C · 2024-02: 1,000")

    def test_three_values_are_too_few_to_stand_out(self):
        run = _Run()
        run.ask("scrap at three plants")
        result = run.tool("anomalies", {"step": 1})
        assert not result.ok
        assert result.error == "There are too few values to tell which ones stand out."

    def test_each_items_share_of_the_total_largest_first(self):
        run = _Run()
        run.ask("revenue by region")
        result = run.tool("contribution", {"step": 1})
        assert result.ok, result.error
        assert result.brief == (
            "REVENUE by REGION: North: 500 (50.0%); South: 300 (30.0%); East: 150 (15.0%); "
            "West: 50 (5.0%); the top 3 hold 95.0%"
        )
        assert sum(row["contribution_pct"] for row in run.kept(result)["rows"]) == pytest.approx(100.0)

    def test_a_months_share_of_the_year(self):
        run = _Run()
        run.ask("revenue by month")
        result = run.tool("contribution", {"step": 1})
        assert result.ok, result.error
        assert result.brief.startswith("REVENUE by MONTH: 2024-12: 1,557 (")

    def test_no_share_is_guessed_when_two_columns_could_name_the_items(self):
        run = _Run()
        run.ask("revenue by region by product line")
        result = run.tool("contribution", {"step": 1})
        assert result.error == (
            "Step 1's result has more than one column that could name its items, so its shares would be a guess."
        )

    def test_no_share_is_given_of_a_balance_across_months(self):
        run = _Run()
        run.ask("stock balance by warehouse and month")
        result = run.tool("contribution", {"step": 1})
        assert not result.ok
        assert result.error == (
            "Shares of a total cannot be given for STOCK_BALANCE: it does not add up across these rows, "
            "or its total is zero."
        )

    def test_two_steps_compared_label_by_label_biggest_change_first(self):
        run = _Run()
        run.ask("revenue by region last year")
        run.ask("revenue by region")
        result = run.tool("compare", {"step": 1, "other_step": 2})
        assert result.ok, result.error
        assert result.brief == (
            "REVENUE, step 1 against step 2: North: 400 then 500 (change 100, 25.0%); "
            "South: 350 then 300 (change -50, -14.3%); West: 100 then 50 (change -50, -50.0%); "
            "East: 150 then 150 (change 0, 0.0%)"
        )
        assert run.warehouse.asked == ["revenue by region last year", "revenue by region"]

    def test_a_result_with_a_row_per_label_per_month_is_not_matched_on_the_label(self):
        run = _Run()
        run.ask("revenue by region")
        run.ask("revenue by region and month")
        result = run.tool("compare", {"step": 1, "other_step": 2})
        assert not result.ok
        assert result.error == "Step 2 has more than one row per REGION, so its rows cannot be matched one to one."

    def test_how_two_columns_move_together(self):
        run = _Run()
        run.ask("units sold by price")
        result = run.tool("correlate", {"step": 1, "column": "price", "other_column": "units"})
        assert result.ok, result.error
        rows = ANSWERS["units sold by price"][0]
        r = statistics.correlation([row["PRICE"] for row in rows], [row["UNITS"] for row in rows])
        assert result.brief == f"PRICE and UNITS: r = {r:.2f}, strong and negative, over 8 pairs"

    def test_a_column_is_not_correlated_with_itself(self):
        run = _Run()
        run.ask("units sold by price")
        result = run.tool("correlate", {"step": 1, "column": "PRICE", "other_column": "price"})
        assert result.error == "A correlation needs two different numeric columns."

    def test_how_the_values_are_spread(self):
        run = _Run()
        run.ask("order values")
        result = run.tool("distribution", {"step": 1})
        assert result.ok, result.error
        bins = run.kept(result)["rows"]
        assert sum(row["count"] for row in bins) == 30
        assert result.brief.startswith("ORDER_VALUE over 30 rows: ")
        assert all(row["bin_label"] in result.brief for row in bins)

    def test_a_cohort_matrix(self):
        run = _Run()
        run.ask("customers by cohort")
        result = run.tool("cohort", {"step": 1})
        assert result.ok, result.error
        matrix = {row["cohort"]: row for row in run.kept(result)["rows"]}
        assert (matrix["2024-01"]["Month 1"], matrix["2024-02"]["Month 1"]) == (80.0, 75.0)
        assert result.brief == (
            "2 cohorts over 4 periods; average retention 63.0%; best cohort 2024-01, weakest 2024-02"
        )

    def test_a_funnel(self):
        run = _Run()
        run.ask("visitors by stage")
        result = run.tool("funnel", {"step": 1})
        assert result.ok, result.error
        assert result.brief == (
            "3 stages: 1,000 at the top, 100 at the bottom, 10.0% overall; "
            "the biggest drop is at Added to cart (60.0%)"
        )

    def test_a_tools_result_is_a_step_a_later_tool_reads(self):
        run = _Run()
        run.ask("revenue by region")
        run.tool("contribution", {"step": 1})
        # contribution_pct is a column of step 2's result, not of step 1's.
        result = run.tool("anomalies", {"step": 2, "column": "contribution_pct"})
        assert result.ok, result.error
        assert "values of contribution_pct" in result.brief
        assert run.warehouse.asked == ["revenue by region"]

    def test_a_french_reader_reads_french_figures_the_check_can_read(self):
        run = _Run()
        run.ask("large sales by region")
        result = run.tool("contribution", {"step": 1}, lang="fr")
        assert result.ok, result.error
        assert "1 234,50" in result.brief.replace(" ", " ").replace(" ", " ")
        assert "61,7" in result.brief and "61.7" not in result.brief
        assert result.question == "Parts du total à l'étape 1"
        assert {1234.5, 61.7} <= {round(value, 2) for value in numbers_in(result.brief)}


class TestDrillAsksAgain:

    def test_the_steps_own_question_by_one_more_dimension(self):
        run = _Run()
        run.ask("revenue by region")
        result = run.tool("drill", {"step": 1, "dimension": "product line"})
        assert result.ok, result.error
        assert run.warehouse.asked == ["revenue by region", "revenue by region by product line"]
        assert result.kind == "drill"
        assert result.question == "Step 1 broken down by product line"
        assert result.columns == ("REGION", "PRODUCT_LINE", "REVENUE")

    def test_a_step_cut_at_the_row_limit_can_still_be_asked_again(self):
        run = _Run(cut={"revenue by region"})
        run.ask("revenue by region")
        assert run.tool("drill", {"step": 1, "dimension": "product line"}).ok


# ── Every input is checked before anything runs ─────────────────────────────


class TestInputsAreCheckedFirst:

    def _run(self) -> _Run:
        run = _Run()
        run.ask("revenue by month")
        return run

    def test_a_tool_there_is_not(self):
        result = self._run().tool("regress", {"step": 1})
        assert (result.ok, result.kind, result.error) == (False, "tool", "There is no tool named regress.")

    def test_an_input_the_tool_does_not_take(self):
        result = self._run().tool("forecast", {"step": 1, "horizon": 3})
        assert result.error == "That tool has no input named horizon."

    def test_an_input_the_tool_needs(self):
        assert self._run().tool("forecast", {"periods": 3}).error == "That tool needs step."

    def test_a_step_this_run_did_not_take(self):
        assert self._run().tool("forecast", {"step": 5}).error == "This investigation has no step 5."

    def test_a_step_that_found_nothing(self):
        run = _Run()
        run.ask("revenue by planet")
        result = run.tool("anomalies", {"step": 1})
        assert result.error == "Step 1 found nothing to work on."

    def test_a_column_the_steps_result_does_not_have(self):
        result = self._run().tool("anomalies", {"step": 1, "column": "MARGIN"})
        assert result.error == "Step 1's result has no column MARGIN."

    @pytest.mark.parametrize("periods", [0, 13, "three"])
    def test_a_count_out_of_range(self, periods):
        result = self._run().tool("forecast", {"step": 1, "periods": periods})
        assert result.error == "periods must be between 1 and 12."

    def test_a_step_number_reads_only_this_runs_own_results(self):
        first = self._run()
        other = _Run()
        other.account_id, other.user = first.account_id, first.user
        other.steps = list(first.steps)
        result = other.tool("forecast", {"step": 1})
        assert result.error == "Step 1's result is no longer held; ask it again."

    def test_nothing_is_asked_of_the_warehouse_for_a_call_that_cannot_run(self):
        run = self._run()
        for name, inputs in (("regress", {}), ("forecast", {"step": 5}), ("anomalies", {"step": 1, "column": "X"})):
            run.tool(name, inputs)
        assert run.warehouse.asked == ["revenue by month"]


class TestAResultCutAtTheRowLimit:

    def test_is_refused_by_every_tool_that_reads_its_rows(self):
        run = _Run(cut={"revenue by region"})
        run.ask("revenue by region")
        run.ask("revenue by region last year")
        for name, inputs in (("contribution", {"step": 1}), ("anomalies", {"step": 1}),
                             ("distribution", {"step": 1}), ("forecast", {"step": 1}),
                             ("compare", {"step": 2, "other_step": 1})):
            assert run.tool(name, inputs).error == (
                "Step 1's result stopped at the row limit, so it is not the whole answer; "
                "ask a narrower question first."
            ), name

    def test_is_marked_so_when_a_channel_adapter_caches_it(self):
        from core.result_renderer import _send_results
        from gateway.session_state import GovernedChannelSessionMixin

        class _Channel(GovernedChannelSessionMixin):
            platform_type = "teams"

            async def send_message(self, event, text):
                return None

        channel = _Channel()
        channel._init_governed_session()
        channel.bind_session(_account(), "7")
        rows, sql = ANSWERS["revenue by region"]
        asyncio.run(_send_results(
            None, channel, "revenue by region", [dict(row) for row in rows], sql, 12, None,
            channel._session_account, {"id": 0, "db_type": "azure_sql"},
            confidence_context={"rows_truncated": True},
        ))
        assert result_cache.get_snapshot(channel.session_id)["metadata"]["rows_truncated"] is True

    def test_what_is_derived_from_it_is_cut_too(self):
        run = _Run(cut={"revenue by region"})
        source = run.ask("revenue by region")
        session = investigation_session_id(run.account_id, run.user, run.run_id)
        child = result_cache.derive_snapshot(session, source.result_id, [{"REGION": "North", "REVENUE": 500.0}],
                                             question="top 1", operation="top")
        assert child["metadata"]["rows_truncated"] is True


# ── The forecast keeps the pipeline's own gates ─────────────────────────────


class TestTheForecastKeepsThePipelinesGates:

    def test_the_policy_can_refuse_a_derived_visual_of_the_result(self):
        from core.compliance import policy_engine
        from core.i18n import t

        rules = [{
            "name": "aggregate_only: revenue", "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "SALES.REVENUE",
            "action": "chart", "effect": "allow", "aggregate_only": True,
        }, {
            "name": "allow: the rest of sales", "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "SALES.*", "action": "chart", "effect": "allow",
        }]
        profile = {"mode": "regulated", "policy_pack_key": "banking_v1", "active_policy_version": 1,
                   "enforcement_mode": "enforce"}
        run = _Run(user=_user(role="analyst"))
        run.ask("raw revenue by month")
        with ExitStack() as stack:
            for name, value in (("get_compliance_profile", profile), ("get_classification_map", {}),
                                ("list_policy_rules", rules), ("list_purposes", []),
                                ("log_policy_decision", "audit-id")):
                stack.enter_context(patch.object(policy_engine.store, name, return_value=value))
            result = run.tool("forecast", {"step": 1})
        assert (result.ok, result.error) == (False, t("caveat.forecast.policy_blocked"))
        assert not result.result_id

    def test_a_series_that_does_not_move_is_not_projected(self):
        from core.i18n import t

        run = _Run()
        run.ask("flat revenue by month")
        assert run.tool("forecast", {"step": 1}).error == t("caveat.forecast.constant_series")

    def test_a_fit_that_does_not_describe_the_series_is_thrown_away(self):
        run = _Run()
        run.ask("erratic revenue by month")
        result = run.tool("forecast", {"step": 1})
        assert not result.ok
        assert not result.result_id
        assert result.error.startswith("I did not project future periods: the trend line explains only ")

    def test_the_horizon_is_the_gates_not_the_planners(self):
        run = _Run()
        run.ask("revenue by month")
        result = run.tool("forecast", {"step": 1, "periods": 12})
        assert result.ok, result.error
        assert len([row for row in run.kept(result)["rows"] if row.get("is_forecast")]) == 6


# ── The loop runs tools, and checks what the summary says against them ─────


@pytest.fixture
def allowed(monkeypatch):
    import core.compliance.policy_engine as pe

    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda account_id: True)


def _planner(replies):
    """A scripted planner: each call returns the next reply; a callable reply
    is given the prompt, the way a model reads it."""
    prompts: list[tuple[str, str]] = []

    async def _complete(system="", user="", *a, **k):
        prompts.append((system, user))
        reply = replies.pop(0)
        return (reply(user) if callable(reply) else reply), 10, 10

    return _complete, prompts


def _found(user_prompt: str, index: int) -> str:
    block = user_prompt.split(f"Step {index} asked: ", 1)[1]
    return block.split("\nFound: ", 1)[1].split("\n", 1)[0]


class TestTheInvestigationCallsTools:

    def _investigate(self, run: _Run, replies, lang=None):
        complete, prompts = _planner(replies)
        with patch.object(dispatcher, "dispatch", run.warehouse.dispatch):
            outcome = asyncio.run(run_investigation(
                objective="revenue by month", account_id=run.account_id, portal_user=run.user,
                run_id=run.run_id, max_steps=4, complete=complete, lang=lang,
            ))
        return outcome, prompts

    def test_a_summary_citing_the_forecast_is_checked_against_it_and_kept(self, allowed):
        run = _Run()

        def _finish(user_prompt):
            first_point = _found(user_prompt, 2).split(": ", 1)[1].split(" (between", 1)[0]
            return ('{"action": "finish", "synthesis": "Revenue is projected at %s next month (step 2)."}'
                    % first_point)

        outcome, prompts = self._investigate(run, [
            '{"action": "tool", "tool": "forecast", "inputs": {"step": 1, "periods": 3}, "reason": "project"}',
            _finish,
        ])
        assert [step.result.kind for step in outcome.steps] == ["query", "forecast"]
        assert run.warehouse.asked == ["revenue by month"]
        assert outcome.phrasing == "llm", outcome.synthesis
        projected = [row for row in run.kept(outcome.steps[1].result)["rows"] if row.get("is_forecast")]
        assert f"{projected[0]['forecast_value']:,.2f}" in outcome.synthesis
        assert "Its result's columns: MONTH, REVENUE" in prompts[0][1]

    def test_a_figure_no_step_computed_is_refused(self, allowed):
        run = _Run()
        outcome, _ = self._investigate(run, [
            '{"action": "tool", "tool": "forecast", "inputs": {"step": 1}}',
            '{"action": "finish", "synthesis": "Revenue is projected at 98,765.43 next month."}',
        ])
        assert outcome.phrasing == "template"
        assert "Forecast of step 1, 3 period(s) ahead" in outcome.synthesis

    def test_a_call_that_cannot_run_is_a_failed_step_and_the_loop_goes_on(self, allowed):
        run = _Run()
        outcome, prompts = self._investigate(run, [
            '{"action": "tool", "tool": "forecast", "inputs": {"step": 9}}',
            '{"action": "query", "question": "revenue by region"}',
            '{"action": "finish", "synthesis": "North leads with 500."}',
        ])
        assert [step.result.ok for step in outcome.steps] == [True, False, True]
        assert outcome.steps[1].result.error == "This investigation has no step 9."
        assert "Could not be answered: This investigation has no step 9." in prompts[1][1]
        assert outcome.phrasing == "llm"

    def test_a_french_reader_gets_a_french_trail_when_the_summary_is_not_used(self, allowed):
        run = _Run(user=_user(lang="fr"))
        outcome, _ = self._investigate(run, [
            '{"action": "tool", "tool": "anomalies", "inputs": {"step": 1}}',
            "not json",
        ], lang="fr")
        assert outcome.phrasing == "template"
        assert "Valeurs qui se démarquent à l'étape 1" in outcome.synthesis
        assert "valeurs de REVENUE sur 12 se démarquent" in outcome.synthesis


# ── The socket's trail names a tool step by its tool ───────────────────────


class TestTheTrail:

    def test_each_step_is_recorded_under_what_it_was(self, monkeypatch):
        from starlette.testclient import TestClient
        from fastapi import FastAPI

        import portal.routes as pr

        account_id = _account()
        store.update_client_meta(account_id, chat_ui_enabled=1, enable_llm_audit=1)
        user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")

        async def _fake_loop(**kwargs):
            return InvestigationOutcome(objective="x", phrasing="template", synthesis="Done.", steps=[
                InvestigationStep(1, "revenue by month", ToolResult(ok=True, kind="query", question="revenue by month")),
                InvestigationStep(2, "Forecast of step 1, 3 period(s) ahead",
                                  ToolResult(ok=True, kind="forecast", question="Forecast of step 1, 3 period(s) ahead")),
                InvestigationStep(3, "anomalies", ToolResult(ok=False, kind="anomalies", question="anomalies",
                                                             error="That tool needs step.")),
                InvestigationStep(4, "revenue by region", ToolResult(ok=True, kind="query", question="revenue by region")),
            ])

        monkeypatch.setattr(investigation_planner, "run_investigation", _fake_loop)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        app = FastAPI()
        app.include_router(wh.router)
        client = TestClient(app)
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        run_id = ""
        with client.websocket_connect(f"/ws/chat/{account_id}") as ws:
            ws.receive_json()
            ws.send_json({"type": "message", "text": "investigate revenue by month"})
            for _ in range(20):
                frame = ws.receive_json()
                run_id = run_id or str(frame.get("run_id") or "")
                if frame.get("type") == "message" and frame.get("content"):
                    break
        steps = store.list_agent_steps(account_id=account_id, portal_user_id=user_id, run_id=run_id)
        assert [(s["tool_name"], s["status"]) for s in steps[1:]] == [
            ("forecast", "completed"), ("anomalies", "failed"), ("query_data", "completed"),
        ]
        assert [s["detail"] for s in steps[1:]] == [
            "Analysing: Forecast of step 1, 3 period(s) ahead", "Analysing: anomalies",
            "Asking: revenue by region",
        ]


def test_the_registry_offers_exactly_these_tools():
    assert sorted(REGISTRY) == sorted(["query", "drill", "compare", "bridge", "price_volume_mix", "forecast",
                                       "anomalies", "correlate", "distribution", "cohort", "funnel",
                                       "contribution"])
