# -*- coding: utf-8 -*-
"""Running the corroborating query — core/corroboration_run.py.

``core.domains`` could already decide that two subject areas might both answer
a question, and compare two results once it had them. What it could not do was
produce the second result: the knowledge base and the prompt are assembled for
the primary area, and a query generated from that context but pointed at
another area's tables differs in ways the comparison cannot attribute.

Four properties are asserted hardest, in this order of seriousness:

  * **the corroborating attempt does not bypass the raw fact-to-fact join
    guard.** That guard reads ``semantic_plan.known_fact_tables`` off the
    semantic context and returns *nothing at all* when fewer than two facts
    are known — so handing a corroborating query an empty context would not
    weaken the guard, it would remove it, for the one query nobody reads
    before it is compared against a user's answer;
  * a second opinion runs under the second area's tables and no others, an
    unrestricted admin included;
  * every failure reports "not checked" rather than degrading the primary
    answer, and nothing is spent once the run is known to be pointless; and
  * the second area is told to refuse rather than guess, because a
    "disagreement" between two different questions is the loudest possible
    way to be wrong.
"""

from __future__ import annotations

import asyncio
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.corroboration_run import (  # noqa: E402
    CARRIED_CONTEXT_KEYS,
    MAX_CONTEXT_DOCUMENTS,
    SecondOpinion,
    build_context,
    corroborating_scope,
    corroborating_semantic_context,
    corroborating_user_message,
    run_second_opinion,
)
from core.sql_attempt import Attempt, ValidationScope  # noqa: E402
from core.validator import validate_sql_detailed  # noqa: E402

# ── A schema two subject areas both live in ──────────────────────────────────
SALES_FACT = "S.F_SALES_INVOICE"
STOCK_FACT = "S.ERP_ITM_BAL_PRD_FCT"
WAREHOUSE = "S.D_WAREHOUSE"

TABLE_COLUMNS = {
    SALES_FACT: {"WAREHOUSE_SK": "int", "NET_REVENUE_AMOUNT": "decimal"},
    STOCK_FACT: {"WHS_DMS_KEY": "int", "INV_VAL_AMT": "decimal"},
    WAREHOUSE: {"WAREHOUSE_SK": "int", "WAREHOUSE_NAME": "varchar"},
}
KNOWN_TABLES = set(TABLE_COLUMNS)

# What the primary answered under: the Sales area, with the workspace's full
# compiled plan behind it.
PRIMARY_SCOPE = ValidationScope(
    known_tables=KNOWN_TABLES,
    db_type="azure_sql",
    allowed_tables={SALES_FACT, WAREHOUSE},
    table_columns=TABLE_COLUMNS,
    semantic_context={
        "question": "revenue by warehouse",
        "intent": {"kind": "aggregate"},
        "top_n": {"n": 5},
        "production_sql": True,
        "graph_context": {"enabled": True, "resolved_edges": [{"from": SALES_FACT}]},
        "resolution_plan": {"expected_tables": [SALES_FACT]},
        "analytical_request_plan": {"selected_facts": [SALES_FACT]},
        "metric_formulas": [{"name": "revenue"}],
        "semantic_plan": {
            "enabled": True,
            "fields": [{"table": SALES_FACT, "column": "NET_REVENUE_AMOUNT"}],
            "known_fact_tables": [SALES_FACT, STOCK_FACT],
        },
    },
)

# The production fan-out: two physical facts joined in one SELECT scope, every
# invoice row duplicated once per stock row for the same warehouse.
FANOUT_SQL = (
    "SELECT w.WAREHOUSE_NAME, SUM(i.NET_REVENUE_AMOUNT) AS TOTAL_REVENUE "
    f"FROM {STOCK_FACT} f "
    f"JOIN {WAREHOUSE} w ON f.WHS_DMS_KEY = w.WAREHOUSE_SK "
    f"JOIN {SALES_FACT} i ON f.WHS_DMS_KEY = i.WAREHOUSE_SK "
    "GROUP BY w.WAREHOUSE_NAME"
)

STOCK_SQL = (
    f"SELECT SUM(INV_VAL_AMT) AS TOTAL FROM {STOCK_FACT}"
)


def _validate(sql, scope):
    return validate_sql_detailed(
        sql, scope.known_tables, scope.db_type, scope.allowed_tables,
        scope.table_columns, scope.semantic_context,
    )


def _attempt(sql=STOCK_SQL, *, ok=True, rows=None, code="ok", reason="OK"):
    return Attempt(sql=sql, ok=ok, rows=rows, code=code, reason=reason,
                   source="corroboration")


# ══════════════════════════════════════════════════════════════════════════════
# The guard that must not go inert
# ══════════════════════════════════════════════════════════════════════════════

class TestTheFactToFactGuardSurvives(unittest.TestCase):
    """The corroborating attempt is validated, not waved through.

    ``_raw_multi_fact_errors`` reads ``semantic_plan.known_fact_tables`` and
    returns ``[]`` when it finds fewer than two — the guard is not weaker
    without the context, it is absent. These two tests are a pair: the second
    is what the first would look like if the fact list were dropped.
    """

    def test_a_fan_out_join_is_refused_under_the_corroborating_scope(self):
        scope = corroborating_scope(PRIMARY_SCOPE, {STOCK_FACT, WAREHOUSE, SALES_FACT})
        result = _validate(FANOUT_SQL, scope)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "raw_fact_to_fact_join")

    def test_the_same_sql_passes_when_the_fact_list_is_dropped(self):
        # Not an aspiration: this is the state the corroborating attempt would
        # have been in had it been validated against an empty context, and it
        # is why the fact list is carried across.
        blind = ValidationScope(
            known_tables=KNOWN_TABLES, db_type="azure_sql",
            allowed_tables={STOCK_FACT, WAREHOUSE, SALES_FACT},
            table_columns=TABLE_COLUMNS,
            semantic_context={"production_sql": True},
        )
        result = _validate(FANOUT_SQL, blind)
        self.assertTrue(result.ok, result.reason)

    def test_the_workspace_fact_list_is_carried_verbatim(self):
        context = corroborating_semantic_context(PRIMARY_SCOPE.semantic_context)
        self.assertEqual(
            context["semantic_plan"]["known_fact_tables"],
            [SALES_FACT, STOCK_FACT],
        )

    def test_a_single_fact_workspace_says_so_rather_than_failing_quietly(self):
        with self.assertLogs("querybot.corroboration", level="INFO") as logged:
            context = corroborating_semantic_context(
                {"semantic_plan": {"known_fact_tables": [SALES_FACT]}}
            )
        self.assertEqual(context["semantic_plan"]["known_fact_tables"], [SALES_FACT])
        self.assertTrue(any("fact-to-fact" in line for line in logged.output))

    def test_a_missing_semantic_plan_still_produces_the_key(self):
        # The validator reads plan["known_fact_tables"]; a context with no
        # semantic_plan at all would raise nothing and check nothing.
        context = corroborating_semantic_context(None)
        self.assertEqual(context["semantic_plan"], {"known_fact_tables": []})
        self.assertTrue(context["production_sql"])


# ══════════════════════════════════════════════════════════════════════════════
# What the corroborating attempt is and is not told
# ══════════════════════════════════════════════════════════════════════════════

class TestTheCorroboratingContext(unittest.TestCase):

    def test_the_primarys_resolved_plan_is_not_carried_over(self):
        # Each of these was resolved against the PRIMARY area's tables. Checking
        # a second-area query against them fails every corroboration on a plan
        # mismatch that says nothing about the data.
        context = corroborating_semantic_context(PRIMARY_SCOPE.semantic_context)
        for key in ("graph_context", "resolution_plan", "analytical_request_plan",
                    "metric_formulas"):
            self.assertNotIn(key, context)

    def test_the_primarys_compiled_fields_are_not_carried_over(self):
        context = corroborating_semantic_context(PRIMARY_SCOPE.semantic_context)
        self.assertNotIn("fields", context["semantic_plan"])

    def test_question_level_facts_are_carried_over(self):
        context = corroborating_semantic_context(PRIMARY_SCOPE.semantic_context)
        for key in CARRIED_CONTEXT_KEYS:
            self.assertIn(key, context)
        self.assertEqual(context["top_n"], {"n": 5})

    def test_the_shape_rules_stay_on(self):
        # production_sql gates the SELECT * and cartesian-product refusals,
        # which are properties of generated SQL rather than of a plan.
        scope = corroborating_scope(PRIMARY_SCOPE, {STOCK_FACT})
        result = _validate(f"SELECT * FROM {STOCK_FACT}", scope)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "production_shape")
        self.assertTrue(any(e.get("code") == "select_star" for e in result.errors))

    def test_the_shape_rules_stay_on_even_when_the_primary_had_them_off(self):
        primary = ValidationScope(
            known_tables=KNOWN_TABLES, db_type="azure_sql", allowed_tables=None,
            table_columns=TABLE_COLUMNS, semantic_context={"production_sql": False},
        )
        scope = corroborating_scope(primary, {STOCK_FACT})
        self.assertTrue(scope.semantic_context["production_sql"])

    def test_the_corroborating_context_is_not_the_primarys(self):
        # An alias would let the corroborating attempt's own bookkeeping reach
        # back into the context the PRIMARY answer is still being validated
        # and repaired against.
        context = corroborating_semantic_context(PRIMARY_SCOPE.semantic_context)
        context["semantic_plan"]["known_fact_tables"].append("SOMETHING.ELSE")
        context["question"] = "a different question"
        self.assertEqual(
            PRIMARY_SCOPE.semantic_context["semantic_plan"]["known_fact_tables"],
            [SALES_FACT, STOCK_FACT],
        )
        self.assertEqual(
            PRIMARY_SCOPE.semantic_context["question"], "revenue by warehouse")


class TestTheCorroboratingScope(unittest.TestCase):

    def test_the_second_area_can_only_read_its_own_tables(self):
        scope = corroborating_scope(PRIMARY_SCOPE, {STOCK_FACT})
        result = _validate(f"SELECT SUM(NET_REVENUE_AMOUNT) FROM {SALES_FACT}", scope)
        self.assertFalse(result.ok)

    def test_the_second_areas_own_query_validates(self):
        scope = corroborating_scope(PRIMARY_SCOPE, {STOCK_FACT})
        result = _validate(STOCK_SQL, scope)
        self.assertTrue(result.ok, result.reason)

    def test_an_unrestricted_admin_is_still_scoped_to_the_second_area(self):
        # allowed_tables=None means "unrestricted" everywhere downstream.
        # Leaving it None here would let the second opinion answer from the
        # FIRST area's fact and agree with itself.
        admin = ValidationScope(
            known_tables=KNOWN_TABLES, db_type="azure_sql", allowed_tables=None,
            table_columns=TABLE_COLUMNS,
            semantic_context=PRIMARY_SCOPE.semantic_context,
        )
        scope = corroborating_scope(admin, {STOCK_FACT})
        self.assertEqual(scope.allowed_tables, {STOCK_FACT})
        result = _validate(f"SELECT SUM(NET_REVENUE_AMOUNT) FROM {SALES_FACT}", scope)
        self.assertFalse(result.ok)

    def test_the_schema_itself_is_unchanged(self):
        # Narrowing known_tables/table_columns would turn a permission failure
        # into an unknown-table error and lose the distinction.
        scope = corroborating_scope(PRIMARY_SCOPE, {STOCK_FACT})
        self.assertEqual(scope.known_tables, KNOWN_TABLES)
        self.assertEqual(scope.table_columns, TABLE_COLUMNS)
        self.assertEqual(scope.db_type, "azure_sql")


# ══════════════════════════════════════════════════════════════════════════════
# The prompt the second area is given
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatTheSecondAreaIsAsked(unittest.TestCase):

    def test_it_is_told_to_refuse_rather_than_answer_a_nearby_question(self):
        message = corroborating_user_message("what was revenue", "Supply Chain")
        self.assertIn("what was revenue", message)
        self.assertIn("Supply Chain", message)
        self.assertIn("CANNOT_GENERATE", message)

    def test_an_unnamed_area_still_produces_a_usable_instruction(self):
        message = corroborating_user_message("what was revenue", "")
        self.assertIn("what was revenue", message)
        self.assertIn("CANNOT_GENERATE", message)

    def test_documents_are_joined_the_way_the_primary_path_joins_them(self):
        self.assertEqual(build_context(["a", "b"]), "a\n\n---\n\nb")

    def test_blank_documents_do_not_become_blank_context(self):
        self.assertEqual(build_context(["", "   ", None]), "")
        self.assertEqual(build_context([]), "")

    def test_the_context_is_bounded(self):
        docs = [f"doc{i}" for i in range(MAX_CONTEXT_DOCUMENTS + 4)]
        self.assertEqual(
            build_context(docs).count("---"), MAX_CONTEXT_DOCUMENTS - 1,
        )


# ══════════════════════════════════════════════════════════════════════════════
# The run itself
# ══════════════════════════════════════════════════════════════════════════════

class _Boundaries:
    """The three things a second opinion actually reaches out to."""

    def __init__(self, *, documents=None, sql=STOCK_SQL, attempt=None,
                 retrieve_error=None, generate_error=None, execute_error=None):
        self.documents = ["## S.ERP_ITM_BAL_PRD_FCT\nINV_VAL_AMT"] if documents is None else documents
        self.sql = sql
        self.attempt = attempt
        self.retrieve_error = retrieve_error
        self.generate_error = generate_error
        self.execute_error = execute_error
        self.retrieved_with = None
        self.generated_with = None
        self.executed_with = None
        self.calls = []

    def retrieve(self, tables):
        self.calls.append("retrieve")
        if self.retrieve_error:
            raise self.retrieve_error
        self.retrieved_with = set(tables)
        return list(self.documents)

    async def generate(self, system, user):
        self.calls.append("generate")
        if self.generate_error:
            raise self.generate_error
        self.generated_with = (system, user)
        return self.sql

    async def execute(self, sql, scope):
        self.calls.append("execute")
        if self.execute_error:
            raise self.execute_error
        self.executed_with = (sql, scope)
        return self.attempt if self.attempt is not None else _attempt(sql, rows=[{"TOTAL": 1000.0}])


def _run(boundaries, *, primary_rows=None, domain="Supply Chain",
         tables=None, question="what was revenue by warehouse"):
    return asyncio.run(run_second_opinion(
        question,
        domain=domain,
        tables={STOCK_FACT, WAREHOUSE} if tables is None else tables,
        primary_rows=[{"TOTAL": 1000.0}] if primary_rows is None else primary_rows,
        primary_domain="Sales",
        scope=PRIMARY_SCOPE,
        retrieve=boundaries.retrieve,
        generate=boundaries.generate,
        execute=boundaries.execute,
    ))


class TestASecondOpinionThatAgrees(unittest.TestCase):

    def test_agreement_is_reported_with_both_figures(self):
        boundaries = _Boundaries(
            attempt=_attempt(rows=[{"TOTAL": 1000.0}]))
        opinion = _run(boundaries)
        self.assertTrue(opinion.checked)
        self.assertTrue(opinion.agrees)
        self.assertEqual(opinion.reason, "agreement")
        self.assertEqual(opinion.domain, "Supply Chain")
        self.assertEqual(opinion.corroboration.primary_value, 1000.0)
        self.assertEqual(opinion.corroboration.secondary_value, 1000.0)
        self.assertEqual(opinion.row_count, 1)

    def test_the_retrieval_ran_under_the_second_areas_tables(self):
        boundaries = _Boundaries()
        _run(boundaries)
        self.assertEqual(boundaries.retrieved_with, {STOCK_FACT, WAREHOUSE})

    def test_the_attempt_ran_under_the_corroborating_scope(self):
        boundaries = _Boundaries()
        _run(boundaries)
        _sql, scope = boundaries.executed_with
        self.assertEqual(scope.allowed_tables, {STOCK_FACT, WAREHOUSE})
        self.assertEqual(
            scope.semantic_context["semantic_plan"]["known_fact_tables"],
            [SALES_FACT, STOCK_FACT],
        )
        self.assertNotIn("graph_context", scope.semantic_context)

    def test_the_second_area_was_asked_the_same_question(self):
        boundaries = _Boundaries()
        _run(boundaries)
        _system, user = boundaries.generated_with
        self.assertIn("what was revenue by warehouse", user)
        self.assertIn("CANNOT_GENERATE", user)

    def test_the_prompt_carries_the_second_areas_knowledge_base(self):
        boundaries = _Boundaries(documents=["## S.ERP_ITM_BAL_PRD_FCT\nINV_VAL_AMT"])
        _run(boundaries)
        system, _user = boundaries.generated_with
        self.assertIn("ERP_ITM_BAL_PRD_FCT", system)


class TestASecondOpinionThatDisagrees(unittest.TestCase):

    def test_a_different_figure_is_reported_as_a_disagreement(self):
        boundaries = _Boundaries(attempt=_attempt(rows=[{"TOTAL": 2700.0}]))
        opinion = _run(boundaries)
        self.assertTrue(opinion.checked)
        self.assertFalse(opinion.agrees)
        self.assertTrue(opinion.disagrees)
        self.assertEqual(opinion.reason, "disagreement")
        self.assertEqual(opinion.corroboration.secondary_value, 2700.0)
        self.assertGreater(opinion.corroboration.relative_difference, 0.5)

    def test_a_rounding_difference_is_not_a_disagreement(self):
        boundaries = _Boundaries(attempt=_attempt(rows=[{"TOTAL": 1000.5}]))
        opinion = _run(boundaries)
        self.assertTrue(opinion.agrees)

    def test_a_disagreement_survives_into_the_dict_the_pipeline_reads(self):
        boundaries = _Boundaries(attempt=_attempt(rows=[{"TOTAL": 2700.0}]))
        payload = _run(boundaries).as_dict()
        self.assertTrue(payload["checked"])
        self.assertFalse(payload["agrees"])
        self.assertEqual(payload["reason"], "disagreement")
        self.assertEqual(payload["domain"], "Supply Chain")
        self.assertEqual(payload["primary_value"], 1000.0)
        self.assertEqual(payload["secondary_value"], 2700.0)


class TestThePrimaryIsUntouched(unittest.TestCase):
    """A second opinion may never change the answer it is checking."""

    def test_a_whole_run_leaves_the_primary_scope_exactly_as_it_was(self):
        before = copy.deepcopy(PRIMARY_SCOPE.semantic_context)
        boundaries = _Boundaries(attempt=_attempt(rows=[{"TOTAL": 2700.0}]))
        primary_rows = [{"TOTAL": 1000.0}]
        opinion = _run(boundaries, primary_rows=primary_rows)
        self.assertTrue(opinion.disagrees)
        self.assertEqual(PRIMARY_SCOPE.semantic_context, before)
        self.assertEqual(PRIMARY_SCOPE.allowed_tables, {SALES_FACT, WAREHOUSE})
        self.assertEqual(primary_rows, [{"TOTAL": 1000.0}])

    def test_not_checked_overrides_a_comparison_that_claims_agreement(self):
        # Constructible state, and the one that matters: whatever a
        # comparison says, a run that did not happen agrees with nothing.
        from core.domains import Corroboration

        opinion = SecondOpinion(
            checked=False, domain="Supply Chain", reason="retrieval_failed",
            corroboration=Corroboration(checked=True, agrees=True),
        )
        self.assertFalse(opinion.agrees)
        self.assertFalse(opinion.disagrees)
        self.assertFalse(opinion.as_dict()["agrees"])


class TestNothingIsSpentOnARunThatCannotHelp(unittest.TestCase):

    def test_an_empty_primary_result_stops_before_retrieval(self):
        boundaries = _Boundaries()
        opinion = _run(boundaries, primary_rows=[])
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "primary_empty")
        self.assertEqual(boundaries.calls, [])

    def test_a_second_area_the_user_cannot_see_stops_before_retrieval(self):
        boundaries = _Boundaries()
        opinion = _run(boundaries, tables=set())
        self.assertEqual(opinion.reason, "no_visible_tables")
        self.assertEqual(boundaries.calls, [])

    def test_no_second_area_stops_before_retrieval(self):
        boundaries = _Boundaries()
        opinion = _run(boundaries, domain="")
        self.assertEqual(opinion.reason, "no_secondary_domain")
        self.assertEqual(boundaries.calls, [])

    def test_an_empty_knowledge_base_stops_before_generation(self):
        # SQL written with no schema in front of it is a guess, and a guess
        # that disagrees with the answer is worse than no second opinion.
        boundaries = _Boundaries(documents=[])
        opinion = _run(boundaries)
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "no_context")
        self.assertEqual(boundaries.calls, ["retrieve"])

    def test_a_refusal_stops_before_execution(self):
        boundaries = _Boundaries(sql="CANNOT_GENERATE")
        opinion = _run(boundaries)
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "not_answerable_there")
        self.assertEqual(boundaries.calls, ["retrieve", "generate"])


class TestAFailedSecondOpinionCostsNothing(unittest.TestCase):
    """"Not checked" is the only honest report of a run that did not happen.

    A user reads "two areas disagree" as evidence about their data. Saying it
    when the second query never ran is worse than saying nothing.
    """

    def _assert_not_checked(self, opinion, reason):
        self.assertFalse(opinion.checked)
        self.assertFalse(opinion.agrees)
        self.assertFalse(opinion.disagrees)
        self.assertEqual(opinion.reason, reason)
        self.assertFalse(opinion.as_dict()["agrees"])

    def test_retrieval_failing_is_not_a_disagreement(self):
        opinion = _run(_Boundaries(retrieve_error=RuntimeError("qdrant down")))
        self._assert_not_checked(opinion, "retrieval_failed")

    def test_generation_failing_is_not_a_disagreement(self):
        opinion = _run(_Boundaries(generate_error=RuntimeError("429")))
        self._assert_not_checked(opinion, "generation_failed")

    def test_the_attempt_raising_is_not_a_disagreement(self):
        opinion = _run(_Boundaries(execute_error=RuntimeError("pool exhausted")))
        self._assert_not_checked(opinion, "attempt_failed")

    def test_invalid_secondary_sql_is_not_a_disagreement(self):
        opinion = _run(_Boundaries(attempt=_attempt(
            ok=False, code="unknown_column", reason="no such column FOO")))
        self._assert_not_checked(opinion, "invalid_there")
        self.assertEqual(opinion.detail["code"], "unknown_column")

    def test_a_valid_query_that_returned_nothing_is_not_a_disagreement(self):
        opinion = _run(_Boundaries(attempt=_attempt(rows=None)))
        self._assert_not_checked(opinion, "execution_failed")

    def test_an_empty_secondary_result_is_not_agreement(self):
        opinion = _run(_Boundaries(attempt=_attempt(rows=[])))
        self._assert_not_checked(opinion, "secondary_empty")

    def test_a_secondary_result_with_no_figure_is_not_agreement(self):
        opinion = _run(_Boundaries(attempt=_attempt(rows=[{"NAME": "West"}])))
        self._assert_not_checked(opinion, "no_comparable_figure")

    def test_a_default_second_opinion_reports_nothing(self):
        self._assert_not_checked(SecondOpinion(), "")


class TestTheGeneratedSqlIsCleanedTheSameWay(unittest.TestCase):

    def test_a_fenced_response_is_unwrapped_before_validation(self):
        boundaries = _Boundaries(sql=f"```sql\n{STOCK_SQL}\n```")
        _run(boundaries)
        sql, _scope = boundaries.executed_with
        self.assertNotIn("```", sql)
        self.assertIn("ERP_ITM_BAL_PRD_FCT", sql)

    def test_a_fenced_refusal_is_still_recognised_as_a_refusal(self):
        boundaries = _Boundaries(sql="```sql\nCANNOT_GENERATE\n```")
        opinion = _run(boundaries)
        self.assertEqual(opinion.reason, "not_answerable_there")
        self.assertEqual(boundaries.calls, ["retrieve", "generate"])


if __name__ == "__main__":
    unittest.main()
