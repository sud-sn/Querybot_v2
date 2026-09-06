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
    describe_second_opinion,
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


class TestTheWholeChainThroughTheRealValidator(unittest.TestCase):
    """Generate → clean → validate → repair → execute, with nothing faked but
    the model and the warehouse.

    The tests above prove the corroborating SCOPE refuses a fan-out join. This
    proves the run actually validates under it — that the scope reaches
    ``run_attempt`` and its verdict reaches the comparison — which is the part
    a canned Attempt cannot show.
    """

    def _run(self, generated_sql, *, tables=None):
        from core.sql_attempt import run_attempt

        boundaries = _Boundaries(sql=generated_sql)
        reached_warehouse = []

        def _executor(sql, semantic_context):
            reached_warehouse.append(sql)
            return SimpleNamespace(rows=[{"TOTAL": 1000.0}], sql=sql, truncated=False)

        async def _execute(sql, scope):
            return await run_attempt(
                sql, scope, executor=_executor, timeout=10,
                timeout_message="timed out", source="corroboration",
            )

        opinion = asyncio.run(run_second_opinion(
            "what was revenue by warehouse",
            domain="Supply Chain",
            tables={STOCK_FACT, WAREHOUSE} if tables is None else tables,
            primary_rows=[{"TOTAL": 1000.0}],
            primary_domain="Sales",
            scope=PRIMARY_SCOPE,
            retrieve=boundaries.retrieve,
            generate=boundaries.generate,
            execute=_execute,
        ))
        return opinion, reached_warehouse

    def test_a_query_the_second_area_can_answer_runs_and_agrees(self):
        opinion, reached = self._run(STOCK_SQL)
        self.assertTrue(opinion.checked, opinion.detail)
        self.assertTrue(opinion.agrees)
        self.assertEqual(len(reached), 1)

    def test_a_fan_out_join_never_reaches_the_warehouse(self):
        # The production incident, generated by the second area this time:
        # every invoice row duplicated once per stock row for the same
        # warehouse. Refused at validation, so no wrong number is ever
        # compared against the answer the user is being shown.
        opinion, reached = self._run(FANOUT_SQL, tables={STOCK_FACT, WAREHOUSE, SALES_FACT})
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "invalid_there")
        self.assertEqual(opinion.detail["code"], "raw_fact_to_fact_join")
        self.assertEqual(reached, [])

    def test_reaching_into_the_first_areas_tables_never_reaches_the_warehouse(self):
        opinion, reached = self._run(
            f"SELECT SUM(NET_REVENUE_AMOUNT) AS TOTAL FROM {SALES_FACT}")
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "invalid_there")
        self.assertEqual(reached, [])

    def test_a_wildcard_projection_never_reaches_the_warehouse(self):
        opinion, reached = self._run(f"SELECT * FROM {STOCK_FACT}")
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "invalid_there")
        self.assertEqual(reached, [])


# ══════════════════════════════════════════════════════════════════════════════
# What the reader is told
# ══════════════════════════════════════════════════════════════════════════════

class TestTheExecutorRunsUnderTheScopeItWasHanded(unittest.TestCase):
    """The half the earlier end-to-end class could not see.

    Those tests injected an executor that returned canned rows and ignored
    table scope. Production's does not: ``execute_governed_query``
    re-validates independently — that is the argument-independent guarantee
    working — so an executor pinned to the PRIMARY area's tables refuses the
    corroborating query as access_denied after it has already passed the
    second area's validation. The attempt then reports "execution failed" and
    corroboration is permanently "not checked" on exactly the workspaces the
    feature exists for.

    So the executor here re-validates against the scope it is given, the way
    the real one does.
    """

    def _run(self, generated_sql, *, executor_scope):
        """`executor_scope` is what the executor believes it may read."""
        from core.sql_attempt import run_attempt

        boundaries = _Boundaries(sql=generated_sql)
        refusals = []

        def _governed_executor(sql, semantic_context):
            # What execute_governed_query does before it runs anything.
            verdict = validate_sql_detailed(
                sql, KNOWN_TABLES, "azure_sql", executor_scope,
                TABLE_COLUMNS, semantic_context,
            )
            if not verdict.ok:
                refusals.append(verdict.code)
                raise ValueError(verdict.reason)
            return SimpleNamespace(rows=[{"TOTAL": 1000.0}], sql=sql, truncated=False)

        async def _execute(sql, scope):
            return await run_attempt(
                sql, scope, executor=_governed_executor, timeout=10,
                timeout_message="timed out", source="corroboration",
            )

        opinion = asyncio.run(run_second_opinion(
            "what was revenue by warehouse",
            domain="Supply Chain", tables={STOCK_FACT},
            primary_rows=[{"TOTAL": 1000.0}], primary_domain="Sales",
            scope=PRIMARY_SCOPE,
            retrieve=boundaries.retrieve, generate=boundaries.generate,
            execute=_execute,
        ))
        return opinion, refusals

    def test_an_executor_scoped_to_the_second_area_completes_the_comparison(self):
        opinion, refusals = self._run(STOCK_SQL, executor_scope={STOCK_FACT})
        self.assertEqual(refusals, [])
        self.assertTrue(opinion.checked, opinion.detail)
        self.assertTrue(opinion.agrees)

    def test_an_executor_still_pinned_to_the_primary_refuses_every_second_opinion(self):
        # The defect, reproduced: the corroborating SQL passes the second
        # area's validation and is then refused by an executor that believes
        # it may only read the first area's tables.
        opinion, refusals = self._run(
            STOCK_SQL, executor_scope=PRIMARY_SCOPE.allowed_tables)
        self.assertEqual(refusals, ["access_denied"])
        self.assertFalse(opinion.checked)
        self.assertEqual(opinion.reason, "execution_failed")

    def test_the_scope_the_module_hands_the_executor_is_the_second_areas(self):
        # The contract this rests on: run_second_opinion passes the
        # corroborating scope to the execute callback, so a caller that reads
        # it gets the right tables without having to know how they were built.
        seen = {}

        async def _capture(sql, scope):
            seen["allowed"] = scope.allowed_tables
            return _attempt(sql, rows=[{"TOTAL": 1000.0}])

        boundaries = _Boundaries()
        asyncio.run(run_second_opinion(
            "q", domain="Supply Chain", tables={STOCK_FACT, WAREHOUSE},
            primary_rows=[{"TOTAL": 1.0}], primary_domain="Sales",
            scope=PRIMARY_SCOPE, retrieve=boundaries.retrieve,
            generate=boundaries.generate, execute=_capture,
        ))
        self.assertEqual(seen["allowed"], {STOCK_FACT, WAREHOUSE})
        self.assertNotEqual(seen["allowed"], PRIMARY_SCOPE.allowed_tables)


class TestTheLineTheReaderGets(unittest.TestCase):

    AGREES = {"checked": True, "agrees": True, "domain": "Supply Chain"}
    DISAGREES = {
        "checked": True, "agrees": False, "domain": "Supply Chain",
        "primary_value": 1000.0, "secondary_value": 2700.0,
        "relative_difference": 0.6296,
    }

    def test_agreement_names_the_area_that_confirmed_it(self):
        line = describe_second_opinion(self.AGREES)
        self.assertIn("Supply Chain", line)

    def test_disagreement_carries_both_figures_and_the_gap(self):
        line = describe_second_opinion(self.DISAGREES)
        self.assertIn("Supply Chain", line)
        self.assertIn("2,700", line)
        self.assertIn("1,000", line)
        self.assertIn("63.0", line)

    def test_a_run_that_did_not_happen_says_nothing(self):
        # Either sentence reads as a statement about the reader's data.
        # "The second area was never asked" is not one of them.
        self.assertEqual(
            describe_second_opinion({"checked": False, "domain": "Supply Chain",
                                     "reason": "retrieval_failed"}), "")
        self.assertEqual(describe_second_opinion({}), "")
        self.assertEqual(describe_second_opinion(None), "")

    def test_the_line_is_translated(self):
        self.assertNotEqual(
            describe_second_opinion(self.AGREES, lang="fr"),
            describe_second_opinion(self.AGREES, lang="en"),
        )
        self.assertIn("Supply Chain", describe_second_opinion(self.AGREES, lang="fr"))

    def test_the_ambient_language_decides_when_none_is_given(self):
        # The renderer runs inside the request's language activation and
        # passes no lang; an "en" default would have pinned every reader to
        # English on the one path that renders this.
        from core.i18n import activate_language, deactivate_language

        token = activate_language("fr")
        try:
            line = describe_second_opinion(self.DISAGREES)
        finally:
            deactivate_language(token)
        self.assertEqual(line, describe_second_opinion(self.DISAGREES, lang="fr"))
        self.assertNotEqual(line, describe_second_opinion(self.DISAGREES, lang="en"))


class TestWhatItDoesToConfidence(unittest.TestCase):
    """The score has to move, and move differently from a candidate check.

    Two candidates disagreeing means the QUESTION was ambiguous. Two subject
    areas disagreeing means the BUSINESS has two answers to it — and only one
    of those is something the reader can resolve by rephrasing.
    """

    def _score(self, corroboration=None):
        from core.answer_confidence import build_answer_confidence
        return build_answer_confidence(
            # No semantic plan and no retry: the baseline lands at 95, below
            # the ceiling and below no cap. A +5 credit is invisible on an
            # answer already scoring 100 and equally invisible under the
            # retry cap at 75 -- either fixture passes whether the credit is
            # applied or not.
            validation_code="ok", row_count=12, tables_used=["S.F"],
            corroboration=corroboration,
        )

    def test_the_baseline_leaves_room_for_a_credit_to_show(self):
        # Guards the fixture, not the feature: this test class proves nothing
        # about the agreement credit if the baseline is already capped.
        self.assertLess(self._score()["score"], 100)

    def test_a_second_area_that_agrees_raises_the_score(self):
        self.assertGreater(
            self._score({"checked": True, "agrees": True})["score"],
            self._score()["score"],
        )

    def test_a_second_area_that_disagrees_lowers_it_and_caps_it(self):
        baseline = self._score()["score"]
        scored = self._score({"checked": True, "agrees": False})
        self.assertLess(scored["score"], baseline)
        self.assertLessEqual(scored["score"], 49)
        self.assertNotEqual(scored["level"], "high")

    def test_a_disagreement_is_scored_low_so_the_reader_sees_it_without_clicking(self):
        """The cap is a display decision, not only a score.

        portal_chat.html renders a warning beside the confidence pill only
        when the verdict is LOW, and puts everything else inside a collapsed
        disclosure. A cap at 59 is "medium", one point above the threshold, so
        the sentence naming both figures sat behind a click the reader has no
        reason to make. The candidate-disagreement branch caps at 49 already.
        """
        self.assertEqual(self._score({"checked": True, "agrees": False})["level"], "low")

    def test_it_is_scored_no_softer_than_two_candidates_disagreeing(self):
        # Two areas disagreeing means the BUSINESS has two answers to the
        # question. That is not the milder of the two findings.
        areas = self._score({"checked": True, "agrees": False})
        candidates = self._score()
        from core.answer_confidence import build_answer_confidence

        candidates = build_answer_confidence(
            validation_code="ok", row_count=12, tables_used=["S.F"],
            candidate_selection={"reason": "verified_candidates_disagree"},
        )
        self.assertLessEqual(areas["score"], candidates["score"])

    def test_a_disagreement_is_stated_not_just_scored(self):
        scored = self._score({"checked": True, "agrees": False})
        self.assertTrue(any("second subject area" in w.lower()
                            for w in scored["warnings"]))

    def test_the_rendered_sentence_is_used_when_there_is_one(self):
        line = describe_second_opinion({
            "checked": True, "agrees": False, "domain": "Supply Chain",
            "primary_value": 1000.0, "secondary_value": 2700.0,
            "relative_difference": 0.6296,
        })
        scored = self._score({"checked": True, "agrees": False, "line": line})
        self.assertIn(line, scored["warnings"])

    def test_a_run_that_did_not_happen_changes_nothing(self):
        # The gate is `checked`, not `agrees`: reading agrees alone would
        # score every failed second opinion as a disagreement.
        baseline = self._score()
        for payload in ({"checked": False, "agrees": False, "reason": "no_context"},
                        {"checked": False, "reason": "retrieval_failed"},
                        {}, None):
            scored = self._score(payload)
            self.assertEqual(scored["score"], baseline["score"], payload)
            self.assertEqual(scored["warnings"], baseline["warnings"], payload)
            self.assertEqual(scored["reasons"], baseline["reasons"], payload)


class TestTheRendererReadsWhatThePipelineWrote(unittest.TestCase):
    """Write API to read API, on the identity key that spans them.

    The pipeline writes ``confidence_context["domain"]["corroboration"]`` and
    the renderer has to read that exact path. A mismatch is the quietest bug
    in this codebase: every lookup misses, nothing raises, and an answer a
    second area contradicts is presented at full confidence.
    """

    @staticmethod
    def _pipeline_payload(agrees):
        """Built the way the pipeline builds it, not hand-written."""
        from core.domains import Corroboration

        opinion = SecondOpinion(
            checked=True, domain="Supply Chain", reason="disagreement",
            corroboration=Corroboration(
                checked=True, agrees=agrees, primary_value=1000.0,
                secondary_value=2700.0, relative_difference=0.6296,
                secondary_source="Supply Chain"),
        )
        return {"name": "Sales", "reason": "two_plausible_domains",
                "second_opinion": "Supply Chain",
                "corroboration": opinion.as_dict()}

    def _render(self, domain_payload):
        from unittest.mock import AsyncMock, MagicMock

        import core.result_renderer as rr

        adapter = MagicMock()
        adapter.send_message = AsyncMock()
        adapter.send_result = AsyncMock()
        adapter.cache_result = None
        with patch.object(rr, "build_answer_confidence",
                          wraps=rr.build_answer_confidence) as confidence:
            try:
                asyncio.run(rr._send_results(
                    MagicMock(), adapter, "revenue by month",
                    [{"MONTH": "2026-01", "REVENUE": 1000.0}],
                    "SELECT 1", 12, None, "acct",
                    {"db_type": "azure_sql", "credentials": {}},
                    confidence_context={"domain": domain_payload},
                    cache_result=False,
                ))
            except Exception:
                # The renderer does far more than build confidence; what it
                # does after is not this test's business, and it has already
                # happened by the time anything else can fail.
                pass
        confidence.assert_called()
        return confidence.call_args.kwargs.get("corroboration")

    def test_a_disagreement_reaches_the_scorer(self):
        received = self._render(self._pipeline_payload(agrees=False))
        self.assertTrue(received.get("checked"))
        self.assertFalse(received.get("agrees"))
        self.assertEqual(received.get("secondary_value"), 2700.0)

    def test_the_reader_facing_sentence_is_attached_on_the_way_through(self):
        received = self._render(self._pipeline_payload(agrees=False))
        self.assertIn("Supply Chain", received.get("line") or "")
        self.assertIn("2,700", received.get("line") or "")

    def test_an_agreement_reaches_the_scorer_too(self):
        received = self._render(self._pipeline_payload(agrees=True))
        self.assertTrue(received.get("agrees"))
        self.assertIn("Supply Chain", received.get("line") or "")

    def test_a_workspace_with_no_domains_passes_an_empty_signal(self):
        self.assertEqual(self._render({}), {})

    def test_a_second_opinion_that_did_not_run_carries_no_sentence(self):
        payload = {"name": "Sales", "second_opinion": "Supply Chain",
                   "corroboration": SecondOpinion(
                       checked=False, domain="Supply Chain",
                       reason="retrieval_failed").as_dict()}
        received = self._render(payload)
        self.assertFalse(received.get("checked"))
        self.assertNotIn("line", received)


class TestThePipelineRunsIt(unittest.TestCase):
    """The block has to be reached, and reached in the right order.

    Source-level, because the block lives inside a 4,000-line function that
    cannot be called from a test. Everything it *decides* was moved into
    core/corroboration_run.py precisely so it could be, and is executed above;
    what is left here is ordering and reachability, which is all a source
    check can honestly assert.
    """

    @staticmethod
    def _source():
        import inspect

        import core.query_pipeline as qp
        return inspect.getsource(qp._handle_query_impl)

    def test_it_runs_only_when_a_second_area_could_answer(self):
        self.assertIn(
            "if _domain_secondary_tables and _domain_routing is not None and rows:",
            self._source())

    def test_the_second_scope_comes_from_the_un_narrowed_decision(self):
        # Recomputing it from `effective` here would intersect the runner-up
        # with the primary domain and be empty on every question.
        self.assertIn("_domain_secondary_tables = set(_scope_decision.secondary)",
                      self._source())

    def test_it_runs_after_the_answer_is_final(self):
        source = self._source()
        settled = source.index("query_row_count=len(rows)")
        corroborated = source.index("_second_opinion = await run_second_opinion(")
        self.assertLess(settled, corroborated)

    def test_it_runs_before_the_answer_is_scored(self):
        source = self._source()
        corroborated = source.index("_second_opinion = await run_second_opinion(")
        scored = source.index("_confidence_context = {")
        self.assertLess(corroborated, scored)

    def test_it_runs_before_post_processing_rewrites_the_rows(self):
        # Forecasting and what-if replace `rows` in place. Comparing against
        # a projected row would corroborate a number the warehouse never
        # returned.
        source = self._source()
        corroborated = source.index("_second_opinion = await run_second_opinion(")
        self.assertLess(corroborated, source.index("compute_whatif(rows"))

    def test_the_result_reaches_answer_confidence(self):
        self.assertIn(
            '"corroboration": (\n                    _second_opinion.as_dict()',
            self._source())

    def test_a_failure_costs_the_second_opinion_and_nothing_else(self):
        source = self._source()
        block = source[source.index("_second_opinion = None"):]
        block = block[:block.index('"formatting_results"')]
        self.assertIn("except Exception as _corroboration_exc", block)
        # Loud: a second opinion that never runs looks identical to one that
        # ran and found nothing, and the confidence panel says nothing either
        # way.
        self.assertIn("log.error(", block)
        self.assertIn("_second_opinion = None", block[block.index("except Exception"):])

    def test_a_run_that_did_not_happen_is_not_traced_as_a_failure(self):
        # "Not checked" is not a failure of this step -- the second area
        # declined, or had no knowledge base of its own. Marking it red tells
        # a reviewer a disagreement was found where none was looked for.
        source = self._source()
        block = source[source.index('_trace_step(\n                trace_id, "corroboration"'):]
        block = block[:block.index("except Exception as _corroboration_exc")]
        self.assertIn('"not_checked" if not _second_opinion.checked', block)
        self.assertIn('else "success" if _second_opinion.agrees', block)
        self.assertIn('else "error"', block)

    def test_the_call_site_matches_the_signature_it_calls(self):
        """The block is fail-open; a bad call would only ever be a log line.

        A missing or renamed keyword raises TypeError on every question,
        the handler catches it, and the feature is dead from the moment it
        merges — which is exactly how a governance gate once shipped inert.
        Bind the real call site's keywords against the real signature.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(self._source()))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "run_second_opinion"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertFalse(any(kw.arg is None for kw in call.keywords),
                         "a **kwargs splat would hide a missing argument")
        bound = inspect.signature(run_second_opinion).bind(
            *[object() for _ in call.args],
            **{kw.arg: object() for kw in call.keywords},
        )
        self.assertIn("scope", bound.arguments)
        self.assertIn("tables", bound.arguments)
        self.assertIn("primary_rows", bound.arguments)

    def test_the_attempt_call_site_matches_run_attempt(self):
        import ast
        import inspect
        import textwrap

        from core.sql_attempt import run_attempt

        tree = ast.parse(textwrap.dedent(self._source()))
        inner = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_run_second_opinion_attempt"
        ]
        self.assertEqual(len(inner), 1)
        calls = [
            node for node in ast.walk(inner[0])
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "run_attempt"
        ]
        self.assertEqual(len(calls), 1)
        inspect.signature(run_attempt).bind(
            *[object() for _ in calls[0].args],
            **{kw.arg: object() for kw in calls[0].keywords},
        )

    def test_it_uses_a_fresh_retriever(self):
        # The primary retrieval's `retriever` is bound inside a try/except
        # thousands of lines above and may never have been assigned.
        source = self._source()
        block = source[source.index("def _retrieve_for_second_opinion"):]
        self.assertIn("load_retriever(account_id)", block[:600])

    def test_the_second_generation_is_audited_under_its_own_component(self):
        source = self._source()
        block = source[source.index("async def _generate_second_opinion"):]
        self.assertIn('component="corroboration"', block[:900])

    def test_the_corroborating_executor_is_not_the_primarys(self):
        """The bug this replaced: executor=_execute_with_policy.

        That closure reads `effective`, which the domain block narrowed to the
        PRIMARY area — so execute_governed_query re-validated the second
        area's query against the first area's tables and refused it every
        time. Pinned through the AST rather than by text: what matters is the
        VALUE of the executor keyword at the call site.
        """
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(self._source()))
        inner = [n for n in ast.walk(tree)
                 if isinstance(n, ast.AsyncFunctionDef)
                 and n.name == "_run_second_opinion_attempt"]
        self.assertEqual(len(inner), 1)
        call = [n for n in ast.walk(inner[0])
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_attempt"][0]
        executor = [kw.value for kw in call.keywords if kw.arg == "executor"][0]
        self.assertIsInstance(executor, ast.Name)
        self.assertNotEqual(
            executor.id, "_execute_with_policy",
            "the second opinion is executing under the primary area's tables",
        )
        # And the executor it does use has to read the scope it was handed.
        helper = [n for n in ast.walk(inner[0])
                  if isinstance(n, ast.FunctionDef) and n.name == executor.id]
        self.assertEqual(len(helper), 1)
        self.assertIn("corroborating_scope.allowed_tables",
                      ast.unparse(helper[0]))

    def test_the_second_query_still_goes_through_the_governed_executor(self):
        """Scoped differently, governed identically.

        This test used to assert `executor=_execute_with_policy` verbatim,
        which certified the defect above as correct: the wiring it pinned was
        the one that ran the second area's query against the first area's
        tables. What actually matters is that the corroborating query reaches
        the same governed executor — policy, row cap, read-only enforcement —
        and it does, through the scoped wrapper.
        """
        source = self._source()
        block = source[source.index("async def _run_second_opinion_attempt"):]
        block = block[:block.index("_second_opinion = await run_second_opinion(")]
        self.assertIn("_execute_with_policy(", block)
        self.assertIn("run_attempt(", block)
        self.assertIn("allowed_tables_override=", block)

    def test_the_override_defaults_to_the_primary_scope_for_everyone_else(self):
        # Every other caller of _execute_with_policy passes no override and
        # must keep running under `effective`. A default of anything else
        # would silently rescope the primary answer.
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(self._source()))
        fn = [n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_execute_with_policy"]
        self.assertEqual(len(fn), 1)
        kwonly = {a.arg: d for a, d in
                  zip(fn[0].args.kwonlyargs, fn[0].args.kw_defaults)}
        self.assertIn("allowed_tables_override", kwonly)
        self.assertIsInstance(kwonly["allowed_tables_override"], ast.Constant)
        self.assertIsNone(kwonly["allowed_tables_override"].value)
        self.assertIn("effective if allowed_tables_override is None",
                      ast.unparse(fn[0]))


if __name__ == "__main__":
    unittest.main()
