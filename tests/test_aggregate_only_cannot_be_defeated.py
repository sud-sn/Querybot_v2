# -*- coding: utf-8 -*-
"""tests/test_aggregate_only_cannot_be_defeated.py

SELECT SUM(SALARY) AS TOTAL, SALARY AS RAW FROM EMPLOYEES passed clean.

`aggregate_only` is the obligation a tenant attaches to a column meaning "this
may only ever leave as an aggregate, never as an individual value" -- salary,
an individual claim amount, a single patient's balance. Enforced in
execute_governed_query (core/compliance/governed_query.py), and independently
re-implemented in chart_policy.aggregate_only_gate_passes and in the dashboard
chart route (portal/routes.py), the check was:

    required_aggregate - aggregate_sources

"aggregate_sources" was the union of resource keys behind every OUTPUT COLUMN
that contains an aggregate function ANYWHERE in its lineage. The question this
answers is "does the restricted resource appear in SOME aggregated output?" --
not "does it appear ONLY in aggregated outputs?". A second, non-aggregated
projection of the same column in the same query was invisible to it:

    SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY
    FROM EMPLOYEES GROUP BY SALARY

TOTAL_SALARY puts EMPLOYEES.SALARY in aggregate_sources; the check saw nothing
missing and RAW_SALARY was never asked about. Reproduced live at HEAD~1.

A second, independent way to defeat it: sqlglot classifies FIRST_VALUE,
LAST_VALUE, LAG and NTH_VALUE as exp.AggFunc subclasses (`type(...).__mro__`
shows AggFunc for all four), because they share aggregate-function SQL
grammar. But wrapped in OVER (...), none of them reduce the row set -- a
window function returns one row per INPUT row, so FIRST_VALUE(SALARY) OVER
(ORDER BY HIRE_DATE) hands back one row's raw salary value for every row in
the result, exactly like a bare column would. analyze_sql's aggregate walk
counted it as a safe aggregate anyway.

Both defects compound a third: when aggregate_only is the ONLY obligation on a
resource (no mask_strategy alongside it -- the shape the seed policy in
admin/routes.py's _default_regulated_rules actually produces for the "chart"
action), result_guard.protect_rows short-circuits entirely
(`if not rows or not decision.masking: return rows unmodified`). There is no
masking backstop. The aggregate_only check IS the only enforcement, so a gap
in it is a gap with nothing behind it.

The fix lives in ONE place -- sql_guard.aggregate_only_violations -- and all
three call sites now use it, so a future gap in the rule is a gap in one
function, not three copies of it that can each drift.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from core.compliance.sql_guard import aggregate_only_violations, analyze_sql


class AnalyzeSqlWindowFunctionTests(unittest.TestCase):
    """A window function never counts as the row-reducing kind of aggregate,
    whatever function is inside OVER (...)."""

    def test_first_value_over_a_window_is_not_an_aggregate_output(self):
        analysis = analyze_sql(
            "SELECT FIRST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE) AS FIRST_SAL "
            "FROM EMPLOYEES",
            "snowflake",
        )
        self.assertNotIn("FIRST_SAL", analysis.aggregate_outputs)
        self.assertIn("EMPLOYEES.SALARY", analysis.lineage["FIRST_SAL"])

    def test_last_value_lag_and_nth_value_are_not_aggregate_outputs_either(self):
        for template in (
            "LAST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE)",
            "LAG(SALARY) OVER (ORDER BY HIRE_DATE)",
            "NTH_VALUE(SALARY, 2) OVER (ORDER BY HIRE_DATE)",
        ):
            with self.subTest(fn=template):
                analysis = analyze_sql(
                    f"SELECT {template} AS OUT_COL FROM EMPLOYEES", "snowflake",
                )
                self.assertNotIn("OUT_COL", analysis.aggregate_outputs, template)

    def test_a_windowed_sum_is_also_not_an_aggregate_output(self):
        """Even a function that IS safe in its non-windowed form (SUM) does
        not reduce the row set when used with OVER (...) -- the obligation
        this protects is about row cardinality, not which function name is
        used."""
        analysis = analyze_sql(
            "SELECT SUM(SALARY) OVER (PARTITION BY DEPT) AS DEPT_TOTAL "
            "FROM EMPLOYEES",
            "snowflake",
        )
        self.assertNotIn("DEPT_TOTAL", analysis.aggregate_outputs)

    def test_the_fallback_lineage_path_also_excludes_window_functions(self):
        """analyze_sql collects AggFunc nodes in two places: the normal
        lineage walk, and a fallback used when sqlglot's lineage builder
        cannot resolve an expression (a real, if rare, path -- see the
        "Conservative fallback" comment in sql_guard.py). Both must exclude
        window-wrapped functions, or a query that happens to land on the
        fallback re-opens the exact bypass the primary path was fixed for.

        Forced deterministically by patching build_lineage to fail -- the
        real external boundary a lineage-resolution failure crosses -- so the
        fallback's OWN code runs for real rather than being asserted on by
        inspection.
        """
        import core.compliance.sql_guard as sql_guard

        with patch.object(sql_guard, "build_lineage", side_effect=RuntimeError("boom")):
            analysis = analyze_sql(
                "SELECT FIRST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE) AS FIRST_SAL "
                "FROM EMPLOYEES",
                "snowflake",
            )
        self.assertNotIn("FIRST_SAL", analysis.aggregate_outputs)
        self.assertIn("EMPLOYEES.SALARY", analysis.lineage["FIRST_SAL"])

    def test_the_same_function_without_over_is_still_a_real_aggregate(self):
        """Proves the exclusion is about the Window wrapper, not the function
        name -- SUM/GROUP BY is untouched by this fix."""
        analysis = analyze_sql(
            "SELECT DEPT, SUM(SALARY) AS TOTAL FROM EMPLOYEES GROUP BY DEPT",
            "snowflake",
        )
        self.assertIn("TOTAL", analysis.aggregate_outputs)
        self.assertIn("TOTAL", analysis.mask_exempt_outputs)


class AggregateOnlyViolationsTests(unittest.TestCase):
    """Unit tests on the shared helper, independent of any policy machinery."""

    def test_raw_passthrough_beside_its_own_aggregate_is_a_violation(self):
        analysis = analyze_sql(
            "SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY "
            "FROM EMPLOYEES GROUP BY SALARY",
            "snowflake",
        )
        self.assertEqual(
            aggregate_only_violations(analysis, {"EMPLOYEES.SALARY"}),
            {"EMPLOYEES.SALARY"},
        )

    def test_a_windowed_pass_through_with_no_group_by_aggregate_is_a_violation(self):
        analysis = analyze_sql(
            "SELECT FIRST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE) AS FIRST_SAL "
            "FROM EMPLOYEES",
            "snowflake",
        )
        self.assertEqual(
            aggregate_only_violations(analysis, {"EMPLOYEES.SALARY"}),
            {"EMPLOYEES.SALARY"},
        )

    def test_a_genuinely_aggregate_only_query_has_no_violation(self):
        analysis = analyze_sql(
            "SELECT DEPT, SUM(SALARY) AS TOTAL FROM EMPLOYEES GROUP BY DEPT",
            "snowflake",
        )
        self.assertEqual(aggregate_only_violations(analysis, {"EMPLOYEES.SALARY"}), set())

    def test_a_resource_never_referenced_at_all_is_still_flagged(self):
        """Preserves the ORIGINAL check's behaviour: a required resource with
        no aggregated output at all is exactly as much a violation as one
        that additionally leaks raw."""
        analysis = analyze_sql("SELECT DEPT FROM EMPLOYEES", "snowflake")
        self.assertEqual(
            aggregate_only_violations(analysis, {"EMPLOYEES.SALARY"}), {"EMPLOYEES.SALARY"},
        )

    def test_only_the_violating_resource_is_named_not_every_required_one(self):
        analysis = analyze_sql(
            "SELECT SUM(SALARY) AS TOTAL, SALARY AS RAW, BONUS "
            "FROM EMPLOYEES GROUP BY SALARY, BONUS",
            "snowflake",
        )
        # BONUS is passed through raw too, but it isn't in `required` -- only
        # a resource actually under the obligation should be named.
        self.assertEqual(
            aggregate_only_violations(
                analysis, {"EMPLOYEES.SALARY", "EMPLOYEES.BONUS"}),
            {"EMPLOYEES.SALARY", "EMPLOYEES.BONUS"},
        )
        self.assertEqual(
            aggregate_only_violations(analysis, {"EMPLOYEES.SALARY"}),
            {"EMPLOYEES.SALARY"},
        )

    def test_nothing_required_is_never_a_violation(self):
        analysis = analyze_sql(
            "SELECT SALARY FROM EMPLOYEES", "snowflake",
        )
        self.assertEqual(aggregate_only_violations(analysis, set()), set())


class GovernedQueryAggregateOnlyTests(unittest.TestCase):
    """Real execution of execute_governed_query with patched store + run_query
    -- the same established pattern as GovernedQueryAttestationTests in
    tests/test_user_attestation.py. run_query is a MagicMock so a test can
    prove the denial happens BEFORE the warehouse is ever touched, not just
    that an exception surfaces."""

    def _run(self, sql: str, *, shadow: bool = False):
        from core.compliance import governed_query, policy_engine, sql_guard
        from core.compliance.models import PolicyContext

        stores = {policy_engine.store, governed_query.store, sql_guard.store}
        rules = [{
            "name": "aggregate_only: salary",
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "EMPLOYEES.SALARY",
            "action": "query_execution", "effect": "allow",
            "aggregate_only": True,
            # Deliberately no mask_strategy: the shape that leaves
            # aggregate_only as the SOLE backstop for this resource.
        }, {
            # Every incidental column the test SQL references besides SALARY
            # (HIRE_DATE for the window-function repro, DEPT/BONUS for the
            # grouping repros) needs SOME matching rule, or evaluate() denies
            # the whole decision on the first resource with none -- before the
            # aggregate_only check for SALARY is ever reached. Listed after
            # the specific SALARY rule so, for SALARY itself, the specific
            # rule is still the one `next()` picks.
            "name": "allow: everything else in employees",
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "EMPLOYEES.*",
            "action": "query_execution", "effect": "allow",
        }, {
            "name": "result_release: employees",
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "EMPLOYEES.*",
            "action": "result_release", "effect": "allow",
        }]
        profile = {
            "mode": "regulated", "policy_pack_key": "banking_v1",
            "active_policy_version": 1,
            "enforcement_mode": "shadow" if shadow else "enforce",
        }
        context = PolicyContext(
            account_id="acct-agg-only", user_id="7", role="analyst",
            purpose_id="internal_ops", action="query_execution", policy_version=1,
        )
        from unittest.mock import MagicMock
        run_query_mock = MagicMock(
            return_value=[{"TOTAL_SALARY": 500000, "RAW_SALARY": 85000}])

        with ExitStack() as stack:
            for st in stores:
                stack.enter_context(patch.object(st, "get_compliance_profile", return_value=profile))
                stack.enter_context(patch.object(st, "get_classification_map", return_value={}))
                stack.enter_context(patch.object(st, "list_policy_rules", return_value=rules))
                stack.enter_context(patch.object(st, "list_purposes", return_value=[]))
                stack.enter_context(patch.object(st, "list_row_policies", return_value=[]))
                stack.enter_context(patch.object(st, "user_attestation_valid", return_value=False))
                stack.enter_context(patch.object(st, "log_policy_decision", return_value="audit-id"))
                stack.enter_context(patch.object(st, "get_classification_map", return_value={}))
            stack.enter_context(patch.object(governed_query, "run_query", run_query_mock))
            result = governed_query.execute_governed_query(
                {}, "snowflake", sql,
                context=context, known_tables={"EMPLOYEES"},
            )
        return result, run_query_mock

    def test_raw_column_beside_its_own_aggregate_is_refused_before_the_warehouse_runs(self):
        from core.compliance.governed_query import PolicyDeniedError

        with self.assertRaises(PolicyDeniedError) as caught:
            self._run(
                "SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY "
                "FROM EMPLOYEES GROUP BY SALARY"
            )
        self.assertEqual(caught.exception.decision.reason_code, "aggregate_only_violation")

    def test_a_windowed_first_value_pass_through_is_refused(self):
        from core.compliance.governed_query import PolicyDeniedError

        with self.assertRaises(PolicyDeniedError) as caught:
            self._run(
                "SELECT FIRST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE) AS FIRST_SAL "
                "FROM EMPLOYEES"
            )
        self.assertEqual(caught.exception.decision.reason_code, "aggregate_only_violation")

    def test_a_genuinely_aggregated_query_is_still_allowed(self):
        """No regression: a query that ONLY aggregates the restricted column
        must keep working exactly as before -- it runs, and the decision
        that comes back is an allow.

        Not asserted on reason_code: a fully-permitted MULTI-resource decision
        leaves `reason` at its unset initial value ("default_deny") even
        though `allowed` is True -- confirmed pre-existing in
        policy_engine.evaluate() by reproducing it with none of this file's
        changes present, so it is a separate, unrelated quirk and not this
        test's concern. `allowed` is the field every caller actually branches
        on (PolicyDecision.effective_allowed reads it, not reason_code).
        """
        result, run_query_mock = self._run(
            "SELECT DEPT, SUM(SALARY) AS TOTAL FROM EMPLOYEES GROUP BY DEPT"
        )
        run_query_mock.assert_called_once()
        self.assertTrue(result.decision.allowed)
        self.assertNotEqual(result.decision.reason_code, "aggregate_only_violation")

    def test_shadow_mode_records_the_violation_but_does_not_block(self):
        """Shadow governs whether a DECISION blocks; it must not silence the
        violation from being detected and recorded in the first place."""
        result, run_query_mock = self._run(
            "SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY "
            "FROM EMPLOYEES GROUP BY SALARY",
            shadow=True,
        )
        run_query_mock.assert_called_once()
        self.assertFalse(result.decision.allowed)
        self.assertEqual(result.decision.reason_code, "aggregate_only_violation")


class DashboardChartRouteWiringTests(unittest.TestCase):
    """The third call site: portal/routes.py's _refresh_chart (the dashboard
    chart renderer _render_dashboard calls per chart) duplicated the same
    inline check a third time. It needs a live db_cfg, store-backed client
    state and a saved dashboard chart record to execute for real -- the same
    shape of dependency that makes core.query_pipeline._handle_query_impl's
    branches get checked as a syntax tree elsewhere in this suite rather than
    executed end to end. Read here for the same reason: does it call the
    FIXED shared helper, not the inline computation this whole file exists
    because of.
    """

    def test_the_dashboard_chart_renderer_calls_the_shared_helper(self):
        import ast
        import inspect

        import portal.routes as routes

        tree = ast.parse(inspect.getsource(routes._refresh_chart))
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("aggregate_only_violations", called)


class ChartPolicyAggregateOnlyTests(unittest.TestCase):
    """aggregate_only_gate_passes (core/chart_policy.py) re-implemented the
    same check independently, for charts and forecasts rather than SQL
    execution, and had the identical bug: nothing here exercised it with a
    real SQL string before, so the fix could regress silently in this one
    call site while the SQL-execution path stayed fixed."""

    def _passes(self, sql: str, *, shadow: bool = False) -> bool:
        from core.chart_policy import aggregate_only_gate_passes
        from core.compliance import policy_engine

        rules = [{
            "name": "aggregate_only: salary",
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "EMPLOYEES.SALARY",
            "action": "chart", "effect": "allow",
            "aggregate_only": True,
        }, {
            "name": "allow: everything else in employees",
            "subject_type": "role", "subject_id": "analyst",
            "resource_type": "column", "resource_pattern": "EMPLOYEES.*",
            "action": "chart", "effect": "allow",
        }]
        profile = {
            "mode": "regulated", "policy_pack_key": "banking_v1",
            "active_policy_version": 1,
            "enforcement_mode": "shadow" if shadow else "enforce",
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(policy_engine.store, "get_compliance_profile", return_value=profile))
            stack.enter_context(patch.object(policy_engine.store, "get_classification_map", return_value={}))
            stack.enter_context(patch.object(policy_engine.store, "list_policy_rules", return_value=rules))
            stack.enter_context(patch.object(policy_engine.store, "list_purposes", return_value=[]))
            stack.enter_context(patch.object(policy_engine.store, "log_policy_decision", return_value="audit-id"))
            return aggregate_only_gate_passes(
                account_id="acct-agg-chart", portal_user={"role": "analyst"},
                event=None, sql=sql, db_type="snowflake", what="Chart",
            )

    def test_raw_column_beside_its_own_aggregate_blocks_the_chart(self):
        self.assertFalse(self._passes(
            "SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY "
            "FROM EMPLOYEES GROUP BY SALARY"
        ))

    def test_a_windowed_first_value_pass_through_blocks_the_chart(self):
        self.assertFalse(self._passes(
            "SELECT FIRST_VALUE(SALARY) OVER (ORDER BY HIRE_DATE) AS FIRST_SAL "
            "FROM EMPLOYEES"
        ))

    def test_a_genuinely_aggregated_chart_still_passes(self):
        self.assertTrue(self._passes(
            "SELECT DEPT, SUM(SALARY) AS TOTAL FROM EMPLOYEES GROUP BY DEPT"
        ))

    def test_shadow_mode_blocks_here_too_unlike_the_sql_execution_path(self):
        """Verified, not assumed: unlike execute_governed_query (which checks
        `decision.shadow` explicitly before raising), this function's
        aggregate_only branch has no shadow carve-out at all -- any detected
        violation is an unconditional `return False`. Confirmed unchanged by
        this fix: the control flow here is identical to before it, only the
        violation computation itself was corrected. A real design
        inconsistency between the two enforcement points (one shadow-aware,
        one not) worth someone's attention, but not part of what this fix
        touches -- recorded as a test so it is judged as a decision if it
        ever changes, not discovered by accident."""
        self.assertFalse(self._passes(
            "SELECT SUM(SALARY) AS TOTAL_SALARY, SALARY AS RAW_SALARY "
            "FROM EMPLOYEES GROUP BY SALARY",
            shadow=True,
        ))


if __name__ == "__main__":
    unittest.main()
