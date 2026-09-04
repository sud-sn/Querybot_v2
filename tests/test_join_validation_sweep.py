"""
tests/test_join_validation_sweep.py

Join-validation sweep: the entity-graph checker rejected correct SQL in two
reproducible ways, and both surface to the user as
"Generated SQL does not follow the resolved entity-graph join plan."

1. Alias collision across CTEs. `alias_to_table` holds one entry per alias for
   the whole statement, so two CTEs that each alias their own table as "f" —
   the natural multi-CTE shape, and one this codebase asks the model to produce
   — collide: the last binding wins, a governed edge that IS present in the
   first CTE resolves against the wrong table, and the edge reads as missing.
   The column checker already fixed this class with build_scope; the join
   checker was still on the flat map.

2. Comma joins. `FROM fact f, dim d WHERE f.key = d.key` is an INNER JOIN by
   any other name and a shape the model produces regularly, but only
   sg_exp.Join nodes were scanned, so the edge read as missing.

Both are false positives on valid SQL. The tests below also pin every genuine
violation the checker must keep catching, so widening it cannot open a hole.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_join_sweep_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.validator import validate_sql_detailed  # noqa: E402

FACT = "ERP.F_SALES"
DATE = "ERP.D_DATE"
WHS = "ERP.D_WHS"
COLUMNS = {
    FACT: {"DATE_SK": "int", "WHS_SK": "int", "AMT": "decimal", "CO": "int", "DIV": "int"},
    DATE: {"DATE_SK": "int", "FULL_DATE": "date"},
    WHS: {"WHS_SK": "int", "CO": "int", "DIV": "int", "WHS_NAME": "varchar"},
}


def _edge(edge_id, from_entity, to_entity, from_table, to_table, conditions,
          join_type="INNER"):
    return {
        "id": edge_id, "from_entity": from_entity, "to_entity": to_entity,
        "from_table": from_table.split(".")[-1], "from_schema": "ERP",
        "to_table": to_table.split(".")[-1], "to_schema": "ERP",
        "conditions": conditions, "join_type": join_type,
        "relationship_key": f"rel{edge_id}",
    }


WHS_EDGE = _edge(1, "Sales", "Whs", FACT, WHS, [["WHS_SK", "WHS_SK"]])
DATE_EDGE = _edge(2, "Sales", "Date", FACT, DATE, [["DATE_SK", "DATE_SK"]])
COMPOSITE_EDGE = _edge(
    3, "Sales", "Whs", FACT, WHS,
    [["WHS_SK", "WHS_SK"], ["CO", "CO"], ["DIV", "DIV"]],
)


def _validate(sql, edges, **graph_overrides):
    graph = {
        "enabled": True, "resolved_edges": list(edges),
        "detected": ["Sales", "Whs"], "anchor": "Sales", "join_skeleton": "x",
    }
    graph.update(graph_overrides)
    return validate_sql_detailed(
        sql, set(COLUMNS), "azure_sql", set(COLUMNS), COLUMNS,
        {"graph_context": graph, "semantic_plan": {}, "question": "revenue by warehouse"},
    )


def _codes(result):
    return {error.get("code") for error in (result.errors or [])}


# ══════════════════════════════════════════════════════════════════════════════
# 1  Correct SQL must pass
# ══════════════════════════════════════════════════════════════════════════════
class TestValidJoinShapesArePermitted(unittest.TestCase):

    def test_plain_join(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f "
            f"INNER JOIN {WHS} w ON f.WHS_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_reversed_operand_order(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {WHS} w "
            f"INNER JOIN {FACT} f ON w.WHS_SK = f.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_composite_edge_with_every_pair(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f INNER JOIN {WHS} w "
            "ON f.WHS_SK = w.WHS_SK AND f.CO = w.CO AND f.DIV = w.DIV "
            "GROUP BY w.WHS_NAME",
            [COMPOSITE_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_join_inside_a_cte(self):
        result = _validate(
            f"WITH scoped AS (SELECT w.WHS_NAME AS N, f.AMT AS A FROM {FACT} f "
            f"INNER JOIN {WHS} w ON f.WHS_SK = w.WHS_SK) "
            "SELECT N, SUM(A) AS T FROM scoped GROUP BY N",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_same_alias_reused_across_ctes(self):
        """The flat alias map made the first CTE's governed edge invisible."""
        result = _validate(
            f"WITH a AS (SELECT f.AMT AS A, w.WHS_NAME AS N FROM {FACT} f "
            f"INNER JOIN {WHS} w ON f.WHS_SK = w.WHS_SK), "
            f"b AS (SELECT f.FULL_DATE AS DT FROM {DATE} f) "
            "SELECT a.N, SUM(a.A) AS T FROM a CROSS JOIN b GROUP BY a.N",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)
        self.assertNotIn("graph_join_missing", _codes(result))

    def test_comma_join_with_the_condition_in_where(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f, {WHS} w "
            "WHERE f.WHS_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_comma_join_with_reversed_operands(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f, {WHS} w "
            "WHERE w.WHS_SK = f.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertTrue(result.ok, result.errors)

    def test_mixed_explicit_join_and_comma_join(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f "
            f"INNER JOIN {DATE} d ON f.DATE_SK = d.DATE_SK, {WHS} w "
            "WHERE f.WHS_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE, DATE_EDGE],
        )
        self.assertTrue(result.ok, result.errors)


# ══════════════════════════════════════════════════════════════════════════════
# 2  Genuine violations must still be caught
# ══════════════════════════════════════════════════════════════════════════════
class TestGenuineJoinViolationsStillFail(unittest.TestCase):
    """Widening the checker must not open a hole."""

    def test_required_edge_entirely_absent(self):
        result = _validate(f"SELECT SUM(f.AMT) AS T FROM {FACT} f", [WHS_EDGE])
        self.assertFalse(result.ok)
        self.assertIn("graph_join_missing", _codes(result))

    def test_wrong_column_pair(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f "
            f"INNER JOIN {WHS} w ON f.DATE_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertFalse(result.ok)
        self.assertIn("graph_join_missing", _codes(result))

    def test_cross_join_with_no_condition(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f CROSS JOIN {WHS} w "
            "GROUP BY w.WHS_NAME",
            [WHS_EDGE],
        )
        self.assertFalse(result.ok)

    def test_composite_edge_with_a_missing_pair(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f INNER JOIN {WHS} w "
            "ON f.WHS_SK = w.WHS_SK AND f.CO = w.CO GROUP BY w.WHS_NAME",
            [COMPOSITE_EDGE],
        )
        self.assertFalse(result.ok)
        self.assertIn("graph_join_missing", _codes(result))

    def test_left_join_requirement_written_as_inner(self):
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f "
            f"INNER JOIN {WHS} w ON f.WHS_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [_edge(1, "Sales", "Whs", FACT, WHS, [["WHS_SK", "WHS_SK"]], join_type="LEFT")],
        )
        self.assertFalse(result.ok)
        self.assertIn("graph_join_type_mismatch", _codes(result))

    def test_a_comma_join_cannot_satisfy_a_left_requirement(self):
        """A comma join is INNER semantics; it must not pass as LEFT."""
        result = _validate(
            f"SELECT w.WHS_NAME, SUM(f.AMT) AS T FROM {FACT} f, {WHS} w "
            "WHERE f.WHS_SK = w.WHS_SK GROUP BY w.WHS_NAME",
            [_edge(1, "Sales", "Whs", FACT, WHS, [["WHS_SK", "WHS_SK"]], join_type="LEFT")],
        )
        self.assertFalse(result.ok)
        self.assertIn("graph_join_type_mismatch", _codes(result))

    def test_an_anchor_subquery_equality_is_not_a_join(self):
        """A scalar anchor's own join must not satisfy the outer query's edge."""
        result = _validate(
            f"SELECT SUM(f.AMT) AS T FROM {FACT} f WHERE f.DATE_SK = ("
            f"  SELECT MAX(d.DATE_SK) FROM {DATE} d, {WHS} w WHERE w.WHS_SK = f.WHS_SK)",
            [WHS_EDGE],
        )
        self.assertFalse(result.ok)
        self.assertIn("graph_join_missing", _codes(result))


# ══════════════════════════════════════════════════════════════════════════════
# 3  Every rejection must be explainable to a business user
# ══════════════════════════════════════════════════════════════════════════════
class TestEveryValidationCodeIsExplained(unittest.TestCase):
    """A user should never see raw developer text.

    Live example: source_fact_mismatch had no translation, so a business user
    was shown "The compiled analytical plan requires
    EMDW_DMART.CUS_ORD_IVC_FCT as measure fact source(s), but the SQL scans …".
    """

    def _result_level_codes(self):
        import re

        source = (ROOT / "core" / "validator.py").read_text(encoding="utf-8")
        codes = set()
        for match in re.finditer(r"SqlValidationResult\(\s*False\s*,", source):
            segment = source[match.start(): match.start() + 1600]
            positional = re.findall(r'^\s*"([a-z_0-9]+)",\s*$', segment, re.M)
            if positional:
                codes.add(positional[0])
        codes |= set(re.findall(
            r'SqlValidationResult\(False,\s*"[^"]*",\s*"([a-z_0-9]+)"', source,
        ))
        return codes

    def test_all_codes_have_a_reason_and_a_next_step(self):
        from core.failure_messages import (
            _VALIDATION_NEXT_STEPS,
            _VALIDATION_REASONS,
            _DEFAULT_VALIDATION_NEXT_STEP,
        )

        missing_reason = sorted(self._result_level_codes() - set(_VALIDATION_REASONS))
        self.assertEqual(
            missing_reason, [],
            f"these validator codes would show raw developer text: {missing_reason}",
        )
        self.assertTrue(_DEFAULT_VALIDATION_NEXT_STEP.strip())
        self.assertTrue(_VALIDATION_NEXT_STEPS)

    def test_join_and_fact_failures_read_as_business_english(self):
        from core.failure_messages import translate_failure

        for code in ("source_fact_mismatch", "raw_fact_to_fact_join",
                     "fanout_aggregate", "graph_plan_mismatch"):
            with self.subTest(code=code):
                rca = translate_failure(
                    kind="validation", code=code, reason="internal detail",
                    sql="SELECT 1", question="q",
                )
                reason = rca.get("most_likely_reason") or ""
                self.assertTrue(reason.strip())
                # No physical identifiers in the business explanation.
                self.assertNotIn("EMDW_DMART", reason)
                self.assertNotIn("_FCT", reason)
                self.assertTrue(rca.get("suggested_next_step", "").strip())


# ══════════════════════════════════════════════════════════════════════════════
# 4  Repair wiring
# ══════════════════════════════════════════════════════════════════════════════
class TestRepairWiring(unittest.TestCase):

    def setUp(self):
        self.source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_a_mechanically_fixable_slip_is_repaired(self):
        """ORDER BY on an undeclared alias is a one-line fix, not terminal."""
        from core.validator import REPAIRABLE_REASON_CODES

        self.assertIn("order_alias_mismatch", REPAIRABLE_REASON_CODES)

    def test_governed_compiler_declines_are_visible(self):
        """~90 guard clauses return "" silently; a skipped contract must log."""
        self.assertIn("Governed compiler declined for", self.source)

    def test_deliberately_terminal_codes_stay_terminal(self):
        """Each is terminal for its own reason, recorded beside it in the
        registry: a policy refusal, an explicit decline, and DDL — which is
        terminal as a governance stance rather than because it is unfixable."""
        from core.validator import REPAIRABLE_REASON_CODES, TERMINAL_REASON_CODES

        for code in ("access_denied", "ddl", "cannot_generate"):
            with self.subTest(code=code):
                self.assertIn(code, TERMINAL_REASON_CODES)
                self.assertNotIn(code, REPAIRABLE_REASON_CODES)


if __name__ == "__main__":
    unittest.main()
