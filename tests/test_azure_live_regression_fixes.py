"""
tests/test_azure_live_regression_fixes.py

Regression tests for the defects found running the QBOT_LIVE_TEST Azure
regression plan against a live instance. Every one of these blocked a
*correct* answer, so each test asserts the good path is accepted while the
guard it relaxed still rejects the genuinely bad shape.

  A. Cross-fact field-plan contamination (live cases 1 & 7): a business term
     ("warehouse") won on a rival fact table, so the validator demanded
     ERP_ITM_BAL_PRD_FCT.WHS_DMS_KEY from a query written entirely against
     F_INVENTORY_DAILY. Requirements may now name at most one fact — the
     anchor; rival facts are demoted to optional.
  B. Semi-join rejected as a missing join (live case 4): the required
     INVOICE_DATE_SK = DATE_SK relationship expressed as
     `IN (SELECT DATE_SK ...)` was rejected for not being a literal JOIN..ON.
  C. Subquery ORDER BY judged against the outer SELECT's aliases (surfaced
     while fixing B).
  D. Surrogate/encoding tokens leaking into user-facing business date names
     (live cases 3 & 5): "Invoice Sk Date", "Posting Yyyymm Date".
  E. Analytical verbs missing from the data-request gate (live case 12):
     "Allocate total revenue by product category." was answered
     conversationally with no SQL.
  F. Period-close wording not recognised as a monthly grain (live case 6):
     "month-end" left the grain blank, so the finest-grain rule handed a
     month-end question to the daily snapshot fact and the monthly period
     role was never offered.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis_contract import measure_class_for_metric  # noqa: E402
from core.contextual_dates import (  # noqa: E402
    _GRAIN_ORDER,
    _role_temporal_grain,
    build_contextual_date_plan,
    metrics_are_semi_additive,
    question_has_explicit_date_filter,
    question_has_snapshot_intent,
    requested_temporal_grain,
)
from core.date_roles import _label_from_column  # noqa: E402
from core.query_pipeline import _date_option_labels  # noqa: E402
from core.dispatcher import _looks_like_data_request  # noqa: E402
from core.response_builder import _period_comparison_from_rows  # noqa: E402
from core.semantic_planner import (  # noqa: E402
    _anchor_fields_to_measure_fact,
    _find_display_field_for_key,
    _is_key_column,
    _key_prefix,
)
from core.semantic_model import (  # noqa: E402
    _is_measure_binding,
    _business_role_from_column,
    _scope_plan_to_single_fact,
)
from core.validator import validate_sql_detailed  # noqa: E402

SCHEMA = "QBOT_LIVE_TEST"
DATABASE = "CHATBOT_DB"


def _dual(name: str, columns: list[str]) -> dict[str, dict[str, str]]:
    """Catalog entries keyed both db-qualified and schema-qualified.

    The live catalog carried both spellings — the error text named
    "CHATBOT_DB.QBOT_LIVE_TEST.F_SALES_INVOICE, QBOT_LIVE_TEST.F_SALES_INVOICE"
    as homes for the same column — so the fixtures reproduce that.
    """
    cols = {c: "decimal" for c in columns}
    return {
        f"{DATABASE}.{SCHEMA}.{name}": dict(cols),
        f"{SCHEMA}.{name}": dict(cols),
    }


TABLE_COLUMNS: dict[str, dict[str, str]] = {}
TABLE_COLUMNS.update(_dual("F_SALES_INVOICE", [
    "SALES_LINE_SK", "CUSTOMER_SK", "PRODUCT_SK", "WAREHOUSE_SK",
    "INVOICE_DATE_SK", "ORDER_DATE_SK", "POSTING_YYYYMM",
    "NET_REVENUE_AMOUNT", "QUANTITY", "UPDATED_AT_UTC",
]))
TABLE_COLUMNS.update(_dual("D_DATE", [
    "DATE_SK", "FULL_DATE", "CALENDAR_YEAR", "CALENDAR_MONTH_NUMBER",
]))
TABLE_COLUMNS.update(_dual("D_WAREHOUSE", [
    "WAREHOUSE_SK", "WAREHOUSE_NAME", "REGION_SK",
]))
TABLE_COLUMNS.update(_dual("F_INVENTORY_DAILY", [
    "DAILY_SNAPSHOT_SK", "PRODUCT_SK", "WAREHOUSE_SK",
    "SNAPSHOT_DATE", "SNAPSHOT_YYYYMMDD", "INVENTORY_VALUE", "UNIT_COST",
]))
TABLE_COLUMNS.update(_dual("ERP_ITM_BAL_PRD_FCT", [
    "ITM_DMS_KEY", "WHS_DMS_KEY", "PRD_DMS_KEY", "INV_VAL_AMT",
]))
KNOWN_TABLES = set(TABLE_COLUMNS)

MODEL_TABLES = [
    {"qualified_name": f"{SCHEMA}.F_INVENTORY_DAILY", "type": "fact"},
    {"qualified_name": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
    {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
    {"qualified_name": f"{SCHEMA}.D_WAREHOUSE", "type": "dimension"},
    {"qualified_name": f"{SCHEMA}.D_DATE", "type": "dimension"},
]


def _validate(sql: str, semantic_context: dict | None = None):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, "mssql", None, TABLE_COLUMNS, semantic_context or {},
    )


class CrossFactPlanScopingTests(unittest.TestCase):
    """A — requirements must not name a fact the answer isn't anchored on."""

    def test_rival_fact_requirement_is_demoted(self):
        fields = [
            {"term": "daily inventory value", "table": f"{SCHEMA}.F_INVENTORY_DAILY",
             "column": "INVENTORY_VALUE", "role": "measure", "enforcement": "required"},
            {"term": "warehouse", "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY", "role": "attribute", "enforcement": "required"},
        ]
        anchor = _scope_plan_to_single_fact(fields, [], MODEL_TABLES)
        self.assertEqual(anchor, f"{SCHEMA}.F_INVENTORY_DAILY".upper())
        self.assertEqual(fields[0]["enforcement"], "required")
        self.assertEqual(fields[1]["enforcement"], "optional")

    def test_scoping_survives_database_qualified_field_tables(self):
        """Field tables and model tables need not agree on qualification.

        Taken verbatim from the production log: the field planner emitted
        CHATBOT_DB.QBOT_LIVE_TEST.ERP_ITM_BAL_PRD_FCT while the model records
        QBOT_LIVE_TEST.ERP_ITM_BAL_PRD_FCT. An exact lookup missed the match,
        so only one fact was ever counted and scoping silently no-opped --
        leaving the ERP field hard-required for a revenue question.
        """
        fields = [
            {"term": "warehouse",
             "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY", "role": "attribute",
             "enforcement": "required"},
            {"term": "Revenue", "table": f"{SCHEMA}.F_SALES_INVOICE",
             "column": "NET_REVENUE_AMOUNT", "role": "measure",
             "enforcement": "required"},
        ]
        tables = [
            {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.D_WAREHOUSE", "type": "dimension"},
        ]
        anchor = _scope_plan_to_single_fact(fields, [], tables)
        self.assertEqual(anchor, f"{SCHEMA}.F_SALES_INVOICE".upper())
        self.assertEqual(fields[0]["enforcement"], "optional")
        self.assertEqual(fields[1]["enforcement"], "required")

    def test_field_with_no_enforcement_key_is_demoted(self):
        """"Required" means *not* enforcement=="optional".

        semantic_planner.py only ever sets "optional" (see its own
        `f.get("enforcement") != "optional"` filter) and the validator skips
        only that value. Testing for enforcement=="required" therefore matched
        nothing the LLM field planner produced — which is every field that
        actually caused the live fan-out.
        """
        fields = [
            {"term": "warehouse",
             "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY"},                      # no enforcement key
            {"term": "Revenue", "table": f"{SCHEMA}.F_SALES_INVOICE",
             "column": "NET_REVENUE_AMOUNT", "role": "measure",
             "enforcement": "required"},
        ]
        tables = [
            {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
        ]
        anchor = _scope_plan_to_single_fact(fields, [], tables)
        self.assertEqual(anchor, f"{SCHEMA}.F_SALES_INVOICE".upper())
        self.assertEqual(fields[0]["enforcement"], "optional")
        self.assertEqual(fields[1]["enforcement"], "required")

    def test_already_optional_fields_are_left_alone(self):
        fields = [
            {"term": "warehouse", "table": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY", "enforcement": "optional"},
            {"term": "Revenue", "table": f"{SCHEMA}.F_SALES_INVOICE",
             "column": "NET_REVENUE_AMOUNT", "role": "measure",
             "enforcement": "required"},
        ]
        tables = [
            {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
        ]
        _scope_plan_to_single_fact(fields, [], tables)
        self.assertEqual(fields[0]["enforcement"], "optional")
        self.assertNotIn("demoted_reason", fields[0])

    def test_live_plan_shape_anchors_on_the_measure_fact(self):
        """Field shapes copied verbatim from the production log.

        The measure arrives as role="attribute" (build_runtime_semantic_plan
        defaults an approved field's role), so anchoring on role=="measure"
        found nothing and the old distinct_facts[0] fallback picked ERP by
        list order -- demoting the real measure and keeping the wrong binding
        required. That is how correct SQL kept being rejected.
        """
        fields = [
            {"term": "warehouse", "role": "dimension",
             "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY"},
            {"term": "Revenue", "role": "attribute", "enforcement": "required",
             "table": f"{SCHEMA}.F_SALES_INVOICE",
             "column": "NET_REVENUE_AMOUNT"},
        ]
        tables = [
            {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
        ]
        anchor = _scope_plan_to_single_fact(fields, [], tables)
        self.assertEqual(anchor, f"{SCHEMA}.F_SALES_INVOICE".upper())
        self.assertEqual(fields[0]["enforcement"], "optional")   # ERP key
        self.assertEqual(fields[1]["enforcement"], "required")   # measure kept

    def test_anchor_order_does_not_decide(self):
        """Same plan, facts listed the other way round -> same anchor."""
        for erp_first in (True, False):
            erp = {"term": "warehouse", "role": "dimension",
                   "table": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "column": "WHS_DMS_KEY"}
            rev = {"term": "Revenue", "role": "attribute", "enforcement": "required",
                   "table": f"{SCHEMA}.F_SALES_INVOICE", "column": "NET_REVENUE_AMOUNT"}
            fields = [erp, rev] if erp_first else [rev, erp]
            tables = [
                {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
                {"qualified_name": f"{SCHEMA}.F_SALES_INVOICE", "type": "fact"},
            ]
            self.assertEqual(
                _scope_plan_to_single_fact(fields, [], tables),
                f"{SCHEMA}.F_SALES_INVOICE".upper(),
                f"anchor changed with field order (erp_first={erp_first})",
            )
            self.assertEqual(erp["enforcement"], "optional")

    def test_no_measure_means_no_demotion(self):
        """Refuse to guess: two key bindings, no measure -> change nothing.

        The old fallback would have picked one arbitrarily and demoted the
        other, which is how a wrong anchor silently inverted the plan.
        """
        fields = [
            {"term": "warehouse", "role": "dimension",
             "table": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "column": "WHS_DMS_KEY"},
            {"term": "warehouse", "role": "dimension",
             "table": f"{SCHEMA}.F_INVENTORY_DAILY", "column": "WAREHOUSE_SK"},
        ]
        tables = [
            {"qualified_name": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "type": "fact"},
            {"qualified_name": f"{SCHEMA}.F_INVENTORY_DAILY", "type": "fact"},
        ]
        self.assertEqual(_scope_plan_to_single_fact(fields, [], tables), "")
        self.assertNotIn("enforcement", fields[0])
        self.assertNotIn("enforcement", fields[1])

    def test_key_columns_are_never_treated_as_measures(self):
        for column, expected in (
            ("NET_REVENUE_AMOUNT", True), ("INVENTORY_VALUE", True),
            ("WHS_DMS_KEY", False), ("WAREHOUSE_SK", False),
            ("CUSTOMER_ID", False), ("PRODUCT_CODE", False),
        ):
            self.assertEqual(
                _is_measure_binding({"column": column}), expected, column,
            )

    def test_single_fact_plan_keeps_every_requirement(self):
        fields = [
            {"term": "daily inventory value", "table": f"{SCHEMA}.F_INVENTORY_DAILY",
             "column": "INVENTORY_VALUE", "role": "measure", "enforcement": "required"},
            {"term": "unit cost", "table": f"{SCHEMA}.F_INVENTORY_DAILY",
             "column": "UNIT_COST", "role": "measure", "enforcement": "required"},
            {"term": "warehouse", "table": f"{SCHEMA}.D_WAREHOUSE",
             "column": "WAREHOUSE_NAME", "role": "attribute", "enforcement": "required"},
        ]
        self.assertEqual(_scope_plan_to_single_fact(fields, [], MODEL_TABLES), "")
        self.assertTrue(all(f["enforcement"] == "required" for f in fields))

    def test_dimension_requirements_survive_scoping(self):
        """Demotion targets rival *facts* only — dimension joins stay required."""
        fields = [
            {"term": "revenue", "table": f"{SCHEMA}.F_SALES_INVOICE",
             "column": "NET_REVENUE_AMOUNT", "role": "measure", "enforcement": "required"},
            {"term": "inventory value", "table": f"{SCHEMA}.F_INVENTORY_DAILY",
             "column": "INVENTORY_VALUE", "role": "measure", "enforcement": "required"},
            {"term": "warehouse", "table": f"{SCHEMA}.D_WAREHOUSE",
             "column": "WAREHOUSE_NAME", "role": "attribute", "enforcement": "required"},
        ]
        _scope_plan_to_single_fact(fields, [], MODEL_TABLES)
        by_table = {f["table"]: f["enforcement"] for f in fields}
        self.assertEqual(by_table[f"{SCHEMA}.D_WAREHOUSE"], "required")

    def test_live_case_7_sql_is_accepted_after_scoping(self):
        fields = [
            {"term": "daily inventory value", "table": f"{SCHEMA}.F_INVENTORY_DAILY",
             "column": "INVENTORY_VALUE", "role": "measure", "enforcement": "required"},
            {"term": "warehouse", "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY", "role": "attribute", "enforcement": "required"},
        ]
        _scope_plan_to_single_fact(fields, [], MODEL_TABLES)
        sql = (
            "SELECT WAREHOUSE_SK, SUM(INVENTORY_VALUE) AS DAILY_INVENTORY_VALUE "
            f"FROM {SCHEMA}.F_INVENTORY_DAILY "
            "WHERE SNAPSHOT_DATE = "
            f"(SELECT MAX(SNAPSHOT_DATE) FROM {SCHEMA}.F_INVENTORY_DAILY) "
            "GROUP BY WAREHOUSE_SK"
        )
        result = _validate(sql, {"semantic_plan": {"enabled": True, "fields": fields}})
        self.assertTrue(result.ok, result.reason)


class SemiJoinProofTests(unittest.TestCase):
    """B — `IN (SELECT col ...)` states the required relationship."""

    PLAN = {"semantic_plan": {
        "enabled": True,
        "fields": [{"term": "revenue", "table": f"{SCHEMA}.F_SALES_INVOICE",
                    "column": "NET_REVENUE_AMOUNT"}],
        "joins": [{"from": f"{SCHEMA}.F_SALES_INVOICE", "to": f"{SCHEMA}.D_DATE",
                   "conditions": [["INVOICE_DATE_SK", "DATE_SK"]],
                   "enforcement": "required"}],
    }}

    _NULL_AWARE = (
        "SELECT COUNT_BIG(*) AS MatchedRows, "
        "COUNT(NET_REVENUE_AMOUNT) AS NonNullMetricRows, "
        "COALESCE(SUM(NET_REVENUE_AMOUNT), 0) AS TOTAL_REVENUE "
        f"FROM {SCHEMA}.F_SALES_INVOICE "
    )

    def test_semi_join_satisfies_required_join(self):
        sql = self._NULL_AWARE + (
            "WHERE INVOICE_DATE_SK IN ("
            f"  SELECT TOP 2 DATE_SK FROM {SCHEMA}.D_DATE"
            "   WHERE DATE_SK IN (SELECT DISTINCT INVOICE_DATE_SK "
            f"                     FROM {SCHEMA}.F_SALES_INVOICE)"
            "   ORDER BY FULL_DATE DESC)"
        )
        result = _validate(sql, self.PLAN)
        self.assertTrue(result.ok, result.reason)

    def test_absent_relationship_still_rejected(self):
        sql = self._NULL_AWARE + "WHERE INVOICE_DATE_SK > 20260301"
        result = _validate(sql, self.PLAN)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")

    def test_in_over_literal_list_proves_nothing(self):
        """A literal list never equates two columns — must not count as a join."""
        sql = self._NULL_AWARE + "WHERE INVOICE_DATE_SK IN (20260302, 20260303)"
        result = _validate(sql, self.PLAN)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")

    def test_in_over_unrelated_column_proves_nothing(self):
        sql = self._NULL_AWARE + (
            "WHERE INVOICE_DATE_SK IN "
            f"(SELECT CALENDAR_YEAR FROM {SCHEMA}.D_DATE)"
        )
        result = _validate(sql, self.PLAN)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")


class SubqueryOrderByScopeTests(unittest.TestCase):
    """C — a subquery's ORDER BY is not scoped to the outer SELECT."""

    def test_subquery_order_by_column_is_allowed(self):
        # Null-aware shape so the aggregate rule (checked earlier) is satisfied
        # and ORDER BY scoping is what this case actually exercises.
        sql = (
            "SELECT COUNT_BIG(*) AS MatchedRows, "
            "COUNT(NET_REVENUE_AMOUNT) AS NonNullMetricRows, "
            "COALESCE(SUM(NET_REVENUE_AMOUNT), 0) AS TOTAL_REVENUE "
            f"FROM {SCHEMA}.F_SALES_INVOICE "
            "WHERE INVOICE_DATE_SK IN ("
            f"  SELECT TOP 2 DATE_SK FROM {SCHEMA}.D_DATE ORDER BY FULL_DATE DESC)"
        )
        result = _validate(sql)
        self.assertTrue(result.ok, result.reason)

    def test_outer_order_by_alias_drift_still_rejected(self):
        # UNIT_COST is a real column on the table (so the unknown-column scan
        # stays quiet) but is neither a SELECT alias nor a grouped output —
        # exactly the drift this rule exists to catch.
        sql = (
            "SELECT WAREHOUSE_SK, SUM(INVENTORY_VALUE) AS TOTAL_VALUE "
            f"FROM {SCHEMA}.F_INVENTORY_DAILY "
            "GROUP BY WAREHOUSE_SK ORDER BY UNIT_COST DESC"
        )
        result = _validate(sql)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "order_alias_mismatch")


class BusinessDateNameTests(unittest.TestCase):
    """D — clarification labels must be business names, not storage detail."""

    def test_surrogate_key_spelling_matches_key_spelling(self):
        # INVOICE_DATE_KEY already rendered correctly; only _SK leaked.
        self.assertEqual(_label_from_column("INVOICE_DATE_SK"), "Invoice Date")
        self.assertEqual(
            _label_from_column("INVOICE_DATE_SK"),
            _label_from_column("INVOICE_DATE_KEY"),
        )

    def test_role_playing_date_labels(self):
        self.assertEqual(_label_from_column("ORDER_DATE_SK"), "Order Date")
        self.assertEqual(_label_from_column("RETURN_DATE_SK"), "Return Date")
        self.assertEqual(
            _label_from_column("ORIGINAL_INVOICE_DATE_SK"), "Original Invoice Date",
        )

    def test_no_physical_token_reaches_a_label(self):
        leaks = ("sk", "fk", "yyyymm", "yyyymmdd", "utc", "key", "dms")
        for column in (
            "INVOICE_DATE_SK", "RETURN_DATE_SK", "ORIGINAL_INVOICE_DATE_SK",
            "POSTING_YYYYMM", "SNAPSHOT_YYYYMMDD", "PERIOD_YYYYMM",
            "PRD_DMS_KEY", "UPDATED_AT_UTC",
        ):
            label = _label_from_column(column).lower()
            for leak in leaks:
                self.assertNotIn(
                    leak, label.split(),
                    f"{column} label {label!r} leaks physical token {leak!r}",
                )

    def test_month_encoded_column_is_named_a_period(self):
        """YYYYMM is a period, not a date — the grain belongs in the name."""
        self.assertEqual(_label_from_column("POSTING_YYYYMM"), "Posting Period")
        self.assertIn("Period", _label_from_column("PERIOD_YYYYMM"))
        self.assertEqual(_label_from_column("SNAPSHOT_YYYYMMDD"), "Snapshot Date")

    def test_business_role_strips_surrogate_suffix(self):
        self.assertEqual(_business_role_from_column("INVOICE_DATE_SK"), "invoice_date")
        self.assertEqual(_business_role_from_column("RETURN_DATE_SK"), "return_date")

    def test_key_tokens_are_stripped_positionally_not_globally(self):
        """Plumbing markers are a *suffix* convention — an interior or leading
        occurrence is real vocabulary. Stripping them everywhere turned
        KEY_ACCOUNT_DATE ("Key Account") into a plain "Account Date"."""
        self.assertEqual(_label_from_column("KEY_ACCOUNT_DATE"), "Key Account Date")
        self.assertEqual(_label_from_column("BID_DATE"), "Bid Date")
        self.assertEqual(
            _label_from_column("IDENTITY_VERIFIED_DATE"), "Identity Verified Date",
        )
        self.assertEqual(_label_from_column("SKU_LAUNCH_DATE"), "Sku Launch Date")

    def test_known_plumbing_prefixes_are_dropped(self):
        self.assertEqual(_label_from_column("DIM_APPROVAL_DATE"), "Approval Date")
        self.assertEqual(_label_from_column("SVC_DT_DMS_KEY"), "Svc Date")


class RawFactToFactJoinTests(unittest.TestCase):
    """H — a fan-out join must be refused, not answered.

    The existing multi-fact guard only fires when the entity graph resolved
    fact entities. With relationships still in review it was inert, and this
    exact SQL executed in production: every invoice row duplicated once per
    inventory row for the same warehouse, reporting 2700.00 where the true
    total is 1050.00. A believable wrong number is worse than a refusal.
    """

    PLAN = {"semantic_plan": {
        "enabled": True,
        "known_fact_tables": [
            f"{SCHEMA}.F_SALES_INVOICE",
            f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
        ],
    }}

    def test_production_fanout_sql_is_rejected(self):
        sql = (
            "SELECT w.WAREHOUSE_NAME, SUM(i.NET_REVENUE_AMOUNT) AS TOTAL_REVENUE "
            f"FROM {SCHEMA}.ERP_ITM_BAL_PRD_FCT f "
            f"JOIN {SCHEMA}.D_WAREHOUSE w ON f.WHS_DMS_KEY = w.WAREHOUSE_SK "
            f"JOIN {SCHEMA}.F_SALES_INVOICE i ON f.WHS_DMS_KEY = i.WAREHOUSE_SK "
            "GROUP BY w.WAREHOUSE_NAME ORDER BY TOTAL_REVENUE DESC"
        )
        result = _validate(sql, self.PLAN)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "raw_fact_to_fact_join")

    def test_correct_single_fact_query_passes(self):
        sql = (
            "SELECT w.WAREHOUSE_NAME, SUM(i.NET_REVENUE_AMOUNT) AS TOTAL_REVENUE "
            f"FROM {SCHEMA}.F_SALES_INVOICE i "
            f"JOIN {SCHEMA}.D_WAREHOUSE w ON i.WAREHOUSE_SK = w.WAREHOUSE_SK "
            "GROUP BY w.WAREHOUSE_NAME ORDER BY TOTAL_REVENUE DESC"
        )
        result = _validate(sql, self.PLAN)
        self.assertTrue(result.ok, result.reason)

    def test_pre_aggregated_ctes_are_still_allowed(self):
        """The sanctioned multi-fact pattern must not be collateral damage."""
        sql = (
            "WITH rev AS (SELECT WAREHOUSE_SK, SUM(NET_REVENUE_AMOUNT) AS R "
            f"             FROM {SCHEMA}.F_SALES_INVOICE GROUP BY WAREHOUSE_SK), "
            "     inv AS (SELECT WHS_DMS_KEY, SUM(INV_VAL_AMT) AS V "
            f"             FROM {SCHEMA}.ERP_ITM_BAL_PRD_FCT GROUP BY WHS_DMS_KEY) "
            "SELECT w.WAREHOUSE_NAME, rev.R, inv.V FROM rev "
            f"JOIN {SCHEMA}.D_WAREHOUSE w ON rev.WAREHOUSE_SK = w.WAREHOUSE_SK "
            "JOIN inv ON inv.WHS_DMS_KEY = rev.WAREHOUSE_SK"
        )
        result = _validate(sql, self.PLAN)
        self.assertTrue(result.ok, result.reason)

    def test_guard_is_inert_without_a_known_fact_list(self):
        """No table roles => no claim. Must not fail closed on every join."""
        sql = (
            "SELECT w.WAREHOUSE_NAME, SUM(i.NET_REVENUE_AMOUNT) AS TOTAL_REVENUE "
            f"FROM {SCHEMA}.ERP_ITM_BAL_PRD_FCT f "
            f"JOIN {SCHEMA}.F_SALES_INVOICE i ON f.WHS_DMS_KEY = i.WAREHOUSE_SK "
            f"JOIN {SCHEMA}.D_WAREHOUSE w ON f.WHS_DMS_KEY = w.WAREHOUSE_SK "
            "GROUP BY w.WAREHOUSE_NAME"
        )
        self.assertTrue(_validate(sql, {"semantic_plan": {"enabled": True}}).ok)

    def test_single_fact_plus_many_dimensions_is_fine(self):
        sql = (
            "SELECT w.WAREHOUSE_NAME, d.CALENDAR_YEAR, "
            "SUM(i.NET_REVENUE_AMOUNT) AS TOTAL_REVENUE "
            f"FROM {SCHEMA}.F_SALES_INVOICE i "
            f"JOIN {SCHEMA}.D_WAREHOUSE w ON i.WAREHOUSE_SK = w.WAREHOUSE_SK "
            f"JOIN {SCHEMA}.D_DATE d ON i.INVOICE_DATE_SK = d.DATE_SK "
            "GROUP BY w.WAREHOUSE_NAME, d.CALENDAR_YEAR"
        )
        result = _validate(sql, self.PLAN)
        self.assertTrue(result.ok, result.reason)


class MeasureFirstFactAnchoringTests(unittest.TestCase):
    """Phase 3 — the resolved measure's fact is authoritative.

    A generic term ("warehouse") scored per-term across every allowed table
    could bind to a fact unrelated to the measure, forcing a fact-to-fact join.
    """

    TABLES = {f"{SCHEMA}.F_SALES_INVOICE", f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT"}

    def test_rival_fact_dimension_is_demoted_to_a_hint(self):
        fields = [
            {"term": "revenue", "role": "measure",
             "table": f"{SCHEMA}.F_SALES_INVOICE", "column": "NET_REVENUE_AMOUNT"},
            {"term": "warehouse", "role": "dimension",
             "table": f"{DATABASE}.{SCHEMA}.ERP_ITM_BAL_PRD_FCT",
             "column": "WHS_DMS_KEY"},
        ]
        anchor = _anchor_fields_to_measure_fact(fields, self.TABLES)
        self.assertEqual(anchor, f"{SCHEMA}.F_SALES_INVOICE".upper())
        self.assertEqual(fields[1]["enforcement"], "optional")
        self.assertNotIn("enforcement", fields[0])   # measure untouched

    def test_dimension_tables_are_never_demoted(self):
        """The star path is the point — only rival *facts* are demoted."""
        fields = [
            {"term": "revenue", "role": "measure",
             "table": f"{SCHEMA}.F_SALES_INVOICE", "column": "NET_REVENUE_AMOUNT"},
            {"term": "warehouse", "role": "dimension",
             "table": f"{SCHEMA}.D_WAREHOUSE", "column": "WAREHOUSE_NAME"},
        ]
        _anchor_fields_to_measure_fact(fields, self.TABLES)
        self.assertNotIn("enforcement", fields[1])

    def test_no_measure_means_no_anchor(self):
        """Cannot lock a fact without a measure — must not guess one."""
        fields = [
            {"term": "warehouse", "role": "dimension",
             "table": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "column": "WHS_DMS_KEY"},
        ]
        self.assertEqual(_anchor_fields_to_measure_fact(fields, self.TABLES), "")
        self.assertNotIn("enforcement", fields[0])

    def test_inert_without_fact_classifications(self):
        """No table roles => no claim; prior behaviour is preserved exactly."""
        fields = [
            {"term": "revenue", "role": "measure",
             "table": f"{SCHEMA}.F_SALES_INVOICE", "column": "NET_REVENUE_AMOUNT"},
            {"term": "warehouse", "role": "dimension",
             "table": f"{SCHEMA}.ERP_ITM_BAL_PRD_FCT", "column": "WHS_DMS_KEY"},
        ]
        self.assertEqual(_anchor_fields_to_measure_fact(fields, None), "")
        self.assertNotIn("enforcement", fields[1])

    def test_measure_on_the_same_fact_is_kept_required(self):
        fields = [
            {"term": "revenue", "role": "measure",
             "table": f"{SCHEMA}.F_SALES_INVOICE", "column": "NET_REVENUE_AMOUNT"},
            {"term": "quantity", "role": "dimension",
             "table": f"{DATABASE}.{SCHEMA}.F_SALES_INVOICE", "column": "QUANTITY"},
        ]
        _anchor_fields_to_measure_fact(fields, self.TABLES)
        self.assertNotIn("enforcement", fields[1])


class SemiAdditiveMetadataTests(unittest.TestCase):
    """G — snapshot detection must not depend on a domain noun list.

    The wording heuristic only ever knew inventory/stock/balance, so a
    semi-additive measure named anything else ("month-end headcount",
    "month-end assets under management") was treated as ordinary additive
    data and could be summed across periods. Governed metric metadata now
    supplies the signal instead.
    """

    HEADCOUNT = {"name": "Headcount", "sql_template": "SUM(ENDING_HEADCOUNT)"}
    AUM = {"name": "AUM", "aggregation_semantics": "semi_additive",
           "sql_template": "SUM(MKT_VAL)"}
    SUBSCRIPTIONS = {"name": "Open Subscriptions",
                     "sql_template": "SUM(CLOSING_SUBSCRIBER_CT)"}
    REVENUE = {"name": "Revenue", "sql_template": "SUM(NET_REVENUE_AMOUNT)"}

    def test_declared_semantics_win(self):
        self.assertEqual(measure_class_for_metric(self.AUM), "semi_additive")
        self.assertEqual(
            measure_class_for_metric(
                {"name": "X", "aggregation": "non-additive",
                 "sql_template": "SUM(ENDING_BALANCE)"}
            ),
            "non_additive",
        )

    def test_derived_from_formula_when_undeclared(self):
        self.assertEqual(measure_class_for_metric(self.HEADCOUNT), "semi_additive")
        self.assertEqual(measure_class_for_metric(self.SUBSCRIPTIONS), "semi_additive")
        self.assertEqual(measure_class_for_metric(self.REVENUE), "additive")

    def test_semi_additive_metric_signals_snapshot_in_any_domain(self):
        for metric, question in (
            (self.HEADCOUNT, "month-end headcount by department"),
            (self.AUM, "month-end assets under management"),
            (self.SUBSCRIPTIONS, "month-end open subscriptions"),
        ):
            self.assertFalse(
                question_has_snapshot_intent(question),
                f"wording alone should not have detected {question!r}",
            )
            self.assertTrue(
                question_has_snapshot_intent(question, matched_metrics=[metric]),
                question,
            )

    def test_additive_metric_does_not_trigger_snapshot(self):
        self.assertFalse(
            question_has_snapshot_intent(
                "month-end revenue by region", matched_metrics=[self.REVENUE]
            )
        )

    def test_metadata_only_promotes_never_suppresses(self):
        """An additive metric must not cancel a genuine wording signal."""
        self.assertTrue(
            question_has_snapshot_intent(
                "inventory by warehouse", matched_metrics=[self.REVENUE]
            )
        )

    def test_wording_only_behaviour_is_unchanged(self):
        self.assertTrue(question_has_snapshot_intent("inventory by warehouse"))
        self.assertTrue(question_has_snapshot_intent("current stock on hand"))
        self.assertFalse(question_has_snapshot_intent("inventory sales revenue"))

    def test_malformed_metric_entries_are_ignored(self):
        self.assertFalse(metrics_are_semi_additive([None, "junk", 42]))
        self.assertFalse(metrics_are_semi_additive(None))

    def test_month_end_headcount_now_selects_the_monthly_role(self):
        """The end-to-end point of G: without it this fell through to daily."""
        roles = [
            {"name": "Snapshot Date", "fact_column": "SNAPSHOT_DATE",
             "date_key_type": "native_date"},
            {"name": "Headcount Period", "fact_column": "PERIOD_YYYYMM",
             "date_key_type": "yyyymm_integer", "temporal_grain": "month"},
        ]
        question = "month-end headcount by department"
        grain = requested_temporal_grain(question)
        self.assertEqual(grain, "month")
        self.assertTrue(
            question_has_snapshot_intent(question, matched_metrics=[self.HEADCOUNT])
        )
        known = [_role_temporal_grain(r) for r in roles if _role_temporal_grain(r)]
        self.assertIn(grain, known)
        selected = [r["fact_column"] for r in roles
                    if _role_temporal_grain(r) == grain]
        self.assertEqual(selected, ["PERIOD_YYYYMM"])


class ForeignSchemaGeneralisationTests(unittest.TestCase):
    """The fixes are schema-driven, not tuned to QBOT_LIVE_TEST.

    Same defects, a different domain (pharmacy claims) and a different mix of
    key conventions (_ID / _KEY / _SK / _DMS_KEY).
    """

    PHARMA = "PHARMA_LAB"

    def _catalog(self):
        def tbl(name, cols):
            return {f"{self.PHARMA}.{name}": {c: "decimal" for c in cols}}
        catalog: dict[str, dict[str, str]] = {}
        catalog.update(tbl("F_RX_FILL", [
            "FILL_SK", "DRUG_SK", "PHARMACY_SK", "FILL_DATE_SK", "NET_PAID_AMT",
        ]))
        catalog.update(tbl("F_CLAIM_MONTHLY", [
            "CLAIM_MTH_SK", "PHARMACY_SK", "PERIOD_YYYYMM", "ENDING_LIABILITY_AMT",
        ]))
        catalog.update(tbl("D_CALENDAR", ["DATE_SK", "CAL_DATE", "CAL_YEAR"]))
        return catalog

    def test_cross_fact_scoping_generalises(self):
        fields = [
            {"term": "net paid", "table": f"{self.PHARMA}.F_RX_FILL",
             "column": "NET_PAID_AMT", "role": "measure", "enforcement": "required"},
            {"term": "pharmacy", "table": f"{self.PHARMA}.F_CLAIM_MONTHLY",
             "column": "PHARMACY_SK", "role": "attribute", "enforcement": "required"},
        ]
        tables = [
            {"qualified_name": f"{self.PHARMA}.F_RX_FILL", "type": "fact"},
            {"qualified_name": f"{self.PHARMA}.F_CLAIM_MONTHLY", "type": "fact"},
        ]
        anchor = _scope_plan_to_single_fact(fields, [], tables)
        self.assertEqual(anchor, f"{self.PHARMA}.F_RX_FILL".upper())
        self.assertEqual(fields[1]["enforcement"], "optional")

    def test_semi_join_proof_generalises(self):
        catalog = self._catalog()
        plan = {"semantic_plan": {
            "enabled": True,
            "fields": [{"term": "net paid", "table": f"{self.PHARMA}.F_RX_FILL",
                        "column": "NET_PAID_AMT"}],
            "joins": [{"from": f"{self.PHARMA}.F_RX_FILL",
                       "to": f"{self.PHARMA}.D_CALENDAR",
                       "conditions": [["FILL_DATE_SK", "DATE_SK"]],
                       "enforcement": "required"}],
        }}
        base = (
            "SELECT COUNT_BIG(*) AS MatchedRows, "
            "COUNT(NET_PAID_AMT) AS NonNullMetricRows, "
            "COALESCE(SUM(NET_PAID_AMT), 0) AS NET_PAID "
            f"FROM {self.PHARMA}.F_RX_FILL "
        )
        accepted = base + (
            "WHERE FILL_DATE_SK IN ("
            f"  SELECT TOP 3 DATE_SK FROM {self.PHARMA}.D_CALENDAR "
            "   ORDER BY CAL_DATE DESC)"
        )
        rejected = base + "WHERE FILL_DATE_SK IN (20260101, 20260102)"
        ok = validate_sql_detailed(
            accepted, set(catalog), "mssql", None, catalog, plan)
        bad = validate_sql_detailed(
            rejected, set(catalog), "mssql", None, catalog, plan)
        self.assertTrue(ok.ok, ok.reason)
        self.assertFalse(bad.ok)

    def test_labels_and_grain_generalise(self):
        self.assertEqual(_label_from_column("FILL_DATE_SK"), "Fill Date")
        self.assertEqual(_label_from_column("ADMIT_DATE_KEY"), "Admit Date")
        self.assertEqual(
            requested_temporal_grain("Show month-end liability for February 2026"),
            "month",
        )
        self.assertTrue(
            _looks_like_data_request("Allocate total net paid by drug class")
        )


class DataRequestVerbTests(unittest.TestCase):
    """E — analytical verbs must route to the pipeline, not the chat analyst."""

    def test_live_case_12_question_is_a_data_request(self):
        self.assertTrue(
            _looks_like_data_request("Allocate total revenue by product category.")
        )

    def test_analytical_verbs_route_to_data(self):
        for question in (
            "Break down revenue by product category",
            "Split revenue by category",
            "Sum inventory value by warehouse",
            "Aggregate revenue by category",
            "Average order amount by customer",
            "Trend revenue by month",
            "Plot revenue by region",
            "Segment customers by revenue",
            "Distribute total revenue by warehouse",
        ):
            self.assertTrue(_looks_like_data_request(question), question)

    def test_previously_supported_phrasings_unchanged(self):
        for question in (
            "Show revenue by region.",
            "What is total revenue by warehouse?",
            "How many orders per customer",
        ):
            self.assertTrue(_looks_like_data_request(question), question)

    def test_non_data_messages_still_excluded(self):
        for message in (
            "hello there", "thanks!", "who are you",
            "what is the weather today", "tell me a joke",
        ):
            self.assertFalse(_looks_like_data_request(message), message)


class MonthEndGrainTests(unittest.TestCase):
    """F — period-close wording is an explicit grain request."""

    def test_month_end_is_a_month_grain(self):
        self.assertEqual(
            requested_temporal_grain("Show month-end inventory value for February 2026."),
            "month",
        )

    def test_period_close_synonyms(self):
        self.assertEqual(requested_temporal_grain("end of month balance"), "month")
        self.assertEqual(requested_temporal_grain("quarter-end inventory"), "quarter")
        self.assertEqual(requested_temporal_grain("year end stock"), "year")

    def test_daily_wording_unchanged(self):
        self.assertEqual(
            requested_temporal_grain(
                "What was inventory value by warehouse on the latest daily snapshot?"
            ),
            "day",
        )

    def test_non_temporal_question_has_no_grain(self):
        self.assertEqual(requested_temporal_grain("Show revenue by region."), "")

    def test_month_end_question_selects_the_monthly_role(self):
        """The narrowing rule in resolve_metric_date_context, in miniature.

        Before the fix the grain came back blank, the finest-grain branch ran,
        and the monthly period role was dropped from the options entirely.
        """
        roles = [
            {"name": "Snapshot Date", "fact_column": "SNAPSHOT_DATE",
             "date_key_type": "native_date"},
            {"name": "Snapshot Date", "fact_column": "SNAPSHOT_YYYYMMDD",
             "date_key_type": "yyyymmdd_integer"},
            {"name": "Inventory Period", "fact_column": "PERIOD_YYYYMM",
             "date_key_type": "yyyymm_integer", "temporal_grain": "month"},
        ]

        def narrow(question, discovered):
            grain = requested_temporal_grain(question)
            if question_has_snapshot_intent(question):
                known = [
                    _role_temporal_grain(r) for r in discovered
                    if _role_temporal_grain(r)
                ]
                if known and not grain:
                    finest = min(known, key=lambda g: _GRAIN_ORDER.get(g, 99))
                    discovered = [
                        r for r in discovered if _role_temporal_grain(r) == finest
                    ]
                elif known and grain in known:
                    discovered = [
                        r for r in discovered if _role_temporal_grain(r) == grain
                    ]
            return [r["fact_column"] for r in discovered]

        self.assertEqual(
            narrow("Show month-end inventory value for February 2026.", list(roles)),
            ["PERIOD_YYYYMM"],
        )
        self.assertEqual(
            narrow(
                "What was inventory value by warehouse on the latest daily snapshot?",
                list(roles),
            ),
            ["SNAPSHOT_DATE", "SNAPSHOT_YYYYMMDD"],
        )


if __name__ == "__main__":
    unittest.main()


class PeriodComparisonNarrationTests(unittest.TestCase):
    """Live case 3 — correct data, contradicting summary.

    The comparison SQL returns one wide row of current/previous pairs. Treated
    as a time series, first and last period are the same cell, so the summary
    read "trended flat 0.0% from 2026-03 to 2026-03" while the table showed
    2026-03 $500.00 against 2026-02 $400.00, +25%. A wrong summary over right
    data is worse than an error: nothing signals the reader to distrust it.
    """

    LIVE_ROW = {
        "CURRENT_MONTH": "2026-03", "CURRENT_REVENUE": 500.0,
        "PREVIOUS_MONTH": "2026-02", "PREVIOUS_REVENUE": 400.0,
        "DIFFERENCE": 100, "PCT_CHANGE": 25.0,
    }

    def test_live_row_is_recognised_as_a_comparison(self):
        found = _period_comparison_from_rows([self.LIVE_ROW])
        self.assertIsNotNone(found)
        self.assertEqual(found["measure_column"], "CURRENT_REVENUE")
        self.assertEqual(found["current_period"], "2026-03")
        self.assertEqual(found["previous_period"], "2026-02")
        self.assertAlmostEqual(found["pct_change"], 25.0)

    def test_percentage_is_derived_when_absent(self):
        found = _period_comparison_from_rows([{
            "CURRENT_MONTH": "2026-03", "CURRENT_SALES": 80,
            "PREVIOUS_MONTH": "2026-02", "PREVIOUS_SALES": 100,
        }])
        self.assertAlmostEqual(found["pct_change"], -20.0)

    def test_label_pair_is_not_mistaken_for_the_measure(self):
        """CURRENT_MONTH/PREVIOUS_MONTH are periods, not the quantity."""
        found = _period_comparison_from_rows([self.LIVE_ROW])
        self.assertNotEqual(found["measure_column"], "CURRENT_MONTH")

    def test_non_comparison_shapes_are_left_alone(self):
        for rows in (
            [],
            [{"WAREHOUSE_NAME": "W01", "TOTAL_REVENUE": 600}],      # grouped
            [{"CURRENT_MONTH": "2026-03", "PREVIOUS_MONTH": "2026-02"}],  # labels only
            [dict(LIVE) for LIVE in (PeriodComparisonNarrationTests.LIVE_ROW,) * 2],  # series
        ):
            self.assertIsNone(_period_comparison_from_rows(rows), rows)


class ExplicitDateBeatsLatestSnapshotTests(unittest.TestCase):
    """Live case 8: 'daily inventory value for March 2, 2026' returned 1485.

    1485 is the 2026-03-03 total. The stated date was dropped and replaced
    with MAX(snapshot date), and the newest snapshot was reported at 75/100
    confidence with no caveat -- a wrong number presented as a good one.

    The cause was an inference, not the model: the question carries snapshot
    intent (semi-additive stock measure) and detect_temporal_window() only
    recognises *relative* wording, so an absolute date produced no window and
    the implicit latest_snapshot branch took over.
    """

    INVENTORY_BINDING = {
        "id": 7,
        "metric_id": 7,
        "metric_name": "Daily Inventory Value",
        "context_name": "Snapshot Date",
        "date_role": "snapshot_date",
        "fact_table": "QBOT_LIVE_TEST.F_INVENTORY_DAILY",
        "fact_column": "SNAPSHOT_DATE",
        "dimension_table": "",
        "dimension_key": "",
        "date_value_column": "SNAPSHOT_DATE",
        "date_key_type": "native_date",
        "is_default": 1,
        "priority": 50,
    }

    def _policies(self, question):
        plan = build_contextual_date_plan(self.INVENTORY_BINDING, question)
        return plan.get("temporal_policies") or []

    def test_the_live_case_8_question_no_longer_anchors_to_latest(self):
        self.assertEqual(self._policies("Show daily inventory value for March 2, 2026."), [])

    def test_latest_snapshot_wording_still_anchors(self):
        """The inference must survive for questions that really do mean 'newest'."""
        policies = self._policies(
            "What was inventory value by warehouse on the latest daily snapshot?"
        )
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["kind"], "latest_snapshot")
        self.assertEqual(policies[0]["anchor_policy"], "latest_available")

    def test_relative_wording_is_untouched(self):
        """detect_temporal_window() runs first; the guard must not shadow it."""
        policies = self._policies("show inventory value for the current month")
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["kind"], "this_month")

    def test_explicit_date_wording_is_recognised(self):
        for question in (
            "Show daily inventory value for March 2, 2026.",
            "month-end inventory value for February 2026",
            "inventory on 2026-03-02",
            "inventory on 03/02/2026",
            "inventory for period 202602",
            "stock on hand in january",
            "balance for 2 march 2026",
        ):
            self.assertTrue(question_has_explicit_date_filter(question), question)

    def test_relative_and_bare_questions_are_not_explicit(self):
        for question in (
            "What was inventory value by warehouse on the latest daily snapshot?",
            "inventory by warehouse",
            "current stock on hand",
            "show me the last 2 days of inventory",
            "month-end inventory value",
        ):
            self.assertFalse(question_has_explicit_date_filter(question), question)

    def test_may_and_march_need_a_number_to_count_as_months(self):
        """Both are ordinary English words; a bare one must not block the anchor."""
        self.assertFalse(question_has_explicit_date_filter("which warehouses may hold stock"))
        self.assertFalse(question_has_explicit_date_filter("stock we may need to march forward"))
        self.assertTrue(question_has_explicit_date_filter("inventory for may 2026"))
        self.assertTrue(question_has_explicit_date_filter("inventory for march 2"))


class StableDateOptionLabelTests(unittest.TestCase):
    """Live cases 7/8: the date picker offered three entries named the same.

        Snapshot At Date
        Snapshot Date (Day data 1)
        Snapshot Date (Day data 2)

    Nothing there lets a user choose, and the ordinal came from the order the
    bindings arrived in rather than from the role -- so clicking "(Day data 2)"
    bound SNAPSHOT_YYYYMMDD on one request and could bind SNAPSHOT_DATE on the
    next. A menu entry that does not always mean the same column is worse than
    an ugly one.
    """

    NATIVE = {
        "context_name": "Snapshot Date",
        "fact_table": "QBOT_LIVE_TEST.F_INVENTORY_DAILY",
        "fact_column": "SNAPSHOT_DATE",
        "date_key_type": "native_date",
        "temporal_grain": "day",
    }
    ENCODED = {
        "context_name": "Snapshot Date",
        "fact_table": "QBOT_LIVE_TEST.F_INVENTORY_DAILY",
        "fact_column": "SNAPSHOT_YYYYMMDD",
        "date_key_type": "yyyymmdd_integer",
        "temporal_grain": "day",
    }
    INVOICE = {
        "context_name": "Invoice Date",
        "fact_table": "QBOT_LIVE_TEST.F_SALES_INVOICE",
        "fact_column": "INVOICE_DATE_SK",
        "date_key_type": "surrogate_fk",
        "temporal_grain": "day",
    }

    def test_a_unique_business_name_is_left_alone(self):
        labels = _date_option_labels([self.INVOICE])
        self.assertEqual(list(labels.values()), ["Invoice Date"])

    def test_colliding_names_are_split_by_how_the_date_is_stored(self):
        labels = _date_option_labels([self.NATIVE, self.ENCODED])
        self.assertEqual(
            labels[("QBOT_LIVE_TEST.F_INVENTORY_DAILY", "SNAPSHOT_DATE")],
            "Snapshot Date (calendar date)",
        )
        self.assertEqual(
            labels[("QBOT_LIVE_TEST.F_INVENTORY_DAILY", "SNAPSHOT_YYYYMMDD")],
            "Snapshot Date (date code)",
        )

    def test_labels_do_not_depend_on_binding_order(self):
        """The regression: same roles, different arrival order, same labels."""
        forward = _date_option_labels([self.NATIVE, self.ENCODED, self.INVOICE])
        reverse = _date_option_labels([self.INVOICE, self.ENCODED, self.NATIVE])
        self.assertEqual(forward, reverse)

    def test_every_physical_role_gets_its_own_label(self):
        labels = _date_option_labels([self.NATIVE, self.ENCODED, self.INVOICE])
        self.assertEqual(len(set(labels.values())), len(labels))

    def test_a_repeated_binding_does_not_create_a_phantom_option(self):
        """list_metric_date_contexts can return one role under two metrics."""
        labels = _date_option_labels([self.NATIVE, dict(self.NATIVE), self.ENCODED])
        self.assertEqual(len(labels), 2)

    def test_identical_storage_falls_back_to_a_reproducible_ordinal(self):
        twin_a = {**self.NATIVE, "fact_column": "OPENED_ON"}
        twin_b = {**self.NATIVE, "fact_column": "CLOSED_ON"}
        forward = _date_option_labels([twin_a, twin_b])
        reverse = _date_option_labels([twin_b, twin_a])
        self.assertEqual(forward, reverse)
        self.assertEqual(len(set(forward.values())), 2)
        for label in forward.values():
            self.assertIn("calendar date", label)

    def test_no_label_leaks_a_column_or_table_name(self):
        """Clarification A forbids showing tables, columns, keys or SQL."""
        labels = _date_option_labels([self.NATIVE, self.ENCODED, self.INVOICE])
        for label in labels.values():
            lowered = label.lower()
            for leak in ("snapshot_", "invoice_date_sk", "yyyymmdd", "_sk", "f_", "qbot"):
                self.assertNotIn(leak, lowered, label)


class SurrogateKeyDisplayNameTests(unittest.TestCase):
    """Live cases 7/12: correct totals reported against raw surrogate keys.

        WAREHOUSE_SK  DAILY_INVENTORY_VALUE        CATEGORY_ID  TOTAL_REVENUE
        10            $1,200.00                    1,000        $760.00

    The planner already knows how to swap a key for its dimension's display
    name, but every gate on that path tested for "_KEY"/"_DMS_KEY" -- the
    Infor M3 convention. A Kimball star schema names its keys "_SK", so
    _key_prefix returned "", the finder bailed, and the upgrade never ran.
    A fix that only recognises one customer's naming is not a fix.
    """

    STAR = {
        "SALES.F_INVENTORY_DAILY": {
            "WAREHOUSE_SK": "int", "INVENTORY_VALUE": "decimal",
        },
        "SALES.D_WAREHOUSE": {"WAREHOUSE_SK": "int", "WAREHOUSE_NAME": "varchar"},
        "SALES.D_CATEGORY": {"CATEGORY_SK": "int", "CATEGORY_NAME": "varchar"},
    }

    def test_kimball_surrogate_key_resolves_to_its_display_name(self):
        found = _find_display_field_for_key(
            "WAREHOUSE_SK", "", "inventory by warehouse", self.STAR, None, ""
        )
        self.assertEqual(found["table"], "SALES.D_WAREHOUSE")
        self.assertEqual(found["column"], "WAREHOUSE_NAME")
        self.assertEqual(found["source_key_column"], "WAREHOUSE_SK")

    def test_the_m3_convention_still_works(self):
        """The path this originally served must not regress."""
        m3 = {
            "ERP.ITM_BAL_FCT": {"WHS_DMS_KEY": "int", "INV_VAL_AMT": "decimal"},
            "ERP.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_DSC": "varchar"},
        }
        found = _find_display_field_for_key(
            "WHS_DMS_KEY", "", "inventory by warehouse", m3, None, ""
        )
        self.assertEqual(found["column"], "WHS_DSC")

    def test_a_foreign_schema_and_convention_also_resolves(self):
        pharma = {
            "RWE.FACT_RX": {"PRESCRIBER_ID": "int", "TRX": "int"},
            "RWE.DIM_PRESCRIBER": {"PRESCRIBER_ID": "int", "PRESCRIBER_NAME": "varchar"},
        }
        found = _find_display_field_for_key(
            "PRESCRIBER_ID", "", "trx by prescriber", pharma, None, ""
        )
        self.assertEqual(found["column"], "PRESCRIBER_NAME")

    def test_display_and_degenerate_columns_are_not_keys(self):
        """_CODE/_CD/_NO are display or degenerate; treating them as keys
        would let a column become its own display field."""
        for column in ("WAREHOUSE_CODE", "STATUS_CD", "ORDER_NO", "INVENTORY_VALUE"):
            self.assertFalse(_is_key_column(column), column)

    def test_a_bare_suffix_is_not_a_key(self):
        for column in ("_SK", "_KEY", "_ID", ""):
            self.assertFalse(_is_key_column(column), column)

    def test_key_prefix_covers_every_recognised_suffix(self):
        self.assertEqual(_key_prefix("WAREHOUSE_SK"), "WAREHOUSE")
        self.assertEqual(_key_prefix("PRESCRIBER_ID"), "PRESCRIBER")
        self.assertEqual(_key_prefix("CUSTOMER_FK"), "CUSTOMER")
        self.assertEqual(_key_prefix("PRODUCT_KEY"), "PRODUCT")
        self.assertEqual(_key_prefix("WHS_DMS_KEY"), "WHS")
        self.assertEqual(_key_prefix("INVENTORY_VALUE"), "")

    def test_no_display_table_means_no_upgrade_rather_than_a_guess(self):
        orphan = {"SALES.F_ORDERS": {"CARRIER_SK": "int", "FREIGHT_AMT": "decimal"}}
        self.assertIsNone(
            _find_display_field_for_key("CARRIER_SK", "", "freight by carrier", orphan, None, "")
        )

    def test_asking_for_the_key_itself_is_respected(self):
        """The caller passes the business term; the question names the key."""
        self.assertIsNone(_find_display_field_for_key(
            "WAREHOUSE_SK", "warehouse", "show the warehouse sk", self.STAR, None, ""
        ))
        self.assertIsNone(_find_display_field_for_key(
            "WAREHOUSE_SK", "warehouse", "list the warehouse id", self.STAR, None, ""
        ))
        # ...but an ordinary breakdown still gets the name.
        self.assertIsNotNone(_find_display_field_for_key(
            "WAREHOUSE_SK", "warehouse", "inventory by warehouse", self.STAR, None, ""
        ))


class SurrogateKeyDisplayPromptRuleTests(unittest.TestCase):
    """Live cases 7/12 again: the planner never emits the dimension field, so
    the display-name resolver is never reached. The decision is made by the
    generator, which grouped by WAREHOUSE_SK / CATEGORY_SK -- valid SQL that
    reports integers to a user. The existing CROSS-TABLE rule does not bite
    because it assumes the grouping column lives in a dimension table; here
    the key sits on the fact, so grouping by it "works".
    """

    def _prompt(self, db_type="azure_sql"):
        from core.llm import build_sql_system_prompt
        return build_sql_system_prompt(db_type, "TABLE CONTEXT")

    def test_the_rule_is_present(self):
        self.assertIn("SURROGATE KEY DISPLAY RULE", self._prompt())

    def test_it_names_the_key_suffixes_the_resolver_recognises(self):
        prompt = self._prompt()
        for suffix in ("_SK", "_KEY", "_ID", "_FK", "_DMS_KEY"):
            self.assertIn(suffix, prompt, suffix)

    def test_it_names_the_display_suffixes_the_resolver_looks_for(self):
        prompt = self._prompt()
        for suffix in ("_NAME", "_DESC", "_DSC", "_DESCRIPTION", "_NM"):
            self.assertIn(suffix, prompt, suffix)

    def test_it_keeps_the_explicit_key_request_escape_hatch(self):
        """Must agree with _question_asks_for_key, or the two layers fight."""
        prompt = self._prompt().lower()
        self.assertIn("explicitly asked for the key", prompt)

    def test_it_forbids_inventing_a_dimension(self):
        prompt = self._prompt().lower()
        self.assertIn("never invent a table or column", prompt)

    def test_the_rule_reaches_every_dialect(self):
        for db_type in ("azure_sql", "postgres", "snowflake"):
            self.assertIn("SURROGATE KEY DISPLAY RULE", self._prompt(db_type), db_type)
