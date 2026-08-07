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
    metrics_are_semi_additive,
    question_has_snapshot_intent,
    requested_temporal_grain,
)
from core.date_roles import _label_from_column  # noqa: E402
from core.dispatcher import _looks_like_data_request  # noqa: E402
from core.semantic_model import (  # noqa: E402
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
