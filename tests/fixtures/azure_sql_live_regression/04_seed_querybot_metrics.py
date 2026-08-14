"""Seed governed metrics for the Azure SQL live-regression client.

Run from the QueryBot repository root after schema discovery / KB generation:

    python tests/fixtures/azure_sql_live_regression/04_seed_querybot_metrics.py --account Test_Az

The script is idempotent. It updates only metrics with the names declared
below for the supplied account and does not touch users or tenant governance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import store  # noqa: E402


FACT_TABLE = "QBOT_LIVE_TEST.F_SALES_INVOICE"
DATE_TABLE = "QBOT_LIVE_TEST.D_DATE"

METRICS = (
    {
        "name": "Revenue",
        "synonyms": "sales revenue, net revenue, invoiced revenue, sales value",
        "sql_template": "SUM(NET_REVENUE_AMOUNT)",
        "description": "Net invoiced sales revenue after discounts.",
        "formula_type": "expression",
        "result_format": "currency",
        "required_columns": "NET_REVENUE_AMOUNT",
        "allowed_dimensions": "customer, warehouse, product, region, invoice date",
        "example_questions": (
            "What is total revenue?\n"
            "Show revenue trend for the last 6 months\n"
            "Show the top 10 warehouses by revenue"
        ),
        "grain": "invoice line; aggregatable by governed dimensions",
        "category": "Sales",
        "default_time_column": "INVOICE_DATE_SK",
        "base_entity": "Sales Invoice",
        "base_table": FACT_TABLE,
        "owner": "live-regression-fixture",
    },
    {
        "name": "Gross Sales",
        "synonyms": "gross revenue, gross invoice amount, sales before discounts",
        "sql_template": "SUM(GROSS_AMOUNT)",
        "description": "Gross invoiced sales amount before discounts.",
        "formula_type": "expression",
        "result_format": "currency",
        "required_columns": "GROSS_AMOUNT",
        "allowed_dimensions": "customer, warehouse, product, region, invoice date",
        "example_questions": "Show gross sales and revenue trend for the last 6 months",
        "grain": "invoice line; aggregatable by governed dimensions",
        "category": "Sales",
        "default_time_column": "INVOICE_DATE_SK",
        "base_entity": "Sales Invoice",
        "base_table": FACT_TABLE,
        "owner": "live-regression-fixture",
    },
    {
        "name": "Discount Amount",
        "synonyms": "sales discount, customer discount, invoice discount",
        "sql_template": "SUM(DISCOUNT_AMOUNT)",
        "description": "Total discount amount applied to invoiced sales lines.",
        "formula_type": "expression",
        "result_format": "currency",
        "required_columns": "DISCOUNT_AMOUNT",
        "allowed_dimensions": "customer, warehouse, product, region, invoice date",
        "example_questions": "Show discount amount by customer",
        "grain": "invoice line; aggregatable by governed dimensions",
        "category": "Sales",
        "default_time_column": "INVOICE_DATE_SK",
        "base_entity": "Sales Invoice",
        "base_table": FACT_TABLE,
        "owner": "live-regression-fixture",
    },
)


def seed(account_id: str, db_type: str = "azure_sql") -> list[tuple[str, int, str]]:
    existing = {
        str(metric.get("name") or "").casefold(): metric
        for metric in store.list_metrics(account_id, active_only=False)
    }
    results: list[tuple[str, int, str]] = []
    for definition in METRICS:
        current = existing.get(definition["name"].casefold())
        if current:
            metric_id = int(current["id"])
            store.update_metric(
                metric_id,
                {**definition, "is_active": 1},
                account_id=account_id,
                db_type=db_type,
            )
            action = "updated"
        else:
            metric_id = int(store.save_metric(account_id, dict(definition), db_type=db_type))
            action = "created"

        store.save_metric_date_context(account_id, {
            "metric_id": metric_id,
            "context_name": "Invoice Date",
            "aliases": "invoice date, invoiced date, billing date",
            "date_role": "invoice_date",
            "fact_table": FACT_TABLE,
            "fact_column": "INVOICE_DATE_SK",
            "dimension_table": DATE_TABLE,
            "dimension_key": "DATE_SK",
            "date_value_column": "FULL_DATE",
            "date_key_type": "surrogate_fk",
            "is_default": True,
            "priority": 100,
            "is_active": True,
        })
        saved = store.get_metric(metric_id) or {}
        results.append((definition["name"], metric_id, str(saved.get("metric_status") or action)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, help="QueryBot client account_id")
    parser.add_argument("--db-type", default="azure_sql")
    args = parser.parse_args()
    for name, metric_id, status in seed(args.account.strip(), args.db_type.strip()):
        print(f"{name}: id={metric_id} status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
