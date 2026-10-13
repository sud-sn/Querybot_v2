"""A compounding batch's status is a field to ask by, and a table of monthly targets is dated by its month.

Found by the sector stress test (October 2026):

* On a compounding pharmacy's database, the batches' BATCH_STATUS (released, rejected, quarantined) and
  BATCH_NUMBER were left out of what readers can ask by: "batch" is also a word of data loading (an ETL batch),
  and columns named with it were taken for load stamps. A word naming the table's own rows is its business.
* On a retail bank's database, the branches' monthly targets (TARGET_MONTH) had no date: a "target" date is a
  planned one, never a table's own. Its only date is the period each row plans for, so "the targets for 2025"
  and "actual against target by month" could not be told apart by period.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE formulas AS SELECT range + 1 AS formula_id, 'Formula ' || range || repeat('x', range % 5) AS formula_name
        FROM range(40);
        CREATE TABLE batches AS SELECT range + 1 AS batch_id, 'B' || lpad(range::VARCHAR, 6, '0') AS batch_number,
               range % 40 + 1 AS formula_id, DATE '2024-01-01' + (range % 600)::INTEGER AS compounded_date,
               ['Released', 'Released', 'Released', 'Rejected', 'Quarantined'][range % 5 + 1] AS batch_status,
               50 + range % 400 AS quantity_made FROM range(3000);
        CREATE TABLE branches AS SELECT range + 1 AS branch_id, 'Branch ' || range || repeat('!', range % 4) AS branch_name
        FROM range(12);
        CREATE TABLE branch_targets AS
        SELECT b.range + 1 AS branch_id, (DATE '2024-01-01' + INTERVAL (m.range) MONTH)::DATE AS target_month,
               20 + (b.range * 3 + m.range) % 15 AS new_accounts_target,
               1000000 + ((b.range * 7 + m.range) % 30) * 25000 AS loan_volume_target
        FROM range(12) b, range(24) m;
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-sx", options=BuildOptions(workers=1))


def test_a_batchs_status_is_a_field_to_ask_by(model):
    fields = {model.columns[a.column].name for a in model.attributes.values()}
    assert {"batch_status", "batch_number"} <= fields, sorted(fields)


def test_monthly_targets_are_dated_by_their_month(model):
    targets = next(t for k, t in model.tables.items() if k.endswith(".branch_targets"))
    assert targets.default_date and model.date_roles[targets.default_date].column.endswith("target_month")
