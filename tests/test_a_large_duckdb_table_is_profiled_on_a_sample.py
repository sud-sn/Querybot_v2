"""A DuckDB table too large to read whole is profiled on a sample, its common values among what is read.

Found by the sector stress test (October 2026), on a distribution warehouse's order lines, six million rows:
Learn wrote DuckDB's sample as "USING SAMPLE n ROWS" after the table, and DuckDB reads that clause only at the
end of a query, after GROUP BY. Every query that groups -- the common values of a column, among them -- was a
syntax error on every table over the sampling threshold, and the ship modes, line statuses and instructions of
the order lines were never learned. The sample is now DuckDB's table sample, written after the table, where a
GROUP BY may follow it.

Invented rows only.
"""

from __future__ import annotations

import duckdb

from core2.bootstrap.inventory import InvColumn, InvTable
from core2.bootstrap.journal import Journal, Watched
from core2.bootstrap.profiler import ProfileOptions, profile_table
from core2.warehouse import dialect as D
from core2.warehouse.runner import DuckDBWarehouse

MODES = ["AIR", "MAIL", "RAIL", "SHIP", "TRUCK"]


def _lines(rows: int = 20_000) -> tuple[DuckDBWarehouse, InvTable]:
    con = duckdb.connect()
    con.execute(f"""CREATE TABLE order_lines AS
                    SELECT range AS line_id, range % 97 AS quantity,
                           CASE range % 5 {' '.join(f"WHEN {i} THEN '{m}'" for i, m in enumerate(MODES))} END AS ship_mode
                    FROM range({rows})""")
    table = InvTable(database="memory", schema="main", name="order_lines", row_count=rows,
                     columns=[InvColumn("line_id", "BIGINT", "integer"), InvColumn("quantity", "BIGINT", "integer"),
                              InvColumn("ship_mode", "VARCHAR", "text")])
    return DuckDBWarehouse(con), table


def test_the_common_values_of_a_sampled_table_are_read():
    warehouse, table = _lines()
    lines: list[str] = []
    profile = profile_table(Watched(warehouse, Journal(lines.append)), table,
                            ProfileOptions(sample_threshold=5_000, sample_rows=2_000))

    assert profile.sampled
    assert sorted(t.value for t in profile.columns["ship_mode"].top or []) == MODES
    assert not any("common values" in line for line in lines), lines
    assert any("TABLESAMPLE" in sql for sql in warehouse.log)


def test_a_duckdb_sample_may_be_followed_by_where_and_group_by():
    warehouse, _ = _lines(1_000)
    sample = D.sample_clause("duckdb", rows=100, total_rows=1_000)
    for sql in (f"SELECT ship_mode, COUNT(1) FROM order_lines {sample} GROUP BY ship_mode",
                f"SELECT COUNT(*) FROM order_lines {sample} WHERE quantity > 3",
                f"SELECT o.ship_mode FROM order_lines AS o {sample} GROUP BY o.ship_mode"):
        assert warehouse.query(sql).rows, sql
