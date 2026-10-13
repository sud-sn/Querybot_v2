"""A sale's ticket is counted once, though the ticket number is the second column of the sales' key.

Found by the sector stress test on the TPC-DS retail benchmark (October 2026): store sales are keyed by item and
ticket number, one row per item sold on a ticket, and no table lists the tickets. A number in the second place of
a key was taken for a line number running within its document, so nothing counted tickets: "how many tickets"
counted lines (459,865 where there were 7,987). A column named as a document's number (a ticket number) is
counted once each wherever it sits in the key; a line number still counts nothing.

Invented rows only.
"""

from __future__ import annotations

import duckdb

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


def test_tickets_and_not_lines_are_counted():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE item AS SELECT range + 1 AS i_item_sk, 'Item ' || range || repeat('x', range % 3) AS i_item_desc
        FROM range(300);
        CREATE TABLE store_sales AS SELECT DATE '2024-01-01' + (t.range % 500)::INTEGER AS ss_sold_date,
               (t.range * 7 + l.range * 31) % 300 + 1 AS ss_item_sk, t.range + 1 AS ss_ticket_number,
               l.range % 5 + 1 AS ss_quantity, round(2 + (t.range * 13 + l.range) % 90 + 0.5, 2) AS ss_net_paid
        FROM range(4000) t, range(6) l;
        CREATE TABLE order_lines AS SELECT o.range + 1 AS order_id, l.range + 1 AS line_no,
               (o.range + l.range) % 300 + 1 AS item_sk, DATE '2024-01-01' + (o.range % 400)::INTEGER AS order_date,
               l.range + 2 AS qty FROM range(3000) o, range(4) l;
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-tix", options=BuildOptions(workers=1))
    counted = {m.business_name: m for m in model.measures.values() if getattr(m.expr, "agg", "") == "count_distinct"}
    tickets = counted.get("Number of tickets")
    assert tickets is not None and tickets.expr.column.endswith(".ss_ticket_number"), sorted(counted)
    assert not [n for n, m in counted.items() if m.expr.column.endswith(".line_no")]
