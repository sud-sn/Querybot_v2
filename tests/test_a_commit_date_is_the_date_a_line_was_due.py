"""A commit date is the date a line was promised by, written glued or not, and on-time is counted against it.

Found by the sector stress test on the TPC-H distribution benchmark (October 2026): "what % of them arrived late
(received after the commit date)" and "the overall on-time rate for 1996" were asked back or answered 0.0%. The
names are written in one case with the table's initials (L_COMMITDATE, L_RECEIPTDATE): Learn read COMMITDATE as
one word, so the commit date was an ordinary event and no date was known to be the one the line was due by. The
word before a glued date word is read too, and a commit date is a date promised.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.analyses import due_and_done
from core2.warehouse.runner import DuckDBWarehouse


def _model(columns: dict[str, str]):
    con = duckdb.connect()
    ship, commit, receipt = columns["ship"], columns["commit"], columns["receipt"]
    con.execute(f"""
        CREATE TABLE lineitem AS SELECT range // 4 + 1 AS l_orderkey, range % 4 + 1 AS l_linenumber,
               DATE '2024-01-01' + (range % 700)::INTEGER AS {ship},
               DATE '2024-01-01' + (range % 700)::INTEGER + (range % 9)::INTEGER AS {commit},
               DATE '2024-01-01' + (range % 700)::INTEGER + (range % 13)::INTEGER AS {receipt},
               1 + range % 50 AS l_quantity, round(100 + range % 900, 2) AS l_extendedprice FROM range(4000);
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-due", options=BuildOptions(workers=1))


@pytest.mark.parametrize("columns", [
    {"ship": "l_shipdate", "commit": "l_commitdate", "receipt": "l_receiptdate"},
    {"ship": "ship_date", "commit": "commit_date", "receipt": "receipt_date"},
])
def test_the_commit_date_is_the_date_due_and_the_receipt_the_date_done(columns):
    model = _model(columns)
    found = due_and_done(model)
    assert found is not None, sorted((r.slug, r.kind) for r in model.date_roles.values())
    due, done = found
    assert due.key.endswith("." + columns["commit"]) and done.key.endswith("." + columns["receipt"]), (due.key, done.key)


def test_a_word_that_only_ends_like_a_date_word_is_not_split():
    """UPDATE is no UP date: a word before the date word is read only when it has three letters or more."""
    from core2.bootstrap import dates

    assert "up" not in {w[:-len(d)] for w in ["update"] for d in dates._GLUED_DATE
                        if w.endswith(d) and len(w) - len(d) >= 3}
