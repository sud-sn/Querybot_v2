"""The public sample warehouses are read as a warehouse would hold them, and their truth holds.

The benchmark scores Learn on three databases designed by other people (a music store, a
trading company, a DVD rental chain). Their files are fetched once, outside the repository,
and checked against a hash; the truth about them is written by hand. These checks keep that
honest: the SQLite reader gives each column the type a warehouse would (dates as dates, money
as decimals, pictures left out), a file that is not the one the truth was written for is
refused, and, where the files are cached, every link, key, date, metric and label in the
truth is checked against the rows.

The reader and the refusals run on databases built here; the truth checks need the cached
files (python -m evals.core2.public fetch) and say so when they are absent.
"""

from __future__ import annotations

import datetime as dt
import io
import sqlite3

import pandas as pd
import pytest

from evals.core2 import benchmark, public
from evals.core2.framework import materialize


@pytest.mark.parametrize("name,expected", [
    ("InvoiceLine", "invoice_line"), ("CustomerID", "customer_id"), ("Order Details", "order_details"),
    ("SupportRepId", "support_rep_id"), ("ShipVia", "ship_via"), ("last_update", "last_update"),
    ("HomePage", "home_page"), ("address2", "address2"),
])
def test_names_are_read_in_snake_case(name, expected):
    assert public.snake(name) == expected


def _shop(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE "Shop Orders" (OrderID INTEGER PRIMARY KEY, CustomerCode TEXT, OrderDate DATETIME,
                                    DueOn DATE, Total NUMERIC(10,2), Freight NUMERIC, Lines NUMERIC,
                                    Rate REAL, Logo BLOB, Notes "BLOB SUB_TYPE TEXT", Secret TEXT);
        INSERT INTO "Shop Orders" VALUES (1, 'ALFKI', '2024-01-05 10:00:00', '2024-01-20', 12.50, 3.25, 2, 0.1,
                                          x'00', 'first', 's1');
        INSERT INTO "Shop Orders" VALUES (2, NULL, '2024-02-06 11:30:00.000', '2024-02-21', 7.00, 1, 5, 0.2,
                                          NULL, NULL, 's2');
        CREATE TABLE OrderLines (OrderID INTEGER, LineNo INTEGER, Qty INTEGER, PRIMARY KEY (OrderID, LineNo));
        INSERT INTO OrderLines VALUES (1, 1, 3), (1, 2, NULL);
        CREATE TABLE Unused (Id INTEGER PRIMARY KEY);
    """)
    con.commit()
    con.close()


def test_a_sqlite_database_is_read_with_warehouse_types(tmp_path):
    path = tmp_path / "shop.sqlite"
    _shop(path)
    tables = public.read_sqlite(path, left_out={"shop_orders.secret"})
    assert set(tables) == {"shop_orders", "order_lines"}            # a table with no rows is left out
    frame, types, key = tables["shop_orders"]
    assert key == ["order_id"]
    assert types == {"order_id": "INTEGER", "customer_code": "VARCHAR", "order_date": "TIMESTAMP", "due_on": "DATE",
                     "total": "DECIMAL(10,2)", "freight": "DECIMAL(18,4)", "lines": "INTEGER", "rate": "DOUBLE",
                     "notes": "VARCHAR"}                              # the picture and the secret are not read
    assert frame["order_date"].iloc[1] == pd.Timestamp("2024-02-06 11:30:00")
    assert frame["due_on"].iloc[0] == dt.date(2024, 1, 20)
    assert pd.isna(frame["customer_code"].iloc[1]) and frame["customer_code"].iloc[0] == "ALFKI"
    lines, line_types, line_key = tables["order_lines"]
    assert line_key == ["order_id", "line_no"]
    assert str(lines["qty"].dtype) == "Int64" and pd.isna(lines["qty"].iloc[1])


def test_a_read_database_loads_into_every_naming_style(tmp_path):
    """What the reader gives is what the synthetic domains give: materialize takes it in every style."""
    from evals.core2.framework import Domain, TableDef, Truth
    from evals.core2.naming import STYLES

    path = tmp_path / "shop.sqlite"
    _shop(path)
    tables = [TableDef(name, "fact", frame, types, key) for name, (frame, types, key) in public.read_sqlite(path).items()]
    domain = Domain("shop", tables, Truth({t.name: t.kind for t in tables}, {}, [], None, [], [], {}),
                    today=dt.date(2024, 3, 1))
    for style in STYLES:
        built = materialize(domain, style)
        table, code = built.c("shop_orders.customer_code")
        rows, codes = built.con.execute(f'SELECT COUNT(1), COUNT("{code}") FROM "{table}"').fetchone()
        assert (rows, codes) == (2, 1)                  # a missing code is NULL in the warehouse, not "nan"


def test_a_file_that_is_not_the_one_checked_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYBOT_BENCHMARK_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch chinook"):
        public.build("chinook")
    (tmp_path / public.SOURCES["chinook"].file).write_bytes(b"not the music store")
    assert public.cached() == ["chinook"]
    with pytest.raises(ValueError, match="not the file the truth was written for"):
        public.build("chinook")


def test_a_download_with_another_hash_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYBOT_BENCHMARK_DATA", str(tmp_path))

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(public.urllib.request, "urlopen", lambda *a, **k: Response(b"a different file"))
    with pytest.raises(ValueError, match="hash"):
        public.fetch(["sakila"])
    assert list(tmp_path.iterdir()) == []


def test_the_benchmark_takes_in_only_what_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYBOT_BENCHMARK_DATA", str(tmp_path))
    kinds = {kind for _, kind in benchmark._sources(None)}
    assert kinds == {"gated", "benchmark"}


# ── the truth against the cached rows ───────────────────────────────────────


def _cached(name):
    if name not in public.cached():
        pytest.skip(f"{name} is not fetched (python -m evals.core2.public fetch)")
    return public.build(name)


@pytest.mark.parametrize("name", list(public.SOURCES))
def test_the_truth_holds_against_the_rows(name):
    domain = _cached(name)
    built = materialize(domain, "descriptive")
    truth = domain.truth
    columns = {t.name: set(t.data.columns) for t in domain.tables}

    def exists(ref):
        table, _, column = ref.partition(".")
        return column in columns.get(table, ())

    for table, key in truth.primary_keys.items():
        assert key, f"{table} has no primary key"
        cols = ", ".join(f'"{c}"' for c in key)
        total, distinct = built.con.execute(f'SELECT COUNT(1), COUNT(DISTINCT ({cols})) FROM "{table}"').fetchone()
        assert total == distinct, (table, key)
    for j in truth.joins:
        assert exists(f"{j.from_table}.{j.from_column}") and exists(f"{j.to_table}.{j.to_column}"), j
        rows, matched = built.con.execute(
            f'SELECT COUNT(1), COUNT(t."{j.to_column}") FROM "{j.from_table}" f LEFT JOIN "{j.to_table}" t '
            f'ON t."{j.to_column}" = f."{j.from_column}" WHERE f."{j.from_column}" IS NOT NULL').fetchone()
        assert rows and matched / rows >= 0.99, (j, matched, rows)
    defaults: dict[str, int] = {}
    for d in truth.dates:
        assert exists(f"{d.table}.{d.column}"), d
        assert domain.table(d.table).types[d.column] in ("DATE", "TIMESTAMP"), d
        defaults[d.table] = defaults.get(d.table, 0) + d.default
        assert not (d.kind == "audit" and d.default), d
    assert all(n <= 1 for n in defaults.values()), defaults
    for m in truth.measures:
        assert m.table in columns and (m.column is None or exists(f"{m.table}.{m.column}")), m
    for ref in truth.not_measures:
        assert exists(ref), ref
    for table, label in truth.labels.items():
        distinct = built.con.execute(f'SELECT COUNT(DISTINCT "{label}") = COUNT(1) FROM "{table}"').fetchone()[0]
        assert distinct, f"{table}.{label} does not name one row each"


@pytest.mark.parametrize("name", list(public.SOURCES))
def test_every_link_in_the_rows_is_in_the_truth(name):
    """The truth lists each declared link that holds data: a link the learner finds that the
    publisher declared is never scored as invented."""
    domain = _cached(name)
    path = public.cache_dir() / public.SOURCES[name].file
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    known = {(j.from_table, j.from_column, j.to_table) for j in domain.truth.joins}
    present = {t.name for t in domain.tables}
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        for fk in con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
            ref = (public.snake(table), public.snake(fk[3]), public.snake(fk[2]))
            if ref[0] not in present or ref[2] not in present:
                continue                    # a link of a table with no rows
            filled = con.execute(f'SELECT COUNT("{fk[3]}") FROM "{table}"').fetchone()[0]
            assert ref in known or not filled, ref
