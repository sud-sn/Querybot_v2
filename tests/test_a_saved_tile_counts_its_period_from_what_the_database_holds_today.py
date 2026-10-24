"""A saved tile counts its period from what the database holds today, not from the day Learn read it.

A dashboard tile keeps its question's plan and runs it again each time it is opened, emailed or checked for an
alert. "This month" and "the last 7 days" count back from today -- or, when the data stopped more than a week
before, from the day after it ends, and say so. Where the data ends was the last day Learn read: on a database
loaded every day, a dashboard opened a month after Learn counted every period back from the day Learn ran, said
"the data ends before today", and showed that month's figures on every day after. Now a date whose recorded end
is more than a week old is read again from the database (its latest real day up to today), in the background, one
read at a time, again some hours later, and the tile counts from it. A database that really stopped still counts
back from its last day.

Invented data only: twenty customers and six hundred orders, the last of them thirty days ago when Learn runs.
"""

from __future__ import annotations

import datetime as dt
import types
from collections import OrderedDict
import uuid

import pytest
import sqlglot

TODAY = dt.date.today()


@pytest.fixture
def shop(tmp_path, monkeypatch):
    import duckdb
    import numpy as np
    import pandas as pd

    import store
    import store.crypto
    from core.compliance import governed_query as gq
    from core2 import service
    from core2.bootstrap.service import build_workspace
    from core2.warehouse import querybot
    from tests.test_core2_learned_page_shows_what_was_learned import _Connection, _schema_file

    rng = np.random.default_rng(11)
    con = duckdb.connect()
    n = 600
    first = TODAY - dt.timedelta(days=400)
    frames = {
        "customers": pd.DataFrame({"customer_id": np.arange(1, 21),
                                   "customer_name": [f"Customer {i:02d}" for i in range(1, 21)]}),
        "orders": pd.DataFrame({
            "order_id": np.arange(1, n + 1), "customer_id": rng.integers(1, 21, n),
            "order_date": [first + dt.timedelta(days=int(d)) for d in np.sort(rng.integers(0, 371, n))],
            "order_amount": np.round(rng.uniform(10, 900, n), 2)}),
    }
    frames["orders"].loc[n - 1, "order_date"] = TODAY - dt.timedelta(days=30)     # Learn's last day
    for name, frame in frames.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    con.execute("ALTER TABLE orders ALTER order_date TYPE DATE")
    built = types.SimpleNamespace(con=con, tables={t: t for t in frames}, declared_pks={}, declared_fks=[])

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = f"acct-dates-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "web")
    db_id = store.save_db_config("azure_sql", "shop", {"server": "s", "database": "d", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    schema_dir = tmp_path / "clients" / account / "schema"
    store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    reads = []

    def warehouse(db_type, credentials, **kw):
        reads.append(db_type)
        return _Connection(con)

    monkeypatch.setattr(querybot, "QueryBotWarehouse", warehouse)

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    # Before Learn: its question map checks each question on the database too, never on a server named "s".
    monkeypatch.setattr(gq, "run_query", run_query)
    build_workspace(account)
    monkeypatch.setattr(service, "_DATES", {})
    monkeypatch.setattr(service, "_MOVED", OrderedDict())
    from core2.bootstrap import ai

    # A saved tile asks no AI: the writers are only made, never called.
    for writer in ("workspace_planner", "metric_writer", "beyond_writer"):
        monkeypatch.setattr(ai, writer, lambda *a, **k: (lambda *x, **y: ""))
    admin = store.get_user(store.create_user(account, "Ada Admin", f"ada-{uuid.uuid4().hex[:6]}@test.com",
                                             role="admin", password="a-password-they-chose")[0])
    from core2.bootstrap.service import answering_model

    model = answering_model(account, store.get_client(account))
    amount = next(m.slug for m in model.measures.values() if "amount" in m.slug and "order" in m.slug)
    role = next(r for r in model.date_roles.values() if r.table.endswith("orders") and r.column.endswith("order_date"))
    assert role.last == TODAY - dt.timedelta(days=30), "Learn read the data as ending thirty days ago"

    def tile(window: dict) -> dict:
        reads.clear()
        return service.portal_replay(account, {"kind": "query", "intent": "value", "measures": [amount],
                                               "time": {"window": window}}, admin, question="q")

    def load(*days: int, amount_each: float = 1000.0) -> None:
        start = int(con.execute("SELECT MAX(order_id) FROM orders").fetchone()[0])
        for i, back in enumerate(days, start=1):
            con.execute("INSERT INTO orders VALUES (?, 1, ?, ?)",
                        [start + i, TODAY - dt.timedelta(days=back), amount_each])

    def truth(start: dt.date, end: dt.date) -> float:
        return round(float(con.execute("SELECT COALESCE(SUM(order_amount), 0) FROM orders WHERE order_date >= ? "
                                       "AND order_date < ?", [start, end]).fetchone()[0]), 2)

    return types.SimpleNamespace(account=account, tile=tile, load=load, truth=truth, reads=reads, service=service,
                                 model=model, role=role, db_id=db_id, store=store, con=con)


def _value(payload: dict) -> float:
    return round(float(payload["kpi"]["value"]), 2)


def test_the_last_7_days_hold_the_rows_loaded_since_learn(shop):
    shop.load(2, 1, 0)
    payload = shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    assert _value(payload) == shop.truth(TODAY - dt.timedelta(days=6), TODAY + dt.timedelta(days=1)) == 3000.0
    assert "counted back" not in str(payload), "the data does not end before today any more"


def test_this_month_is_todays_month(shop):
    shop.load(2, 1, 0)
    payload = shop.tile({"kind": "this", "unit": "month"})
    assert _value(payload) == shop.truth(TODAY.replace(day=1), TODAY + dt.timedelta(days=1)) > 0


def test_a_database_that_stopped_still_counts_back_from_its_last_day(shop):
    payload = shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    end = TODAY - dt.timedelta(days=29)
    assert _value(payload) == shop.truth(end - dt.timedelta(days=7), end)
    assert "counted back" in str(payload)


def _settled(shop) -> None:
    """Wait for the read under way, if any (it runs in the background)."""
    for kept in list(shop.service._DATES.values()):
        if kept.reading is not None:
            kept.reading.wait(10)


def test_the_last_day_is_read_again_some_hours_on(shop):
    import time

    key = (shop.account, shop.db_id, shop.model.version)
    shop.load(1)
    shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    _settled(shop)
    assert shop.service._DATES[key].days[shop.role.key] == TODAY - dt.timedelta(days=1), "read where it ends now"
    assert shop.reads == ["azure_sql"], "read once, under the service connection"
    shop.load(0, amount_each=500.0)
    again = shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    assert shop.reads == [], "not read again within the hours"
    assert shop.service._DATES[key].days[shop.role.key] == TODAY - dt.timedelta(days=1)
    assert _value(again) == 1500.0, "the new row is counted all the same: the window is today's"
    # Some hours on, it is read again, in the background: the tile is answered meanwhile.
    assert shop.service._DATES[key].due - time.monotonic() > (shop.service.DATES_HOURS - 0.1) * 3600
    shop.service._DATES[key].due = 0.0
    shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    _settled(shop)
    assert shop.service._DATES[key].days[shop.role.key] == TODAY, "read again after the hours"


def test_one_read_at_a_time_and_answered_meanwhile(shop, monkeypatch):
    """Twenty tiles opened at once on a database that is slow to answer: one read of its dates, not twenty; each
    tile is answered from the days Learn read after a short wait, the read going on for the next one."""
    import threading

    from core2.warehouse import querybot
    from tests.test_core2_learned_page_shows_what_was_learned import _Connection

    release, made = threading.Event(), []

    class Slow(_Connection):
        def query(self, sql, max_rows=None):
            release.wait(20)
            return super().query(sql, max_rows=max_rows)

    def warehouse(db_type, credentials, **kw):
        made.append(kw.get("timeout_seconds"))
        return Slow(shop.con)

    monkeypatch.setattr(querybot, "QueryBotWarehouse", warehouse)
    shop.load(1)
    db_config = shop.store.get_db_config(shop.db_id)
    got: list = []
    callers = [threading.Thread(target=lambda: got.append(shop.service.current_dates(
        shop.account, shop.model, db_config, TODAY, wait=0.2))) for _ in range(20)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(10)
    assert len(got) == 20 and all(m is shop.model for m in got), "answered from the days Learn read meanwhile"
    assert made == [shop.service.DATES_QUERY_SECONDS], "one read, each query given a time limit"
    release.set()
    _settled(shop)
    moved = shop.service.current_dates(shop.account, shop.model, db_config, TODAY, wait=0)
    assert moved.date_roles[shop.role.key].last == TODAY - dt.timedelta(days=1)
    assert shop.service.current_dates(shop.account, shop.model, db_config, TODAY, wait=0) is moved, \
        "the same model while the days stay the same (what is kept by the model stays kept)"
    assert made == [shop.service.DATES_QUERY_SECONDS]


def test_a_day_after_today_is_not_where_the_data_ends(shop):
    """A row dated ahead (a delivery booked for next month) is not data yet: the data ends on the last day up to
    today, and "the last 7 days" are the seven up to today."""
    shop.load(1)
    shop.load(-40, amount_each=7.0)
    payload = shop.tile({"kind": "rolling", "unit": "day", "n": 7})
    _settled(shop)
    key = (shop.account, shop.db_id, shop.model.version)
    assert shop.service._DATES[key].days[shop.role.key] == TODAY - dt.timedelta(days=1)
    assert _value(payload) == 1000.0 and "counted back" not in str(payload)


def test_a_new_learn_starts_the_days_again(shop):
    """The days read are the model's: the next model (a new Learn) is read afresh, and the older one's are let go."""
    shop.load(1)
    db_config = shop.store.get_db_config(shop.db_id)
    shop.service.current_dates(shop.account, shop.model, db_config, TODAY)
    _settled(shop)
    newer = shop.model.model_copy(update={"version": int(shop.model.version or 0) + 1})
    shop.service.current_dates(shop.account, newer, db_config, TODAY)
    _settled(shop)
    assert [k for k in shop.service._DATES if k[0] == shop.account] == [(shop.account, shop.db_id, newer.version)]


def test_the_home_page_shows_where_the_data_ends_today(shop):
    """The welcome line says the data runs to the day the database holds now (once read, never waited for)."""
    from portal import home

    shop.load(1)
    client = shop.store.get_client(shop.account)
    user = {"account_id": shop.account, "id": 0, "name": "Ada", "role": "admin"}
    home.welcome(user, client, today=TODAY)
    _settled(shop)
    assert home.welcome(user, client, today=TODAY)["data_to"] == TODAY - dt.timedelta(days=1)


def test_a_database_that_cannot_be_read_counts_from_the_day_learn_read(shop, monkeypatch, caplog):
    """A database that cannot be read (paused, in its nightly window) is answered from the day Learn read, and is
    tried again minutes later, not hours: the tiles count from today as soon as it answers."""
    import time

    from core2.warehouse import querybot

    working = querybot.QueryBotWarehouse

    def down(db_type, credentials, **kw):
        raise RuntimeError("the database is paused")

    shop.load(1)
    monkeypatch.setattr(querybot, "QueryBotWarehouse", down)
    db_config = shop.store.get_db_config(shop.db_id)
    model = shop.service.current_dates(shop.account, shop.model, db_config, TODAY)
    assert model is shop.model and "could not be read" in caplog.text
    kept = shop.service._DATES[(shop.account, shop.db_id, shop.model.version)]
    assert kept.due - time.monotonic() <= shop.service.DATES_RETRY_SECONDS, "tried again in minutes"
    monkeypatch.setattr(querybot, "QueryBotWarehouse", working)
    assert shop.service.current_dates(shop.account, shop.model, db_config, TODAY) is shop.model, "not before then"
    kept.due = 0.0
    shop.service.current_dates(shop.account, shop.model, db_config, TODAY)
    _settled(shop)
    moved = shop.service.current_dates(shop.account, shop.model, db_config, TODAY, wait=0)
    assert moved.date_roles[shop.role.key].last == TODAY - dt.timedelta(days=1)


def test_moved_models_are_kept_for_so_many_workspaces(shop, monkeypatch):
    """The moved models kept are bounded, as the answering models are: a server answering many workspaces keeps
    the latest ones, not every one it ever answered."""
    monkeypatch.setattr(shop.service, "DATES_MODELS", 1)
    shop.load(1)
    db_config = shop.store.get_db_config(shop.db_id)
    shop.service.current_dates(shop.account, shop.model, db_config, TODAY)
    _settled(shop)
    first = shop.service.current_dates(shop.account, shop.model, db_config, TODAY, wait=0)
    assert first is not shop.model and len(shop.service._MOVED) == 1
    other = shop.model.model_copy(update={"version": int(shop.model.version or 0) + 7})
    shop.service._DATES[("another-workspace", 9, int(other.version))] = shop.service._Dates(
        days=dict(shop.service._DATES[(shop.account, shop.db_id, shop.model.version)].days), due=time_far())
    shop.service.current_dates("another-workspace", other, {**db_config, "id": 9}, TODAY, wait=0)
    assert list(shop.service._MOVED) == [("another-workspace", 9, int(other.version))]


def time_far() -> float:
    import time

    return time.monotonic() + 3600


def test_the_latest_day_is_read_as_answers_read_the_date():
    from core2.compile.compiler import latest_days_sql, text_date
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    built, model = learn(domains.build("retail"), "descriptive")
    roles = list(model.date_roles.values())
    compared = 0
    for sql, keys in latest_days_sql(model, roles, "duckdb"):
        for key, got in zip(keys, built.con.execute(sql).fetchone()):
            role = model.date_roles[key]
            last = got.date() if isinstance(got, dt.datetime) else got if isinstance(got, dt.date) else None
            if role.granularity != "month" and last is not None:
                assert last == role.last, (key, last, role.last)
                compared += 1
    assert compared
    plain: dict[str, list[str]] = {}
    for role in roles:
        if not role.calendar and not role.quarter_column and not text_date(model.columns[role.column]):
            plain.setdefault(role.table, []).append(role.key)
    queries = [sorted(keys) for _, keys in latest_days_sql(model, roles, "duckdb")]
    for table, keys in plain.items():
        assert sorted(keys) in queries, f"one pass over {table} for all its dates"


def test_a_number_that_is_no_date_is_not_the_latest_day():
    """A day kept as a yyyymmdd number: 13 40 of a year (no day) is never where the data ends."""
    from core2.bootstrap.dates import _parse_day
    from core2.compile.compiler import latest_days_sql
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    built, model = learn(domains.build("retail"), "descriptive")
    role = next(r for r in model.date_roles.values() if not r.calendar and not r.quarter_column
                and model.columns[r.column].data_type == "date" and r.last)
    column = model.columns[role.column]
    name = model.tables[role.table].name
    built.con.execute(f"ALTER TABLE {name} ADD COLUMN day_key INTEGER")
    built.con.execute(f"UPDATE {name} SET day_key = CAST(strftime({column.name}, '%Y%m%d') AS INTEGER)")
    built.con.execute(f"UPDATE {name} SET day_key = {role.last.year}1340 WHERE {column.name} = "
                      f"(SELECT MIN({column.name}) FROM {name})")
    key = f"{role.table}.day_key"
    model.columns[key] = column.model_copy(update={"key": key, "name": "day_key", "data_type": "integer"})
    number = role.model_copy(update={"key": "day_key_role", "column": key})
    (sql, keys), = latest_days_sql(model, [number], "duckdb", before=dt.date(role.last.year + 1, 1, 2))
    got = built.con.execute(sql).fetchone()[0]
    assert _parse_day(got, "day") == role.last, got


def test_a_date_kept_as_text_is_read_on_its_own():
    """A date kept as text is read in a query of its own: a value shaped like a date but none can fail its reading
    on some databases, and never keeps the table's other dates from moving."""
    from core2.compile.compiler import latest_days_sql
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    built, model = learn(domains.build("retail"), "descriptive")
    role = next(r for r in model.date_roles.values() if not r.calendar and not r.quarter_column
                and model.columns[r.column].data_type == "date" and r.last)
    column = model.columns[role.column]
    key = f"{role.table}.day_text"
    # As Learn keeps it: a date it found, in a column the database keeps as text.
    model.columns[key] = column.model_copy(update={"key": key, "name": "day_text", "data_type": "date",
                                                   "raw_type": "VARCHAR(10)"})
    text = role.model_copy(update={"key": "day_text_role", "column": key})
    queries = [keys for _, keys in latest_days_sql(model, [role, text], "duckdb")]
    assert sorted(queries) == [["day_text_role"], [role.key]]
