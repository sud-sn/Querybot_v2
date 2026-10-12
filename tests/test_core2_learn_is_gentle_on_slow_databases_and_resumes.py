"""Learn on a small or busy database: a table too slow to read whole is read on a sample at once,
Learn reads as many tables at a time as the server is set to, and a Learn a restart stopped starts
where it was.

On a server test, Learn read each table of a large database whole; the warehouse stopped one at its
time limit, and Learn then asked for each of its columns alone, each waiting out the time limit again:
a wide table took the limit times its columns, and an update that restarted QueryBot meant reading every
table from the first again. Each table read is now kept as Learn goes; the next Learn uses the ones whose
columns, row count and masking are as they were, when the Learn before it did not finish.

The build runs as production runs it, with the synthetic retail warehouse in DuckDB standing in for the
connection (tests/test_core2_learned_page_shows_what_was_learned.py).
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core2.bootstrap.inventory import from_schema_json
from core2.bootstrap.journal import Journal, Watched, timed_out
from core2.bootstrap.profiler import ProfileOptions, profile_table
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_core2_learned_page_shows_what_was_learned import (  # noqa: F401 - the fixture
    ACCOUNT,
    _schema_file,
    workspace,
)

AZURE_TIMEOUT = "('HYT00', '[HYT00] [Microsoft][ODBC Driver 18 for SQL Server]Query timeout expired (0) (SQLExecDirectW)')"


def _largest(built, tmp_path):
    folder = tmp_path / "schema"
    _schema_file(built, folder)
    inventory = from_schema_json(json.loads((folder / "_schema.json").read_text()), "azure_sql")
    return max(inventory.tables.values(), key=lambda t: t.row_count or 0)


class _Busy:
    """The warehouse, but a query over the whole of one table runs past the time limit (or is refused):
    every one, or only the first (the query of all its columns at once)."""

    def __init__(self, con, table: str, error: str, *, once: bool = False):
        self.inner = DuckDBWarehouse(con)
        self.table, self.error, self.once = table, error, once
        self.stopped: list[str] = []

    dialect = "duckdb"
    db_type = "duckdb"

    def query(self, sql: str, *, max_rows: int | None = None):
        whole = self.table in sql and "SAMPLE" not in sql.upper()
        if whole and not (self.once and self.stopped):
            self.stopped.append(sql)
            raise RuntimeError(self.error)
        return self.inner.query(sql, max_rows=max_rows)


@pytest.fixture
def retail():
    from evals.core2.domains import retail as domain
    from evals.core2.framework import materialize

    return materialize(domain.build(), "descriptive")


def test_a_table_too_slow_to_read_whole_is_read_on_a_sample_at_once(retail, tmp_path):
    table = _largest(retail, tmp_path)
    busy = _Busy(retail.con, table.name, AZURE_TIMEOUT)
    lines: list[str] = []
    profile = profile_table(Watched(busy, Journal(lines.append)), table, ProfileOptions(sample_rows=500))

    assert len(busy.stopped) == 1                       # not once more for each of its columns
    assert profile.sampled and not profile.unread
    assert set(profile.columns) == {c.name for c in table.columns}
    assert any(f"reading all {table.row_count:,} rows took longer than the database allows" in line
               and "reading a sample of about 500 rows instead" in line for line in lines)


def test_a_refusal_that_is_not_about_time_is_still_found_column_by_column(retail, tmp_path):
    table = _largest(retail, tmp_path)
    busy = _Busy(retail.con, table.name, "Conversion failed when converting the varchar value", once=True)
    lines: list[str] = []
    profile = profile_table(Watched(busy, Journal(lines.append)), table, ProfileOptions(sample_rows=500))

    assert not profile.sampled and len(busy.stopped) == 1
    assert set(profile.columns) == {c.name for c in table.columns}
    assert not any("reading a sample" in line for line in lines)


def test_a_table_no_larger_than_a_sample_is_not_read_again_as_one(retail, tmp_path):
    table = _largest(retail, tmp_path)
    busy = _Busy(retail.con, table.name, AZURE_TIMEOUT, once=True)
    profile = profile_table(Watched(busy, Journal()), table, ProfileOptions(sample_rows=table.row_count))
    assert not profile.sampled and set(profile.columns) == {c.name for c in table.columns}


@pytest.mark.parametrize("message", [
    AZURE_TIMEOUT,
    "000630 (57014): Statement reached its statement or warehouse timeout of 120 second(s) and was canceled.",
    "DPY-4024: call timeout of 120000 ms exceeded",
    "ORA-03156: OCI call timed out",
    "canceling statement due to statement timeout",
])
def test_each_warehouse_saying_its_time_ran_out_is_understood(message):
    assert timed_out(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "Invalid column name 'X'.", "Conversion failed when converting the varchar value 'a' to data type int.",
    "SQL compilation error: invalid identifier 'X'", "ORA-00904: invalid identifier",
])
def test_a_refusal_is_not_taken_for_the_time_running_out(message):
    assert not timed_out(RuntimeError(message))


@pytest.mark.parametrize("value, workers, seconds", [
    (None, 4, 120), ("1", 1, 10), ("2", 2, 10), ("9000", 16, 3600), ("0", 1, 10), ("many", 4, 120),
])
def test_how_many_tables_at_a_time_and_how_long_each_query_may_run_are_settings(monkeypatch, value, workers,
                                                                                seconds):
    from core2.bootstrap.service import learn_workers
    from core2.warehouse.querybot import QueryBotWarehouse, learn_timeout_seconds

    for name in ("QUERYBOT_LEARN_WORKERS", "QUERYBOT_LEARN_QUERY_TIMEOUT_SECONDS"):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert learn_workers() == workers
    assert learn_timeout_seconds() == seconds
    assert QueryBotWarehouse("azure_sql", {}).timeout_seconds == seconds


# --- A Learn a restart stopped starts where it was ---------------------------------------------------------

READ_AGAIN = r"\(\d+ of \d+\): [\d,]+ rows"


@pytest.fixture
def clock(monkeypatch):
    """Each build a minute after the one before it (a real Learn takes minutes; the store counts in seconds)."""
    import store.core2_store as core2_store

    now = [dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)]

    def tick() -> str:
        now[0] += dt.timedelta(seconds=7)
        return now[0].strftime("%Y-%m-%d %H:%M:%S")

    monkeypatch.setattr(core2_store, "_now", tick)
    return now


def _build(store, *, fail: str = "") -> tuple[dict, list[str]]:
    from unittest.mock import patch

    from core2.bootstrap import service

    if fail:
        with patch.object(store, "save_core2_model", side_effect=RuntimeError(fail)), pytest.raises(RuntimeError):
            service.build_workspace(ACCOUNT)
    else:
        service.build_workspace(ACCOUNT)
    build = store.latest_core2_build(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    return build, [line[9:] for line in build["log"].splitlines() if line]


def _kept(store) -> int:
    from store.db import get_db

    with get_db() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM core2_learn_table WHERE account_id = ?",
                                (ACCOUNT,)).fetchone()["n"])


def _model(store) -> dict:
    found = store.load_core2_model(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    model = json.loads(found["model_json"])
    return {part: sorted(model.get(part) or {}) for part in ("tables", "columns", "measures", "joins", "attributes")}


def test_a_stopped_learn_is_picked_up_where_it_was(workspace, clock):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    tables = len(built.tables)

    build, said = _build(store, fail="QueryBot was restarted")
    assert build["status"] == "failed" and _kept(store) == tables

    build, said = _build(store)
    assert build["status"] == "done"
    assert f"Starting where the Learn that stopped was: {tables} table(s) it read are kept, unless their " \
           "columns or masking changed since" in said
    picked = [line for line in said if line.startswith("Read ") and line.endswith("as the Learn that stopped read it")]
    assert len(picked) == tables
    assert _kept(store) == 0                            # a finished Learn's tables are in its model
    resumed = _model(store)

    # The next Learn after a finished one reads everything again, and learns the same.
    build, said = _build(store)
    assert build["status"] == "done"
    assert not any("Learn that stopped" in line for line in said)
    assert sum(line.startswith("Read ") and re.search(READ_AGAIN, line) is not None for line in said) == tables
    assert _model(store) == resumed


def test_a_table_whose_shape_changed_since_is_read_again(workspace, clock):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    _build(store, fail="QueryBot was restarted")

    schema = json.loads((schema_dir / "_schema.json").read_text())
    changed = next(key for key in schema if not key.startswith("__"))
    schema[changed]["row_count"] += 1                   # discovery ran again and the table had grown
    (schema_dir / "_schema.json").write_text(json.dumps(schema))

    build, said = _build(store)
    name = changed.split(".")[-1]
    assert build["status"] == "done"
    mine = [line for line in said if line.startswith(f"Read {name} (")]
    assert len(mine) == 1 and not mine[0].endswith("as the Learn that stopped read it")
    assert sum(line.endswith("as the Learn that stopped read it") for line in said) == len(built.tables) - 1


def test_tables_kept_from_a_learn_more_than_a_day_ago_are_read_again(workspace, clock):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    _build(store, fail="QueryBot was restarted")
    from store.db import get_db

    with get_db() as conn:
        conn.execute("UPDATE core2_learn_table SET read_at = '2000-01-01 00:00:00'")
    build, said = _build(store)
    assert build["status"] == "done" and not any("Learn that stopped" in line for line in said)
