"""The learned page shows what Learn is doing while it runs, and what it left out.

Learn on a real warehouse takes minutes, and the page said only "QueryBot is
studying the database" -- the same words beside a failure that had already
happened. Each step is now written as it happens (connecting, each table read,
keys, calendars, the joins tested, dates, measures, checks, naming, saving,
bringing over today's decisions, done or stopped), the page shows the steps and
adds new ones while Learn runs, and a run cut off by a service restart is said
to be interrupted instead of blocking the button.

The build runs as production runs it, with the synthetic retail warehouse in
DuckDB standing in for the connection (tests/test_core2_learned_page_shows_what_was_learned.py).
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import patch

from core2.warehouse.runner import DuckDBWarehouse
from tests.test_core2_learned_page_shows_what_was_learned import (  # noqa: F401 - the fixture
    ACCOUNT,
    _Connection,
    _page,
    _request,
    _schema_file,
    workspace,
)

STAMP = re.compile(r"^\d\d:\d\d:\d\d ")


def _lines(store):
    build = store.latest_core2_build(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    return build, [line for line in build["log"].splitlines() if line]


def _progress():
    from admin import core2_routes

    with patch.object(core2_routes, "_is_auth", return_value=True):
        response = asyncio.run(core2_routes.learned_progress(
            _request(f"/admin/clients/{ACCOUNT}/learned/progress", method="GET"), ACCOUNT))
    return json.loads(response.body)


def test_each_step_is_written_as_it_happens_and_the_page_keeps_them(workspace):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    build, lines = _lines(store)
    assert build["status"] == "done" and all(STAMP.match(line) for line in lines)
    said = [line[9:] for line in lines]
    tables = len(built.tables)
    for expected in ("Connecting to the azure sql database", f"Reading {tables} tables", "Finding each table's key",
                     "Looking for calendar and period tables", "Testing ", "Finding the dates each table is about",
                     "Sorting tables into events", "Checking the data for things worth a look",
                     "Saving what was learned", "Bringing over the decisions already made", "Done in "):
        assert any(line.startswith(expected) for line in said), expected
    assert sum(line.startswith("Read ") and f" of {tables}):" in line for line in said) == tables
    assert said[-1].startswith("Done in ") and ": version 1, " in said[-1] and "left out" not in said[-1]
    assert said.index("Saving what was learned") < len(said) - 1

    page = _page()
    assert "The last learning run" in page and f"{len(lines)} steps" in page
    assert "Learning now" not in page and "lq-log" in page and 'id="lq-log"' not in page   # folded, not live
    assert _progress() == {"building": False, "status": "done", "lines": lines}


def test_while_learn_runs_the_page_shows_the_steps_and_asks_for_more(workspace):  # noqa: F811
    store, _built, _schema_dir = workspace
    db_id = store.get_client(ACCOUNT)["db_config_id"]
    started = store.start_core2_build(ACCOUNT, db_id)
    store.add_core2_build_line(ACCOUNT, db_id, started, "Reading 9 tables")
    store.add_core2_build_line(ACCOUNT, db_id, started, "Read  ORDER_LINES (1 of 9):   4,000 rows")
    page = _page(query="saved=building")
    assert "Learning now" in page and "each step appears below as it happens" in page
    assert "What QueryBot is doing now" in page and 'id="lq-log"' in page
    assert "Reading 9 tables" in page and "Read ORDER_LINES (1 of 9): 4,000 rows" in page   # one line, spaces kept tidy
    assert f"/admin/clients/{ACCOUNT}/learned/progress" in page and "window.location.replace" in page
    assert "QueryBot is studying the database. Each step appears below" in page
    assert "has not studied this database yet" not in page   # it is studying it
    assert "disabled" in page.split("Learn this database")[0].rsplit("<button", 1)[1]
    progress = _progress()
    assert progress["building"] is True and progress["status"] == "running"
    assert [line[9:] for line in progress["lines"]] == ["Reading 9 tables", "Read ORDER_LINES (1 of 9): 4,000 rows"]

    store.finish_core2_build(ACCOUNT, db_id, started, status="done", version=1)
    page = _page(query="saved=building")
    assert "QueryBot is studying the database" not in page   # the notice goes with the run


def test_a_run_that_stops_says_where(workspace):  # noqa: F811
    store, _built, _schema_dir = workspace
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)   # no discovery yet
    build, lines = _lines(store)
    assert build["status"] == "failed"
    assert lines[-1][9:].startswith("Stopped after ") and "Run discovery first" in lines[-1]
    page = _page()
    assert "The last learning run, where it stopped" in page and 'class="lq-log-stop"' in page


def test_what_was_left_out_is_in_the_steps_and_the_notes(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    secret = built.t("stores")

    class Partial(_Connection):
        def query(self, sql, *, max_rows=None):
            if secret.upper() in sql.upper():
                raise RuntimeError(f"[42000] The SELECT permission was denied on the object '{secret}' (229)")
            return DuckDBWarehouse.query(self, sql, max_rows=max_rows)

    from core2.warehouse import querybot

    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials: Partial(built.con))
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    build, lines = _lines(store)
    assert build["status"] == "done"
    assert any(line[9:].startswith(f"Left out the table {secret}: the database refused it") for line in lines)
    assert lines[-1].endswith("1 left out, each named in the notes below")
    page = _page()
    assert re.search(rf"<summary>{len(lines)} steps, [0-9: -]+ UTC; 1 left out</summary>", page)
    assert 'class="lq-log-left"' in page
    assert f"The table {secret} was left out: the database refused it" in page


def test_a_run_cut_off_by_a_restart_is_said_and_does_not_block_learning_again(workspace):  # noqa: F811
    store, _built, _schema_dir = workspace
    from admin import core2_routes
    from store.db import get_db

    db_id = store.get_client(ACCOUNT)["db_config_id"]
    started = store.start_core2_build(ACCOUNT, db_id)
    store.add_core2_build_line(ACCOUNT, db_id, started, "Reading 9 tables")
    with get_db() as conn:   # started this morning by the process a restart replaced
        conn.execute("UPDATE core2_build SET runner = 'a process before the restart', "
                     "started_at = '2026-01-01 08:00:00' WHERE started_at = ?", (started,))
    page = _page()
    assert "The last attempt was interrupted: the service restarted" in page
    assert "The last learning run, where it stopped" in page and "Reading 9 tables" in page
    assert "Learning now" not in page
    assert _progress()["building"] is False
    with patch.object(core2_routes, "_is_auth", return_value=True), \
            patch.object(core2_routes, "_run_build") as run:
        from fastapi import BackgroundTasks

        tasks = BackgroundTasks()
        response = asyncio.run(core2_routes.learned_build(
            _request(f"/admin/clients/{ACCOUNT}/learned/build"), ACCOUNT, tasks))
    assert "saved=building" in response.headers["location"] and len(tasks.tasks) == 1
    assert not run.called   # queued, not run here
    # ...and already recorded as running, so the page the browser goes to shows it at once.
    page = _page(query="saved=building")
    assert "What QueryBot is doing now" in page and "QueryBot is studying the database" in page
    assert tasks.tasks[0].args == (ACCOUNT, store.latest_core2_build(ACCOUNT, db_id)["started_at"])
