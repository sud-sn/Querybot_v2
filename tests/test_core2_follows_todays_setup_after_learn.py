"""A decision made in today's setup after Learn reaches the new core's next answer.

Learn brings today's decisions over (links, default dates, metrics, names,
descriptions, confirmed meanings). One made afterwards - a metric added in the
metric builder, a link confirmed, a metric retired - reached the new core only
when an admin pressed "Bring them over again" or learned again. Now an answer
first compares today's setup with what came over and brings a change over:

* a metric added after Learn is in the very next answer's catalog;
* one retired after Learn is gone from it;
* nothing changed writes nothing;
* today's setup is read at most every DECISIONS_EVERY seconds per workspace;
* a setup that cannot be read leaves the answer standing on what came over.
"""

from __future__ import annotations

import json

import pytest

from tests.test_core2_brings_over_todays_decisions import _seed_todays_setup
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _schema_file,
    workspace,  # noqa: F401 - the fixture
)


@pytest.fixture
def learned(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes
    from core2.bootstrap import service as bootstrap

    monkeypatch.setattr(bootstrap, "_LOOKED", {})
    core2_routes._run_build(ACCOUNT)
    return store, built, schema_dir


def _db(store) -> int:
    return int(store.get_client(ACCOUNT)["db_config_id"])


def _measures(store) -> set[str]:
    from core2.bootstrap.service import load_model

    return {m.business_name for m in load_model(ACCOUNT, _db(store)).measures.values()}


def _imports(store) -> list[dict]:
    return [o for o in store.list_core2_overrides(ACCOUNT, _db(store)) if o["author"] == "import"]


def _look(monkeypatch=None) -> bool:
    """An answer's look at today's setup, as the portal makes it (the wait between looks is spent)."""
    import store
    from core2.bootstrap import service as bootstrap

    bootstrap._LOOKED.clear()
    return bootstrap.keep_decisions_current(ACCOUNT, _db(store), store.get_client(ACCOUNT))


def _retire_metrics() -> None:
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute("UPDATE metric_registry SET is_active=0, metric_status='deprecated' WHERE account_id=?",
                     (ACCOUNT,))


def test_a_metric_added_after_learn_is_in_the_next_answer(learned, tmp_path, monkeypatch):
    store, built, _ = learned
    import core.compliance.governed_query as gq
    import core2.bootstrap.ai as ai
    import sqlglot
    from core2 import service

    assert "Delivered sales" not in _measures(store)
    _seed_todays_setup(store, tmp_path)          # the admin's work in today's setup, after Learn

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = built.con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    seen: list[str] = []

    def planner(*args, **kwargs):
        def complete(stable, tail):
            seen.append(stable)
            return json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"]})
        return complete

    monkeypatch.setattr(ai, "workspace_planner", planner)
    from core2.bootstrap import service as bootstrap

    bootstrap._LOOKED.clear()
    service.portal_answer(ACCOUNT, "delivered sales", {"id": 1, "role": "admin"}, session_key="d")
    assert seen and "Delivered sales" in seen[0], "the answer's catalog has the metric added after Learn"
    assert "Delivered sales" in _measures(store)
    report = json.loads(store.latest_core2_build(ACCOUNT, _db(store))["import_report"])
    assert report["written"] and report["counts"], "the learned page shows what came over"


def test_a_metric_retired_after_learn_is_gone_from_the_next_answer(learned, tmp_path):
    store, *_ = learned
    _seed_todays_setup(store, tmp_path)
    assert _look() and "Delivered sales" in _measures(store)
    _retire_metrics()
    assert _look() and "Delivered sales" not in _measures(store)


def test_nothing_changed_writes_nothing(learned, tmp_path):
    store, *_ = learned
    _seed_todays_setup(store, tmp_path)
    assert _look()
    before = _imports(store)
    assert before and not _look(), "what came over is what today's setup holds"
    assert _imports(store) == before


def test_todays_setup_is_read_at_most_every_few_seconds(learned, tmp_path, monkeypatch):
    store, *_ = learned
    from core2.bootstrap import service as bootstrap
    from core2.model import imports

    reads: list[int] = []
    real = imports.read_legacy
    monkeypatch.setattr(imports, "read_legacy", lambda *a, **k: reads.append(1) or real(*a, **k))
    bootstrap._LOOKED.clear()
    client = store.get_client(ACCOUNT)
    assert not bootstrap.keep_decisions_current(ACCOUNT, _db(store), client) and len(reads) == 1
    _seed_todays_setup(store, tmp_path)
    assert not bootstrap.keep_decisions_current(ACCOUNT, _db(store), store.get_client(ACCOUNT))
    assert len(reads) == 1, "inside the wait today's setup is not read again"
    bootstrap._LOOKED[(ACCOUNT, _db(store))] -= bootstrap.DECISIONS_EVERY + 1
    assert bootstrap.keep_decisions_current(ACCOUNT, _db(store), store.get_client(ACCOUNT)) and len(reads) == 2
    assert "Delivered sales" in _measures(store)


def test_a_setup_that_cannot_be_read_leaves_the_answer_on_what_came_over(learned, tmp_path, monkeypatch, caplog):
    store, *_ = learned
    _seed_todays_setup(store, tmp_path)
    assert _look()
    from core2.model import imports

    def broken(*args, **kwargs):
        raise RuntimeError("the store is locked")

    monkeypatch.setattr(imports, "read_legacy", broken)
    with caplog.at_level("WARNING", logger="querybot.core2"):
        assert not _look()
    assert "Delivered sales" in _measures(store) and "the store is locked" in caplog.text
