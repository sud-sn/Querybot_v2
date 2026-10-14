"""What came over from the earlier setup at Learn is the new core's own: nothing in the earlier setup changes it.

The new core is the only engine. Learn brings the earlier setup's decisions over (links, default dates, metrics,
names, descriptions, confirmed meanings) and adopts them: authored "adopted", they are the new core's decisions,
changed on its own pages. Before, an answer looked every few seconds for decisions made in the earlier setup since
and brought them over; its pages are hidden now, so:

* an answer never reads the earlier setup's stores;
* a metric retired there after Learn stays in the next answer's catalog;
* a re-import never takes an adopted decision back;
* a database where the decisions were held as imports adopts them when the service starts.
"""

from __future__ import annotations

import pytest

from tests.test_core2_brings_over_todays_decisions import _seed_todays_setup
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _schema_file,
    workspace,  # noqa: F401 - the fixture
)


@pytest.fixture
def learned(workspace, tmp_path):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    _seed_todays_setup(store, tmp_path)          # the earlier setup's decisions, before Learn
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    return store


def _db(store) -> int:
    return int(store.get_client(ACCOUNT)["db_config_id"])


def _measures(store) -> set[str]:
    from core2.bootstrap.service import load_model

    return {m.business_name for m in load_model(ACCOUNT, _db(store)).measures.values()}


def _retire_metrics() -> None:
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute("UPDATE metric_registry SET is_active=0, metric_status='deprecated' WHERE account_id=?",
                     (ACCOUNT,))


def test_what_came_over_at_learn_is_adopted(learned):
    store = learned
    authors = {o["author"] for o in store.list_core2_overrides(ACCOUNT, _db(store))}
    assert authors == {"adopted"} and "Delivered sales" in _measures(store)


def test_an_answer_never_reads_the_earlier_setup(learned, monkeypatch):
    store = learned
    from core2.bootstrap import service as bootstrap
    from core2.model import imports

    monkeypatch.setattr(imports, "read_legacy", lambda *a, **k: pytest.fail("the earlier setup was read"))
    assert not bootstrap.keep_decisions_current(ACCOUNT, _db(store), store.get_client(ACCOUNT))


def test_a_metric_retired_in_the_earlier_setup_stays(learned):
    store = learned
    from core2.bootstrap import service as bootstrap

    _retire_metrics()
    assert not bootstrap.keep_decisions_current(ACCOUNT, _db(store), store.get_client(ACCOUNT))
    assert "Delivered sales" in _measures(store)


def test_a_reimport_never_takes_an_adopted_decision_back(learned):
    store = learned
    from core2.bootstrap.service import bring_over, learned_model

    _retire_metrics()
    done = bring_over(ACCOUNT, _db(store), learned_model(ACCOUNT, _db(store)), store.get_client(ACCOUNT))
    assert "error" not in done
    assert "Delivered sales" in _measures(store)


def test_decisions_held_as_imports_are_adopted_when_the_service_starts(learned):
    store = learned
    db_id = _db(store)
    held = store.list_core2_overrides(ACCOUNT, db_id)
    for o in held:          # as a database from before the new core became the only engine holds them
        store.set_core2_override(ACCOUNT, db_id, o["object_key"], o["field"], o["value"], author="import",
                                 note=(o.get("note") or "").replace("the earlier setup", "today's setup"))
    store.init_db()
    after = store.list_core2_overrides(ACCOUNT, db_id)
    assert {o["author"] for o in after} == {"adopted"} and len(after) == len(held)
    assert not any("today's setup" in (o.get("note") or "") for o in after)
