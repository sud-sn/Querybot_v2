"""The member names the new core matches questions against stay current.

Names ("Northline Distribution 58") are read from the warehouse, not from the
model, so a question can name a member the profile never sampled. They used to
be read once per model version and kept until the next Learn or a restart. Now:

* an attribute the admin hides or marks sensitive is left out of the very next
  answer, without reading anything again; one the admin allows again is back;
* an attribute newly allowed is read at once;
* the names are read again in the background after MEMBERS_HOURS, so a customer
  added since is found by name; the question that notices waits for nothing;
* Learn reads them as its last step, so the first question does not wait, and
  the admin's "Refresh values" reads them again;
* an admin who turned value indexing off gets none of it: Learn keeps no
  column's values, no name is read, and none reaches the AI.
"""

from __future__ import annotations

import json
import threading

import pytest

from tests.test_core2_answers_in_the_portal import learned  # noqa: F401 - the fixture
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _schema_file,
    workspace,  # noqa: F401 - the fixture
)


MATCHES = "VALUE MATCHES ("      # the block of member names found in the question, not the rules naming it


def _db(store):
    db_id = int(store.get_client(ACCOUNT)["db_config_id"])
    return db_id, store.get_db_config(db_id)


def _model(store):
    from core2.bootstrap.service import load_model

    return load_model(ACCOUNT, _db(store)[0])


def _names_attribute(model):
    """The listable attribute with the most members (the retail customers' codes)."""
    from core2.plan.values import listable

    found = [(slug, model.attributes[slug]) for slug in listable(model)]
    assert found, "the retail warehouse has members that may be listed"
    return max(found, key=lambda item: (item[1].members or 0, item[0]))


def _reads(monkeypatch) -> list[int]:
    from core2 import service

    reads: list[int] = []
    real = service.build_index

    def counting(*args, **kwargs):
        reads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(service, "build_index", counting)
    return reads


def _named(index, attribute: str) -> set[str]:
    return {value for entries in index.names.values() for a, value in entries if a == attribute}


@pytest.fixture
def fresh(learned, monkeypatch):  # noqa: F811
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    return learned


def test_learn_reads_the_names_so_the_first_question_does_not_wait(learned):  # noqa: F811
    store, *_ = learned
    from core2 import service

    model = _model(store)
    slug, _ = _names_attribute(model)
    kept = service._INDEXES.get((ACCOUNT, model.version))
    assert kept is not None and _named(kept.index, slug)
    build = store.latest_core2_build(ACCOUNT, _db(store)[0])
    assert "Reading the member names" in json.dumps(build)


def test_a_column_the_admin_hides_leaves_the_next_answer_without_a_new_read(fresh, monkeypatch):
    store, *_ = fresh
    from core2 import service

    reads = _reads(monkeypatch)
    db_id, config = _db(store)
    model = _model(store)
    slug, attribute = _names_attribute(model)
    assert _named(service._member_index(ACCOUNT, model, config), slug) and len(reads) == 1

    store.set_core2_override(ACCOUNT, db_id, f"column:{attribute.column}", "sensitivity", "pii", author="admin")
    hidden = _model(store)
    assert not _named(service._member_index(ACCOUNT, hidden, config), slug)
    assert len(reads) == 1, "leaving a column out reads nothing"

    store.delete_core2_override(ACCOUNT, db_id, f"column:{attribute.column}", "sensitivity")
    assert _named(service._member_index(ACCOUNT, _model(store), config), slug) and len(reads) == 1


def test_a_column_the_admin_allows_again_after_the_read_is_read_at_once(fresh, monkeypatch):
    store, *_ = fresh
    from core2 import service

    reads = _reads(monkeypatch)
    db_id, config = _db(store)
    slug, attribute = _names_attribute(_model(store))
    store.set_core2_override(ACCOUNT, db_id, f"column:{attribute.column}", "hidden", True, author="admin")
    assert not _named(service._member_index(ACCOUNT, _model(store), config), slug) and len(reads) == 1

    store.delete_core2_override(ACCOUNT, db_id, f"column:{attribute.column}", "hidden")
    assert _named(service._member_index(ACCOUNT, _model(store), config), slug) and len(reads) == 2


def _join_reads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("core2-members-"):
            thread.join(10)


def test_names_are_read_again_in_the_background_after_the_wait(fresh, monkeypatch):
    store, built, _ = fresh
    from core2 import service

    reads = _reads(monkeypatch)
    _, config = _db(store)
    model = _model(store)
    slug, attribute = _names_attribute(model)
    column = model.columns[attribute.column]
    table = model.tables[column.table]
    service._member_index(ACCOUNT, model, config)

    # A new member arrives in the warehouse after the names were read.
    first = built.con.execute(f'SELECT * FROM "{table.name}" LIMIT 1').fetchdf()
    first[column.name] = "ZZ-NEW-9001"
    for (keys,) in built.con.execute("SELECT constraint_column_names FROM duckdb_constraints() WHERE table_name = ? "
                                     "AND constraint_type = 'PRIMARY KEY'", [table.name]).fetchall():
        for key in keys:
            first[key] = built.con.execute(f'SELECT MAX("{key}") + 1 FROM "{table.name}"').fetchone()[0]
    built.con.register("arrival", first)
    built.con.execute(f'INSERT INTO "{table.name}" SELECT * FROM arrival')

    assert "ZZ-NEW-9001" not in _named(service._member_index(ACCOUNT, model, config), slug)
    assert len(reads) == 1, "inside the wait nothing is read again"

    service._INDEXES[(ACCOUNT, model.version)].read_at -= service.MEMBERS_HOURS * 3600 + 1
    noticed = service._member_index(ACCOUNT, model, config)
    assert "ZZ-NEW-9001" not in _named(noticed, slug), "the question that notices does not wait"
    _join_reads()
    assert len(reads) == 2
    assert "ZZ-NEW-9001" in _named(service._member_index(ACCOUNT, model, config), slug)
    assert len(reads) == 2, "read once, not at every question"


def test_a_failed_read_keeps_the_names_and_tries_again_after_the_same_wait(fresh, monkeypatch):
    store, *_ = fresh
    from core2 import service

    _, config = _db(store)
    model = _model(store)
    slug, _ = _names_attribute(model)
    service._member_index(ACCOUNT, model, config)
    kept = service._INDEXES[(ACCOUNT, model.version)]
    kept.read_at -= service.MEMBERS_HOURS * 3600 + 1

    def broken(*args, **kwargs):
        raise RuntimeError("the warehouse is asleep")

    monkeypatch.setattr(service, "build_index", broken)
    assert _named(service._member_index(ACCOUNT, model, config), slug)
    _join_reads()
    again = service._INDEXES[(ACCOUNT, model.version)]
    assert again is kept and not again.refreshing and _named(again.index, slug)
    assert again.read_at > kept.read_at - 1 and not service._member_index(ACCOUNT, model, config) is None


def test_refresh_values_reads_the_new_cores_names_again(fresh, monkeypatch):
    import asyncio
    from unittest.mock import patch

    from starlette.background import BackgroundTasks

    store, *_ = fresh
    import core.value_index as value_index
    from admin import routes
    from core2 import service
    from tests.test_core2_learned_page_shows_what_was_learned import _request

    monkeypatch.setattr(value_index, "build_value_index", lambda *a, **k: {})
    model = _model(store)
    assert (ACCOUNT, model.version) not in service._INDEXES
    tasks = BackgroundTasks()
    with patch.object(routes, "_is_auth", return_value=True):
        asyncio.run(routes.admin_refresh_value_index(
            _request(f"/admin/clients/{ACCOUNT}/value-index/refresh"), ACCOUNT, tasks))
    asyncio.run(tasks())
    slug, _ = _names_attribute(model)
    assert _named(service._INDEXES[(ACCOUNT, model.version)].index, slug)


# ── value indexing turned off ──────────────────────────────────────────────


@pytest.fixture
def unindexed(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    state = store.get_client_state(ACCOUNT)
    store.update_client_state(ACCOUNT, "READY", {**state, "value_index_enabled": False})
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    return store, built, schema_dir


def _prompts(monkeypatch) -> list[str]:
    import core2.bootstrap.ai as ai

    seen: list[str] = []

    def planner(*args, **kwargs):
        def complete(stable, tail):
            seen.append(stable + tail)
            return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                               "group_by": ["customer"]})
        return complete

    monkeypatch.setattr(ai, "workspace_planner", planner)
    return seen


def _a_member(built, model) -> str:
    """A customer's code, as a reader would type it (whether or not QueryBot may list the codes)."""
    column = next(c for c in model.columns.values() if c.name == "customer_code")
    return str(built.con.execute(f'SELECT MIN("{column.name}") FROM "{model.tables[column.table].name}"').fetchone()[0])


def _learn_and_ask(store, built, schema_dir, monkeypatch):
    import core.compliance.governed_query as gq
    import sqlglot
    from admin import core2_routes
    from core2 import service

    _schema_file(built, schema_dir)
    core2_routes._run_build(ACCOUNT)

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = built.con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    seen = _prompts(monkeypatch)
    model = _model(store)
    member = _a_member(built, model)
    service.portal_answer(ACCOUNT, f'net sales for "{member}"', {"id": 1, "role": "admin"}, session_key="v")
    return model, member, seen


def test_with_value_indexing_on_member_names_reach_the_planner(workspace, monkeypatch):  # noqa: F811
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    model, member, seen = _learn_and_ask(*workspace, monkeypatch)
    assert seen and member in seen[0] and MATCHES in seen[0]


def test_value_indexing_turned_off_keeps_reads_and_sends_no_member_name(unindexed, monkeypatch):
    store, *_ = unindexed
    from core2 import service

    reads = _reads(monkeypatch)
    model, member, seen = _learn_and_ask(*unindexed, monkeypatch)
    assert not any(c.values_allowed for c in model.columns.values()), "Learn keeps no column's values"
    assert not any(c.profile and c.profile.top for c in model.columns.values())
    assert reads == [] and not service._INDEXES, "no name is read"
    assert service.read_members(ACCOUNT) == 0
    # The reader's own words are the question; nothing QueryBot read is added to them.
    assert seen and MATCHES not in seen[0]
    assert seen[0].count(member) == 1, seen[0][-600:]


def test_value_indexing_turned_off_after_learn_stops_the_names_at_the_next_question(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    _learn_and_ask(store, built, schema_dir, monkeypatch)
    state = store.get_client_state(ACCOUNT)
    store.update_client_state(ACCOUNT, "READY", {**state, "value_index_enabled": False})
    seen = _prompts(monkeypatch)
    member = _a_member(built, _model(store))
    service.portal_answer(ACCOUNT, f'net sales for "{member}"', {"id": 1, "role": "admin"}, session_key="w")
    assert seen and MATCHES not in seen[0] and seen[0].count(member) == 1


# ── a reader's own metric, made in the chat ──────────────────────────────────

def _writes(monkeypatch) -> list[str]:
    import core2.bootstrap.ai as ai

    written: list[str] = []

    def writer(account_id, client, *, description=""):
        written.append(description)
        return lambda stable, tail: "{}"

    monkeypatch.setattr(ai, "metric_writer", writer)
    return written


DEFINITION = "Margin after returns = net amount minus refunds, divided by net amount"


def test_a_metric_of_the_readers_own_is_written_by_the_ai_where_questions_reach_it(workspace, monkeypatch):  # noqa: F811
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    _learn_and_ask(*workspace, monkeypatch)
    written = _writes(monkeypatch)
    service.portal_answer(ACCOUNT, DEFINITION, {"id": 1, "role": "admin"}, session_key="own-on")
    assert written == ["net amount minus refunds, divided by net amount"]


def test_value_indexing_turned_off_offers_no_metric_of_the_readers_own(unindexed, monkeypatch):
    from core2 import service

    _learn_and_ask(*unindexed, monkeypatch)
    written = _writes(monkeypatch)
    payload = service.portal_answer(ACCOUNT, DEFINITION, {"id": 1, "role": "admin"}, session_key="own-off")
    assert written == [] and "own_metric" not in payload
