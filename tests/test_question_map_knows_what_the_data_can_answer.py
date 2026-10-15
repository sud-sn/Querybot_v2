"""Every question a workspace's data answers is worked out from the model, checked, and kept with it.

"The patient count by each doctor" failed because nothing knew, before a reader asked, which questions the data
held. The question map (core2/question_map.py) writes them down: every figure on its own, every figure by every
grouping a reader can split it by (and how: through a link, or counted on the rows that name both), every analysis
the tables support; each resolved and compiled with no AI, then run on the data. What cannot be answered says why.
Learn builds it; an admin's change rebuilds it; a reader only ever gets the questions of the tables they may use.
The planner is told the routes by name, and a question that cannot be answered offers the nearest that can.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from core2 import question_map as Q
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
_LEARNED: dict[str, tuple] = {}


def _domain(name: str):
    if name not in _LEARNED:
        built, model = learn(domains.build(name), "descriptive")
        qmap = Q.build(model, today=TODAY)
        Q.check(qmap, model, DuckDBWarehouse(built.con), today=TODAY, budget=None, most=100000)
        _LEARNED[name] = (built, model, qmap)
    return _LEARNED[name]


def _entry(qmap: Q.QuestionMap, key: str) -> Q.Entry:
    return next(e for e in qmap.entries if e.key == key)


def test_it_knows_how_each_question_is_answered_and_why_one_is_not():
    _, _, qmap = _domain("compounding_pharmacy")
    patients = _entry(qmap, "number_of_patients|prescriber")
    assert patients.status == Q.ANSWERABLE and patients.route == "rows" and patients.rows_on == "Prescription"
    assert patients.question == "Number of patients by prescriber" and patients.ran is True
    tests = _entry(qmap, "number_of_quality_tests|prescriber")
    assert tests.status == Q.NOT_LINKED and "is not linked to" in tests.reason
    payers = _entry(qmap, "number_of_payers|patient")
    assert payers.status == Q.CANNOT_SPLIT and "cannot split" in payers.reason
    kinds = {e.analysis for e in qmap.entries if e.kind == "analysis" and e.status == Q.ANSWERABLE}
    assert {"lifecycle", "cohort", "gap"} <= kinds
    figures = [e for e in qmap.entries if e.kind == "figure"]
    assert figures and all(e.status == Q.ANSWERABLE for e in figures)


@pytest.mark.parametrize("name", ["compounding_pharmacy", "retail", "bank", "finance", "home_services"])
def test_every_question_it_calls_answerable_runs_on_the_data(name):
    _, _, qmap = _domain(name)
    ok = [e for e in qmap.entries if e.status == Q.ANSWERABLE]
    assert len(ok) > 50
    assert all(e.ran is not None for e in ok)                 # every one was run
    assert not [e for e in qmap.entries if e.status == Q.REFUSED], \
        [(e.question, e.reason) for e in qmap.entries if e.status == Q.REFUSED][:5]


def test_it_holds_names_and_plans_never_a_members_value_or_a_persons_column():
    built, model, qmap = _domain("compounding_pharmacy")
    text = qmap.to_json()
    names = [r[0] for r in built.con.execute("SELECT prescriber_name FROM prescribers").fetchall()]
    assert not any(n in text for n in names)
    personal = {c.key for c in model.columns.values() if c.personal != "none"}
    groups = {e.group for e in qmap.entries if e.group}
    attrs = {a.slug: a.column for a in model.attributes.values()}
    assert not any(attrs.get(g) in personal for g in groups)


def test_a_question_through_a_link_waiting_for_an_admin_is_a_gap_to_confirm_not_a_failure():
    _, model, _ = _domain("compounding_pharmacy")
    copy = model.model_copy(deep=True)
    from core2.model.schema import Evidence

    for link in copy.joins.values():
        if link.to_table.endswith(".prescribers"):        # a column two tables' keys fit: an admin decides
            link.trust, link.status = "proposed", "proposed"
            link.evidence = [*link.evidence, Evidence(kind="ambiguous", detail="fits two tables")]
    qmap = Q.build(copy, today=TODAY)
    entry = _entry(qmap, "number_of_prescriptions|prescriber")
    assert entry.status == Q.UNCONFIRMED and "confirmed" in entry.reason


def test_a_map_reads_the_links_once_and_a_changed_link_is_read_again():
    # A schema of 70 tables took nine minutes to map: every one of its thousands of questions read the whole link
    # graph again, hundreds of times. While a map is made the model does not change, so each way of reading its
    # links is built once; outside, and for any other model, they are read as they stand.
    from core2.resolve import paths as P

    _, model, _ = _domain("compounding_pharmacy")
    read = []
    real = P._read_edges

    def counting(*args):
        read.append(args[1:])
        return real(*args)

    with patch.object(P, "_read_edges", counting):
        qmap = Q.build(model, today=TODAY)
        assert qmap.entries and len(read) == len(set(read)) <= 8
        tables = sorted(model.tables)
        with P.fixed(model):
            inside = [[p.joins for p in P.all_paths(model, a, b)] for a in tables for b in tables]
        assert inside == [[p.joins for p in P.all_paths(model, a, b)] for a in tables for b in tables]
        copy = model.model_copy(deep=True)
        with P.fixed(model):
            for link in copy.joins.values():
                link.trust = "rejected"
            assert all(not P.all_paths(copy, a, b) for a in tables for b in tables if a != b)
            assert any(P.all_paths(model, a, b) for a in tables for b in tables if a != b)


def test_a_reader_gets_only_the_questions_of_tables_they_may_use():
    _, model, qmap = _domain("compounding_pharmacy")
    allowed = {k for k in model.tables if not k.endswith(".prescriptions")}
    keys = {e.key for e in Q.answerable(qmap, allowed)}
    assert "number_of_patients|prescriber" not in keys
    assert "number_of_patients|prescriber" in {e.key for e in Q.answerable(qmap)}
    assert all(set(e.tables) <= allowed for e in Q.answerable(qmap, allowed))


def test_the_nearest_answerable_questions_to_one_that_is_not():
    _, _, qmap = _domain("compounding_pharmacy")
    found = Q.nearest(qmap, measure="number_of_quality_tests", group="prescriber")
    assert found and all(e.status == Q.ANSWERABLE for e in found)
    assert found[0].measure == "number_of_quality_tests"          # the same figure by another grouping first


def test_the_planner_is_told_the_routes_by_name():
    from core2.plan.planner import stable_prompt

    _, model, qmap = _domain("compounding_pharmacy")
    routes = Q.routes_text(qmap)
    line = next(x for x in routes.splitlines() if x.startswith("- Number of patients by "))
    assert "prescriber" in line and "pharmacist" in line and "prescription" in line
    assert routes in stable_prompt(model, routes=routes)
    assert "COUNTED THROUGH ROWS" not in stable_prompt(model)      # unchanged without a map


def test_a_question_that_cannot_be_answered_offers_the_nearest_that_can():
    built, model, qmap = _domain("compounding_pharmacy")

    def ai(stable: str, tail: str) -> str:
        assert "COUNTED THROUGH ROWS THAT NAME BOTH" in stable and "Number of patients by " in stable
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["number_of_quality_tests"],
                           "group_by": ["prescriber"]})

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(),
                        today=TODAY, question_map=qmap)
    payload = answer_question("quality tests by prescriber", services, Session())
    offered = [c["question"] for c in payload["follow_up_suggestions"]]
    nearest = [e.question for e in Q.nearest(qmap, measure="number_of_quality_tests", group="prescriber")]
    assert offered[:len(nearest)] == nearest
    assert "Number of quality tests by prescriber" not in offered        # never the question that failed


@pytest.fixture
def site(tmp_path, monkeypatch):
    built, model, _ = _domain("compounding_pharmacy")
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_questions, core2_routes, routes as admin_routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    store.save_core2_model(account, None, model.model_dump_json())
    app = FastAPI()
    app.include_router(admin_routes.router)
    with ExitStack() as stack:
        for module in (admin_routes, core2_routes, core2_questions):
            stack.enter_context(patch.object(module, "_is_auth", return_value=True))
        stack.enter_context(patch("core2.kpi_proposals.governed_warehouse",
                                  lambda account_id, client: DuckDBWarehouse(built.con)))
        yield TestClient(app), account, store


def test_it_is_kept_with_the_model_and_rebuilt_when_an_admin_changes_it(site):
    _, account, store = site
    assert Q.current(account, rebuild=False) is None
    qmap = Q.build_for(account, today=TODAY)
    assert qmap is not None and qmap.checked_at
    kept = Q.current(account, rebuild=False)
    assert kept is not None and kept.counts() == qmap.counts()
    store.set_core2_override(account, None, "memory.main.formulas", "business_name", json.dumps("Recipe"))
    assert Q.current(account, rebuild=False) is None            # the model changed: rebuilt before it is used


def test_the_admin_page_lists_what_readers_can_ask_and_why_not(site):
    client, account, _ = site
    page = client.get(f"/admin/clients/{account}/askable").text
    assert "What readers can ask" in page and "Not mapped yet" in page
    said = client.post(f"/admin/clients/{account}/askable/api/check").json()
    assert said["ok"] and said["answerable"] > 100 and said["gaps"] > 0
    page = client.get(f"/admin/clients/{account}/askable").text
    assert "questions answered" in page and "counted on rows that name both lists" in page
    assert "Number of quality tests by prescriber" in page and "Not linked" in page


def test_learn_ends_by_mapping_the_questions():
    import inspect

    from core2.bootstrap import service

    source = inspect.getsource(service.build_workspace)
    assert "from core2.question_map import build_for" in source
