"""A reader suggests a change to what the data says; an admin accepts or rejects it.

"What you can ask" shows a reader each subject's metrics (what each means, how it is
counted, the date it is counted by, the other names people use), what they break down by
and the dates. Wherever something is wrong or missing, the reader suggests a change, or
describes a metric that is not there. The suggestion waits under Admin → Data → Requests,
beside what is there now. Accepting it (as sent, or edited first) writes admin decisions,
so it holds from the next answer and through every Learn, and the reader is told; rejecting
it tells them why; an accepted request can be undone. A reader with the portal's admin role
edits directly.

Each test starts at the reader's own route and ends where the change is read: the model
answers are built from, the planner's catalog, the reader's page, the admin's queue.
Invented retail data only.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from evals.core2 import domains
from evals.core2.compile_eval import learn

ORDER_TABLES = {"main.order_lines", "main.customers", "main.products", "main.stores", "main.categories", "main.regions"}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


class Site:
    """Both consoles over one store, with a reader signed in to the portal and an admin to the admin pages."""

    def __init__(self, client, account, store, readers):
        self.client, self.account, self.store, self.readers = client, account, store, readers
        self.reader = readers["riley"]

    def as_reader(self, name: str):
        self.reader = self.readers[name]
        return self

    def suggest(self, **form):
        return self.client.post("/portal/api/requests", json=form).json()

    def model(self):
        from core2.bootstrap.service import load_model

        return load_model(self.account, None)

    def accept(self, request_id, **body):
        return self.client.post(f"/admin/clients/{self.account}/requests/api/{request_id}/accept", json=body).json()


@pytest.fixture
def site(retail, tmp_path, monkeypatch):
    _, learned = retail
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_metrics, core2_requests, core2_routes, routes as admin_routes
    from portal import routes as portal_routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    store.update_client_meta(account, client_name="Retail (invented)")
    store.save_core2_model(account, None, learned.model_dump_json())
    store.set_query_engine(account, "core2")
    readers = {
        "riley": {"id": 7, "account_id": account, "name": "Riley Reader", "email": "riley@example.test", "role": "analyst"},
        "sam": {"id": 8, "account_id": account, "name": "Sam Store-Manager", "email": "sam@example.test", "role": "analyst"},
        "ada": {"id": 9, "account_id": account, "name": "Ada Admin-Reader", "email": "ada@example.test", "role": "admin"},
    }
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.include_router(portal_routes.router)
    holder: dict = {}
    site = Site(TestClient(app), account, store, readers)
    holder["site"] = site
    with ExitStack() as stack:
        stack.enter_context(patch.object(portal_routes, "_get_portal_user", lambda *a, **k: site.reader))
        stack.enter_context(patch.object(admin_routes, "_is_auth", return_value=True))
        stack.enter_context(patch.object(core2_routes, "_is_auth", return_value=True))
        stack.enter_context(patch.object(core2_requests, "_is_auth", return_value=True))
        stack.enter_context(patch.object(core2_metrics, "_is_auth", return_value=True))
        # Every store the code reads through: tests elsewhere delete the store modules and import them again,
        # so a route can hold an older store than the one imported here.
        import sys

        import core2.request_service
        from portal import core2_requests as portal_requests

        stores = {id(m): m for m in (sys.modules["store"], store, portal_routes.store, portal_requests.store,
                                     core2.request_service.store, core2_requests.store)}
        for module in stores.values():
            stack.enter_context(patch.object(module, "get_allowed_tables",
                                             lambda user: None if user.get("role") == "admin" else set(ORDER_TABLES)))
        yield site


def _measure(model, name):
    return next(m for m in model.measures.values() if m.business_name == name)


def _date(model, measure, name):
    return next(r for r in model.date_roles.values() if r.table == measure.table and r.name == name)


# ── what the reader sees ─────────────────────────────────────────────────────

def test_the_reader_sees_their_subjects_metrics_fields_and_dates_and_nothing_else(retail):
    from core2.model.requests import reader_view

    _, model = retail
    model = model.model_copy(deep=True)
    hidden = _measure(model, "Gross amount")
    hidden.hidden = True
    allowed = {t for t in model.tables if model.tables[t].name in {n.split(".")[1] for n in ORDER_TABLES}}
    view = reader_view(model, today=dt.date(2026, 6, 15), allowed=allowed)
    (subject,) = view["subjects"]                      # returns and targets are not this reader's
    assert subject["name"] == "Order line" and subject["grain"] == "one row per order line"
    names = [m["name"] for m in subject["metrics"]]
    assert "Net amount" in names and "Gross amount" not in names
    net = next(m for m in subject["metrics"] if m["name"] == "Net amount")
    assert net["how"] == "Net amount is the total net amount across order lines." and net["date"] == "Order date"
    assert {b["name"] for b in subject["breakdowns"]} >= {"Customer", "Store", "Product"}
    assert [d["name"] for d in subject["dates"]][0] == "Order date"


# ── a suggestion, sent and waiting ───────────────────────────────────────────

def test_a_suggestion_waits_beside_what_is_there_now_and_changes_nothing_yet(site):
    model = site.model()
    net = _measure(model, "Net amount")
    ship = _date(model, net, "Ship date")
    sent = site.suggest(kind="change", target_kind="measure", target_key=net.key,
                        meaning="Sale value after discounts, before tax.", names="sales, turnover, Net amount",
                        date=ship.key, note="Leave out refunded orders.", example="Net sales by store last quarter")
    assert sent["ok"] and sent["request"]["status"] == "waiting"
    (waiting,) = site.store.list_core2_requests(site.account, status="waiting")
    assert waiting["user_name"] == "Riley Reader" and waiting["target_name"] == "Net amount"
    assert [(c["field"], c["value"]) for c in waiting["changes"]] == [
        ("description", "Sale value after discounts, before tax."), ("synonyms", ["sales", "turnover"]),
        ("default_date", ship.key)]
    assert waiting["note"] == "Leave out refunded orders."
    after = _measure(site.model(), "Net amount")
    assert (after.description, after.synonyms, after.default_date) == (net.description, net.synonyms, net.default_date)


def test_the_admin_sees_it_under_requests_with_now_and_suggested(site):
    net = _measure(site.model(), "Net amount")
    site.suggest(kind="change", target_kind="measure", target_key=net.key, meaning="Sale value after discounts.")
    page = site.client.get(f"/admin/clients/{site.account}/requests")
    assert page.status_code == 200
    html = page.text
    assert "Net amount" in html and "Riley Reader" in html and "Sale value after discounts." in html
    assert "Accept for everyone" in html and "Open in the editor" in html
    assert '<span class="client-nav-count">1</span>' in html          # the Data menu counts what waits
    assert "Sale value" not in site.client.get(f"/admin/clients/{site.account}/requests?status=rejected").text
    assert "Sale value" not in site.client.get(f"/admin/clients/{site.account}/requests?kind=new").text
    assert "Sale value" in site.client.get(f"/admin/clients/{site.account}/requests?kind=meanings").text
    assert "Sale value" not in site.client.get(f"/admin/clients/{site.account}/requests?kind=names").text


# ── accepted: it holds from the next answer ──────────────────────────────────

def test_accepting_writes_it_for_everyone_and_the_planner_reads_the_new_name(site):
    from core2.plan.catalog import catalog_text

    model = site.model()
    net = _measure(model, "Net amount")
    ship = _date(model, net, "Ship date")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, meaning="Sale value after "
                              "discounts, before tax.", names="sales, turnover", date=ship.key)["request"]["id"]
    assert site.accept(request_id)["ok"]
    after = _measure(site.model(), "Net amount")
    assert after.description == "Sale value after discounts, before tax."
    assert sorted(w for ws in after.synonyms.values() for w in ws) == ["sales", "turnover"]
    assert after.default_date == ship.key
    assert "also called: sales, turnover" in catalog_text(site.model())
    decided = site.store.get_core2_request(site.account, request_id)
    assert decided["status"] == "accepted" and [a["field"] for a in decided["applied"]] == [
        "description", "synonyms", "default_date"]
    # The reader is told: on their list, and in the notice the portal polls for.
    mine = site.client.get("/portal/api/requests").json()["items"]
    assert [(r["id"], r["status"]) for r in mine] == [(request_id, "accepted")]
    notices = site.client.get("/portal/api/semantic-feedback/updates").json()["items"]
    assert {"id": f"c2r-{request_id}", "status": "approved", "column_name": "Net amount"}.items() <= next(
        n for n in notices if n["id"] == f"c2r-{request_id}").items()


def test_added_names_join_the_words_already_there(site):
    net = _measure(site.model(), "Net amount")
    site.store.set_core2_override(site.account, None, f"measure:{net.key}", "synonyms", {"en": ["net sales"]})
    first = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="revenue")["request"]["id"]
    second = site.as_reader("sam").suggest(kind="change", target_kind="measure", target_key=net.key,
                                           names="sales")["request"]["id"]
    site.accept(second)
    site.accept(first)                  # accepted after another name was added: nothing is written over
    words = sorted(w for ws in _measure(site.model(), "Net amount").synonyms.values() for w in ws)
    assert words == ["net sales", "revenue", "sales"]


def test_the_admin_can_edit_it_or_leave_part_of_it_out_before_accepting(site):
    model = site.model()
    net = _measure(model, "Net amount")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, meaning="Sales.",
                              names="turnover", date=_date(model, net, "Ship date").key)["request"]["id"]
    assert site.accept(request_id, edits={"0": "Sale value after discounts, before tax."}, skip=[2])["ok"]
    after = _measure(site.model(), "Net amount")
    assert after.description == "Sale value after discounts, before tax."
    assert after.default_date == net.default_date               # the date was left out
    assert site.store.get_core2_request(site.account, request_id)["changes"][0]["value"].startswith("Sale value")


def test_a_field_and_its_entity_take_meanings_and_names_too(site):
    from core2.plan.catalog import catalog_text

    request_id = site.suggest(kind="change", target_kind="entity", target_key="store", names="shop, outlet",
                              meaning="Where the sale was made.")["request"]["id"]
    site.accept(request_id)
    model = site.model()
    assert sorted(w for ws in model.entities["store"].synonyms.values() for w in ws) == ["outlet", "shop"]
    assert model.columns[model.entities["store"].label_column].description == "Where the sale was made."
    assert "also called: outlet, shop" in catalog_text(model)


# ── rejected, withdrawn, decided once ────────────────────────────────────────

def test_a_rejected_request_changes_nothing_and_the_reader_is_told_why(site):
    net = _measure(site.model(), "Net amount")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="profit")["request"]["id"]
    rejected = site.client.post(f"/admin/clients/{site.account}/requests/api/{request_id}/reject",
                                json={"reason": "Profit is a different metric."}).json()
    assert rejected["ok"]
    assert not _measure(site.model(), "Net amount").synonyms
    (mine,) = site.client.get("/portal/api/requests").json()["items"]
    assert (mine["status"], mine["reason"]) == ("rejected", "Profit is a different metric.")


def test_a_request_is_decided_once(site):
    net = _measure(site.model(), "Net amount")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")["request"]["id"]
    assert site.accept(request_id)["ok"]
    again = site.accept(request_id)
    assert not again["ok"] and "already been decided" in again["error"]
    late = site.client.post(f"/admin/clients/{site.account}/requests/api/{request_id}/reject", json={}).json()
    assert not late["ok"]


def test_only_the_reader_who_sent_it_can_take_it_back_and_only_while_it_waits(site):
    net = _measure(site.model(), "Net amount")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")["request"]["id"]
    refused = site.as_reader("sam").client.post(f"/portal/api/requests/{request_id}/withdraw").json()
    assert not refused["ok"]
    assert site.as_reader("riley").client.post(f"/portal/api/requests/{request_id}/withdraw").json()["ok"]
    assert site.store.get_core2_request(site.account, request_id)["status"] == "withdrawn"
    assert site.client.get("/portal/api/requests").json()["items"] == []
    assert not site.accept(request_id)["ok"]


# ── undone ───────────────────────────────────────────────────────────────────

def test_undo_puts_back_what_was_there_unless_it_changed_since(site):
    net = _measure(site.model(), "Net amount")
    site.store.set_core2_override(site.account, None, f"measure:{net.key}", "description", "The admin's words.")
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key, meaning="A reader's words.",
                              names="sales")["request"]["id"]
    site.accept(request_id)
    # The names are changed again by the admin since; the meaning is not.
    site.store.set_core2_override(site.account, None, f"measure:{net.key}", "synonyms", {"en": ["sales", "takings"]})
    undone = site.client.post(f"/admin/clients/{site.account}/requests/api/{request_id}/undo").json()
    assert undone == {"ok": True, "kept": ["synonyms"]}
    after = _measure(site.model(), "Net amount")
    assert after.description == "The admin's words."
    assert sorted(w for ws in after.synonyms.values() for w in ws) == ["sales", "takings"]
    assert site.store.get_core2_request(site.account, request_id)["status"] == "undone"


def test_undo_of_a_field_nobody_had_decided_removes_the_decision(site):
    net = _measure(site.model(), "Net amount")
    before = net.default_date
    request_id = site.suggest(kind="change", target_kind="measure", target_key=net.key,
                              date=_date(site.model(), net, "Ship date").key)["request"]["id"]
    site.accept(request_id)
    site.client.post(f"/admin/clients/{site.account}/requests/api/{request_id}/undo")
    assert site.store.get_core2_override(site.account, None, f"measure:{net.key}", "default_date") is None
    assert _measure(site.model(), "Net amount").default_date == before


# ── a reader who edits directly ──────────────────────────────────────────────

def test_a_reader_with_the_admin_role_edits_directly_and_it_can_still_be_undone(site):
    net = _measure(site.model(), "Net amount")
    sent = site.as_reader("ada").suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")
    assert sent["request"]["status"] == "accepted"
    assert [w for ws in _measure(site.model(), "Net amount").synonyms.values() for w in ws] == ["sales"]
    decided = site.store.get_core2_request(site.account, sent["request"]["id"])
    assert decided["decided_by"] == "Ada Admin-Reader (edits directly)"
    accepted = site.client.get(f"/admin/clients/{site.account}/requests?status=accepted").text
    assert "Ada Admin-Reader" in accepted and "Undo" in accepted
    # Their own edit is not announced back to them as someone else's decision.
    assert not [n for n in site.client.get("/portal/api/semantic-feedback/updates").json()["items"]
                if str(n["id"]).startswith("c2r-")]


# ── what a reader cannot suggest ─────────────────────────────────────────────

@pytest.mark.parametrize("form,said", [
    (lambda m: {"target_kind": "measure", "target_key": _measure(m, "Refund amount").key, "names": "refunds"},
     "not in the data you can ask about"),        # returns are not this reader's table
    (lambda m: {"target_kind": "measure", "target_key": "no-such-metric", "names": "x"}, "not in the data"),
    (lambda m: {"target_kind": "measure", "target_key": _measure(m, "Net amount").key}, "Nothing to send"),
    (lambda m: {"target_kind": "measure", "target_key": _measure(m, "Net amount").key,
                "date": next(r.key for r in m.date_roles.values() if r.name == "Return date")}, "cannot be counted by"),
    (lambda m: {"target_kind": "date_role", "target_key": next(r.key for r in m.date_roles.values()
                                                               if r.name == "Order date"), "date": "x"},
     "Only a metric"),
    (lambda m: {"target_kind": "measure", "target_key": _measure(m, "Net amount").key,
                "names": ", ".join(f"w{i}" for i in range(11))}, "at most 10"),
])
def test_what_cannot_be_suggested_is_refused_with_the_reason(site, form, said):
    sent = site.suggest(kind="change", **form(site.model()))
    assert not sent["ok"] and said in sent["error"], sent
    assert site.store.list_core2_requests(site.account) == []


def test_a_hidden_metric_cannot_be_suggested_on(site):
    net = _measure(site.model(), "Net amount")
    site.store.set_core2_override(site.account, None, f"measure:{net.key}", "hidden", True)
    sent = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")
    assert not sent["ok"]


def test_twenty_waiting_suggestions_are_enough(site):
    net = _measure(site.model(), "Net amount")
    for i in range(20):
        assert site.suggest(kind="change", target_kind="measure", target_key=net.key, names=f"word{i}")["ok"]
    sent = site.suggest(kind="change", target_kind="measure", target_key=net.key, names="one more")
    assert not sent["ok"] and "20 suggestions waiting" in sent["error"]


def test_a_name_another_metric_already_has_is_flagged_for_the_admin(site):
    model = site.model()
    gross = _measure(model, "Gross amount")
    sent = site.suggest(kind="change", target_kind="measure", target_key=gross.key, names="Net amount, gross sales")
    (change,) = site.store.get_core2_request(site.account, sent["request"]["id"])["changes"]
    assert change["clashes"] == {"Net amount": "Net amount"}
    assert '"Net amount" is already a name of Net amount.' in site.client.get(
        f"/admin/clients/{site.account}/requests").text


# ── a metric that is not there ───────────────────────────────────────────────

def test_a_described_metric_goes_to_the_editor_with_the_readers_words(site):
    sent = site.suggest(kind="new_metric", name="Margin after returns",
                        description="Net amount minus refunds, divided by net amount, completed orders only.",
                        example="Margin after returns by month")
    assert sent["ok"]
    page = site.client.get(f"/admin/clients/{site.account}/requests?kind=new").text
    assert "Margin after returns" in page and "Write it in the editor" in page and "Mark as done" in page
    editor = site.client.get(f"/admin/clients/{site.account}/measures/edit",
                             params={"describe": "Net amount minus refunds, divided by net amount.",
                                     "name": "Margin after returns"}).text
    assert ">Net amount minus refunds, divided by net amount.</textarea>" in editor
    assert '"name": "Margin after returns"' in editor
    assert '"description": "Net amount minus refunds, divided by net amount."' in editor
    assert site.accept(sent["request"]["id"])["ok"]             # marked done once written
    assert site.store.get_core2_request(site.account, sent["request"]["id"])["applied"] == []


def test_a_metric_needs_a_name_and_a_sentence(site):
    sent = site.suggest(kind="new_metric", name="", description="short")
    assert not sent["ok"] and "name" in sent["error"]


# ── the reader's page ────────────────────────────────────────────────────────

def test_the_readers_page_lists_what_they_can_suggest_on_and_where_their_suggestion_stands(site):
    net = _measure(site.model(), "Net amount")
    site.suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")
    html = site.client.get("/portal/kb").text
    assert "Describe a new metric" in html and "Suggest a change" in html
    assert f'data-kind="measure" data-key="{net.key}"' in html
    assert "Your suggestion is waiting" in html and "1 waiting" in html
    assert 'id="ask-dialog"' in html and "Send to your admin" in html
    assert "Refund amount" not in html                       # not this reader's table


def test_a_reader_who_edits_directly_is_offered_edit_and_save(site):
    html = site.as_reader("ada").client.get("/portal/kb").text
    assert ">Save</button>" in html and "You edit directly" in html


def test_the_readers_page_in_french(site):
    site.client.cookies.set("qb_lang", "fr")
    html = site.client.get("/portal/kb").text
    assert "Proposer une modification" in html and "Décrire un nouvel indicateur" in html


def test_without_the_new_core_a_suggestion_is_not_taken(site):
    site.store.set_query_engine(site.account, "legacy")
    net = _measure(site.model(), "Net amount")
    assert not site.suggest(kind="change", target_kind="measure", target_key=net.key, names="sales")["ok"]


# ── the admin's inbox ────────────────────────────────────────────────────────

def test_waiting_requests_reach_the_admin_inbox_as_a_warning():
    from admin.inbox import build_inbox
    from tests.test_admin_inbox import _DB, _clients, _Signals

    with _Signals(reader_requests={"c0": 2}, access={"c0": 1}):
        items = build_inbox(_clients(2), _DB)
    (item,) = [i for i in items if i["kind"] == "reader-request"]
    assert (item["total"], item["severity"], item["href"]) == (2, "warn", "/admin/clients/c0/requests")
    kinds = [i["kind"] for i in items]
    assert kinds.index("access") < kinds.index("reader-request")
