"""Business metrics are suggested from the data, accepted by an admin, and shown on the executive dashboard.

A bank is read by its approval rate and its delinquent loans, a pharmacy by its margin and its rejected claims, a
shop by its average order and its returns, a ledger by its profit margin. Learn makes sums and counts; the business
metrics of the workspace's kind (core2/kpis.py) are written over its fields in the metric editor's own language,
checked on the data, and filed on the Requests page as suggestions from Solviz (core2/kpi_proposals.py). None is
a metric until an admin accepts it. Accepted, it is on the executive dashboard: the most important beside the
overview's totals, and each on a tab of its area (Lending, Risk, Claims), its numbers first, then by month, then
by the business's grouping. A suggestion is never made twice, and one the data does not hold is never made.

A bank is a kind of business of its own, and the industry an admin sets on the Compliance page weighs for its kind
where the names alone leave it open.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from core2 import dashboards, kpi_proposals, kpis, recipes
from core2.model import authoring
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
BANK = {"bank.approval_rate", "bank.delinquent_loans", "bank.average_loan_size", "bank.loan_to_value",
        "bank.charge_off_rate", "bank.late_payments", "bank.collection_rate", "bank.average_credit_limit",
        "bank.active_cards", "bank.compensation_per_complaint", "bank.average_transaction"}
PHARMACY = {"pharmacy.gross_margin_pct", "pharmacy.gross_margin", "pharmacy.claim_rejection_rate",
            "pharmacy.paid_to_billed", "pharmacy.price_per_fill", "pharmacy.quality_failure_rate"}
# The shop keeps its quantities in several units (each, case): units per order and the price per unit would mix
# them, so they are not suggested there.
SALES = {"sales.average_order_value", "sales.gross_margin_pct", "sales.gross_margin", "sales.return_rate",
         "sales.discount_rate", "sales.sales_per_customer"}


@pytest.fixture(scope="module")
def learned():
    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = learn(domains.build(name), "descriptive")
        return cache[name]

    return get


def _no_ai(stable: str, tail: str) -> str:
    raise AssertionError(f"the AI was asked: {tail[:80]}")


# ── the kind of business ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name, kind", [
    ("bank", "bank"), ("retail", "sales"), ("compounding_pharmacy", "pharmacy"), ("finance", "finance"),
    ("purchasing", "purchasing"), ("hr", "workforce"), ("subscriptions", "subscriptions"),
    ("inventory", "inventory"), ("networking", "network"), ("home_services", "field_service")])
def test_a_bank_is_a_bank_and_every_other_kind_stays_as_it_was(name, kind, learned):
    _, model = learned(name)
    assert recipes.kind_of(model).kind == kind
    # The industry set on the Compliance page never turns a business into another kind.
    assert recipes.kind_of(model, industry="banking").kind == kind
    assert recipes.kind_of(model, industry="healthcare_pharmacy").kind == kind


def test_the_industry_setting_decides_where_the_names_leave_it_open(learned):
    _, model = learned("bank")

    def few(table: str) -> bool:        # a reader who sees only the transactions, branches and customers
        return table.rsplit(".", 1)[-1] in ("transactions", "branches", "customers")

    assert recipes.kind_of(model, few).kind == "business"
    assert recipes.kind_of(model, few, industry="banking").kind == "bank"


# ── the metrics each kind is run by ──────────────────────────────────────────


@pytest.mark.parametrize("name, kind, keys", [("bank", "bank", BANK), ("compounding_pharmacy", "pharmacy", PHARMACY),
                                              ("retail", "sales", SALES)])
def test_each_kinds_metrics_are_written_as_the_editor_writes_them_and_run_on_its_data(name, kind, keys, learned):
    built, model = learned(name)
    found = kpis.proposals(model, kind)
    assert {p.key for p in found} == keys
    assert [p.rank for p in found] == sorted(p.rank for p in found)          # the most important first
    warehouse = DuckDBWarehouse(built.con)
    for p in found:
        measure = kpis.measure_of(model, p)                                   # read as the editor reads it
        assert measure.kpi == p.key and measure.status == "proposed" and measure.format == p.fmt
        result = authoring.check(model.model_copy(deep=True), measure, warehouse, TODAY)
        assert not result["problem"], (p.key, result["problem"])
        assert kpi_proposals.judge(p, result) == "", (p.key, result["months"])


def test_codes_are_read_by_the_words_they_stand_for_and_never_by_a_negated_one(learned):
    _, model = learned("compounding_pharmacy")
    claims = next(p for p in kpis.proposals(model, "pharmacy") if p.key == "pharmacy.claim_rejection_rate")
    assert "= 'REJ'" in claims.formula.text                                  # REJ is rejected; PEND is not
    _, bank = learned("bank")
    copy = bank.model_copy(deep=True)
    decision = next(c for c in copy.columns.values() if c.name == "decision")
    from core2.model.schema import TopValue

    decision.profile.top = [TopValue(value=v, count=10) for v in
                            ("Approved", "Pending approval", "Not approved", "Declined", "Approved with conditions")]
    approval = next(p for p in kpis.proposals(copy, "bank") if p.key == "bank.approval_rate")
    assert "IN ('Approved', 'Approved with conditions')" in approval.formula.text


def test_a_value_people_may_not_see_is_never_read_into_a_metric(learned):
    _, bank = learned("bank")
    copy = bank.model_copy(deep=True)
    for c in copy.columns.values():
        if c.name == "decision":
            c.values_allowed = False
    assert "bank.approval_rate" not in {p.key for p in kpis.proposals(copy, "bank")}


def test_a_ledger_in_one_currency_gets_its_margin_and_one_in_several_does_not(learned):
    _, model = learned("finance")
    # The synthetic ledger keeps its amounts in several currencies: a margin of them would mix them.
    assert not [p for p in kpis.proposals(model, "finance") if p.key.startswith("finance.")]
    single = model.model_copy(deep=True)
    single.quality = [q for q in single.quality if q.kind != "unit_mix"]
    for m in single.measures.values():
        m.unit_column = None
    found = {p.key: p for p in kpis.proposals(single, "finance")}
    assert found["finance.profit_margin"].formula.text == "[Profit] / [Revenue] * 100"
    assert found["finance.expense_ratio"].formula.text == "[Expenses] / [Revenue] * 100"
    for p in found.values():
        kpis.measure_of(single, p)


def test_documents_are_counted_by_their_number_never_by_each_lines_own_id(learned):
    _, model = learned("retail")
    found = {p.key: p.formula.text for p in kpis.proposals(model, "sales")}
    assert found["sales.average_order_value"].endswith("/ COUNT(DISTINCT [Order line · Order number])")
    assert found["sales.sales_per_customer"].endswith("/ COUNT(DISTINCT [Order line · Customer id])")


def test_approval_is_a_share_of_the_applications_decided(learned):
    """Pending and withdrawn applications are no refusals: counting them would show the last months, still being
    decided, as a collapse. Without a declined value there is nothing decided to read it over."""
    _, bank = learned("bank")
    approval = next(p for p in kpis.proposals(bank, "bank") if p.key == "bank.approval_rate")
    assert approval.formula.text.endswith("IN ('Approved', 'Declined') THEN 1 ELSE 0 END) * 100")
    from core2.model.schema import TopValue

    copy = bank.model_copy(deep=True)
    decision = next(c for c in copy.columns.values() if c.name == "decision")
    decision.profile.top = [TopValue(value=v, count=10) for v in ("Approved", "Pending", "Withdrawn")]
    assert "bank.approval_rate" not in {p.key for p in kpis.proposals(copy, "bank")}


@pytest.mark.parametrize("change", ["integer codes", "formula marks"])
def test_a_status_is_read_only_by_words_a_formula_can_hold(change, learned):
    """Integer codes say nothing of what they stand for; a value with a formula's own marks would be read as a
    field inside it."""
    _, bank = learned("bank")
    copy = bank.model_copy(deep=True)
    decision = next(c for c in copy.columns.values() if c.name == "decision")
    if change == "integer codes":
        decision.data_type = "integer"
    else:
        from core2.model.schema import TopValue

        decision.profile.top = [TopValue(value=v, count=10) for v in ("Approved [A]", "Declined", "Pending")]
    assert "bank.approval_rate" not in {p.key for p in kpis.proposals(copy, "bank")}


def test_a_metric_with_conditions_is_never_read_as_its_plain_sum(learned):
    _, model = learned("bank")
    bind = kpis.Bind(model)
    plain = next(m for m in model.measures.values()
                 if getattr(m.expr, "agg", "") == "sum" and not m.filters and not m.hidden)
    assert bind.named(plain).startswith("SUM([")
    from core2.model.schema import ColumnFilter

    kept = plain.model_copy(deep=True)
    kept.expr = kept.expr.model_copy(update={"filters": [ColumnFilter(column=plain.expr.column, op="gt", values=[0])]})
    assert bind.named(kept) == f"[{plain.business_name}]"


def test_one_metric_the_data_trips_up_leaves_the_others(learned, monkeypatch):
    import dataclasses

    _, model = learned("bank")

    def broken(b):
        raise RuntimeError("an odd column")

    monkeypatch.setattr(kpis, "PACKS", [dataclasses.replace(k, write=broken) if k.key == "bank.approval_rate" else k
                                        for k in kpis.PACKS])
    found = {p.key for p in kpis.proposals(model, "bank")}
    assert {k for k in found if not k.startswith("any.")} == BANK - {"bank.approval_rate"}
    # Its status read by no business metric, a share of it is suggested in its place.
    assert {k for k in found if k.startswith("any.")} == {"any.share.memory.main.loan_applications.decision.declin"}


def test_a_metric_over_two_tables_is_waiting_only_for_readers_of_both(learned):
    _, model = learned("retail")
    returns = next(p for p in kpis.proposals(model, "sales") if p.key == "sales.return_rate")
    definition = kpis.measure_of(model, returns).model_dump(mode="json")
    read = kpi_proposals._tables_read(definition)
    assert {t.rsplit(".", 1)[-1] for t in read} == {"order_lines", "returns"}


def test_french_names_for_a_french_workspace(learned):
    _, model = learned("bank")
    names = {p.key: (p.name, p.area) for p in kpis.proposals(model, "bank", lang="fr")}
    assert names["bank.approval_rate"] == ("Taux d'approbation", "Prêts")


@pytest.mark.parametrize("months, share, fmt, why", [
    ([54.0, 52.8, 47.1, 46.5, 48.3, 39.0], True, "percent", ""),
    ([None, None, None, None, None, None], False, "percent", "it has no value in the data"),
    ([None, None, 5.0, None, 4.0, None], False, "percent", "it has a value in only 2 of the last 6 months"),
    ([120.0, 90.0, 80.0], True, "percent", "it is meant to be a share between 0 and 100, and is not"),
    ([0.0, 0.0, 0.0, 0.0], True, "percent", "it is 0 in every month"),
    ([0.0, 2.3, 0.0, 0.0, 0.0, 0.0], True, "percent", "it is 0 in most of the last months"),
    ([100.0, 100.0, 100.0, 100.0], False, "percent", "it is 100% every month: its two fields hold the same numbers"),
    ([38.0, 38.0, 38.0, 38.0], False, "percent", ""),           # a steady margin is a figure, not a fault
    ([-48.0, -48.6, -47.5], False, "percent", ""),              # so is a loss
])
def test_a_metric_the_data_does_not_hold_is_not_suggested(months, share, fmt, why):
    proposal = kpis.Proposal("x", "X", "", "", 1, fmt, "higher", share, kpis.Formula("SUM([a])"))
    result = {"months": [{"period": f"2026-0{i + 1}", "value": v} for i, v in enumerate(months)], "problem": ""}
    assert kpi_proposals.judge(proposal, result) == why
    assert kpi_proposals.judge(proposal, {"months": [], "problem": "no such field"}).startswith("it could not run")


# ── suggested to the admin, accepted, on the dashboard ───────────────────────


@pytest.fixture
def site(learned, tmp_path, monkeypatch):
    built, model = learned("bank")
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_metrics, core2_requests, core2_routes, routes as admin_routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    store.save_core2_model(account, None, model.model_dump_json())
    app = FastAPI()
    app.include_router(admin_routes.router)
    with ExitStack() as stack:
        for module in (admin_routes, core2_routes, core2_requests, core2_metrics):
            stack.enter_context(patch.object(module, "_is_auth", return_value=True))
        stack.enter_context(patch.object(kpi_proposals, "governed_warehouse",
                                         lambda account_id, client: DuckDBWarehouse(built.con)))
        yield TestClient(app), account, store, built


def _propose(store, account: str, built):
    return kpi_proposals.propose(account, store.get_client(account), warehouse=DuckDBWarehouse(built.con),
                                 today=TODAY)


def test_suggestions_are_filed_once_and_one_rejected_is_never_filed_again(site):
    client, account, store, built = site
    first = _propose(store, account, built)
    assert first.kind == "bank" and set(first.filed) == {
        "Approval rate", "Delinquent loans", "Average loan size", "Loan-to-value", "Charge-off rate",
        "Late payment rate", "Collection rate", "Average credit limit", "Active cards",
        "Compensation per complaint", "Average transaction"} and not first.skipped
    waiting = store.list_core2_requests(account, status="waiting")
    assert {(r["kind"], r["source"], r["user_name"], r["user_id"]) for r in waiting} == {
        ("new_metric", "learn", "Solviz", "")}
    approval = next(r for r in waiting if r["target_name"] == "Approval rate")
    assert approval["definition"]["kpi"] == "bank.approval_rate"
    assert approval["note"] == "Loan applications approved, as a share of those decided (approved or declined)."
    checked = approval["definition"]["checked"]
    assert len(checked["months"]) == authoring.CHECK_MONTHS and checked["months"][0]["shown"].endswith("%")
    assert checked["area"] == "Lending" and checked["rank"] == 1
    # A reader is told only of the metrics over the tables they may use.
    cards = {approval["definition"]["table"].rsplit(".", 1)[0] + ".cards"}
    assert kpi_proposals.waiting_metrics(account, cards) == ["Average credit limit", "Active cards"]
    # Run again (after the next Learn): nothing new.
    again = _propose(store, account, built)
    assert again.filed == [] and again.already == 11
    # Rejected, it is never suggested again.
    client.post(f"/admin/clients/{account}/requests/api/{approval['id']}/reject", json={"reason": "Not ours"})
    assert _propose(store, account, built).filed == []
    assert "Approval rate" not in kpi_proposals.waiting_metrics(account)


def test_learn_and_an_admin_at_once_file_each_suggestion_once(site):
    import threading

    _, account, store, built = site
    outcomes = []
    runs = [threading.Thread(target=lambda: outcomes.append(_propose(store, account, built))) for _ in range(2)]
    for run in runs:
        run.start()
    for run in runs:
        run.join()
    assert sorted(len(o.filed) for o in outcomes) == [0, 11]
    assert len(store.list_core2_requests(account, status="waiting")) == 11


def test_the_requests_page_says_where_a_suggestion_came_from_and_shows_its_last_months(site):
    client, account, store, built = site
    page = client.get(f"/admin/clients/{account}/requests").text
    assert "Suggest business metrics" in page and "rq-suggest" in page
    said = client.post(f"/admin/clients/{account}/requests/api/suggest", json={}).json()
    assert said["ok"] and said["kind"] == "bank" and len(said["filed"]) == 11
    page = client.get(f"/admin/clients/{account}/requests").text
    assert "Solviz, from what Learn found" in page
    assert "SUM(CASE WHEN [Loan application · Decision] = &#39;Approved&#39; THEN 1 ELSE 0 END)" in page
    assert "Its last months on the data" in page and "Accept for everyone" in page
    assert "Write it in the editor" not in page                       # it is written already
    again = client.post(f"/admin/clients/{account}/requests/api/suggest", json={}).json()
    assert again["ok"] and again["filed"] == []


def test_before_any_is_accepted_the_dashboard_is_as_it_was(site):
    _, account, store, built = site
    from core2.bootstrap.service import load_model

    before = recipes.executive(load_model(account, None), today=TODAY)
    _propose(store, account, built)
    after = recipes.executive(load_model(account, None), today=TODAY)
    assert [(t.title, t.tab) for t in before[1]] == [(t.title, t.tab) for t in after[1]]


def test_accepted_they_are_the_admins_metrics_and_the_dashboard_shows_them_in_their_areas(site):
    client, account, store, built = site
    from core2.bootstrap.service import load_model

    _propose(store, account, built)
    for r in store.list_core2_requests(account, status="waiting"):
        assert client.post(f"/admin/clients/{account}/requests/api/{r['id']}/accept", json={}).json()["ok"]
    model = load_model(account, None)
    kept = {m.kpi: m for m in model.measures.values() if m.kpi}
    assert set(kept) == BANK
    approval = kept["bank.approval_rate"]
    assert approval.key == "admin:approval_rate" and approval.provenance == "admin" and approval.format == "percent"
    name, made = recipes.executive(model, today=TODAY)
    assert name == "Banking executive dashboard"
    tiles = [dashboards.Tile(title=t.title, kind=t.kind, question=t.question, tab=t.tab, planned=t.plan)
             for t in made]
    tabs = dashboards.lay_out(tiles)
    assert tabs[0] == "Overview" and len(tabs) <= recipes.MOST_TABS
    # Eight tabs: the areas of the leading metrics, then the tables the overview's totals are counted on; the area
    # of a supporting metric (Service: compensation per complaint) gives way when they are full.
    assert tabs == ["Overview", "Lending", "Risk", "Repayments", "Cards", "Transactions", "Loans", "Loan payments"]
    assert len(tabs) == recipes.MOST_TABS == dashboards.MOST_TABS
    # Each tab's tiles as they are placed: row by row, left to right.
    on = {tab: sorted((t for t in tiles if t.tab == tab), key=lambda t: (t.layout["grid_y"], t.layout["grid_x"]))
          for tab in tabs}
    # The overview keeps its totals and adds the business's headline ratios, in one row of numbers.
    overview = [t.title for t in on["Overview"] if t.kind == "number"]
    assert overview[-2:] == ["Approval rate", "Delinquent loans"] and len(overview) <= recipes.MOST_NUMBERS
    # Each area: its numbers first, then by month, then by the business's grouping.
    assert [(t.kind, t.title) for t in on["Risk"]] == [
        ("number", "Delinquent loans"), ("number", "Loan-to-value"), ("number", "Charge-off rate"),
        ("trend", "Delinquent loans by month"), ("trend", "Loan-to-value by month"),
        ("breakdown", "Delinquent loans by product")]
    for tab in tabs:
        kinds = [t.kind for t in on[tab]]
        assert kinds == sorted(kinds, key=["number", "trend", "breakdown", "table"].index)
        assert len(kinds) <= dashboards.MOST_TILES
    # Every tile is a plan the resolver answers: drawn on the data, with no AI.
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_no_ai, index=MemberIndex(),
                        today=TODAY)
    payload = answer_question("Create an executive dashboard", services, Session())
    # Every tile built; one left out shows nothing another does not (core2/dashboards.py scored), and says so.
    assert payload["type"] == "dashboard_built" and [t for t in payload["not_built"] if not t["left_out"]] == []
    assert payload["waiting"] == []
    risk = next(t for t in payload["dashboard_tiles"] if t["title"] == "Delinquent loans" and t["tab"] == "Risk")
    assert risk["chart_type"] == "kpi"


def test_accepted_the_status_they_read_is_not_suggested_again_as_a_share(site):
    client, account, store, built = site
    from core2.bootstrap.service import load_model

    _propose(store, account, built)
    for r in store.list_core2_requests(account, status="waiting"):
        client.post(f"/admin/clients/{account}/requests/api/{r['id']}/accept", json={})
    model = load_model(account, None)
    assert kpis.proposals(model, "bank") == []
    # In French, the areas are named in French and a table's tab of the same subject is still the area's.
    name, made = recipes.executive(model, today=TODAY, language="fr")
    tabs = dashboards.lay_out([dashboards.Tile(title=t.title, kind=t.kind, question=t.question, tab=t.tab,
                                               planned=t.plan) for t in made])
    assert "Prêts" in tabs and "Risque" in tabs and len(tabs) == len(set(tabs)) <= recipes.MOST_TABS
    assert not {"Lending", "Risk"} & set(tabs)


def test_an_accepted_metric_edited_in_the_editor_stays_a_business_metric(site):
    client, account, store, built = site
    from core2.bootstrap.service import load_model

    _propose(store, account, built)
    request = next(r for r in store.list_core2_requests(account, status="waiting")
                   if r["target_name"] == "Average loan size")
    client.post(f"/admin/clients/{account}/requests/api/{request['id']}/accept", json={})
    model = load_model(account, None)
    measure = model.measures["admin:average_loan_size"]
    saved = client.post(f"/admin/clients/{account}/measures/api/save", json={
        "key": measure.key, "name": "Average loan", "formula": authoring.to_text(model, measure.expr),
        "format": "currency", "better": "neither"}).json()
    assert saved["ok"], saved
    edited = load_model(account, None).measures["admin:average_loan_size"]
    assert edited.business_name == "Average loan" and edited.kpi == "bank.average_loan_size"


@pytest.mark.parametrize("area, tab, one", [
    ("Shipping", "Shipment", True), ("Quality", "Quality tests", True), ("Claims", "Claims", True),
    ("Repayments", "Loan payments", False), ("Lending", "Loans", False), ("Risk", "Transactions", False)])
def test_a_tables_tab_goes_on_the_area_of_its_subject(area, tab, one):
    assert (recipes._stems(area) <= recipes._stems(tab)) is one


def test_an_executive_dashboard_names_the_metrics_waiting_for_an_admin(learned):
    built, model = learned("retail")

    def services(complete=_no_ai) -> Services:
        return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete, index=MemberIndex(),
                        today=TODAY, waiting=lambda: ["Average order value", "Return rate"])

    payload = answer_question("Create an executive dashboard", services(), Session())
    assert payload["waiting"] == ["Average order value", "Return rate"]
    import json

    def ai(stable: str, tail: str) -> str:
        if tail.startswith("REQUEST:"):
            return json.dumps({"name": "Revenue", "tiles": [
                {"title": "Net amount", "kind": "number", "question": "Net amount this year"}]})
        return json.dumps({"kind": "query", "measures": ["net_amount"],
                           "time": {"window": {"kind": "this", "unit": "year"}}})

    designed = answer_question("I need a dashboard with net amount, orders", services(ai), Session())
    assert designed["type"] == "dashboard_built" and designed["waiting"] == []


def test_the_chat_card_and_the_saved_card_carry_the_waiting_metrics():
    from pathlib import Path

    from core import i18n

    root = Path(__file__).resolve().parents[1]
    chat = (root / "portal/templates/portal_chat.html").read_text(encoding="utf-8")
    assert "ui.chat.dashmade.waiting" in chat and "msg.waiting" in chat
    en, fr = (i18n.t("ui.chat.dashmade.waiting", lang=lang, items="Approval rate") for lang in ("en", "fr"))
    assert "Approval rate" in en and "Approval rate" in fr and en != fr
    bridge = (root / "gateway/core2_bridge.py").read_text(encoding="utf-8")
    assert '"waiting": list(payload.get("waiting") or [])' in bridge


def test_learn_ends_by_suggesting_the_business_metrics():
    import inspect

    from core2.bootstrap import service

    source = inspect.getsource(service.build_workspace)
    assert "from core2.kpi_proposals import propose" in source and "say=say" in source
