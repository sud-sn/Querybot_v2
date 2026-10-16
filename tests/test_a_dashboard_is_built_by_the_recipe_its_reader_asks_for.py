"""A dashboard asked for by subject or by reader is built by its recipe from the catalogue, in code.

"Make me a dashboard" built one of eleven executive overviews, all the same shape; any other request was designed
by the AI afresh each time. The catalogue adds 24 recipes by subject and reader (sales performance for the head
of sales, customer health for marketing, a store scorecard for regional managers, supplier performance, budget
against actual, workforce, recurring revenue, network operations...). Each finds its figures, groupings and
lists by the workspace's own names, keeps a tile only when the resolver answers its plan, and says what the data
lacks: ready, in part ("no targets in this data"), or not for this data ("needs employees with start and end
dates"). A request names its recipe by its words ("for the sales team", "customer retention", "a CFO
dashboard"); a request that lists its tiles is still built as it lists them.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2 import dashboards, recipe_book
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def learned():
    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = learn(domains.build(name), "descriptive")
        return cache[name]

    return get


def _states(learned, name: str) -> dict[str, str]:
    _, model = learned(name)
    return {c["key"]: c["state"] for c in recipe_book.catalogue(model, today=TODAY)}


# Each business: the recipes its data gives in full.
READY = {
    "retail": {"sales_overview", "sales_performance", "customer_health", "product_and_category", "store_scorecard",
               "returns_and_quality", "monthly_review"},
    "hr": {"workforce_overview", "workforce_cost"},
    "subscriptions": {"recurring_revenue", "subscriber_retention"},
    "networking": {"network_operations"},
    "purchasing": {"spend_analysis"},
    "inventory": {"inventory_health"},
    "compounding_pharmacy": {"pharmacy_operations"},
}


@pytest.mark.parametrize("name", sorted(READY))
def test_each_business_has_the_recipes_its_data_gives(learned, name):
    states = _states(learned, name)
    assert READY[name] <= {k for k, s in states.items() if s == "ready"}, states


@pytest.mark.parametrize("name", ["hr", "purchasing", "inventory", "finance"])
def test_no_sales_recipe_is_built_from_pay_purchases_stock_or_a_ledger(learned, name):
    states = _states(learned, name)
    sales = [b.key for b in recipe_book.BOOKS if b.group == "sales"]
    assert all(states[k] == "no" for k in sales), {k: states[k] for k in sales}


def test_a_recipe_the_data_cannot_give_says_what_it_needs(learned):
    _, model = learned("retail")
    built = recipe_book.build(model, "workforce_overview", today=TODAY)
    assert built.state == "no" and built.tiles == [] and built.missing == ["employees with start and end dates"]
    assert recipe_book.build(model, "subscriber_retention", today=TODAY).state == "no"   # customers are no subscribers


def test_a_ledger_kept_in_several_currencies_says_so(learned):
    _, model = learned("finance")
    built = recipe_book.build(model, "profit_and_loss", today=TODAY)
    assert built.state == "no" and "one currency" in built.missing[0]


def test_a_recipe_given_in_part_says_what_is_missing(learned):
    _, model = learned("retail")
    pnl = recipe_book.build(model, "profit_and_loss", today=TODAY)
    assert pnl.state == "partly" and pnl.missing == ["operating costs"]
    assert {"Net amount", "Cost amount", "Gross margin"} <= {t.title for t in pnl.tiles}


def test_every_tile_is_planned_in_code_and_answered_with_no_ai(learned):
    built, model = learned("retail")
    recipe = recipe_book.build(model, "sales_performance", today=TODAY)
    titles = [t.title for t in recipe.tiles]
    assert {"Net amount", "Number of orders", "Average order", "Net amount by month against the year before",
            "Net amount by region and month", "Stores ranked by net amount, with change",
            "Net amount against target amount by store", "Biggest risers: products",
            "Biggest fallers: products"} <= set(titles), titles

    def no_ai(stable: str, tail: str) -> str:
        raise AssertionError("the AI was asked")

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=no_ai, index=MemberIndex(),
                        today=TODAY)
    from core2.plan.ir import Plan
    from core2.service import _answer_question

    for tile in recipe.tiles:
        payload = _answer_question(tile.question, services, Session(), planned=Plan.model_validate(tile.plan))
        assert payload["data"]["rows"], tile.title


def test_the_numbers_lead_each_tab_and_the_same_data_gives_the_same_dashboard(learned):
    _, model = learned("retail")
    first, again = (recipe_book.build(model, "sales_overview", today=TODAY) for _ in range(2))
    assert [(t.title, t.tab, t.plan) for t in first.tiles] == [(t.title, t.tab, t.plan) for t in again.tiles]
    for tab in dict.fromkeys(t.tab for t in first.tiles):
        kinds = [t.kind for t in first.tiles if t.tab == tab]
        assert kinds == sorted(kinds, key=["number", "trend", "breakdown", "table"].index)


def test_staff_and_subscriptions_are_told_in_their_own_words(learned):
    _, hr = learned("hr")
    titles = {t.title for t in recipe_book.build(hr, "workforce_overview", today=TODAY).tiles}
    assert {"Hires by month", "Leavers by month", "Attrition rate by month", "Employees at each month end"} <= titles
    _, subs = learned("subscriptions")
    titles = {t.title for t in recipe_book.build(subs, "recurring_revenue", today=TODAY).tiles}
    assert {"MRR", "ARR", "New subscriptions by month", "Subscriptions ended by month"} <= titles
    _, network = learned("networking")
    titles = {t.title for t in recipe_book.build(network, "network_operations", today=TODAY).tiles}
    assert "Average days from opened to resolved" in titles


def test_no_person_is_named_on_a_workforce_dashboard(learned):
    _, hr = learned("hr")
    for key in ("workforce_overview", "workforce_cost"):
        for tile in recipe_book.build(hr, key, today=TODAY).tiles:
            assert "employee" not in (tile.plan.get("group_by") or []), tile.title


def test_a_reader_is_given_only_what_their_tables_hold(learned):
    _, model = learned("retail")
    allowed = {k for k in model.tables if not k.endswith(".returns")}
    built = recipe_book.build(model, "returns_and_quality", today=TODAY, allowed=allowed)
    assert built.state == "no"


def test_a_recipe_in_french_is_named_and_titled_in_french(learned):
    _, model = learned("retail")
    built = recipe_book.build(model, "sales_performance", today=TODAY, lang="fr")
    assert built.name == "Performance commerciale"
    assert "Net amount par mois par rapport à l'année précédente" in {t.title for t in built.tiles}
    assert built.tiles[0].tab == "Vue d'ensemble"


@pytest.mark.parametrize("request_, first", [
    ("a dashboard for the sales team", "sales_performance"),
    ("customer retention dashboard", "customer_health"),
    ("something for my store managers", "store_scorecard"),
    ("CFO dashboard", "profit_and_loss"),
    ("board pack for this month", "monthly_review"),
    ("budget vs actual by cost centre", "budget_against_actual"),
    ("headcount and attrition", "workforce_overview"),
    ("MRR", "recurring_revenue"),
    ("the Sales performance dashboard", "sales_performance"),
    ("tableau de bord pour l'équipe commerciale", "sales_performance"),
])
def test_a_request_finds_its_recipe_by_its_words(request_, first):
    assert recipe_book.find(request_)[0] == first


@pytest.mark.parametrize("request_", ["returns by courier and weekday", "net amount, orders and refunds", "sales",
                                      "revenue"])
def test_a_request_that_lists_its_tiles_or_names_a_figure_is_no_recipe(request_):
    from core2.dashboards import _lists

    assert recipe_book.find(request_, lists=_lists(request_)) == []


def _no_ai(stable: str, tail: str) -> str:
    raise AssertionError(f"the AI was asked: {tail[:80]}")


def test_a_reader_asking_for_their_recipe_gets_it_built_with_no_ai(learned):
    built, model = learned("retail")
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_no_ai, index=MemberIndex(),
                        today=TODAY)
    payload = answer_question("Create a dashboard for the sales team", services, Session())
    assert payload["type"] == "dashboard_built" and payload["name"] == "Sales performance"
    assert payload["not_built"] == [] and len(payload["dashboard_tiles"]) >= 9
    assert payload["tabs"] == ["Overview", "Stores and products"]


def test_a_recipe_the_data_cannot_give_leaves_the_request_to_the_design(learned):
    built, model = learned("retail")
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        if tail.startswith("REQUEST:"):
            return json.dumps({"name": "People", "tiles": [
                {"title": "Net amount", "kind": "number", "question": "Net amount this year"},
                {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month this year"},
                {"title": "Net amount by region", "kind": "breakdown", "question": "Net amount by region this year"}]})
        return json.dumps({"kind": "query", "measures": ["net_amount"],
                           "time": {"window": {"kind": "this", "unit": "year"}}})

    assert dashboards.from_book(model, "Create a workforce dashboard", today=TODAY) is None
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(),
                        today=TODAY)
    answer_question("Create a workforce dashboard", services, Session())
    assert asked and asked[0].startswith("REQUEST:")


def test_the_catalogue_says_for_each_recipe_what_the_data_gives(learned):
    _, model = learned("retail")
    catalogue = recipe_book.catalogue(model, today=TODAY)
    assert len(catalogue) == 24 == len({c["key"] for c in catalogue})
    performance = next(c for c in catalogue if c["key"] == "sales_performance")
    assert performance["group"] == "Sales and customers" and performance["reader"] == "Head of sales"
    assert performance["state"] == "ready" and performance["tiles"] >= 9 and "order line" in performance["uses"]
    assert recipe_book.find("Create a Sales performance dashboard")[0] == "sales_performance"
    workforce = next(c for c in catalogue if c["key"] == "workforce_overview")
    assert workforce["state"] == "no" and workforce["needs"] == "employees with start and end dates"


def test_a_trend_against_the_year_before_draws_the_year_before_too(learned):
    """A recipe's "net amount by month against the year before" drew this year's line alone: the year before is
    now drawn too, each named by its period -- as bars side by side with the change under them (BA2), a line a
    click away."""
    built, model = learned("retail")
    from core2.plan.ir import Plan
    from core2.service import _answer_question

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_no_ai, index=MemberIndex(),
                        today=TODAY)
    plan = {"kind": "query", "intent": "trend", "measures": ["net_amount"],
            "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12},
                     "compare": {"kind": "same_period_last_year"}}}
    chart = _answer_question("q", services, Session(), planned=Plan.model_validate(plan))["chart"]
    assert chart["chart_type"] == "bar" and chart["y_keys"] == ["net_amount", "net_amount_prior"]
    assert chart["compare"]["strip"] is True and "line" in chart["renderable_types"]
    assert chart["compare"]["prior"] == "net_amount_prior"
    assert chart["compare"]["prior_label"] == "Jun 2024 – May 2025"


def test_a_target_answer_is_read_by_its_own_headline_not_as_a_breakdown(learned):
    """ "Net amount against target amount by store" was read as a breakdown of net amount ("Old Town Store leads
    with 12% of $1.0M"): its own headline says how many stores reached their target."""
    built, model = learned("retail")
    from core2.plan.ir import Plan
    from core2.service import _answer_question

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_no_ai, index=MemberIndex(),
                        today=TODAY)
    plan = {"kind": "query", "intent": "breakdown", "target": {"actual": "net_amount", "target": "target_amount"},
            "group_by": ["store"], "time": {"window": {"kind": "to_date", "unit": "year"}}}
    payload = _answer_question("q", services, Session(), planned=Plan.model_validate(plan))
    assert "reached their target amount" in payload["answer"]["headline"]
    assert not (payload.get("reading") or {}).get("facts")
