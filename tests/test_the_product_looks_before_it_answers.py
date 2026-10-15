"""Before an answer is shown, the product looks at it as an analyst would, and says what it saw.

Nothing found: is there anything for those conditions in another period (when was the last, or the first), or
for the others in this period? A figure resting on two days of a month says so, and names the days. A period that
starts before the data does says where the data starts. What it checked is listed, and the next questions it
offers run with no AI -- clicked, typed, or answered "yes". A busy answer gets no reply: nothing to say.

All on invented retail data, each figure checked against SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 10, 15)          # four months after the invented retail data's last day (14 Jun 2026)
LINES = """SELECT o.*, d.full_date AS day, s.store_name, p.product_name, c.customer_name FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id
           LEFT JOIN products p ON o.product_id = p.product_id LEFT JOIN customers c ON o.customer_id = c.customer_id"""
MAY = {"window": {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}}
IN_MAY = "l.day >= DATE '2026-05-01' AND l.day < DATE '2026-06-01'"
LIVE = "TRUE"           # this learned model leaves no order line out by default (cancelled ones count)


def _net(product: str | None = None, customer: str | None = None) -> dict:
    filters = [{"field": "product.name", "op": "eq", "values": [product]}] if product else []
    filters += [{"field": "customer.name", "op": "eq", "values": [customer]}] if customer else []
    return {"measures": ["net_amount"], "filters": filters}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


class Chat:
    """One conversation, the AI's plans scripted; an AI call with no plan scripted fails the test."""

    def __init__(self, retail, today: dt.date = TODAY):
        built, model, warehouse, index = retail
        self.plans: list[dict] = []
        self.asked = 0

        def complete(system: str, tail: str) -> str:
            self.asked += 1
            assert self.plans, "the AI was asked, with no plan scripted"
            return json.dumps({"kind": "query", **self.plans.pop(0)})

        self.services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=today)
        self.session = Session()

    def ask(self, question: str, plan: dict | None = None) -> dict:
        if plan is not None:
            self.plans.append(plan)
        return answer_question(question, self.services, self.session)


def _one(retail, sql: str):
    return retail[0].con.execute(sql).fetchone()[0]


def _seen(got: dict, kind: str) -> dict:
    found = [s for s in (got.get("reply") or {}).get("seen") or [] if s["kind"] == kind]
    assert found, f"nothing of kind {kind} was seen: {got.get('reply')}"
    return found[0]


# ── nothing found ─────────────────────────────────────────────────────────────


def test_nothing_for_a_customer_who_bought_before_says_when_they_last_did(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales for "Lakeside Wholesale" in May 2026', {**_net(customer="Lakeside Wholesale"), "time": MAY})
    last = _one(retail, f"SELECT MAX(day) FROM ({LINES}) l WHERE customer_name = 'Lakeside Wholesale' AND {LIVE}")
    seen = _seen(got, "last_seen")
    assert seen["days"] == [last.isoformat()]
    reply = got["reply"]
    assert f"{last.day} {last:%b %Y}" in reply["en"]
    # The others, in the same month: so it is this customer, not the month.
    others = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {IN_MAY} AND {LIVE}")
    assert _seen(got, "elsewhere")["figures"][0] == pytest.approx(float(others))
    # Offers that look further, never "why did net amount change" about a figure that has none.
    labels = [o["label"] for o in reply["offers"]]
    assert any("by month" in o and "Lakeside Wholesale" in o for o in labels)
    assert any(f"{last:%B %Y}" in o for o in labels)
    assert not any(s["label"].startswith("Why did") for s in got["follow_up_suggestions"])
    assert reply["checked"], "what was checked is listed"


def test_a_product_never_sold_is_said_so(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales of "Orange Juice Small" in May 2026', {**_net(product="Orange Juice Small"), "time": MAY})
    assert _one(retail, f"SELECT COUNT(*) FROM ({LINES}) l WHERE product_name = 'Orange Juice Small'") == 0
    _seen(got, "never")
    others = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {IN_MAY} AND {LIVE}")
    assert _seen(got, "elsewhere")["figures"][0] == pytest.approx(float(others))
    assert "Orange Juice Small" in got["reply"]["en"]
    assert "any period" in got["reply"]["en"]


# ── values on only a few days ─────────────────────────────────────────────────


def _days(retail, product: str) -> list[str]:
    rows = retail[0].con.execute(f"""SELECT DISTINCT day FROM ({LINES}) l WHERE product_name = '{product}'
        AND {IN_MAY} AND {LIVE} ORDER BY 1""").fetchall()
    return [r[0].isoformat() for r in rows]


def test_a_figure_resting_on_two_days_of_the_month_says_so(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    days = _days(retail, "Toothpaste Small")
    assert len(days) == 2
    seen = _seen(got, "few_days")
    assert seen["days"] == days and seen["figures"][:2] == [2, 31]
    reply = got["reply"]
    assert "Only 2 of the 31 days in May 2026" in reply["en"]
    for day in days:
        assert f"{dt.date.fromisoformat(day).day} May" in reply["en"]
    assert "Seuls 2 des 31 jours" in reply["fr"]
    assert any("by day" in o["label"] for o in reply["offers"])
    # The figure itself is the hand-written one.
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE product_name = 'Toothpaste Small' AND {IN_MAY} AND {LIVE}")
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))


def test_a_series_with_few_days_says_so_from_its_own_rows(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" by day in May 2026',
                   {**_net(product="Toothpaste Small"), "time": {**MAY, "grain": "day"}})
    seen = _seen(got, "few_days")
    assert seen["days"] == _days(retail, "Toothpaste Small")


def test_a_period_starting_before_the_data_says_where_the_data_starts(retail):
    chat = Chat(retail)
    got = chat.ask("Net sales from October 2022 to March 2023", {
        "measures": ["net_amount"], "time": {"window": {"kind": "between", "start": "2022-10-01", "end": "2023-03-31"}}})
    first = _one(retail, f"SELECT MIN(day) FROM ({LINES}) l WHERE {LIVE}")
    seen = _seen(got, "starts_late")
    assert seen["days"] == [first.isoformat()]
    assert f"{first.day} {first:%b %Y}" in got["reply"]["en"]


def test_a_busy_answer_gets_no_reply(retail):
    chat = Chat(retail)
    got = chat.ask("Net sales in May 2026", {"measures": ["net_amount"], "time": MAY})
    assert not got.get("reply")


# ── the next question, with no AI ────────────────────────────────────────────


def test_yes_runs_the_first_offer_with_no_ai(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    first = got["reply"]["offers"][0]
    asked = chat.asked
    again = chat.ask("yes please")
    assert chat.asked == asked, "a yes is no new question for the AI"
    assert again["question"] == "yes please"
    rows = again["data"]["rows"]
    assert len([r for r in rows if r.get("net_amount")]) == 2, first
    # And the offer clicked by its words runs the same, with no AI either.
    chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    clicked = chat.ask(first["question"])
    assert chat.asked == asked + 1 and clicked["data"]["rows"] == rows


def test_yes_with_nothing_offered_is_a_question_like_any_other(retail):
    chat = Chat(retail)
    chat.ask("Net sales in May 2026", {"measures": ["net_amount"], "time": MAY})
    chat.ask("yes, and by store", {"measures": ["net_amount"], "group_by": ["store.name"], "time": MAY,
                                    "follow_up": "refine"})
    assert chat.asked == 2


# ── the reply in the workspace's AI's words, checked ─────────────────────────


def test_the_ai_says_the_reply_in_its_own_words_and_every_figure_is_checked(retail):
    from core2.agent import talk

    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    reply, headline = got["reply"], got["answer"]["headline"]
    asked: list[str] = []

    def says(text: str):
        def complete(system: str, given: str) -> str:
            asked.append(given)
            return text
        return complete

    good = ("Toothpaste Small only sold on 2 of the 31 days in May 2026, the 17th and the 27th, so the $211.97 comes "
            "from those two days alone. Want to see them day by day?")
    assert talk.write(reply, "Net sales of Toothpaste Small in May 2026", headline, says(good)) == good
    # What the AI was given: the checks' sentences and the next questions, never the rows.
    assert "Only 2 of the 31 days" in asked[0] and "by day" in asked[0] and "net_amount" not in asked[0]
    # A figure the checks did not find, a cause they do not give, a list: the code's own sentences stand.
    for bad in ("Toothpaste Small sold on 2 days, about 6% of the month.",
                "Sales were low because the product was out of stock on 29 of the days.",
                "- 17 May\n- 27 May"):
        assert talk.write(reply, "q", headline, says(bad)) is None, bad


def test_the_ai_is_never_asked_where_the_reply_names_people_or_kept_members():
    from core2.agent import talk

    reply = {"seen": [{"kind": "last_seen", "en": "Nothing for customer Ann Lee in May 2026.", "fr": "",
                       "figures": [], "days": [], "members": ["Ann Lee"]}], "offers": []}
    assert talk.names_members(reply)


# ── small talk, with no AI ────────────────────────────────────────────────────


@pytest.mark.parametrize("said,lang", [("thanks!", "en"), ("Thank you so much", "en"), ("merci beaucoup", "fr"),
                                       ("hi", "en"), ("bonjour", "fr")])
def test_thanks_and_hello_are_answered_in_the_readers_language_with_no_ai(retail, said, lang):
    chat = Chat(retail)
    got = chat.ask(said)
    assert chat.asked == 0 and got.get("kind") == "smalltalk"
    text = got["answer"]["headline"]
    if lang == "fr":
        assert any(w in text for w in ("Avec plaisir", "Bonjour"))
    else:
        assert any(w in text for w in ("welcome", "Hello"))


# ── found while looking: a target is set per what its rows reach ─────────────


def test_one_products_sales_are_never_set_against_the_stores_whole_target(retail):
    # The analyst note under "net sales of Toothpaste Small in May 2026" said "at 0.1% of its target: $212 against
    # $196.4K": the store's whole target for the month, against one product's sales. A target is set per store and
    # month here; a figure for one product has none, and is never set against one.
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    assert not any(f.get("kind") == "target" for f in (got.get("reading") or {}).get("facts") or [])
    asked = chat.ask('Net sales of "Toothpaste Small" against target in May 2026', {
        "target": {"actual": "net_amount", "target": "target_amount"}, "time": MAY,
        "filters": [{"field": "product.name", "op": "eq", "values": ["Toothpaste Small"]}]})
    assert "no target amount per product name" in asked["answer"]["headline"]
    # A store keeps one: its figure is set against it (a conversation of its own: the product is not carried).
    store = Chat(retail).ask('Net sales of "Airport Store" against target in May 2026', {
        "target": {"actual": "net_amount", "target": "target_amount"}, "time": MAY,
        "filters": [{"field": "store.name", "op": "eq", "values": ["Airport Store"]}]})
    target = _one(retail, """SELECT SUM(t.target_amount) FROM sales_targets t JOIN stores s ON t.store_id = s.store_id
        WHERE s.store_name = 'Airport Store' AND t.period_key = 202605""")     # targets are kept by month
    assert float(store["export_rows"][0]["target_amount"]) == pytest.approx(float(target))


def test_a_reason_guessed_rather_than_found_is_never_said(retail):
    from core2.agent import talk

    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {**_net(product="Toothpaste Small"), "time": MAY})
    for guess in ("Only 2 of the 31 days had sales, which could indicate partial data for this product.",
                  "Seuls 2 des 31 jours ont des ventes : cela pourrait indiquer des données partielles."):
        assert talk.write(got["reply"], "q", got["answer"]["headline"], lambda s, g, t=guess: t) is None, guess
