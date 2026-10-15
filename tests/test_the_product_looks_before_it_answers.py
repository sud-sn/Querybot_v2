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


def test_a_why_naming_one_month_is_that_month_not_the_year_kept_from_above(retail):
    # "why did spend spike in nov 2025??" after an answer for 2025 was planned for all of 2025 by month, and the why
    # compared December with November.
    chat = Chat(retail)
    got = chat.ask("why did net sales drop in may 2026??", {
        "intent": "drivers", "measures": ["net_amount"], "follow_up": "refine",
        "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}}})
    whole = float(_one(retail, f"SELECT {CHANGE} FROM ({LINES}) l"))
    assert float(got["drivers"]["change"]) == pytest.approx(whole, abs=0.01)
    # A span is the AI's to read.
    span = Chat(retail).ask("net sales from jan 2026 to may 2026", {
        "measures": ["net_amount"], "time": {"window": {"kind": "between", "start": "2026-01-01",
                                                        "end": "2026-05-31"}}})
    assert span["plan"]["time"]["window"]["start"] == "2026-01-01"


def test_by_day_is_each_day_never_the_days_of_the_week(retail):
    # "... by day in June 2026" was planned with the day of the week too: the answer ranked Monday to Sunday.
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" by day in May 2026', {
        **_net(product="Toothpaste Small"), "intent": "breakdown", "group_by": ["time:day_of_week"],
        "time": {**MAY, "grain": "day"}})
    days = _days(retail, "Toothpaste Small")
    assert [str(r["period"])[:10] for r in got["export_rows"]] == days
    seen = _seen(got, "few_days")
    assert seen["days"] == days
    # Asked by weekday, it stays by weekday.
    weekday = Chat(retail).ask("Net sales by day of the week in May 2026", {
        "measures": ["net_amount"], "intent": "breakdown", "group_by": ["time:day_of_week"],
        "time": {**MAY, "grain": "day"}})
    assert "day_of_week" in weekday["export_rows"][0]


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


def test_the_ais_words_are_checked_as_people_write_figures():
    from core2.agent import talk

    steps = {"seen": [{"kind": "steps", "figures": [-12330.23, -5796.81, -5053.9], "days": [],
                       "members": ["Lakeshore Store"]}], "offers": [{"label": "Why did it change, for store?"}]}
    # A fall said as a fall ("down $12.3K"), and French figures ("5 797 $", "12,3 k $") are the checks' own.
    for said, lang in (("Net sales fell by $5,796.81: Lakeshore Store was down $12.3K, Online $5,054. Look closer?",
                        "en"),
                       ("Les ventes ont baissé de 5 797 $, dont 12,3 k $ pour Lakeshore Store. On regarde de plus près ?",
                        "fr")):
        assert talk.checked(said, steps, "", lang) == said, said
    assert talk.checked("Lakeshore Store fell by 5 900 $.", steps, "", "fr") is None
    # No next question offered: a question it ends on has no button to answer it, and is left out.
    late = {"seen": [{"kind": "starts_late", "figures": [], "days": ["2023-01-01"]}], "offers": []}
    assert talk.checked("The data starts on 1 Jan 2023. Would you like to change the period?", late, "") == \
        "The data starts on 1 Jan 2023."
    # An inference the checks never made: data missing, demand low.
    few = {"seen": [{"kind": "few_days", "figures": [2, 31], "days": ["2026-05-17", "2026-05-27"]}], "offers": []}
    for guess, lang in (("Sales came on 2 of the 31 days. This suggests demand was limited.", "en"),
                        ("Seuls 2 des 31 jours ont des ventes : les chiffres pourraient ne pas refléter tout le mois.",
                         "fr"),
                        ("Only 2 of the 31 days have sales; the figures may not reflect the whole month.", "en"),
                        # the pharmacy book: lots received in 3 months of 2025 said as data missing for the others
                        ("Nous avons reçu 3 lots en 2025, mais les données ne couvrent que les mois d'avril, "
                         "septembre et décembre.", "fr"),
                        ("Les informations pour les autres mois ne sont pas disponibles.", "fr"),
                        ("Sales happened on 2 of the 31 days, so the data is incomplete for May.", "en")):
        assert talk.checked(guess, few, "", lang) is None, guess
    # A day said as an ordinal is one of the days seen: "the first 10 May" was read back as "the 1st to the 10th".
    seven = {"seen": [{"kind": "few_days", "figures": [7, 31], "days": ["2026-05-10", "2026-05-28"]}], "offers": []}
    assert talk.checked("It comes from 7 days in May: the 1st to the 10th and the 28th.", seven, "") is None
    assert talk.checked("It comes from 7 days in May, the first on the 10th and the last on the 28th.", seven, "")
    # Where the data itself starts inside the period, saying it does not cover all of it is what was found.
    assert talk.checked("The figure is only from 2 Oct 2023: the data does not cover the months before.",
                        {"seen": [{"kind": "starts_late", "figures": [], "days": ["2023-10-02"]}], "offers": []},
                        "") is not None


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


# ── a name the data does not have: the nearest, checked first ────────────────


def test_a_name_written_wrong_offers_the_nearest_with_its_figure(retail):
    chat = Chat(retail)
    got = chat.ask('Net sales of "Downtwn Store" in May 2026', {
        "measures": ["net_amount"], "time": MAY, "filters": [{"field": "store.name", "op": "eq",
                                                               "values": ["Downtwn Store"]}]})
    text = got["answer"]["headline"]
    assert 'called "Downtwn Store"' in text and 'Did you mean "Downtown Store"?' in text
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE store_name = 'Downtown Store' AND {IN_MAY}")
    assert f"{float(want):,.2f}"[:5] in text.replace("$", "") or "K" in text     # its figure, as the answer says it
    offer = got["follow_up_suggestions"][0]["question"]
    assert offer == 'Net sales of "Downtown Store" in May 2026'
    asked = chat.asked
    again = chat.ask("yes")
    assert chat.asked == asked
    assert float(again["kpi"]["value"]) == pytest.approx(float(want))


def test_a_name_written_wrong_on_the_store_itself_offers_the_nearest(retail):
    # gpt-4o put "Lakeshore Stor" on the store, not the store's name field: the answer said "there is no store
    # called ..." and offered nothing. The store's members are its names.
    chat = Chat(retail)
    got = chat.ask('Net sales of "Lakeshore Stor" in May 2026', {
        "measures": ["net_amount"], "time": MAY, "filters": [{"field": "store", "op": "eq", "values": ["Lakeshore Stor"]}]})
    assert 'Did you mean "Lakeshore Store"?' in got["answer"]["headline"]
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE store_name = 'Lakeshore Store' AND {IN_MAY}")
    asked = chat.asked
    again = chat.ask("yes")
    assert chat.asked == asked and float(again["kpi"]["value"]) == pytest.approx(float(want))


def test_a_quoted_name_the_ai_left_out_is_never_answered_for_every_one(retail):
    # 'How much did customers spend on "Home improvment" in May 2026?' was planned with no condition at all, and
    # answered for every category: the name written wrong was dropped without a word.
    chat = Chat(retail)
    got = chat.ask('Net sales of "Downtwn Store" in May 2026', {"measures": ["net_amount"], "time": MAY})
    assert 'Did you mean "Downtown Store"?' in got["answer"]["headline"]
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE store_name = 'Downtown Store' AND {IN_MAY}")
    asked = chat.asked
    again = chat.ask("yes")
    assert chat.asked == asked and float(again["kpi"]["value"]) == pytest.approx(float(want))
    # Quoted words that are no name of the data, nor written like one, are left as the AI read them.
    plain = Chat(retail).ask('What were "net sales" in May 2026?', {"measures": ["net_amount"], "time": MAY})
    total = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {IN_MAY}")
    assert float(plain["kpi"]["value"]) == pytest.approx(float(total))


def test_a_name_written_wrong_on_another_grouping_offers_the_nearest_where_it_is(retail):
    # "Key Acounts" was put on the account, where no name is written like it: the cost centre "Key Accounts" is.
    chat = Chat(retail)
    got = chat.ask('Net sales of "Downtwn Store" in May 2026', {
        "measures": ["net_amount"], "time": MAY, "filters": [{"field": "product.name", "op": "eq",
                                                               "values": ["Downtwn Store"]}]})
    assert 'Did you mean "Downtown Store"?' in got["answer"]["headline"]
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE store_name = 'Downtown Store' AND {IN_MAY}")
    asked = chat.asked
    again = chat.ask("yes")
    assert chat.asked == asked and float(again["kpi"]["value"]) == pytest.approx(float(want))
    assert again["plan"]["filters"] == [{"field": "store.name", "op": "eq", "values": ["Downtown Store"]}]


def test_a_quoted_name_is_filtered_whole_never_as_its_pieces(retail):
    # gpt-4o split "VersaBase cream" (an ingredient) into an FDA name "VersaBase" and a dosage form "Cream", and
    # the answer said there was no "VersaBase".
    chat = Chat(retail)
    got = chat.ask('Net sales of "Toothpaste Small" in May 2026', {
        "measures": ["net_amount"], "time": MAY, "filters": [{"field": "product.name", "op": "eq",
                                                               "values": ["Toothpaste"]}]})
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE product_name = 'Toothpaste Small' AND {IN_MAY}")
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    assert got["plan"]["filters"] == [{"field": "product.name", "op": "eq", "values": ["Toothpaste Small"]}]


def test_a_name_never_offers_one_the_reader_cannot_see(retail):
    # Each name offered is asked first through the reader's own access: a member their rows do not reach has none.
    built, model, warehouse, index = retail

    class Blind:
        dialect = warehouse.dialect

        def query(self, sql: str, max_rows: int | None = None):
            result = warehouse.query(sql, max_rows=max_rows)
            if "Downtown Store" in sql:
                return type(result)(result.columns, [tuple(None for _ in r) for r in result.rows][:1],
                                    *([] if not hasattr(result, "truncated") else [False]))
            return result

    chat = Chat(retail)
    chat.services.warehouse = Blind()
    got = chat.ask('Net sales of "Downtwn Store" in May 2026', {
        "measures": ["net_amount"], "time": MAY, "filters": [{"field": "store.name", "op": "eq",
                                                               "values": ["Downtwn Store"]}]})
    assert "Did you mean" not in got["answer"]["headline"]


# ── looking again: the AI asked once more when its plan looks misread ────────


class Told(Chat):
    """A conversation that keeps what the AI was told at each call."""

    def __init__(self, retail, today: dt.date = TODAY):
        super().__init__(retail, today)
        self.told: list[str] = []
        first = self.services.complete

        def complete(system: str, tail: str) -> str:
            self.told.append(tail)
            return first(system, tail)

        self.services.complete = complete


def _rate(status: str) -> dict:
    return {"rates": [{"name": "Cancelled share", "measure": "number_of_order_lines",
                       "where": [{"field": "order_line.status_code", "op": "eq", "values": [status]}]}], "time": MAY}


def _share(retail, status: str) -> float:
    return float(_one(retail, f"""SELECT 100.0 * COUNT(*) FILTER (WHERE status_code = '{status}') / COUNT(*)
        FROM ({LINES}) l WHERE {IN_MAY}"""))


def test_a_share_of_exactly_none_is_read_again_and_the_second_reading_kept(retail):
    chat = Told(retail)
    chat.plans += [_rate("S"), _rate("C")]       # the first reading took the wrong status; the second the right one
    got = chat.ask("What share of order lines were cancelled in May 2026?")
    assert len(chat.told) == 2 and "A FIRST PLAN FOR THIS QUESTION" in chat.told[1] and "exactly 0%" in chat.told[1]
    assert float(got["kpi"]["value"]) == pytest.approx(_share(retail, "C"), abs=0.01)
    reply = got["reply"]
    assert reply["seen"][0]["kind"] == "looked_again" and "read your question again" in reply["en"]
    assert any(o["label"] == "Show my first reading" for o in reply["offers"])
    assert got["trust"]["looked_again"]["first_plan"]["rates"][0]["where"][0]["values"] == ["S"]
    # The first reading, one click away, with no AI.
    first = chat.ask("Show my first reading")
    assert len(chat.told) == 2 and float(first["kpi"]["value"]) == 0


def test_nothing_found_by_a_condition_the_question_never_states_is_read_again(retail):
    chat = Told(retail)
    chat.plans += [{**_net(customer="Lakeside Wholesale"), "time": MAY}, {"measures": ["net_amount"], "time": MAY}]
    got = chat.ask("Net sales in May 2026")
    assert "Customer name is Lakeside Wholesale" in chat.told[1] or "customer name is Lakeside" in chat.told[1].lower()
    want = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {IN_MAY}")
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    assert got["reply"]["seen"][0]["kind"] == "looked_again"


def test_a_condition_the_question_states_is_never_second_guessed(retail):
    chat = Told(retail)
    got = chat.ask('Net sales for "Lakeside Wholesale" in May 2026', {**_net(customer="Lakeside Wholesale"), "time": MAY})
    assert len(chat.told) == 1 and "looked_again" not in got.get("trust", {})
    _seen(got, "last_seen")


def test_a_second_reading_that_puts_nothing_right_leaves_the_first_answer(retail):
    chat = Told(retail)
    chat.plans += [_rate("S"), _rate("O")]       # read again, still a share of none
    got = chat.ask("What share of order lines were cancelled in May 2026?")
    assert len(chat.told) == 2 and float(got["kpi"]["value"]) == 0
    assert got["trust"]["looked_again"]["kept"] == "first"
    # The conversation is the first reading's: a follow-up goes on from it.
    assert chat.session.turns[-1].question == "What share of order lines were cancelled in May 2026?"
    assert len(chat.session.turns) == 1


def test_member_names_never_reach_the_ai_when_it_looks_again_where_they_may_not(retail):
    chat = Told(retail)
    chat.services.values_allowed = False
    chat.plans += [{**_net(customer="Lakeside Wholesale"), "time": MAY}, {"measures": ["net_amount"], "time": MAY}]
    chat.ask("Net sales in May 2026")
    assert len(chat.told) == 2 and "Lakeside" not in chat.told[1] and "<value>" in chat.told[1]


def test_nothing_found_because_of_a_condition_kept_from_the_question_before_is_said(retail):
    chat = Chat(retail)
    chat.ask('Net sales for "Lakeside Wholesale" in January 2026', {
        **_net(customer="Lakeside Wholesale"), "time": {"window": {"kind": "between", "start": "2026-01-01",
                                                                   "end": "2026-01-31"}}})
    got = chat.ask("and in May 2026?", {**_net(customer="Lakeside Wholesale"), "time": MAY, "follow_up": "refine"})
    seen = _seen(got, "carried")
    assert "I kept the condition for customer Lakeside Wholesale from the question before" in seen["en"]
    others = _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {IN_MAY}")
    assert seen["figures"][0] == pytest.approx(float(others))
    offer = next(o for o in got["reply"]["offers"] if "Lakeside" not in o["label"])
    assert got["reply"]["offers"][0] == offer        # the first button: likely the condition is no longer meant
    asked = chat.asked
    plain = chat.ask(offer["question"])
    assert chat.asked == asked and float(plain["kpi"]["value"]) == pytest.approx(float(others))
    # "yes" asks the same
    again = Chat(retail)
    again.ask('Net sales for "Lakeside Wholesale" in January 2026', {
        **_net(customer="Lakeside Wholesale"), "time": {"window": {"kind": "between", "start": "2026-01-01",
                                                                   "end": "2026-01-31"}}})
    again.ask("and in May 2026?", {**_net(customer="Lakeside Wholesale"), "time": MAY, "follow_up": "refine"})
    asked = again.asked
    said_yes = again.ask("yes")
    assert again.asked == asked and float(said_yes["kpi"]["value"]) == pytest.approx(float(others))


# ── two fair readings: each one worked out before asking ─────────────────────


def test_a_question_two_dates_could_answer_is_asked_with_each_ones_figure(retail):
    chat = Chat(retail)
    got = chat.ask("How many customers in May 2026?", {"measures": ["number_of_customers"], "time": MAY})
    figures = got["clarify"]["figures"]
    ordered = _one(retail, f"SELECT COUNT(DISTINCT customer_id) FROM ({LINES}) l WHERE {IN_MAY}")
    returned = _one(retail, """SELECT COUNT(DISTINCT r.customer_id) FROM returns r JOIN calendar d
        ON r.return_date_key = d.date_key WHERE d.full_date >= DATE '2026-05-01' AND d.full_date < DATE '2026-06-01'""")
    assert figures["Order date"] == f"{ordered:,}" and figures["Return date"] == f"{returned:,}"
    assert f"Order date ({ordered:,})" in got["answer"]["headline"]
    # The reply picks one by name, with no AI: the waiting question is answered by it.
    asked = chat.asked
    picked = chat.ask("Order date")
    assert chat.asked == asked and float(picked["kpi"]["value"]) == ordered


def _same_orders():
    import duckdb
    import numpy as np
    import pandas as pd

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb

    rng = np.random.default_rng(5)
    customers = pd.DataFrame({"customer_id": np.arange(1, 21), "customer_name": [f"Client {i:02d}" for i in range(1, 21)]})
    bill = rng.integers(1, 21, 800)
    orders = pd.DataFrame({"order_id": np.arange(1, 801), "bill_to_customer_id": bill, "ship_to_customer_id": bill,
                           "order_amount": np.round(rng.uniform(10, 900, 800), 2)})
    con = duckdb.connect()
    for name, frame in {"customers": customers, "orders": orders}.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    warehouse = DuckDBWarehouse(con)
    return warehouse, build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))


def test_a_question_two_links_answer_alike_is_answered_and_says_so():
    from core2.plan.values import MemberIndex

    warehouse, model = _same_orders()
    measure = next(m.slug for m in model.measures.values()
                   if (getattr(m.expr, "column", None) or "").endswith("order_amount"))
    name = next(a for a in model.attributes if a.endswith("customer_name") or a.endswith(".name"))
    index = MemberIndex()
    index.add(name, [f"Client {i:02d}" for i in range(1, 21)])
    plan = {"kind": "query", "measures": [measure], "filters": [{"field": name, "op": "eq", "values": ["Client 07"]}]}
    sent: list[str] = []

    def complete(system: str, tail: str) -> str:
        sent.append(tail)
        return json.dumps(plan)

    services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=TODAY)
    got = answer_question('Sales for "Client 07"', services, Session())
    want = warehouse.query("SELECT SUM(order_amount) FROM orders WHERE bill_to_customer_id = 7").rows[0][0]
    assert not got.get("clarify"), "the same figure either way: answered, not asked"
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    assert got["reply"]["seen"][0]["kind"] == "same_either_way" and "same either way" in got["reply"]["en"]


# ── why, in steps: the member that carried the change, then what moved it ─────


APRIL = "l.day >= DATE '2026-04-01' AND l.day < DATE '2026-05-01'"
CHANGE = f"SUM(net_amount) FILTER (WHERE {IN_MAY}) - COALESCE(SUM(net_amount) FILTER (WHERE {APRIL}), 0)"
SEGMENTS = f"""SELECT l.*, c.segment FROM ({LINES}) l LEFT JOIN customers c ON l.customer_id = c.customer_id"""


def _moved(retail, by: str, where: str = "TRUE") -> list[tuple[str, float]]:
    rows = retail[0].con.execute(f"""SELECT {by}, {CHANGE} AS change FROM ({SEGMENTS}) l
        WHERE {where} AND ({IN_MAY} OR {APRIL}) GROUP BY 1 ORDER BY abs(change) DESC""").fetchall()
    return [(r[0], float(r[1])) for r in rows]


def test_why_a_figure_changed_is_followed_one_step_down(retail):
    chat = Chat(retail)
    got = chat.ask("Why did net sales change in May 2026?",
                   {"intent": "drivers", "measures": ["net_amount"], "time": MAY})
    whole = float(_one(retail, f"SELECT {CHANGE} FROM ({LINES}) l"))
    store, moved = _moved(retail, "store_name")[0]
    segment, inside = _moved(retail, "segment", f"store_name = '{store}'")[0]
    seen = _seen(got, "steps")
    assert seen["members"] == [store]
    assert seen["figures"] == pytest.approx([moved, whole, inside], abs=0.01)
    reply = got["reply"]
    assert f"Store {store}" in reply["en"] or f"store {store}" in reply["en"]
    assert f"Inside {store}, segment {segment} moved most" in reply["en"]
    assert f"Dans {store}" in reply["fr"]
    if abs(moved) > abs(whole):
        assert "more than the whole change" in reply["en"] and "the rest going the other way" in reply["en"]
    assert [c["en"].split(",")[0] for c in reply["checked"]] == ["Step 1", "Step 2"]
    # The step itself, one click away -- or "yes" -- with no AI: the same "why", for that store alone.
    offer = reply["offers"][0]
    assert offer["label"] == f'Why did it change, for store "{store}"?'
    asked = chat.asked
    step = chat.ask("yes")
    assert chat.asked == asked
    assert float(step["drivers"]["change"]) == pytest.approx(
        float(_one(retail, f"SELECT {CHANGE} FROM ({LINES}) l WHERE store_name = '{store}'")), abs=0.01)
    assert any(f["values"] == [store] for f in step["plan"]["filters"])

