"""A standing condition holds through every later answer, shows on each, and goes when removed.

"From now on use the last 2 months", "only the Downtown store from now on", "for this conversation leave out
'Wholesale'": every question after it in the conversation is answered under it, and says so; "always ..." keeps it
in the reader's profile, for every conversation of theirs. A question that names its own period, or its own member
of the same field, is answered for those, and the answer says the standing one was set aside. "Stop using the last
2 months" (or the chip's x) removes it, from the follow-ups too.

Read in code (core2/standing.py), never by the AI: the AI is asked only the questions, and no standing words or
names reach it. Asked of the invented retail warehouse; every figure against SQL written by hand. Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Window
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.standing import Standing, instruction, letting_go
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
LINES = """SELECT o.*, d.full_date AS day, s.store_name, c.segment FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id
           JOIN customers c ON o.customer_id = c.customer_id"""


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
    """One conversation: the AI's plans scripted, each question asked through the whole answer path."""

    def __init__(self, retail, always: list[Standing] | None = None, today: dt.date = TODAY):
        built, model, warehouse, index = retail
        self.plans: list[dict] = []
        self.asked: list[str] = []
        self.kept: list[dict] = []
        self.dropped: list[str] = []

        def complete(system: str, tail: str) -> str:
            self.asked.append(tail)
            return json.dumps({"kind": "query", **self.plans.pop(0)})

        self.services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=today,
                                 always=list(always or []), keep_always=lambda s: self.kept.append(s.data()),
                                 drop_always=self.dropped.append)
        self.session = Session()

    def ask(self, question: str, plan: dict | None = None) -> dict:
        if plan is not None:
            self.plans.append(plan)
        payload = answer_question(question, self.services, self.session)
        assert not self.plans, "the AI was not asked"
        return payload


def _sql(retail, where: str, by: str = "") -> object:
    con = retail[0].con
    if by:
        return {(k if isinstance(k, str) else str(k)[:10]): float(v) for k, v in con.execute(
            f"SELECT {by}, SUM(net_amount) FROM ({LINES}) l WHERE {where} GROUP BY 1").fetchall()}
    return float(con.execute(f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {where}").fetchone()[0])


def _by(rows: list[dict], key_hint: str) -> dict:
    key = next(k for k in rows[0] if key_hint in k.casefold())
    value = next(k for k in rows[0] if k != key)
    return {(str(r[key])[:10] if key == "period" else r[key]): float(r[value]) for r in rows}


APR_MAY = "l.day >= DATE '2026-04-01' AND l.day < DATE '2026-06-01'"
NET = {"measures": ["net_amount"]}


# ── read in code ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("question, scope, words", [
    ("From now on use the last 2 months", "conversation", "use the last 2 months"),
    ("from now on, only the \"Downtown Store\"", "conversation", "only the \"Downtown Store\""),
    ("Only the \"Downtown Store\" from now on.", "conversation", "Only the \"Downtown Store\""),
    ("For the rest of this conversation leave out \"Wholesale\" customers", "conversation",
     "leave out \"Wholesale\" customers"),
    ("Going forward, use the last 2 months data from the current day", "conversation",
     "use the last 2 months data from the current day"),
    ("Always use the last 2 months", "always", "use the last 2 months"),
    ("From now on always leave out \"Wholesale\"", "always", "leave out \"Wholesale\""),
    ("Désormais, seulement les 2 derniers mois", "conversation", "seulement les 2 derniers mois"),
    ("Toujours exclure \"Wholesale\"", "always", "exclure \"Wholesale\""),
])
def test_an_instruction_is_read_in_code(question, scope, words):
    assert instruction(question) == (scope, words)


@pytest.mark.parametrize("question", [
    "net sales by store", "Always-on customers by month", "always on time deliveries?", "from now",
    "what did we sell from now on to the end of the year?",
])
def test_a_question_is_no_instruction(question):
    assert instruction(question) is None


def test_letting_go_names_the_condition_or_all_of_them():
    period = Standing("a", "the last 2 months", window=Window(kind="last", unit="month", n=2))
    store = Standing("b", "Store name is Downtown Store")
    kept = [period, store]
    assert letting_go("Stop using the last 2 months", kept) == [period]
    assert letting_go("remove the Downtown Store condition", kept) == [store]
    assert letting_go("clear the standing conditions", kept) == kept
    assert letting_go("forget them", kept) == kept
    assert letting_go("stop", kept) == kept
    # About a condition none of them is: said so (an empty list), never answered as a question.
    assert letting_go("remove the Airport filter", kept) == []
    assert letting_go("remove cancelled orders", kept) is None
    assert letting_go("Stop using the last 2 months", []) is None


# ── a period that holds ─────────────────────────────────────────────────────


def test_a_standing_period_holds_shows_and_goes(retail):
    chat = Chat(retail)
    set_ = chat.ask("From now on use the last 2 months")
    assert set_["answer"]["headline"].startswith("From now on in this conversation: the last 2 months (1 Apr 2026 "
                                                 "– 31 May 2026)")
    assert [s["words"] for s in set_["standing"]["kept"]] == ["the last 2 months"]
    assert chat.asked == []                       # read in code, the AI never asked

    by_store = chat.ask("net sales by store", {**NET, "group_by": ["store.name"]})
    assert _by(by_store["export_rows"], "store") == pytest.approx(_sql(retail, APR_MAY, "l.store_name"))
    assert by_store["standing"]["applied"] == ["the last 2 months"]
    assert [b["text"] for b in by_store["answer"]["badges"] if b["kind"] == "period"] == ["Apr–May 2026"]
    assert "last 2 months" not in chat.asked[-1]  # the AI is asked the question alone

    # A follow-up holds it too.
    monthly = chat.ask("and by month?", {**NET, "follow_up": "refine", "time": {"grain": "month"}})
    assert _by(monthly["export_rows"], "period") == pytest.approx(
        _sql(retail, APR_MAY, "date_trunc('month', l.day)::DATE"))

    # A question with its own period is answered for it, and says the standing one was set aside.
    own = chat.ask("net sales in 2024", {**NET, "time": {"window": {"kind": "between", "start": "2024-01-01",
                                                                    "end": "2024-12-31"}}})
    assert float(own["kpi"]["value"]) == pytest.approx(
        _sql(retail, "l.day >= DATE '2024-01-01' AND l.day < DATE '2025-01-01'"))
    assert own["standing"]["applied"] == []
    assert own["standing"]["aside"] == ["the last 2 months (your question names its own period)"]

    gone = chat.ask("Stop using the last 2 months")
    assert gone["answer"]["headline"].startswith("Removed: the last 2 months.")
    assert gone["standing"]["kept"] == []
    every = chat.ask("New question: net sales", NET)
    assert float(every["kpi"]["value"]) == pytest.approx(_sql(retail, "1 = 1"))
    assert "standing" not in every


def test_a_follow_up_after_removal_has_none_of_it(retail):
    chat = Chat(retail)
    chat.ask("From now on only the \"Downtown Store\"")
    first = chat.ask("net sales by month", {**NET, "time": {"grain": "month"}})
    assert first["standing"]["applied"] == ["Store name is Downtown Store"]
    chat.ask("remove the Downtown Store condition")
    after = chat.ask("and by store?", {**NET, "follow_up": "refine", "group_by": ["store.name"]})
    assert _by(after["export_rows"], "store") == pytest.approx(_sql(retail, "1 = 1", "l.store_name"))
    assert "standing" not in after


def test_rolling_from_the_current_day(retail):
    chat = Chat(retail)
    chat.ask("Going forward, use the last 2 months data from the current day")
    got = chat.ask("net sales", NET)
    assert float(got["kpi"]["value"]) == pytest.approx(
        _sql(retail, "l.day >= DATE '2026-04-16' AND l.day <= DATE '2026-06-15'"))


# ── names that hold ────────────────────────────────────────────────────────


def test_a_standing_store_holds_and_a_question_naming_another_wins(retail):
    chat = Chat(retail)
    set_ = chat.ask("only the \"Downtown Store\" from now on")
    assert set_["standing"]["kept"][0]["words"] == "Store name is Downtown Store"
    got = chat.ask("net sales by month in 2025", {**NET, "time": {"grain": "month", "window": {
        "kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert _by(got["export_rows"], "period") == pytest.approx(_sql(
        retail, "l.store_name = 'Downtown Store' AND l.day >= DATE '2025-01-01' AND l.day < DATE '2026-01-01'",
        "date_trunc('month', l.day)::DATE"))
    other = chat.ask("New question: net sales for \"Airport Store\"", {**NET, "filters": [
        {"field": "store.name", "op": "eq", "values": ["Airport Store"]}]})
    assert float(other["kpi"]["value"]) == pytest.approx(_sql(retail, "l.store_name = 'Airport Store'"))
    assert other["standing"]["aside"] == ["Store name is Downtown Store (your question names its own)"]


def test_another_member_named_by_the_list_itself_wins_too(retail):
    # 'and for "Airport Store"?' planned on the stores list, not its name field: the same field all the same, so
    # never both stores at once (an answer of nothing).
    chat = Chat(retail)
    chat.ask("only the \"Downtown Store\" from now on")
    chat.ask("net sales in 2025", {**NET, "time": {"window": {"kind": "between", "start": "2025-01-01",
                                                               "end": "2025-12-31"}}})
    other = chat.ask("and for \"Airport Store\"?", {**NET, "follow_up": "refine", "filters": [
        {"field": "store", "op": "eq", "values": ["Airport Store"]}], "time": {"window": {
            "kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert float(other["kpi"]["value"]) == pytest.approx(_sql(
        retail, "l.store_name = 'Airport Store' AND l.day >= DATE '2025-01-01' AND l.day < DATE '2026-01-01'"))
    assert other["standing"]["aside"] == ["Store name is Downtown Store (your question names its own)"]


def test_leave_out_a_segment_and_a_period_together(retail):
    chat = Chat(retail)
    set_ = chat.ask("From now on use the last 2 months and leave out \"Wholesale\" customers")
    assert sorted(s["words"] for s in set_["standing"]["kept"]) == ["Segment is not Wholesale", "the last 2 months"]
    got = chat.ask("net sales by store", {**NET, "group_by": ["store.name"]})
    assert _by(got["export_rows"], "store") == pytest.approx(
        _sql(retail, f"{APR_MAY} AND l.segment <> 'Wholesale'", "l.store_name"))


def test_a_name_the_data_does_not_hold_is_said(retail):
    chat = Chat(retail)
    got = chat.ask("From now on only \"Moon Base\"")
    assert "Nothing in the data is called “Moon Base”" in got["answer"]["headline"]
    assert "standing" not in got or got["standing"]["kept"] == []


# ── always: the reader's profile ───────────────────────────────────────────


def test_always_is_kept_in_the_profile_and_every_conversation_starts_with_it(retail):
    chat = Chat(retail)
    said = chat.ask("Always use the last 2 months")
    assert said["answer"]["headline"].startswith("Saved to your profile: the last 2 months")
    assert [k["words"] for k in chat.kept] == ["the last 2 months"]
    assert chat.kept[0]["window"] == {"kind": "last", "unit": "month", "n": 2}

    # Another conversation of the same reader, the profile read again.
    later = Chat(retail, always=[Standing.of(k) for k in chat.kept])
    got = later.ask("net sales", NET)
    assert float(got["kpi"]["value"]) == pytest.approx(_sql(retail, APR_MAY))
    assert got["standing"]["kept"][0]["scope"] == "always"

    # Taken off this conversation only, then out of the profile for good.
    off = later.ask("stop using the last 2 months")
    assert "stays in your profile" in off["answer"]["headline"]
    assert later.dropped == []
    assert float(later.ask("New question: net sales", NET)["kpi"]["value"]) == pytest.approx(_sql(retail, "1 = 1"))
    third = Chat(retail, always=[Standing.of(k) for k in chat.kept])
    third.ask("remove the last 2 months for good")
    assert third.dropped == [chat.kept[0]["id"]]


# ── said in the reader's language, with the dates the answers use ─────────


def test_the_dates_said_are_the_ones_the_answers_use_when_the_data_ended_a_while_ago(retail):
    # Four months after the data's last day: the last 2 months are counted back from where it ends, as answers are.
    chat = Chat(retail, today=dt.date(2026, 10, 15))
    said = chat.ask("From now on use the last 2 months")
    assert "the last 2 months (1 Apr 2026 \u2013 31 May 2026)" in said["answer"]["headline"]
    got = chat.ask("net sales", NET)
    assert float(got["kpi"]["value"]) == pytest.approx(_sql(retail, APR_MAY))


def test_in_french(retail):
    chat = Chat(retail)
    said = chat.ask("Désormais, seulement les 2 derniers mois")
    assert said["answer"]["headline"].startswith("Désormais dans cette conversation : les 2 derniers mois "
                                                 "(1er avr. 2026 \u2013 31 mai 2026)")
    assert said["standing"]["kept"][0]["words_fr"] == "les 2 derniers mois"
    got = chat.ask("ventes nettes en 2025", {**NET, "time": {"window": {"kind": "between", "start": "2025-01-01",
                                                                         "end": "2025-12-31"}}})
    assert got["standing"]["aside_fr"] == ["les 2 derniers mois (votre question nomme sa propre période)"]
    gone = chat.ask("Enlève les 2 derniers mois")
    assert gone["answer"]["headline"].startswith("Retiré : les 2 derniers mois. Les prochaines réponses")
    saved = chat.ask("Toujours exclure \"Wholesale\"")
    assert saved["answer"]["headline"].startswith("Enregistré dans votre profil : Segment n’est pas Wholesale")
