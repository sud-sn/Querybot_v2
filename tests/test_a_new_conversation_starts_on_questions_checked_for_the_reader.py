"""A new conversation opens on questions the product answers, checked for the reader who sees them.

The new thread's suggestions came from the old knowledge-base suggester: a workspace learned only by the new engine
got none (a heading and an empty box), and where it did offer questions nothing checked the new engine could answer
them. They now come from the question map (core2/suggest.py), chosen for each reader from what they ask, and every
one is run for them -- on their tables, under their row rules -- before it is shown; one that fails or comes back
empty is dropped. A card asked as the first question of a new conversation is answered from the plan it was checked
with, with no AI.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from core2 import question_map as Q
from core2 import suggest
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question, parse_plan
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
_LEARNED: dict[str, tuple] = {}


def _domain(name: str = "compounding_pharmacy"):
    if name not in _LEARNED:
        built, model = learn(domains.build(name), "descriptive")
        qmap = Q.build(model, today=TODAY)
        Q.check(qmap, model, DuckDBWarehouse(built.con), today=TODAY, budget=None, most=100000)
        _LEARNED[name] = (built, model, qmap)
    return _LEARNED[name]


def _no_ai(*_args, **_kwargs):
    raise AssertionError("a start-page question was sent to the AI")


class Site:
    """A workspace on the store, with its learned model and map, its readers, and a runner that answers a plan on
    its data the way the portal does (no AI)."""

    def __init__(self, store, account: str, built, model) -> None:
        self.store, self.account, self.built, self.model = store, account, built, model
        self.ran: list[str] = []

    def reader(self, tables: list[str] | None, name: str = "Ada") -> dict:
        email = f"{os.urandom(4).hex()}@example.com"
        user_id, _ = self.store.create_user(self.account, name, email, password="a-password-they-chose")
        if tables is not None:
            group = self.store.create_group(self.account, f"Group {os.urandom(2).hex()}")
            self.store.set_group_tables(group, self.account, tables)
            self.store.update_user(user_id, group_id=group)
        return self.store.get_user_by_email(self.account, email)

    def tables(self, *keys: str) -> list[str]:
        return [f"{t.schema_name}.{t.name}" for k, t in self.model.tables.items() if not keys or k in keys]

    def run(self, plan: dict, question: str) -> dict:
        self.ran.append(question)
        services = Services(model=self.model, warehouse=DuckDBWarehouse(self.built.con), complete=_no_ai,
                            index=MemberIndex(), today=TODAY)
        return answer_question(question, services, Session(), planned=parse_plan({"kind": "query", **plan}))

    def asked(self, user: dict, question: str, plan: dict, days_ago: int) -> None:
        self.store.log_core2_answer(self.account, user_id=str(user["id"]), mode="core2", question=question,
                                    status="answered", headline="", sql="", row_count=1, plan=plan, duration_ms=1,
                                    model_version=1)
        when = (dt.datetime.combine(TODAY, dt.time(9)) - dt.timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        with self.store.get_db() as conn:
            conn.execute("UPDATE core2_answer SET created_at = ? WHERE id = (SELECT MAX(id) FROM core2_answer)",
                         (when,))

    def build(self, user: dict, **kw) -> dict:
        kw.setdefault("run", self.run)
        return suggest.build(self.account, user, today=TODAY, **kw)


@pytest.fixture
def site(tmp_path, monkeypatch):
    built, model, qmap = _domain()
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Compounding")
    store.save_core2_model(account, None, model.model_dump_json())
    kept = Q.QuestionMap.from_json(qmap.to_json())
    kept.stamp = Q.stamp_of(account, None)
    Q.save(account, None, kept)
    return Site(store, account, built, model)


def _cards(built: dict) -> list[dict]:
    return [c for s in built.get("for_you") or [] for c in s] + \
           [c for s in built.get("explore") or [] for c in s["cards"]]


def test_a_reader_with_no_questions_yet_is_offered_the_main_figures_of_their_tables_each_answered_for_them(site):
    reader = site.reader(site.tables())
    built = site.build(reader)
    assert built["state"] == "ready"
    for_you = [c for s in built["for_you"] for c in s]
    assert len(built["for_you"]) >= 2 and all(1 <= len(s) <= 3 for s in built["for_you"])
    assert all(c["why"].startswith(("A main figure of", "An analysis you have not run yet")) for c in for_you)
    # Each was run for the reader before it was offered, and a number card carries its answer now.
    assert all(c["question"] in site.ran for c in _cards(built))
    numbers = [c for c in for_you if c["kind"] == "number"]
    assert numbers and all(c["value"] for c in numbers)
    assert built["explore"] and all(2 <= len(s["cards"]) <= 4 for s in built["explore"])
    assert built["data"]["tables"] == len(site.model.tables)


def test_what_a_reader_asks_steers_for_you_to_an_angle_they_have_not_looked_at(site):
    reader = site.reader(site.tables())
    figure = next(e for e in Q.answerable(Q.load(site.account, None)) if e.kind == "split" and e.rows
                  and 2 <= e.rows <= 30)
    measure = site.model.measures[next(k for k, m in site.model.measures.items() if m.slug == figure.measure)]
    for days in (1, 3, 20):
        site.asked(reader, f"{measure.business_name} by {figure.group}", {"intent": "breakdown",
                   "measures": [measure.slug], "group_by": [figure.group]}, days)
    built = site.build(reader)
    for_you = [c for s in built["for_you"] for c in s]
    mine = [c for c in for_you if c["measure"] == measure.slug]
    assert mine and all(c["why"] == f"From your questions on {measure.business_name.lower()}" for c in mine)
    # Not the split they asked again, but the figure from new angles: last month, another split, over time.
    assert not any(c["plan"].get("group_by") == [figure.group] for c in mine)
    assert {c["kind"] for c in mine} & {"number", "trend", "breakdown", "rank"}


def test_a_question_asked_in_the_last_fourteen_days_is_not_offered_again(site):
    reader = site.reader(site.tables())
    first = site.build(reader)
    card = first["for_you"][0][0]
    site.asked(reader, card["question"], card["plan"], 2)
    again = site.build(reader)
    assert card["question"] not in [c["question"] for c in _cards(again)]
    other = site.reader(site.tables(), name="Grace")
    site.asked(other, card["question"], card["plan"], 40)      # long ago: it may be offered again
    assert card["question"] in [c["question"] for c in _cards(site.build(other))]


def test_a_card_that_fails_or_comes_back_empty_for_the_reader_is_not_shown(site):
    reader = site.reader(site.tables())
    shown = [c["question"] for c in _cards(site.build(reader))]
    empty, broken = shown[0], shown[1]

    def run(plan: dict, question: str) -> dict:
        if question == empty:
            return {"type": "assistant_response", "data": {"rows": []}}
        if question == broken:
            raise RuntimeError("the warehouse said no")
        return site.run(plan, question)

    again = [c["question"] for c in _cards(site.build(reader, run=run))]
    assert empty not in again and broken not in again and again


def test_readers_with_different_tables_get_different_questions_and_one_with_none_gets_a_clear_state(site):
    everyone = site.reader(site.tables())
    one = next(k for k, t in site.model.tables.items()
               if sum(m.table == k for m in site.model.measures.values()) >= 3)
    some = site.reader(site.tables(one), name="Grace")
    nothing = site.reader([], name="Linus")
    wide, narrow = site.build(everyone), site.build(some)
    assert {c["question"] for c in _cards(wide)} != {c["question"] for c in _cards(narrow)}
    allowed = {site.model.tables[one].business_name}
    assert {s["subject"] for s in narrow["explore"]} <= allowed
    assert site.build(nothing)["state"] == "no_access"
    assert "for_you" not in site.build(nothing)


def test_no_question_offered_names_a_member_or_a_person(site):
    reader = site.reader(site.tables())
    built = site.build(reader)
    text = json.dumps([[c["question"], c["question_fr"], c["why"], c["why_fr"]] for c in _cards(built)])
    for table, column in (("prescribers", "prescriber_name"), ("patients", "last_name")):
        try:
            values = [str(r[0]) for r in site.built.con.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()]
        except Exception:   # noqa: BLE001 - a domain without the column
            continue
        assert not [v for v in values if v and len(v) > 3 and v in text]


def test_a_card_asked_first_in_a_new_conversation_is_answered_from_the_plan_it_was_checked_with(site):
    from core2.service import _card_asked

    reader = site.reader(site.tables())
    built = suggest._build_and_keep(site.account, reader, TODAY, None, "", run=site.run)
    card = built["for_you"][0][0]
    plan = _card_asked(site.account, reader, card["question"], Session())
    assert plan is not None and plan.measures == card["plan"].get("measures", [])
    services = Services(model=site.model, warehouse=DuckDBWarehouse(site.built.con), complete=_no_ai,
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(card["question"], services, Session(), planned=plan)     # no AI asked
    assert suggest.answered(payload)
    # Asked later in a conversation, the same words are read with what came before.
    going = Session()
    answer_question(card["question"], services, going, planned=plan)
    assert _card_asked(site.account, reader, card["question"], going) is None
    # A reader's own words are no card; another reader's card is not theirs.
    assert _card_asked(site.account, reader, "something else entirely", Session()) is None
    assert _card_asked(site.account, site.reader(site.tables(), name="Grace"), card["question"], Session()) is None
    assert site.store.core2_suggest_kinds(site.account, str(reader["id"]))[card["kind"]][1] == 1


def test_kinds_a_reader_asks_rise_and_kinds_they_leave_fall():
    cards = [suggest.Card(kind=k, question=f"{k} {i}", question_fr="", plan={"i": i, "k": k})
             for i, k in enumerate(["number", "breakdown", "trend", "rank", "segment", "number"])]
    plain = suggest._sets(cards, {k: 1.0 for k in suggest.KINDS})
    assert [c.kind for c in plain[0]] == ["number", "breakdown", "trend"]
    leaned = suggest._sets(cards, {**{k: 0.2 for k in suggest.KINDS}, "rank": 0.9, "segment": 0.8})
    assert [c.kind for c in leaned[0]][:2] == ["rank", "segment"]
    assert all(len({c.kind for c in s}) == len(s) for s in leaned)     # a set mixes kinds


def test_a_page_checked_today_runs_only_the_cards_it_has_not_checked(site):
    reader = site.reader(site.tables())
    first = suggest._build_and_keep(site.account, reader, TODAY, None, "", run=site.run)
    ran = len(site.ran)
    kept = site.store.load_core2_start_page(site.account, str(reader["id"]))
    again = suggest._build_and_keep(site.account, reader, TODAY, kept, "", run=site.run)
    assert len(site.ran) == ran                                   # nothing run twice in a day
    assert [c["question"] for c in _cards(again)] == [c["question"] for c in _cards(first)]


def test_new_since_the_last_visit_names_the_metrics_an_admin_accepted(site):
    reader = site.reader(site.tables())
    suggest._build_and_keep(site.account, reader, TODAY, None, "", run=site.run)
    with site.store.get_db() as conn:          # the last visit was yesterday
        conn.execute("UPDATE core2_start_page SET day = ?, updated_at = ?",
                     ((TODAY - dt.timedelta(days=1)).isoformat(), "2026-06-14 09:00:00"))
    measure = next(m for m in site.model.measures.values() if m.default_date and not m.hidden
                   and m.kind == "column")
    site.store.set_core2_override(site.account, None, f"measure:{measure.key}", "status", "approved")
    kept = site.store.load_core2_start_page(site.account, str(reader["id"]))
    built = suggest._build_and_keep(site.account, reader, TODAY, kept, "", run=site.run)
    assert built["new"]["count"] == 1 and built["new"]["metrics"] == [measure.business_name]
    assert any(c["measure"] == measure.slug and c["why"] == "A metric an admin accepted lately"
               for s in built["for_you"] for c in s)


def test_the_page_is_said_in_the_readers_language(site):
    reader = site.reader(site.tables())
    built = site.build(reader)
    fr = suggest.view(built, "fr")
    cards = [c for s in fr["for_you"] for c in s]
    assert {c["label"] for c in cards} <= {"Nombre", "Tendance", "Répartition", "Classement", "Segment", "Analyse"}
    assert any("le mois dernier" in c["question"] or " par " in c["question"] for c in cards)
    assert not any("plan" in c for c in cards)                  # the plans stay on the server


@pytest.mark.parametrize("english, french", [
    ("Average days between orders per customer in 2026", "Jours moyens entre deux orders par customer en 2026"),
    ("New, returning and lapsed customers by month", "Customers nouveaux, de retour et perdus par mois"),
    ("Customer retention by month of first order", "Rétention des customer selon leur premier mois"),
    ("Share of order lines on time in 2026", "Part des order lines à temps en 2026"),
    ("Products making 80% of net amount", "Products qui font 80 % de net amount"),
    ("Net amount against sales target by month in 2025", "Net amount par rapport à sales target par mois en 2025"),
    ("Orders through placed date, shipped date and delivered date",
     "Orders : étapes placed date, shipped date et delivered date"),
    ("Net amount", "Net amount"),
])
def test_analysis_questions_are_said_in_french(english, french):
    from core2.plan.analyses import in_french

    assert in_french(english) == french


# ── the page's own functions, run as the browser runs them ──────────────────

START_FUNCTIONS = ("function escHtml(str)", "function _startGreeting(now, name)", "function _startDate(iso)",
                   "function _startDaysAgo(iso, today)", "function _startDataLine(data, today)",
                   "function _startWhen(at, today)", "function _startCardHtml(card, withNumber)")


def _js(script: str, lang: str = "en"):
    from tests.chat_js import run

    return run(script, lang=lang, functions=START_FUNCTIONS, consts=("_START_ICONS",),
               preamble="window.qbPlural = (stem, count, vars) => plural(stem, count, vars);")


def test_the_data_line_says_how_fresh_the_data_is_and_what_the_reader_may_use():
    today = "new Date(2026, 9, 15)"
    said = _js(f"""JSON.stringify([
        _startDataLine({{until: '2026-09-14', tables: 10, learned: '2026-10-13'}}, {today}),
        _startDataLine({{until: '', tables: 1, learned: '2026-10-15'}}, {today}),
        _startDataLine({{learned: '2026-01-02'}}, {today})])""")
    assert said[0] == ["Data up to 14 Sep 2026", "10 tables you can use", "Relearned 2 days ago"]
    assert said[1] == ["1 table you can use", "Relearned today"]
    assert said[2][0].startswith("Relearned on 02 Jan 2026")
    fr = _js(f"JSON.stringify(_startDataLine({{until: '2026-09-14', tables: 1, learned: '2026-10-14'}}, {today}))",
             lang="fr")
    assert fr == ["Données jusqu'au 14 sept. 2026", "1 table accessible", "Réappris hier"]


def test_a_for_you_card_shows_its_answer_now_and_an_explore_card_its_question_only():
    card = {"kind": "number", "label": "Number", "question": "Refunds last month <b>", "why": "From your questions",
            "value": "$8.07M", "change": {"direction": "down", "pct": "-28.9%", "good": False, "vs": "August 2026"},
            "spark": {"path": "M2,2 L10,8", "area": "", "tail": ""}}
    html = _js(f"JSON.stringify([_startCardHtml({json.dumps(card)}, true), _startCardHtml({json.dumps(card)}, false)])")
    shown, plain = html
    assert "Refunds last month &lt;b&gt;" in shown and "<b>$8.07M</b>" in shown
    assert 'class="start-chg bad"' in shown and "▼ 28.9%" in shown and "start-spark" in shown
    assert 'data-question="Refunds last month &lt;b&gt;"' in shown
    assert "$8.07M" not in plain and "start-spark" not in plain and "From your questions" in plain


def test_the_greeting_follows_the_readers_clock_and_names_them():
    said = _js("JSON.stringify([_startGreeting(new Date(2026, 9, 15, 8), 'Riley'),"
               " _startGreeting(new Date(2026, 9, 15, 14), ''), _startGreeting(new Date(2026, 9, 15, 20), 'Riley')])")
    assert said == ["Good morning, Riley", "Good afternoon", "Good evening, Riley"]
    assert _js("JSON.stringify(_startGreeting(new Date(2026, 9, 15, 8), 'Riley'))", lang="fr") == "Bonjour, Riley"


def test_a_conversation_to_continue_says_when_it_was():
    today = "new Date(2026, 9, 15)"
    said = _js(f"JSON.stringify([_startWhen('2026-10-15 09:12', {today}), _startWhen('2026-10-14 18:00', {today}),"
               f" _startWhen('2026-10-12 09:00', {today}), _startWhen('', {today})])")
    assert said == ["Today", "Yesterday", "3 days ago", ""]


def test_the_chat_page_asks_for_the_start_page_and_answers_a_card_by_its_words():
    from tests.chat_js import source

    src = source()
    assert "fetch('/portal/api/start'" in src
    assert "function askStartCard(button)" in src and "prefillComposer(question);" in src
    from tests.chat_render import render as render_chat, visible as visible_chat
    page = render_chat()
    assert "Typing your own question needs the AI connection" in visible_chat(page)     # no AI set up here
