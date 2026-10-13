"""
Home is where a reader who asks starts: a welcome, what to do next, the new questions Learn found, the headline
figures and their dashboards (portal/home.py, portal_home.html).

    who            a reader who may ask lands on Home, and may choose it as their start page; a Viewer, or a
                   workspace with its chat off, lands where it always did and Home sends them there: its figures
                   are new questions run for them, which a Viewer may not ask
    figures        the executive dashboard's number tiles, each run as a pinned answer is, as the reader may see the
                   data; kept a few minutes for that reader under the same rules, never for another, and never once
                   people's data was released to them
    new questions  shown when Learn ran since the reader last signed in, the Data guide's questions, each a link
                   that asks it
    dashboards     the reader's own and those shared with them, each with a miniature of its layout and no data
    the welcome    "data up to" is the latest date a subject they may see reaches, never past today

Invented people and data only (the retail domain).
"""

from __future__ import annotations

import datetime as dt
import json
from unittest.mock import patch

import pytest

from tests.test_a_new_core_answer_can_be_pinned import _replay, retail  # noqa: F401
from tests.test_a_readers_own_page import _board, _profile_form, _sign_in, _signed_in, app  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner, _person


@pytest.fixture(autouse=True)
def _nothing_kept():
    from portal import home

    home.forget()
    yield
    home.forget()


def _counting(retail):  # noqa: F811
    """portal_replay on the invented warehouse, counting who it ran for."""
    calls: list[dict] = []
    replay = _replay(retail)

    def counted(account_id, plan_data, portal_user, *, question=""):
        calls.append({"user": portal_user, "plan": plan_data, "question": question})
        return replay(account_id, plan_data, portal_user, question=question)
    return counted, calls


class TestWhoStartsAtHome:

    def test_a_reader_who_asks_lands_on_home_and_home_is_a_page(self, app):  # noqa: F811
        sam = _owner(app.store)
        client, landed = _sign_in(app, sam)
        assert landed == "/portal/home"
        page = client.get("/portal/home")
        assert page.status_code == 200
        assert "Welcome back, Sam" in page.text
        for link in ('href="/portal/chat?new=1"', 'href="/portal/chat?new=1&amp;compose=dashboard"',
                     'href="/portal/kb"', 'href="/portal/notifications"', 'href="/portal/dashboard"'):
            assert link in page.text, link
        assert 'aria-current="page"' in page.text.split('href="/portal/home" title=')[1][:200]

    def test_a_viewer_lands_on_their_dashboards_and_home_sends_them_there(self, app):  # noqa: F811
        sam = _owner(app.store)
        viv = _person(app.store, sam, "Viv Viewer")
        client, landed = _sign_in(app, viv)
        assert landed == "/portal/dashboard"
        assert client.get("/portal/home", follow_redirects=False).headers["location"] == "/portal/dashboard"
        assert client.get("/portal/home/figures").status_code == 403

    def test_with_the_chat_off_home_sends_an_asker_to_the_dashboards(self, app):  # noqa: F811
        sam = _owner(app.store)
        app.store.update_client_meta(sam["account_id"], chat_ui_enabled=0)
        client, landed = _sign_in(app, sam)
        assert landed == "/portal/dashboard"
        assert client.get("/portal/home", follow_redirects=False).headers["location"] == "/portal/dashboard"

    def test_signed_out_there_is_nothing(self, app):  # noqa: F811
        from starlette.testclient import TestClient

        client = TestClient(app.app, base_url="https://testserver")
        assert client.get("/portal/home", follow_redirects=False).headers["location"] == "/portal/login"
        assert client.get("/portal/home/figures").status_code == 401

    def test_home_is_a_start_page_an_asker_may_choose_and_a_viewer_is_not_offered(self, app):  # noqa: F811
        sam = _owner(app.store)
        viv = _person(app.store, sam, "Viv Viewer")
        client = _signed_in(app, sam)
        assert 'value="home"' in client.get("/portal/settings").text
        client.post("/portal/settings/profile", data=_profile_form(sam, start_page="dashboards"))
        assert _sign_in(app, sam)[1] == "/portal/dashboard"
        client.post("/portal/settings/profile", data=_profile_form(sam, start_page="home"))
        assert _sign_in(app, sam)[1] == "/portal/home"
        assert 'value="home"' not in _signed_in(app, viv).get("/portal/settings").text


class TestTheHeadlineFigures:

    def _figures(self, app, retail, user, **kw):  # noqa: F811
        from portal import home

        _, model = retail
        counted, calls = _counting(retail)
        with patch.object(home, "_model", return_value=(model, None)), \
             patch("core2.service.portal_replay", counted):
            out = home.figures(user, {}, today=dt.date.today(), **kw)
        return out, calls

    def test_they_are_the_executive_number_tiles_run_as_the_reader(self, app, retail):  # noqa: F811
        from core2 import recipes

        sam = _owner(app.store)
        out, calls = self._figures(app, retail, sam)
        _, model = retail
        _, tiles = recipes.executive(model, today=dt.date.today())
        numbers = [m for m in tiles if m.kind == "number"][:4]
        assert out and len(out) <= 4
        assert [f["title"] for f in out] == [m.title for m in numbers][:len(out)]
        assert all(c["user"] is sam for c in calls), "each figure is run as the reader, under their rules"
        assert sorted(json.dumps(c["plan"], sort_keys=True) for c in calls) == \
            sorted(json.dumps(m.plan, sort_keys=True) for m in numbers), "drawn side by side, in any order"
        for figure in out:
            assert figure["value"] and figure["full"] and figure["question"]

    def test_each_figure_is_run_in_the_readers_language(self, app, retail):  # noqa: F811
        from core.i18n import activate_language, deactivate_language, get_active_language
        from portal import home

        sam = _owner(app.store)
        _, model = retail
        replay = _replay(retail)
        seen = []

        def noting(account_id, plan_data, portal_user, *, question=""):
            seen.append(get_active_language())
            return replay(account_id, plan_data, portal_user, question=question)
        token = activate_language("fr")
        try:
            with patch.object(home, "_model", return_value=(model, None)), \
                 patch("core2.service.portal_replay", noting):
                home.figures(sam, {}, today=dt.date.today(), lang="fr")
        finally:
            deactivate_language(token)
        assert seen and set(seen) == {"fr"}

    def test_they_are_kept_for_the_same_reader_and_never_for_another(self, app, retail):  # noqa: F811
        sam = _owner(app.store)
        ann = _person(app.store, sam, "Ann Asker", portal_role="asker")
        first, calls = self._figures(app, retail, sam)
        again, more = self._figures(app, retail, sam)
        assert again == first and not more, "the same reader's figures are kept a few minutes"
        _, theirs = self._figures(app, retail, ann)
        assert theirs and all(c["user"] is ann for c in theirs), "another reader's are run for them"

    def test_a_change_in_the_rules_runs_them_again(self, app, retail):  # noqa: F811
        sam = _owner(app.store)
        self._figures(app, retail, sam)
        app.store.save_compliance_profile(sam["account_id"], mode="standard", active_policy_version=7)
        _, calls = self._figures(app, retail, sam)
        assert calls

    def test_once_peoples_data_was_released_nothing_is_kept(self, app, retail):  # noqa: F811
        from portal import home

        sam = _owner(app.store)
        _, model = retail
        replay = _replay(retail)
        calls = []

        def released(account_id, plan_data, portal_user, *, question=""):
            calls.append(1)
            return {**replay(account_id, plan_data, portal_user, question=question), "released": True}
        with patch.object(home, "_model", return_value=(model, None)), \
             patch("core2.service.portal_replay", released):
            home.figures(sam, {}, today=dt.date.today())
            ran = len(calls)
            home.figures(sam, {}, today=dt.date.today())
        assert ran and len(calls) == 2 * ran

    def test_a_workspace_the_new_core_does_not_answer_has_none(self, app):  # noqa: F811
        from portal import home

        sam = _owner(app.store)
        with patch("core2.service.portal_replay", side_effect=AssertionError("never run")):
            assert home.figures(sam, app.store.get_client(sam["account_id"]), today=dt.date.today()) == []

    def test_the_cards_say_the_change_and_escape_what_they_show(self):
        from tests.portal_render import render

        figure = {"title": "<b>Net amount</b>", "question": "Net amount over the last 12 months", "period": "",
                  "value": "$1.2K", "full": "$1,234.00", "trend": None,
                  "change": {"direction": "down", "change": "-$100.00", "pct": "-7.5%", "vs": "the 12 months before",
                             "before": "$1,334.00", "good": False}}
        html = render("home_figures.html", figures=[figure], asks=True)
        assert "<b>Net amount</b>" not in html and "&lt;b&gt;Net amount&lt;/b&gt;" in html
        assert 'class="home-figure-change is-bad"' in html and "7.5%" in html
        assert 'href="/portal/chat?new=1&amp;ask=Net%20amount%20over%20the%20last%2012%20months"' in html
        assert "could not be drawn" in render("home_figures.html", figures=None, asks=True)


class TestNewQuestions:

    def _news(self, app, retail, user, *, learned, last_visit):  # noqa: F811
        from portal import home

        _, model = retail
        learned_model = model.model_copy(update={"built_at": learned, "version": 5})
        with patch.object(home, "_model", return_value=(learned_model, None)), \
             patch.object(home, "_last_visit", return_value=last_visit):
            return home.news(user, app.store.get_client(user["account_id"]), today=dt.date.today())

    def test_learn_since_the_last_visit_brings_its_questions(self, app, retail):  # noqa: F811
        sam = _owner(app.store)
        news = self._news(app, retail, sam, learned=dt.datetime(2026, 5, 2, 3), last_visit=dt.datetime(2026, 5, 1, 9))
        assert news and 1 <= len(news["questions"]) <= 3 and news["version"] == 5

    def test_nothing_new_when_learn_ran_before_it(self, app, retail):  # noqa: F811
        sam = _owner(app.store)
        assert self._news(app, retail, sam, learned=dt.datetime(2026, 5, 1, 3),
                          last_visit=dt.datetime(2026, 5, 2, 9)) is None

    def test_a_viewer_is_never_offered_questions(self, app, retail):  # noqa: F811
        sam = _owner(app.store)
        viv = _person(app.store, sam, "Viv Viewer")
        assert self._news(app, retail, viv, learned=dt.datetime(2026, 5, 2, 3), last_visit=None) is None

    def test_the_banner_asks_each_question_and_can_be_put_away(self, app, retail):  # noqa: F811
        from portal import home

        sam = _owner(app.store)
        client = _signed_in(app, sam)
        news = {"questions": ["Net amount by region in 2025", "How long between orders?"], "learned": None,
                "version": 9}
        with patch.object(home, "news", return_value=news):
            page = client.get("/portal/home").text
        assert "2 new questions your data can answer" in page
        assert 'href="/portal/chat?new=1&amp;ask=Net%20amount%20by%20region%20in%202025"' in page
        assert 'data-version="9"' in page and 'id="homeNewsLater"' in page


class TestTheDashboards:

    def test_each_is_listed_with_its_layout_and_no_data(self, app, retail):  # noqa: F811
        from portal import home

        sam = _owner(app.store)
        board, _ = _board(app.store, retail, sam)
        with patch("core2.service.portal_replay", side_effect=AssertionError("no data is read")):
            found = home.boards(sam)
        assert [b["name"] for b in found] == ["Sales pulse"]
        assert found[0]["tiles"] == 2 and found[0]["mine"]
        assert {box["kind"] for box in found[0]["layout"]} <= {"number", "line", "bars", "table"}
        assert len(found[0]["layout"]) == 2
        page = _signed_in(app, sam).get("/portal/home").text
        assert f'href="/portal/dashboard?dashboard_id={board["id"]}"' in page and "Sales pulse" in page
        assert "2 tiles" in page

    def test_one_shared_with_the_reader_says_whose_it_is(self, app, retail):  # noqa: F811
        from portal import home

        sam = _owner(app.store)
        ann = _person(app.store, sam, "Ann Asker", portal_role="asker")
        board, _ = _board(app.store, retail, sam)
        app.store.add_dashboard_share(board["id"], sam["id"], sam["account_id"], "user", ann["id"])
        found = home.boards(ann)
        assert found and not found[0]["mine"] and found[0]["owner"]


class TestTheWelcome:

    def test_data_up_to_is_the_latest_a_subject_reaches_and_never_past_today(self, retail):  # noqa: F811
        from portal import home

        _, model = retail
        got = home._data_to(model, None, dt.date(2100, 1, 1))
        assert got is not None and got in {r.last for r in model.date_roles.values()}
        # Dates set into the future (a budget for next year) are not data yet.
        future = model.model_copy(deep=True)
        for role in future.date_roles.values():
            role.last = dt.date(2101, 1, 1)
        assert home._data_to(future, None, dt.date(2100, 1, 1)) is None

    def test_only_the_readers_tables_count(self, retail):  # noqa: F811
        from portal import home

        _, model = retail
        assert home._data_to(model, set(), dt.date(2100, 1, 1)) is None

    def test_learned_says_when_in_words(self):
        from portal.routes import _learned_when

        today = dt.date(2026, 6, 10)
        assert _learned_when(dt.datetime(2026, 6, 10, 2), today, "en") == "Learned today"
        assert _learned_when(dt.datetime(2026, 6, 9, 23), today, "en") == "Learned yesterday"
        assert _learned_when(dt.datetime(2026, 6, 6, 1), today, "fr") == "Appris il y a 4 jours"
        assert _learned_when(dt.datetime(2026, 5, 1, 1), today, "en").startswith("Learned on 01")
        assert _learned_when(None, today, "en") == ""

    def test_the_page_speaks_french(self, app):  # noqa: F811
        sam = _owner(app.store)
        client = _signed_in(app, sam)
        client.cookies.set("qb_lang", "fr")
        page = client.get("/portal/home").text
        assert "Bon retour, Sam" in page and "Poser une question" in page and "Vos tableaux de bord" in page
