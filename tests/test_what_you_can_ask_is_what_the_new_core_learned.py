"""A reader's "What you can ask" page says what the new core learned, as they may see it.

The page listed the earlier pipeline's knowledge-base fields, table by table,
column by column -- a schema browser, under a menu entry that now promises
"What you can ask". For a workspace the new core answers, it opens with the
description the chat gives to "what data do you have?": a lead, a card per
subject with its measures, what they break down by and their dates, and
questions to try, each a link that opens a new thread with the question in the chat's box. A
reader limited to some tables sees only those. The field definitions stay
below, where corrections are proposed.

Run through the real route with the invented retail warehouse's learned model.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from evals.core2 import domains
from evals.core2.compile_eval import learn


@pytest.fixture(scope="module")
def model():
    _built, learned = learn(domains.build("retail"), "descriptive")
    return learned


def _page(model, *, engine="core2", allowed=None):
    from portal import routes

    user = {"id": 1, "account_id": "acct", "role": "analyst", "name": "Ada"}
    request = MagicMock()
    request.query_params = {}
    with patch.object(routes, "_get_portal_user", return_value=user), \
            patch.object(routes.store, "get_client", return_value={"account_id": "acct", "state_data": "{}"}), \
            patch.object(routes.store, "get_allowed_tables", return_value=allowed), \
            patch.object(routes.store, "semantic_feedback_maps", return_value=({}, {})), \
            patch.object(routes.store, "count_semantic_field_feedback", return_value=0), \
            patch.object(routes.store, "get_query_engine", return_value=engine), \
            patch("core2.bootstrap.service.answering_model", return_value=model), \
            patch.object(routes, "_resp", side_effect=lambda req, name, ctx: ctx):
        return asyncio.run(routes.portal_kb(request))


def test_it_says_what_can_be_asked(model):
    learned = _page(model)["learned"]
    assert learned.headline.startswith("This data covers 3 subjects")
    titles = [s["title"] for s in learned.sections]
    assert titles == ["Order line", "Return", "Sales target"]
    assert any(line.startswith("Measures: net amount") for line in learned.sections[0]["bullets"])
    assert "Net amount by store in 2026" in learned.examples


def test_a_reader_limited_to_some_tables_sees_only_those(model):
    learned = _page(model, allowed={"ORDER_LINES", "STORES", "REGIONS", "CUSTOMERS", "PRODUCTS", "CATEGORIES",
                                    "CALENDAR"})["learned"]
    assert [s["title"] for s in learned.sections] == ["Order line"]


def test_todays_pipeline_keeps_its_field_list_alone(model):
    assert _page(model, engine="legacy")["learned"] is None


def test_the_questions_to_try_open_in_the_chat():
    from tests.portal_render import render

    described = MagicMock(headline="This data covers 1 subject.", note="",
                          sections=[{"title": "Order line", "body": "", "bullets": ["Measures: net amount"]}],
                          examples=["Net amount by store in 2026"])
    html = render("portal_kb.html", path="/portal/kb", user={"id": 1, "name": "Ada", "account_id": "a"},
                  learned=described, semantic_tables=[], visible_tables=[], schemas=[], pending_count=0)
    assert 'href="/portal/chat?new=1&amp;ask=Net%20amount%20by%20store%20in%202026"' in html, \
        "a question to try starts its own thread, never a follow-up to the last one"
    assert "Measures: net amount" in html
    start = html.index('<section class="semantic-hero">')
    hero = html[start:html.index("</section>", start)]
    assert "questions to try" in hero, "the page says what it now is"
    assert "status-pill" not in hero, "nothing pending, so no pill"


def test_the_chat_puts_a_tried_question_in_its_box():
    page = open("portal/templates/portal_chat.html", encoding="utf-8").read()
    assert "get('ask')" in page and "box.value = _questionFromUrl" in page
