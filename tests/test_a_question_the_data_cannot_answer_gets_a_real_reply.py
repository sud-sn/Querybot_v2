"""A question the data cannot answer gets a reply a person would give, not one fixed sentence.

Before, anything the catalog could not answer was "I cannot answer that from this data: ...". Now
(core2/answer/beyond.py):

* a question about Solviz itself is answered in code, before any AI, in the reader's language, and never takes a
  question about the data ("market share by region", "export sales by country");
* a question about business in general is answered from the AI's general knowledge, labelled "General knowledge,
  not from your data", where the admin allows it (on by default; a switch on the workspace's settings);
* anything else gets a friendly decline, worded a few ways, with questions to try;
* data the workspace does not hold is said, with what it does hold, the nearest questions and a button that asks
  the admin for it (a New metric request);
* the AI that sorts them is given the question as the planner was (scrubbed where the workspace scrubs) and the
  reader's table names, never a row or a value, and is audited as its own call.

Synthetic data only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from unittest.mock import patch

import pytest

from core2.answer import beyond
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question, general_knowledge_enabled
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
REFUSED = '{"kind": "unsupported", "notes": ["nothing in this data counts employees"]}'


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.sent: list[tuple[str, str]] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append((stable, tail))
        return self.answers.pop(0)


class Logged(DuckDBWarehouse):
    def __init__(self, con):
        super().__init__(con)
        self.asked: list[str] = []

    def query(self, sql, *args, **kwargs):
        self.asked.append(sql)
        return super().query(sql, *args, **kwargs)


def _ask(retail, question: str, *, planner: Recorded | None = None, sorter: Recorded | None = None, **kw) -> dict:
    built, model = retail
    kw.setdefault("index", MemberIndex())
    services = Services(model=model, warehouse=Logged(built.con), complete=planner or Recorded(), today=TODAY,
                        beyond=sorter, **kw)
    said = answer_question(question, services, Session(), question_id="c2-q")
    said["_asked"] = services.warehouse.asked
    return said


# ── Solviz itself ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,topic,words", [
    ("How do I share a dashboard?", "share", "press Share"),
    ("can I share this with my team?", "share", "people or groups"),
    ("How do I download this table?", "download", "Download as CSV"),
    ("how can I export the results to Excel?", "download", "Download as CSV"),
    ("How do I add this to a dashboard?", "pin", "Add to dashboard"),
    ("how do I create a dashboard?", "dashboard", "create a dashboard of"),
    ("How can I get this report emailed every Monday?", "schedule", "Subscribe"),
    ("how do I set an alert?", "alert", "press Alert"),
    ("How do I start a new question?", "new_question", "New chat"),
    ("how do I add my own metric?", "metric", "name = how it is counted"),
    ("how do I change the chart type?", "chart", "as a line"),
    ("How can I change my password?", "password", "Security"),
    ("what can you do?", "about", "from your own data"),
    ("comment partager ce tableau de bord ?", "share", "appuyez sur Partager"),
    ("comment télécharger le tableau ?", "download", "Télécharger en CSV"),
    # Found with gpt-4o: "un" was not read, and the AI recommended other tools' sharing.
    ("comment partager un tableau de bord ?", "share", "appuyez sur Partager"),
    ("que peux-tu faire ?", "about", "vos propres données"),
])
def test_a_question_about_solviz_is_answered_in_code_in_the_readers_language(retail, question, topic, words):
    said = _ask(retail, question)            # the AI has no answer to give: it is never asked
    assert said["kind"] == "product_help" and said["help_topic"] == topic, said["answer"]["headline"]
    assert words in said["answer"]["headline"], said["answer"]["headline"]
    assert said["follow_up_suggestions"] and not said["_asked"]


@pytest.mark.parametrize("question", [
    "market share by region", "can I see market share by region?", "export sales by country in 2025",
    "email opens by month", "show sales as a line chart", "can we see net sales by store?",
    "how did sales do last month?", "how do I see sales by store?", "how can I improve margin?",
    "comment vont les ventes ?", "peut-on voir les ventes par magasin ?", "comment exporter les ventes par pays ?",
])
def test_a_question_about_the_data_is_never_taken_for_one_about_solviz(question):
    assert beyond.product_help(question) is None, question


# ── Business in general ──────────────────────────────────────────────────────

GENERAL = json.dumps({"kind": "general_knowledge", "answer": (
    "Inventory turnover is cost of goods sold divided by average inventory. Grocery retailers often turn stock "
    "a dozen times a year or more, while slower categories turn far less; compare against your own category mix.")})


def test_a_question_about_business_in_general_is_answered_and_labelled_as_such(retail):
    sorter = Recorded(GENERAL)
    said = _ask(retail, "what is a good inventory turnover for a grocery retailer?", planner=Recorded(REFUSED),
                sorter=sorter)
    assert said["kind"] == "general_knowledge" and said["unsupported"]
    assert said["general"] == {"label": "General knowledge, not from your data"}
    assert said["answer"]["headline"].startswith("Inventory turnover is cost of goods sold")
    assert said["follow_up_suggestions"] and not said["_asked"]
    # The sorter is given the question, why the plan stopped and the reader's tables by name: never a value.
    system, given = sorter.sent[0]
    assert "Question: what is a good inventory turnover" in given and "nothing in this data counts employees" in given
    assert "The reader's tables: Order line, Return, Sales target, Customer" in given
    from core.llm_audit import build_egress_manifest

    assert not build_egress_manifest(system, given)["values_sent"]


def test_where_the_admin_switched_it_off_a_general_question_is_said_to_be_beyond_the_data(retail):
    said = _ask(retail, "what does EBITDA mean?", planner=Recorded(REFUSED), sorter=Recorded(GENERAL),
                general_knowledge=False)
    assert said["kind"] == beyond.GENERAL and said["unsupported"]
    assert "general" not in said and said["answer"]["headline"].startswith("That's a question about business")


def test_a_general_answer_that_claims_to_come_from_the_data_is_not_shown(retail):
    claims = json.dumps({"kind": "general_knowledge",
                         "answer": "Your data shows a turnover of 14, which is healthy for a grocer."})
    said = _ask(retail, "is our stock turn healthy?", planner=Recorded(REFUSED), sorter=Recorded(claims))
    assert said["kind"] == "data_not_here" and "general" not in said


# ── Anything else ────────────────────────────────────────────────────────────

OFF = '{"kind": "off_topic"}'


def test_anything_else_gets_a_friendly_decline_worded_a_few_ways(retail):
    said = _ask(retail, "what's the weather in Lyon tomorrow?", planner=Recorded(REFUSED), sorter=Recorded(OFF))
    assert said["kind"] == "off_topic" and said["answer"]["headline"] in beyond._DECLINES["en"]
    assert said["follow_up_suggestions"] and "request_data" not in said
    again = _ask(retail, "what's the weather in Lyon tomorrow?", planner=Recorded(REFUSED), sorter=Recorded(OFF))
    assert again["answer"]["headline"] == said["answer"]["headline"]          # the same question, the same words
    asked = ["tell me a joke", "who won the match last night?", "write me a poem", "what's the capital of Peru?",
             "can you fix my python code?", "recommend a good film", "what time is it in Tokyo?", "how tall is K2?"]
    assert len({beyond.decline(q, lang="en") for q in asked}) >= 3
    french = _ask(retail, "quel temps fera-t-il demain ?", planner=Recorded(REFUSED), sorter=Recorded(OFF))
    assert french["answer"]["headline"] in beyond._DECLINES["fr"]


# ── Data the workspace does not hold ─────────────────────────────────────────

def test_data_not_held_is_said_with_what_is_held_the_nearest_questions_and_a_request(retail):
    wanted = json.dumps({"kind": "data_not_here", "wanted": "employee headcount"})
    said = _ask(retail, "how many employees work in each store?", planner=Recorded(REFUSED), sorter=Recorded(wanted))
    assert said["kind"] == "data_not_here" and said["unsupported"]
    assert said["answer"]["headline"] == "Your data does not hold employee headcount."
    assert said["answer"]["scope_note"] == (
        "It holds order lines, returns and sales targets, with customers, products, stores, categories and regions. "
        "You can ask your admin to add this data.")
    assert said["request_data"] == {"name": "employee headcount", "question": "how many employees work in each store?"}
    # The questions to try sharing its words come first ("... by store").
    assert said["follow_up_suggestions"] and "store" in said["follow_up_suggestions"][0]["question"].lower()
    french = _ask(retail, "combien d'employés par magasin ?", planner=Recorded(REFUSED),
                  sorter=Recorded(json.dumps({"kind": "data_not_here", "wanted": "l'effectif des employés"})))
    assert french["answer"]["headline"] == "Ces données ne couvrent pas ce sujet : l'effectif des employés."
    assert french["answer"]["scope_note"].startswith("Elles contiennent order lines")


def test_with_no_ai_to_sort_it_the_refusal_says_why_and_what_is_held(retail):
    said = _ask(retail, "how many employees work in each store?", planner=Recorded(REFUSED))
    assert said["answer"]["headline"] == "I cannot answer that from this data: nothing in this data counts employees"
    assert said["answer"]["scope_note"].startswith("It holds order lines")
    sorter = Recorded("not json at all")
    assert _ask(retail, "headcount by store", planner=Recorded(REFUSED), sorter=sorter)["kind"] == "data_not_here"

    def down(stable, tail):
        raise RuntimeError("the AI is down")

    assert _ask(retail, "headcount by store", planner=Recorded(REFUSED), sorter=down)["kind"] == "data_not_here"


def test_the_sorter_is_given_the_question_as_scrubbed(retail):
    sorter = Recorded(OFF)
    _ask(retail, "what does Jane Doe think of the weather?", planner=Recorded(REFUSED), sorter=sorter,
         scrub=lambda q: q.replace("Jane Doe", "[PERSON]"), values_allowed=False)
    assert "Jane Doe" not in sorter.sent[0][1] and "[PERSON]" in sorter.sent[0][1]


def test_the_request_button_files_a_new_metric_request_for_the_admin(retail):
    import store
    from core.i18n import t
    from core2.request_service import file_request
    from tests.test_the_answer_card import Elements, card

    wanted = json.dumps({"kind": "data_not_here", "wanted": "employee headcount"})
    said = _ask(retail, "how many employees work in each store?", planner=Recorded(REFUSED), sorter=Recorded(wanted))
    said.pop("_asked")
    shown = Elements(card(said))
    box = shown.first("request-data")
    assert box is not None and json.loads(box["attrs"]["data-request-data"])["name"] == "employee headcount"
    # What the button posts, as the page builds it, is a request the admin sees in their queue.
    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    with patch("core2.request_service._model", return_value=retail[1]):
        filed = file_request(account, {}, {"id": "u1", "name": "Ada"},
                             {"kind": "new_metric", "name": "employee headcount", "source": "chat",
                              "example": said["question"],
                              "description": t("ui.chat.request_data_note", question=said["question"])},
                             allowed=None)
    assert filed["kind"] == "new_metric" and filed["target_name"] == "employee headcount"


def test_the_general_answer_shows_its_label_first_on_the_card(retail):
    from tests.test_the_answer_card import Elements, card

    said = _ask(retail, "what is a good inventory turnover?", planner=Recorded(REFUSED), sorter=Recorded(GENERAL))
    said.pop("_asked")
    shown = Elements(card(said))
    label = next((e for e in shown.found if e["attrs"].get("data-badge") == "general"), None)
    assert label is not None and "badge-info" in label["classes"], label
    assert label["text"].strip() == "General knowledge, not from your data"
    assert shown.found.index(label) < shown.order("answer-headline")[0], "the label comes before the answer"


# ── The admin's switch ───────────────────────────────────────────────────────

def test_the_switch_is_on_unless_the_admin_turns_it_off():
    assert general_knowledge_enabled({}) and general_knowledge_enabled(None)
    assert not general_knowledge_enabled({"general_knowledge_enabled": False})
    assert not general_knowledge_enabled({"general_knowledge_enabled": "off"})


def test_the_settings_form_saves_the_switch_and_other_forms_leave_it(tmp_path):
    import admin.routes as routes
    import store
    from starlette.datastructures import FormData

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")

    def save(pairs):
        async def form():
            return FormData(pairs)

        request = type("R", (), {"form": staticmethod(form)})()
        with patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.client_update(request, account, client_name="", db_config_id="", llm_provider="",
                                             llm_model="", query_limit_monthly="", token_limit_monthly="",
                                             enable_llm_audit="", portal_only=None, teams_platform_config_id=None))
        return general_knowledge_enabled(store.get_client_state(account) or {})

    assert save([("general_knowledge_shown", "1"), ("general_knowledge", "1")]) is True
    assert save([]) is True                                                      # a form without it: unchanged
    assert save([("general_knowledge_shown", "1")]) is False                     # the box left unticked
    assert save([]) is False


# ── "What does X mean?" ──────────────────────────────────────────────────────

DESCRIBE = '{"kind": "describe_data"}'


def test_a_meaning_asked_of_nothing_in_the_data_is_answered_beyond_it_not_with_everything_else(retail):
    # Found with gpt-4o: "what does EBITDA mean?" was planned as a question about the data (about nothing) and
    # answered with the description of every table.
    said = _ask(retail, "what does EBITDA mean?", planner=Recorded(DESCRIBE), sorter=Recorded(GENERAL))
    assert said["kind"] == "general_knowledge" and said["general"]
    french = _ask(retail, "que signifie EBITDA ?", planner=Recorded(DESCRIBE), sorter=Recorded(OFF))
    assert french["kind"] == "off_topic" and french["answer"]["headline"] in beyond._DECLINES["fr"]


@pytest.mark.parametrize("question,starts", [
    ("what does net sales mean?", "Net amount"),           # the reader's word for Net amount
    ("what is margin?", "Margin percent"),                 # a name written in part
    ("what is the data about?", "This data has 8 tables"),  # the data as a whole
])
def test_a_meaning_asked_in_the_readers_words_is_the_data_described(retail, question, starts):
    said = _ask(retail, question, planner=Recorded(DESCRIBE), sorter=Recorded(OFF))
    assert said["kind"] == "describe_data" and said["answer"]["headline"].startswith(starts), said["answer"]


def test_a_message_read_as_small_talk_that_greets_or_thanks_nobody_is_declined(retail):
    # Found with gpt-4o: "tell me a joke" was planned as small talk and answered "Hello! Ask me about your data".
    said = _ask(retail, "tell me a joke", planner=Recorded('{"kind": "smalltalk"}'), sorter=Recorded(OFF))
    assert said["kind"] == "off_topic" and said["answer"]["headline"] in beyond._DECLINES["en"]
    hello = _ask(retail, "hello there", planner=Recorded('{"kind": "smalltalk"}'))
    assert hello["kind"] == "smalltalk" and hello["answer"]["headline"].startswith("Hello")


def test_a_question_about_solviz_the_topics_miss_is_answered_with_what_it_does(retail):
    said = _ask(retail, "does this tool work on my phone?", planner=Recorded(REFUSED),
                sorter=Recorded('{"kind": "product"}'))
    assert said["kind"] == "product_help" and said["answer"]["headline"].startswith("I answer questions about your")
    assert "Tableau" not in said["answer"]["headline"] and "general" not in said


def test_data_that_is_held_is_never_said_not_to_be(retail):
    # Found with Claude on an unseen warehouse: "What was our MRR at the end of September 2026, by product?" was told
    # "Your data does not hold monthly recurring revenue", where a field of that name was there (Learn had kept it as
    # each subscription's property). The retail data holds a "Gross amount": asked about it the same way, never
    # said not to hold it.
    wanted = json.dumps({"kind": "data_not_here", "wanted": "gross amount by week of the year"})
    said = _ask(retail, "what was our gross amount by week of the year?", planner=Recorded(REFUSED),
                sorter=Recorded(wanted))
    assert said["answer"]["headline"].startswith("This data holds Gross amount, but I could not work the answer out")
    assert "does not hold" not in said["answer"]["headline"]
    # A field the reader may not see is never named.
    hidden = _ask(retail, "what was our gross amount by week of the year?", planner=Recorded(REFUSED),
                  sorter=Recorded(wanted), allowed_tables={"customers"})
    assert "Gross amount" not in hidden["answer"]["headline"]


@pytest.mark.parametrize("words,named", [
    ("subscription price or recurring revenue amounts by product", ["Monthly recurring revenue"]),  # no qualifier
    ("What was our MRR at the end of September 2026?", ["Monthly recurring revenue"]),            # its initials
    ("employee headcount by department", []), ("website traffic", []), ("marketing budget 2026", []),
])
def test_what_the_data_holds_is_found_in_the_readers_words(words, named):
    names = ["Monthly recurring revenue", "Average download speed", "Line amount", "SLA target", "Provider name"]
    assert beyond.held(names, words) == named


# ── Found in review ──────────────────────────────────────────────────────────

def _indexed(retail):
    """The retail model's member names, as the portal indexes them."""
    from core2.plan.values import build_index

    built, model = retail

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in built.con.execute(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').fetchall()]

    return build_index(model, fetch)


def test_the_sorter_never_sees_a_member_name_the_planner_was_kept_from(retail):
    # Values kept from the AI: the planner saw a placeholder, the sorter was sent the store's name.
    planner, sorter = Recorded(REFUSED), Recorded(json.dumps({"kind": "data_not_here", "wanted": "headcount"}))
    said = _ask(retail, "how many employees work at Downtown Store?", planner=planner, sorter=sorter,
                index=_indexed(retail), values_allowed=False)
    assert "Downtown Store" not in planner.sent[0][1]
    assert sorter.sent and "Downtown Store" not in sorter.sent[0][1], sorter.sent[0][1]
    assert said["kind"] == "data_not_here"


@pytest.mark.parametrize("question,name", [
    ("what is Jane Doe's loyalty tier?", "Jane Doe"),
    ("qu'est-ce que Jane Doe a acheté ?", "Jane Doe"),
    ("define Jane Doe", "Jane Doe"),
])
def test_a_meaning_asked_of_a_scrubbed_name_never_reaches_the_sorter(retail, question, name):
    sorter = Recorded(OFF)
    _ask(retail, question, planner=Recorded('{"kind": "describe_data"}'), sorter=sorter,
         scrub=lambda q: q.replace(name, "[PERSON]"), values_allowed=False)
    assert sorter.sent and all(name not in part for part in sorter.sent[0]), sorter.sent[0][1]


def test_a_meaning_asked_of_a_masked_member_never_reaches_the_sorter(retail):
    sorter = Recorded(OFF)
    _ask(retail, "what is Downtown Store?", planner=Recorded('{"kind": "describe_data"}'), sorter=sorter,
         index=_indexed(retail), values_allowed=False)
    assert all("Downtown Store" not in part for sent in sorter.sent for part in sent), sorter.sent


@pytest.mark.parametrize("question", [
    "can I get the results for each store?", "can I get this broken down for each region?",
    "can we get a table of sales for each month?", "can I see the shipping schedule for next week?",
    "can we see our market share this year?", "can I see which customers always pay late?",
    "puis-je voir qui paie toujours en retard ?", "can we see sales by default reason?",
    "can I see customers by language?", "peut-on voir les clients par langue ?",
    "can I see password resets by month?", "logouts per day", "can I see the alerts raised last month?",
    "comment prévenir les ruptures de stock ?", "app download numbers by country",
])
def test_a_data_question_with_a_product_word_is_never_taken_for_one_about_solviz(question):
    assert beyond.product_help(question) is None, (question, beyond.product_help(question))


@pytest.mark.parametrize("question,topic", [
    ("how do I log out?", "password"), ("how do I subscribe to this dashboard?", "schedule"),
    ("how do I schedule a report?", "schedule"), ("can I always use the last 2 months?", "always"),
    ("how do I set a default period?", "always"), ("how do I get alerted when sales drop?", "alert"),
    ("how do I get this report emailed to me every Monday?", "schedule"),
    ("is there a way to export this to Excel?", "download"), ("où puis-je télécharger le tableau ?", "download"),
])
def test_a_question_about_solviz_is_still_answered(question, topic):
    found = beyond.product_help(question)
    assert found is not None and found[1] == topic, (question, found)


@pytest.mark.parametrize("question", [
    "what is available?", "what is in the database?", "what are the fields?", "what is covered?",
    "qu'est-ce que je peux demander ?", "qu'est-ce que vous avez ?", "what are the metrics?", "what are the KPIs?",
    "what are the measures?",
])
def test_a_question_about_what_the_data_holds_is_never_read_as_a_meaning(question):
    assert beyond.meaning_asked(question) == "", question


def test_a_meaning_asked_of_our_own_figure_drops_the_possessive():
    assert beyond.meaning_asked("what is our margin?") == "margin"
    assert beyond.meaning_asked("qu'est-ce que notre marge ?") == "marge"


def test_what_is_in_the_database_is_described_never_sent_beyond(retail):
    sorter = Recorded(OFF)
    said = _ask(retail, "what is in the database?", planner=Recorded('{"kind": "describe_data"}'), sorter=sorter)
    assert said["kind"] == "describe_data" and not sorter.sent, said["answer"]["headline"]


@pytest.mark.parametrize("question", ["combien d'employés par magasin ?", "what's our headcount by store?",
                                      "x' onmouseover='alert(1)"])
def test_the_request_button_carries_any_question_whole(question):
    from tests.test_the_answer_card import Elements, card

    msg = {"kind": "data_not_here", "unsupported": True, "question": question,
           "answer": {"headline": "Your data does not hold it."},
           "request_data": {"name": f"what {question}", "question": question}}
    shown = Elements(card(msg))
    box = shown.first("request-data")
    assert box is not None and json.loads(box["attrs"]["data-request-data"]) == {"name": f"what {question}",
                                                                                 "question": question}
    assert not any(k.startswith("on") for e in shown.found for k in e["attrs"]), [e["attrs"] for e in shown.found]


def test_a_general_answer_says_so_in_teams_and_slack_too(retail):
    from gateway.core2_bridge import channel_text

    said = _ask(retail, "what is a good inventory turnover?", planner=Recorded(REFUSED), sorter=Recorded(GENERAL))
    text = channel_text(said)
    assert "General knowledge, not from your data" in text, text
    assert text.index("General knowledge") < text.index("Inventory turnover is"), text


def test_a_sort_written_after_a_line_of_prose_is_read(retail):
    said = _ask(retail, "what does EBITDA mean?", planner=Recorded(REFUSED),
                sorter=Recorded("Here is my answer:\n" + GENERAL))
    assert said["kind"] == "general_knowledge", said["answer"]["headline"]


@pytest.mark.parametrize("answer", [
    "According to your data, your turnover is 14, which is healthy for a grocer.",
    "Looking at your order lines, net sales grew 12% last year, ahead of most grocers.",
    "Based on your sales, your stores turn stock 9 times a year, below the grocery norm.",
    "D'après vos données, la marge est de 31 %, ce qui est sain pour un épicier.",
    "Vos chiffres montrent une rotation de 14, ce qui est sain.",
    "Your turnover is 14 times a year, which is healthy for a grocer.",
])
def test_a_general_answer_that_states_the_readers_own_figures_is_not_shown(retail, answer):
    claims = json.dumps({"kind": "general_knowledge", "answer": answer})
    said = _ask(retail, "is our stock turn healthy?", planner=Recorded(REFUSED), sorter=Recorded(claims))
    assert said["kind"] == "data_not_here" and "general" not in said, said["answer"]["headline"]


def test_a_general_answer_with_the_usual_figures_is_shown(retail):
    usual = json.dumps({"kind": "general_knowledge", "answer": (
        "Grocers usually turn stock 12 to 15 times a year, and specialty retailers 4 to 6 times. For your "
        "business, compare with stores that sell a similar mix.")})
    said = _ask(retail, "what is a good inventory turnover?", planner=Recorded(REFUSED), sorter=Recorded(usual))
    assert said["kind"] == "general_knowledge", said["answer"]["headline"]


@pytest.mark.parametrize("question", ["bye!", "ok thanks", "hmm ok"])
def test_small_talk_is_declined_when_the_ai_cannot_sort_it(retail, question):
    def down(stable, tail):
        raise RuntimeError("the AI is down")

    for sorter in (down, Recorded("sure thing!")):
        said = _ask(retail, question, planner=Recorded('{"kind": "smalltalk"}'), sorter=sorter)
        assert said["kind"] in ("off_topic", "smalltalk") and "request_data" not in said, (question, said["kind"])
