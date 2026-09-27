"""
A count is never shown as money.

"Number of profit centres by province" was answered on the sample tenant with
"ON leads at $171.00" -- "ON arrive en tête avec 171,00 $" in French -- and its
chart's axis in dollars. The count's column, PROFIT_CENTRE_COUNT, was given a
format by the words in its name, and "profit" is a money word. So would have
been an INVOICE_COUNT, a SALES_QTY, a PAYMENT_NUMBER.

The last word of a column's name is what it holds: a count or a quantity is a
number, whatever it counts. The answer card, the chart the server describes and
the chart the browser draws all read it so.

The result renderer, the chart builder and the browser's renderer
(static/js/qb-charts.js) are the product's own; the chat channel is a stand-in
that keeps what it is sent.
"""

from __future__ import annotations

import asyncio
import json

import pytest

@pytest.fixture(scope="module", autouse=True)
def _a_store():
    """The card reads the workspace's compliance profile from the store."""
    import store

    store.init_db()


ROWS = [{"PROVINCE": "ON", "PROFIT_CENTRE_COUNT": 171}, {"PROVINCE": "BC", "PROFIT_CENTRE_COUNT": 81},
        {"PROVINCE": "AB", "PROFIT_CENTRE_COUNT": 37}]


class _Chat:
    def __init__(self):
        self.answers: list[dict] = []
        self.messages: list[str] = []

    async def send_message(self, event, text, **kwargs):
        self.messages.append(str(text))

    async def send_assistant_response(self, event, payload, *args, **kwargs):
        self.answers.append(payload)


def _card(rows: list[dict], question: str, lang: str = "en") -> dict:
    from core.i18n import activate_language, deactivate_language
    from core.result_renderer import _send_results

    chat = _Chat()
    token = activate_language(lang)
    try:
        asyncio.run(_send_results({}, chat, question, rows, "SELECT 1", 900, None, "acct-counts",
                                  {"id": 1, "db_type": "azure_sql"}, question_id=None, cache_result=False))
    finally:
        deactivate_language(token)
    assert len(chat.answers) == 1
    return chat.answers[0]


class TestTheAnswer:

    @pytest.mark.parametrize("lang,headline", [
        ("en", "ON leads at 171."),
        ("fr", "ON arrive en tête avec 171."),
    ])
    def test_the_count_of_profit_centres(self, lang, headline):
        answer = _card(ROWS, "Number of profit centres by province", lang)["answer"]
        assert answer["headline"] == headline
        assert "$" not in json.dumps(answer, ensure_ascii=False)

    def test_an_amount_is_still_money(self):
        rows = [{"PROVINCE": "ON", "INVENTORY_COST": 7338871.86}, {"PROVINCE": "BC", "INVENTORY_COST": 17171.03}]
        assert _card(rows, "Inventory cost by province")["answer"]["headline"] == "ON leads at $7,338,871.86."


class TestTheName:

    @pytest.mark.parametrize("column,kind", [
        ("PROFIT_CENTRE_COUNT", "number"),
        ("INVOICE_COUNT", "number"),
        ("SALES_QTY", "number"),
        ("PAYMENT_NUMBER", "number"),
        ("salesUnits", "number"),
        ("INVENTORY_COST", "currency"),
        ("NET_SLS_AMT", "currency"),
        ("PROFIT", "currency"),
        ("NUM_OF_SALES", "number"),
        ("COUNT_OF_SALES", "number"),
        ("COUNT_OFFSET_AMT", "currency"),
        ("GRS_MRGN_PCT", "percent"),
    ])
    def test_the_last_word_is_what_it_holds(self, column, kind):
        from core.result_renderer import _detect_column_format

        assert _detect_column_format(column) == kind


class TestTheChart:

    def test_the_server_describes_a_count(self):
        from core.chart import build_chart_payload

        payload = build_chart_payload(ROWS, None, question="number of profit centres by province")
        roles = payload.get("column_roles") or payload.get("roles") or {}
        assert roles["PROFIT_CENTRE_COUNT"]["format"] == "number"

    @pytest.mark.parametrize("column,kind", [
        ("PROFIT_CENTRE_COUNT", "number"),
        ("salesQty", "number"),
        ("COUNT_OF_SALES", "number"),
        ("NET_SLS_AMT", "currency"),
        ("PROFIT", "currency"),
    ])
    def test_the_browser_draws_a_count(self, column, kind):
        pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it does")
        from tests.test_chart_annotation_language import _run

        assert _run("portal_chat.html", "en", f"QBCharts.formatFor({{}}, {json.dumps(column)})") == kind
