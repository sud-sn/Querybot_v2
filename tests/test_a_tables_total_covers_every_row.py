"""The total under an answer's table is the answer's total, over every row it read, never the first 200 shown.

Found on a report of every order line in a year: the table sends the page its first 200 rows, and the footer's
"Σ Net amount" added those up -- a fraction of the year's sales, under a headline that gave the whole. The answer now
carries each figure's total over every row it read, the footer shows it while nothing is filtered, and says so when
it adds up only the rows on screen (filtered, or more rows than the page holds) or when the rows read were cut
short. A figure that does not add up (a rate, a percentage) has no total.

Synthetic data only.
"""

from __future__ import annotations

import json

import pytest

from core import i18n
from tests import chat_js, shop_data
from tests.shop_data import ask, rows

IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
FOOTER = ["function _footerTotal(data, key, shownSum, filtered, rowsHere)"]


@pytest.fixture(scope="module")
def shop():
    return shop_data.build()


def _lines_2025(shop) -> tuple[int, float]:
    count, net = shop[0].execute(
        "SELECT COUNT(*), SUM(net_amount) FROM (SELECT order_id, line, SUM(net_amount) net_amount FROM order_lines "
        "WHERE status <> 'Cancelled' AND order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' "
        "GROUP BY 1, 2)").fetchone()
    return int(count), float(net)


def test_a_long_report_carries_the_total_of_every_row_it_read(shop):
    payload = ask(shop, "every order line in 2025 with its order, product and net amount",
                  {"intent": "list", "measures": ["net_amount"], "group_by": ["order_line.order_id", "product.name"],
                   "time": IN_2025})
    count, net = _lines_2025(shop)
    data = payload["data"]
    assert len(data["rows"]) == 200 < data["total_rows"], "the page is sent the first 200 rows"
    assert data["column_totals"]["net_amount"] == pytest.approx(net), "the total of every row, not of 200"
    assert data["totals_complete"] is True
    shown = sum(float(r["net_amount"]) for r in data["rows"])
    assert shown < net


def test_a_rate_has_no_total(shop):
    payload = ask(shop, "net sales and the share of each store in 2025",
                  {"intent": "share", "measures": ["net_amount"], "group_by": ["store.name"], "time": IN_2025})
    totals = payload["data"]["column_totals"]
    assert set(totals) == {"net_amount"}, totals
    assert totals["net_amount"] == pytest.approx(sum(float(r["net_amount"]) for r in rows(payload)))


def _footer(script: str, lang: str = "en") -> dict:
    return chat_js.run(f"JSON.stringify({script})", lang=lang, functions=FOOTER)


DATA = {"headers": ["store_name", "net_amount"], "total_rows": 640, "column_totals": {"net_amount": 98765.4},
        "totals_complete": True}


def test_the_footer_shows_the_answers_own_total_while_nothing_is_filtered():
    out = _footer(f"_footerTotal({json.dumps(DATA)}, 'net_amount', 120.5, false, 200)")
    assert out == {"sum": 98765.4, "note": ""}


def test_filtered_the_footer_adds_up_the_rows_on_screen_and_says_so():
    out = _footer(f"_footerTotal({json.dumps(DATA)}, 'net_amount', 120.5, true, 200)")
    assert out == {"sum": 120.5, "note": " " + i18n.catalogue_for("en")["ui.chat.table.sum_rows_shown"]}


def test_a_total_over_rows_cut_short_says_how_many_it_read():
    cut = {**DATA, "total_rows": 5000, "totals_complete": False}
    out = _footer(f"_footerTotal({json.dumps(cut)}, 'net_amount', 1, false, 200)")
    assert out["sum"] == 98765.4 and "5,000" in out["note"], out
    french = _footer(f"_footerTotal({json.dumps(cut)}, 'net_amount', 1, false, 200)", lang="fr")
    assert "premières lignes" in french["note"], french


def test_a_figure_with_no_total_adds_up_the_rows_on_screen_saying_so_when_rows_are_held_back():
    out = _footer(f"_footerTotal({json.dumps(DATA)}, 'share', 55.0, false, 200)")
    assert out == {"sum": 55.0, "note": " " + i18n.catalogue_for("en")["ui.chat.table.sum_rows_shown"]}
    # Every row on screen: nothing to say.
    whole = {**DATA, "total_rows": 8}
    assert _footer(f"_footerTotal({json.dumps(whole)}, 'share', 100.0, false, 8)") == {"sum": 100.0, "note": ""}
    # An answer from before the totals were carried: the rows on screen, as before.
    assert _footer("_footerTotal({}, undefined, 7, false, 7)") == {"sum": 7, "note": ""}


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_the_footer_words_are_in_the_catalogue(lang):
    catalogue = i18n.catalogue_for(lang)
    assert catalogue["ui.chat.table.sum_rows_shown"] and "{count}" in catalogue["ui.chat.table.sum_rows_read"]
