"""A comparison by period totals only what adds up, and says only what its periods hold (BA2, after review).

A review of the comparison answers (this year so far against the same days last year, by month) found:

* a quantity kept apart by unit (EA, KG, BOX) was added across its units in the headline, the figures under the
  lead, the totals row and "what the rest of the year needs", while the note read EA alone;
* a weekly comparison, or one with no data the year before, read "0 of 0 months ahead" and showed last year as
  $0, and lost the findings a series has;
* a balance (stock on hand, month-end MRR) would be totalled over its months;
* a quarter in progress was named "1–14" (its days in June), and in French a quarter was "q1 2026";
* a change in % from a figure below zero has the wrong sign;
* the French "data ends" warning named its months in English, "and 1 more" read "X and Y and 1 more", "et 1
  autres"; a one-day cut read "1–1 April";
* last year's later months, drawn beside this year's, were named "December 2026" in the tooltip under the first
  window's name, and offered to drill into;
* data ending on 28 February of a leap year compared last year's February over its first 27 days.

Invented retail data (its last day 14 Jun 2026), every figure checked against SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.resolve.time import Range, shift
from core2.service import Services, Session, answer_question
from tests.test_a_comparison_shows_last_year_in_full import COMPARE, _ask
from tests.test_the_product_looks_before_it_answers import LINES, LIVE, TODAY
from tests.test_the_product_looks_before_it_answers import retail as retail  # noqa: F401 - fixture


def _sum(retail, what: str, start: str, end: str, where: str = "TRUE") -> float:
    sql = (f"SELECT SUM({what}) FROM ({LINES}) l LEFT JOIN products pu ON l.product_id = pu.product_id "
           f"WHERE day >= DATE '{start}' AND day < DATE '{end}' AND {LIVE} AND {where}")
    return float(retail[0].con.execute(sql).fetchone()[0] or 0.0)


def test_a_quantity_in_several_units_is_never_added_across_them(retail):
    got = _ask(retail, "What is my quantity comparison?", {**COMPARE, "measures": ["quantity"]})
    each = "pu.unit_of_measure = 'EA'"
    now, before = _sum(retail, "quantity", "2026-01-01", "2026-06-15", each), \
        _sum(retail, "quantity", "2025-01-01", "2025-06-15", each)
    headline = got["answer"]["headline"]
    assert headline.startswith(f"Quantity from 1 Jan 2026 to 14 Jun 2026: {round(now):,} EA"), headline
    assert f"({round(before):,} EA)" in headline, headline
    # No total of EA, KG and BOX anywhere: no figures under the lead, no totals row, no last year in full.
    assert not got.get("compared") and not (got.get("data") or {}).get("totals")
    assert not (got.get("data") or {}).get("reference_rows")


def test_with_no_data_the_year_before_it_is_read_as_the_series_it_is(retail):
    # The data starts in January 2023: the first quarter of 2023 has nothing a year before.
    got = _ask(retail, "Net amount by month for Q1 2023 against a year before",
               {**COMPARE, "time": {"window": {"kind": "between", "start": "2023-01-01", "end": "2023-03-31"},
                                    "grain": "month", "compare": {"kind": "same_period_last_year"}}})
    assert " of 0 " not in got["answer"]["headline"], got["answer"]["headline"]
    assert not got.get("compared"), got.get("compared")          # never "Same days of 2022: $0"
    assert got["reading"]["shape"] == "series" and got["key_insights"], got["reading"]
    assert not (got["chart"].get("compare") or {}).get("strip"), got["chart"]["compare"]
    # Said once, in both languages: nothing in the year before to compare with.
    said = next(c for c in got["coverage_caveats"] if c.startswith("There is no net amount "))
    assert said == "There is no net amount in Q1 2022 to compare it with: the data holds none.", said
    assert got["coverage_caveats_fr"][said].startswith("Il n'y a aucune donnée de Net amount "), got["coverage_caveats_fr"]


def test_a_weekly_comparison_never_reads_none_of_none(retail):
    got = _ask(retail, "Net amount by week, last 8 weeks against a year before",
               {**COMPARE, "time": {"window": {"kind": "last", "unit": "week", "n": 8}, "grain": "week",
                                    "compare": {"kind": "same_period_last_year"}}})
    assert " of 0 " not in got["answer"]["headline"], got["answer"]["headline"]
    assert got["key_insights"], got


def test_a_balance_is_never_totalled_over_its_months(retail):
    built, model, warehouse, index = retail
    balance = model.model_copy(deep=True)
    net = next(m for m in balance.measures.values() if m.slug == "net_amount")
    net.additivity, net.time_aggregation = "semi_additive", "last"      # read as a month-end balance
    answers = [json.dumps(COMPARE)]
    services = Services(model=balance, warehouse=warehouse, complete=lambda s, t: answers.pop(0), index=index,
                        today=TODAY)
    got = answer_question("What is my revenue comparison?", services, Session())
    assert not got.get("compared") and not (got.get("data") or {}).get("totals")
    kinds = {f["kind"] for f in got["reading"]["facts"]}
    assert "total" not in kinds and "rest_of_year" not in kinds, kinds


def test_a_quarter_in_progress_is_named_with_its_months_and_in_french_as_t(retail):
    got = _ask(retail, "Net amount by quarter this year against last year",
               {**COMPARE, "time": {"window": {"kind": "to_date", "unit": "year"}, "grain": "quarter",
                                    "compare": {"kind": "same_period_last_year"}}})
    cmp = got["chart"]["compare"]
    assert cmp["partial"] == {"2026-04-01": "1 Apr–14 Jun"}, cmp
    assert cmp["partial_fr"] == {"2026-04-01": "1er avr.–14 juin"}, cmp
    assert "T1 2026 a le plus progressé" in got["reading"]["lead"]["fr"], got["reading"]["lead"]["fr"]
    assert "Q1" not in got["reading"]["lead"]["fr"] and "q1" not in got["reading"]["lead"]["fr"]


def test_the_data_ends_warning_names_its_months_in_french(retail):
    got = _ask(retail, "Net amount by month in 2026",
               {"kind": "query", "intent": "trend", "measures": ["net_amount"],
                "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}, "grain": "month"}})
    en = next(c for c in got["coverage_caveats"] if c.startswith("The data ends"))
    assert got["coverage_caveats_fr"][en] == ("Les données s'arrêtent le 14 juin 2026 : il n'y a pas encore de "
                                              "données de juil. 2026 à déc. 2026."), got["coverage_caveats_fr"]


def test_three_behind_read_as_a_list(retail):
    got = _ask(retail, "What is my unit price comparison?", {**COMPARE, "measures": ["unit_price"]})
    ahead = next(f for f in got["reading"]["facts"] if f["kind"] == "ahead")
    assert "; March 2026, May 2026 and 1 more are behind." in ahead["en"], ahead["en"]
    assert "; mars 2026, mai 2026 et 1 autre sont en retrait." in ahead["fr"], ahead["fr"]


def test_a_cut_of_one_day_is_named_as_one_day(retail):
    got = _ask(retail, "Net amount by month from 1 Jan to 1 Apr 2024 against a year before",
               {**COMPARE, "time": {"window": {"kind": "between", "start": "2024-01-01", "end": "2024-04-01"},
                                    "grain": "month", "compare": {"kind": "same_period_last_year"}}})
    assert "April 2024 is compared over 1 April in both years." in got["coverage_caveats"], got["coverage_caveats"]
    assert got["coverage_caveats_fr"]["April 2024 is compared over 1 April in both years."] == \
        "Avril 2024 est comparé sur le 1er avril, les deux années."
    assert got["chart"]["compare"]["partial"] == {"2024-04-01": "1"}


def test_last_years_later_months_are_named_as_themselves_and_never_drilled_into(retail):
    from tests.chat_js import run
    from tests.test_chart_annotation_language import _build

    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    chart = got["chart"]
    cmp = chart["compare"]
    assert cmp["prior_label"] == "2025"                     # the quiet bars are all of 2025 now
    assert cmp["reference"]["2026-07-01"] == {"en": "July 2025", "fr": "juillet 2025"}
    july = [r[chart["x_key"]] for r in chart["rows"]].index("2026-07-01")
    tip = json.loads(_build("portal_chat.html", "en", chart, f"""JSON.stringify(opt.tooltip.formatter(
        opt.series.map(function (s) {{ return {{dataIndex: {july}, axisValue: '2026-07-01', seriesName: s.name,
            value: s.data[{july}] && s.data[{july}].value !== undefined ? s.data[{july}].value : s.data[{july}],
            color: '#000'}}; }})))"""))
    assert "July 2025" in tip and "2026" not in tip, tip
    drill = {**chart, "drill": {"on": "period", "items": [{"label": "{period} by store", "question": "q"}]}}
    choices = run(f"JSON.stringify(_drillChoices({json.dumps(drill)}, {json.dumps({'name': '2026-07-01', 'value': 1})}));",
                  functions=["function _drillChoices(payload, params)", "function _inLang(entry, key)"],
                  preamble="var window = this; window.QBCharts = {periodLabel: function (raw) { return raw; }};")
    assert choices == []


def test_no_change_in_percent_from_a_figure_below_zero():
    from core2.answer.builder import change_ratio
    from core2.answer.shape_of_data import settle
    from tests.test_a_comparison_by_month_is_drawn_side_by_side import STRIP, _payload
    from tests.test_chart_annotation_language import _build

    assert change_ratio(-50.0, -100.0) is None and change_ratio(5.0, 0.0) is None
    assert change_ratio(110.0, 100.0) == pytest.approx(0.10)
    loss = _payload(6)
    loss["rows"][1]["before"], loss["rows"][1]["now"] = -100.0, -50.0       # a loss halved: no "−50%" in red
    strip = json.loads(_build("portal_chat.html", "en", settle(loss, grain="month"), STRIP))
    assert strip[1] == "" or strip[1] == "Feb", strip


def test_data_to_the_last_day_of_february_compares_the_same_days_a_year_before():
    # Data to 28 Feb 2028 (a leap year): the window ends on the 29th (exclusive); a year before it ends after the
    # 28th of February 2027, not after the 27th. Whole months stay whole.
    from core2.resolve.time import year_back_end

    assert year_back_end(dt.date(2028, 2, 29)) == dt.date(2027, 3, 1)
    assert year_back_end(dt.date(2028, 3, 1)) == dt.date(2027, 3, 1)
    assert year_back_end(dt.date(2026, 6, 15)) == dt.date(2025, 6, 15)
    assert shift(Range(dt.date(2028, 1, 1), dt.date(2028, 2, 29)), "same_period_last_year") == \
        Range(dt.date(2027, 1, 1), dt.date(2027, 3, 1))
    assert shift(Range(dt.date(2029, 2, 1), dt.date(2029, 3, 1)), "same_period_last_year") == \
        Range(dt.date(2028, 2, 1), dt.date(2028, 3, 1))           # all of February 2028, its 29th too
