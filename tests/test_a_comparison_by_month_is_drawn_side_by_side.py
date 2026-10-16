"""A comparison by month is drawn as bars side by side, last year's beside this year's, the change in % under them.

"What is my revenue comparison?" was drawn as two lines, this year's and a dashed one for last year: the change
was the gap between them, read by eye, and the month in progress looked like a fall. Now (BA2, as recommended):

* bars side by side, last year's quiet and first, this year's in the main colour, named by their periods;
* under them, a strip with each month's change in %, in the colours of good and bad news;
* the month in progress, compared over the same days in both years, hatched and named for its days ("1–14");
* past 13 periods the bars would crowd: it stays two lines; a reader can still pick a line.

Invented data only.
"""

from __future__ import annotations

import json

import pytest

from core2.answer.shape_of_data import settle
from tests.test_a_comparison_by_month_reads_like_an_analysts import COMPARE, _ask
from tests.test_the_product_looks_before_it_answers import retail as retail  # noqa: F401 - fixture

dukpy = pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
from tests.test_chart_annotation_language import _build  # noqa: E402


def test_the_comparison_by_month_is_bars_side_by_side_with_the_change_under_them(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    chart = got["chart"]
    assert chart["chart_type"] == "bar", chart["chart_type"]
    assert chart["compare"]["strip"] is True
    assert set(chart["renderable_types"][:2]) == {"bar", "line"}
    june = [r["period"] for r in chart["rows"] if str(r["period"]).startswith("2026-06")]
    assert chart["compare"]["partial"] == {june[0]: "1–14"}, chart["compare"]
    assert chart["why"]["en"].startswith("Bars side by side:"), chart["why"]
    assert chart["why"]["fr"].startswith("Des barres côte à côte :"), chart["why"]


def _payload(months: int, partial: dict | None = None) -> dict:
    rows = [{"m": f"{2025 + (i // 12)}-{i % 12 + 1:02d}-01", "now": 100.0 + 10 * i, "before": 100.0 + 8 * i - (30 if i == 2 else 0)}
            for i in range(months)]
    return {"chart_type": "line", "x_key": "m", "y_keys": ["now", "before"], "rows": rows,
            "column_formats": {"now": "currency", "before": "currency"}, "x_style": "month_year_long",
            "renderable_types": ["line", "area", "bar"], "allowed_types": ["line", "area", "bar"],
            "chart_spec": {"x": {"column": "m", "role": "temporal"}},
            "compare": {"prior": "before", "current": "now", "prior_label": "Jan–Jun 2024",
                        "current_label": "Jan–Jun 2025", **({"partial": partial} if partial else {})}}


def test_past_thirteen_periods_it_stays_two_lines():
    assert settle(_payload(6), grain="month")["chart_type"] == "bar"
    assert settle(_payload(18), grain="month")["chart_type"] == "line"


def _drawn(payload: dict, expr: str):
    payload = settle(payload, grain="month")
    return json.loads(_build("portal_chat.html", "en", payload, expr))


def test_last_years_bars_come_first_and_quiet_named_by_their_periods():
    names = _drawn(_payload(6), "JSON.stringify(opt.series.filter(function (s) { return s.type === 'bar'; })"
                                ".map(function (s) { return [s.name, (s.itemStyle || {}).color]; }))")
    assert [n for n, _ in names] == ["Jan–Jun 2024", "Jan–Jun 2025"], names
    assert names[0][1] != names[1][1]


STRIP = """JSON.stringify((function () {
    var out = [];
    for (var i = 0; i < opt.xAxis.data.length; i++) {
        out.push(opt.xAxis.axisLabel.formatter(opt.xAxis.data[i], i).split(String.fromCharCode(10)).pop());
    }
    return out;
})())"""


def test_the_strip_says_each_months_change_in_the_colours_of_good_and_bad_news():
    strip = _drawn(_payload(6), STRIP)
    # Each month's change on the same month before, in % (month 3: 120 against 86), a rise in good news's colour
    # and no change in plain ink.
    assert strip == ["{flat|+0.0%}", "{good|+1.9%}", "{good|+39.5%}", "{good|+4.8%}", "{good|+6.1%}",
                     "{good|+7.1%}"], strip
    down = _payload(6)
    down["rows"][1]["now"] = 50.0
    assert _drawn(down, STRIP)[1] == "{bad|−53.7%}"
    costs = _payload(6)
    costs["compare"]["better"] = "lower"           # a rising cost is bad news
    assert _drawn(costs, STRIP)[2] == "{bad|+39.5%}"
    styles = _drawn(_payload(6), "JSON.stringify([opt.xAxis.axisLabel.rich.good.color, opt.xAxis.axisLabel.rich.bad.color])")
    assert styles[0] != styles[1]


def test_the_month_in_progress_is_drawn_lighter_and_named_for_its_days():
    payload = _payload(6, partial={"2025-06-01": "1–14"})
    lighter = _drawn(payload, """JSON.stringify(opt.series.filter(function (s) { return s.type === 'bar'; })
        .map(function (s) { var d = s.data[5]; return [d.itemStyle.opacity, d.itemStyle.borderType]; }))""")
    assert lighter == [[0.5, "dashed"], [0.5, "dashed"]], lighter
    assert _drawn(payload, STRIP)[5] == "{good|+7.1% (1–14)}"


def test_on_a_narrow_card_the_strip_says_whole_percents_and_leaves_the_days_to_the_bars():
    # A phone or a small dashboard tile: twelve months in 360px leave each about 24px, room for "+5%" and not
    # for "+4.8% (1–14)"; the tooltip still says it in full.
    payload = settle(_payload(12, partial={"2025-12-01": "1–14"}), grain="month")
    narrow = f"var opt = QBCharts.buildOption({json.dumps(payload)}, {{width: 360}});"
    strip = json.loads(_build("portal_chat.html", "en", payload, narrow + STRIP))
    assert strip[3] == "{good|+5%}" and strip[11].startswith("{good|+") and "(" not in strip[11], strip
    wide = json.loads(_build("portal_chat.html", "en", payload,
                             f"var opt = QBCharts.buildOption({json.dumps(payload)}, {{width: 1400}});" + STRIP))
    assert wide[3] == "{good|+4.8%}" and wide[11].endswith(" (1–14)}"), wide
    tip = json.loads(_build("portal_chat.html", "en", payload, narrow + """JSON.stringify(opt.tooltip.formatter(
        opt.series.map(function (s) { return {dataIndex: 11, axisValue: '2025-12-01', seriesName: s.name,
                                              value: s.data[11].value, color: '#000'}; })))"""))
    assert "(1–14)" in tip, tip


LABELS = """JSON.stringify((function () {
    var out = [];
    for (var i = 0; i < opt.xAxis.data.length; i++) out.push(opt.xAxis.axisLabel.formatter(opt.xAxis.data[i], i));
    return [out, opt.xAxis.axisLabel.hideOverlap];
})())"""


def test_every_change_sits_on_one_line_whatever_the_name_above_it_holds():
    # January carries its year under it; the other months' changes were drawn a line higher than January's.
    labels, hidden = _drawn(_payload(6), LABELS)
    rows = [label.split("\n") for label in labels]
    assert [len(r) for r in rows] == [3] * 6, labels
    assert rows[0][:2] == ["Jan", "2025"] and rows[1][0] == "Feb", labels
    assert hidden is True



def test_on_a_narrow_card_every_month_keeps_its_name_and_its_change():
    # Twelve months in 360px: each month is its initial, the year under January, and the changes alternate between
    # two rows; none is hidden for running into its neighbour.
    payload = settle(_payload(12), grain="month")
    labels, hidden = json.loads(_build("portal_chat.html", "en", payload,
                                       f"var opt = QBCharts.buildOption({json.dumps(payload)}, {{width: 360}});" + LABELS))
    assert hidden is False
    names = [label.split("\n")[0] for label in labels]
    assert names == list("JFMAMJJASOND"), names
    assert labels[0].split("\n")[1] == "2025"
    rows = [len(label.split("\n")) - 1 for label in labels]          # the line each change is on
    assert rows == [2, 3] * 6, labels
    fr = json.loads(_build("portal_chat.html", "fr", payload,
                           f"var opt = QBCharts.buildOption({json.dumps(payload)}, {{width: 360}});" + LABELS))[0]
    assert [label.split("\n")[0] for label in fr] == list("JFMAMJJASOND"), fr


def test_a_french_reader_reads_the_compared_periods_in_french(retail):
    # The legend and the why named the periods "Jan–Jun 2025" under a French answer.
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    chart = got["chart"]
    cmp = chart["compare"]
    # Last year's bars are all of 2025 (the months this year has not reached are drawn too): named by the year.
    assert cmp["prior_label"] == cmp["prior_label_fr"] == "2025", cmp
    assert cmp["current_label"] == "1 Jan 2026 – 14 Jun 2026", cmp
    assert cmp["current_label_fr"] == "1er janv. 2026 – 14 juin 2026", cmp
    assert "chaque mois du 1er janv. 2026 au 14 juin 2026 face à la période du 1er janv. 2025 au 14 juin 2025" \
        in chart["why"]["fr"], chart["why"]["fr"]
    assert "each month from 1 Jan 2026 to 14 Jun 2026 against 1 Jan 2025 to 14 Jun 2025" in chart["why"]["en"]
    names = json.loads(_build("portal_chat.html", "fr", chart,
                              "JSON.stringify(opt.series.filter(function (s) { return s.type === 'bar'; })"
                              ".map(function (s) { return s.name; }))"))
    assert names == [cmp["prior_label_fr"], cmp["current_label_fr"]], names
