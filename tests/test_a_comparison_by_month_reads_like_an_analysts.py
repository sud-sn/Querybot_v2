"""A comparison by month says how this year stands against last year, as an analyst would.

"What is my revenue comparison?" is answered month by month, this year so far against the same days last year
(BA1). Its headline, analyst note, dashboard tile line and email line read it as a plain series: "Net amount held
near $13.1K a month from Jan 2026 to Sep 2026", with no word of last year, of the change, or of the months ahead
and behind (found on the sample workbook). The month in progress was said to be "only partly covered by the data:
not compared as whole periods", which is true and says nothing of how it was compared -- over the same days in
both years -- and in English on a French answer.

Now the reading of a comparison by period leads with the change on last year (total against total), says how
many months are ahead, which gained and which fell the most, and that the month in progress is compared over the
same days in both years; in English and in French. A plain series still reads as one.

Invented retail data whose last day is 14 Jun 2026, asked on 15 Oct 2026; every figure checked against SQL
written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.facts import amount, percent
from core2.service import Services, Session, answer_question
from tests.test_the_product_looks_before_it_answers import LINES, LIVE, TODAY
from tests.test_the_product_looks_before_it_answers import retail as retail  # noqa: F401 - fixture

SO_FAR, SO_FAR_LY = ("2026-01-01", "2026-06-15"), ("2025-01-01", "2025-06-15")
COMPARE = {"kind": "query", "intent": "compare", "measures": ["net_amount"]}


def _ask(retail, question: str, *plans: dict) -> dict:
    _built, model, warehouse, index = retail
    answers = [json.dumps(p) for p in plans]
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: answers.pop(0), index=index,
                        today=TODAY)
    return answer_question(question, services, Session())


def _in(span: tuple[str, str]) -> str:
    return f"day >= DATE '{span[0]}' AND day < DATE '{span[1]}'"


def _months(retail) -> list[tuple[int, float, float]]:
    rows = retail[0].con.execute(f"""SELECT month(day), SUM(CASE WHEN {_in(SO_FAR)} THEN net_amount END),
        SUM(CASE WHEN {_in(SO_FAR_LY)} THEN net_amount END) FROM ({LINES}) l
        WHERE ({_in(SO_FAR)} OR {_in(SO_FAR_LY)}) AND {LIVE} GROUP BY 1 ORDER BY 1""").fetchall()
    return [(m, float(now), float(before)) for m, now, before in rows]


def _facts(got: dict) -> dict[str, dict]:
    return {f["kind"]: f for f in (got.get("reading") or {}).get("facts") or []}


MONTHS = ["January", "February", "March", "April", "May", "June"]


def test_the_lead_says_the_change_on_last_year_total_against_total(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    months = _months(retail)
    now, before = sum(m[1] for m in months), sum(m[2] for m in months)
    reading = got["reading"]
    assert reading["shape"] == "compared"
    lead = reading["lead"]["en"]
    way = "up" if now >= before else "down"
    assert lead.startswith(f"Net amount is {way} {percent(abs(now / before - 1))} on the same days a year before: "
                           f"{amount(now, 'currency')} from 1 Jan 2026 to 14 Jun 2026, against "
                           f"{amount(before, 'currency')}."), lead
    total = _facts(got)["total"]
    assert total["figures"][0] == pytest.approx(now) and total["figures"][-1] == pytest.approx(before)
    assert "held near" not in lead


def test_the_months_ahead_and_the_biggest_gain_and_fall_are_named(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    months = _months(retail)
    ahead = [m for m in months if m[1] > m[2]]
    facts = _facts(got)
    said = facts["ahead"]["en"]
    if len(ahead) == len(months):
        assert said.startswith("Every month is ahead of the same month a year before"), said
    else:
        assert said.startswith(f"{len(ahead)} of the {len(months)} months are ahead of the same months a year "
                               "before"), said
    best = max(months, key=lambda m: m[1] - m[2])
    assert facts["gain"]["en"].startswith(f"{MONTHS[best[0] - 1]} 2026 gained the most "
                                          f"(+{amount(best[1] - best[2], 'currency')}"), facts["gain"]["en"]
    assert facts["gain"]["figures"][0] == pytest.approx(best[1] - best[2])
    behind = [m for m in months if m[1] < m[2]]
    if behind:
        worst = min(months, key=lambda m: m[1] - m[2])
        assert facts["fall"]["en"].startswith(f"{MONTHS[worst[0] - 1]} 2026 fell the most"), facts["fall"]["en"]
    else:
        assert "fall" not in facts
    # The second sentence of the lead says the months.
    assert ("Every month is ahead" if not behind else f"{len(ahead)} of the {len(months)} months are ahead") \
        in got["reading"]["lead"]["en"]


def test_the_month_in_progress_is_compared_over_the_same_days_and_said_so(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    june = _months(retail)[-1]
    assert june[0] == 6
    caveats = got["coverage_caveats"]
    assert "June 2026 is compared over 1–14 June in both years." in caveats, caveats
    assert not any("not compared as whole periods" in c for c in caveats)
    fr = got["coverage_caveats_fr"]          # each caveat's French twin, by its English words
    assert fr["June 2026 is compared over 1–14 June in both years."] == \
        "Juin 2026 est comparé du 1er au 14 juin, les deux années.", fr
    like = _facts(got)["like_for_like"]
    assert like["figures"][:3] == [14, pytest.approx(june[1]), pytest.approx(june[2])]
    assert like["en"].startswith(f"June 2026 is compared over its first 14 days in both years: "
                                 f"{amount(june[1], 'currency')} against {amount(june[2], 'currency')}"), like["en"]


def test_the_comparison_reads_in_french(retail):
    got = _ask(retail, "Quelle est ma comparaison du chiffre d'affaires ?", COMPARE)
    months = _months(retail)
    now, before = sum(m[1] for m in months), sum(m[2] for m in months)
    fr = got["reading"]["lead"]["fr"]
    way = "en hausse" if now >= before else "en baisse"
    assert fr.startswith(f"Net amount est {way} de {percent(abs(now / before - 1), lang='fr')} sur les mêmes jours "
                         f"un an plus tôt : {amount(now, 'currency', lang='fr')} du 1er janv. 2026 au 14 juin 2026, "
                         f"contre {amount(before, 'currency', lang='fr')}."), fr


def test_the_headline_sent_to_teams_and_slack_says_the_change(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    months = _months(retail)
    now, before = sum(m[1] for m in months), sum(m[2] for m in months)
    headline = got["answer"]["headline"]
    assert "against 1 Jan 2025 to 14 Jun 2025" in headline or "on 1 Jan 2025 to 14 Jun 2025" in headline, headline
    assert f"{(now / before - 1) * 100:+.1f}%" in headline, headline
    ahead = sum(1 for m in months if m[1] > m[2])
    assert f"{ahead} of {len(months)} months ahead" in headline, headline
    # The findings follow the comparison, not the series of this year alone.
    assert not any("held near" in i or "in a row" in i for i in got["key_insights"]), got["key_insights"]
    assert got["key_insights"] and got["key_insights"][0] == _facts(got)["total"]["en"]


def test_a_plain_series_still_reads_as_one(retail):
    got = _ask(retail, "Net amount by month in 2025",
               {"kind": "query", "intent": "trend", "measures": ["net_amount"],
                "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}, "grain": "month"}})
    assert got["reading"]["shape"] == "series"
    assert not any("in both years" in c for c in got["coverage_caveats"])


def test_the_table_headers_say_a_year_before_and_read_in_french(retail):
    # The table's figure before was "Net amount (before)", and a French reader read "Period", "(before)" and
    # ": change" in English under a French answer.
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    en, fr = got["data"]["header_labels"], got["data"]["header_labels_fr"]
    assert en["period"] == "Period" and en["net_amount_prior"] == "Net amount, a year before", en
    assert fr["period"] == "Période" and fr["net_amount_prior"] == "Net amount, un an plus tôt", fr
    assert fr["net_amount_change"] == "Net amount : écart" and fr["net_amount_pct_change"] == "Net amount : variation %"
