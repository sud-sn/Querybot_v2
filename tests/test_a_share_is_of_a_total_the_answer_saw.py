"""
A share is of a total the answer saw, in one unit.

"Top 5 item groups by number of receipts in 2022" came back as the five
groups the limit kept, and the card's summary said "WATERMAIN FITTINGS leads at
69 (62.2% of total)": 62.2% of the five rows returned, not of every group's
receipts -- a total the answer never saw, printed as if it were the whole. The
same brief shared out sums of quantities in different units: items ranked by
units sold, each counted in its own unit, had a leader "holding" a share of a
total that added eaches to feet.

A result the limit cut, or a quantity its rows keep in more than one unit, now
has no total and no share of one. The ranking, the leader and its lead over the
runner-up stand. A complete result in one unit keeps its shares, and so does
money beside a unit column: a value adds up across units.
"""

from __future__ import annotations

import pytest

from core import i18n

_GROUPS = [
    {"ITEM_GROUP": "WATERMAIN FITTINGS", "NUMBER_OF_RECEIPTS": 69},
    {"ITEM_GROUP": "PIPE", "NUMBER_OF_RECEIPTS": 20},
    {"ITEM_GROUP": "VALVES", "NUMBER_OF_RECEIPTS": 10},
    {"ITEM_GROUP": "HYDRANTS", "NUMBER_OF_RECEIPTS": 7},
    {"ITEM_GROUP": "TOOLS", "NUMBER_OF_RECEIPTS": 5},
]
_TOP_5 = "SELECT TOP 5 ITEM_GROUP, COUNT(*) AS NUMBER_OF_RECEIPTS FROM RECEIPTS GROUP BY ITEM_GROUP"
_ALL = "SELECT ITEM_GROUP, COUNT(*) AS NUMBER_OF_RECEIPTS FROM RECEIPTS GROUP BY ITEM_GROUP"

_ITEMS = [
    {"ITEM": "COPPER PIPE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 900.0},
    {"ITEM": "BRASS ELBOW", "UNT_OF_MSR": "EA", "UNITS_SOLD": 160.0},
    {"ITEM": "STEEL TEE", "UNT_OF_MSR": "EA", "UNITS_SOLD": 40.0},
]


def _answer(rows: list[dict], question: str, sql: str, lang: str = "en") -> dict:
    """The summary, callouts, signal and brief of the card, as the renderer builds them."""
    from core.insight import compute_data_brief
    from core.response_builder import (
        _build_anomaly_callouts, _build_decision_signal, _build_insight_summary, summarize_result_context,
    )

    token = i18n.activate_language(lang)
    try:
        ctx = summarize_result_context(rows, question, sql)
        brief = compute_data_brief(rows, question, result_scope=ctx.get("result_scope"), context=ctx)
        callouts = _build_anomaly_callouts(brief)
        return {
            "brief": brief,
            "summary": _build_insight_summary(rows, ctx, brief),
            "callouts": [callout["message"] for callout in callouts],
            "signal": _build_decision_signal(ctx, brief, callouts),
        }
    finally:
        i18n.deactivate_language(token)


class TestATopNSlice:

    def test_has_no_share_of_a_total_it_never_saw(self):
        answer = _answer(_GROUPS, "Top 5 item groups by number of receipts in 2022", _TOP_5)
        breakdown = answer["brief"]["category_breakdown"]
        assert {"total", "leader_share_pct", "top_3_share_pct"}.isdisjoint(breakdown)
        assert answer["summary"].startswith("WATERMAIN FITTINGS leads at 69")
        assert "%" not in answer["summary"]
        assert not any("%" in callout for callout in answer["callouts"])

    def test_keeps_the_leaders_lead(self):
        breakdown = _answer(_GROUPS, "Top 5 item groups by number of receipts in 2022", _TOP_5)["brief"][
            "category_breakdown"]
        assert breakdown["leader_vs_runner_up_gap"] == 49

    def test_in_french(self):
        answer = _answer(_GROUPS, "Les 5 principaux groupes d'articles par nombre de réceptions en 2022", _TOP_5,
                         "fr")
        assert "%" not in answer["summary"]

    def test_a_complete_result_keeps_its_share(self):
        answer = _answer(_GROUPS, "Number of receipts by item group in 2022", _ALL)
        assert answer["brief"]["category_breakdown"]["leader_share_pct"] == pytest.approx(62.2)
        assert "(62.2% of total)" in answer["summary"]


class TestABalance:
    """Inventory value is a balance: it adds up across warehouses at one date,
    and never across dates. Every balance was withheld, and "inventory value by
    warehouse" lost its shares and the % contribution chip."""

    _BY_WAREHOUSE = [{"WAREHOUSE": "NORTH DEPOT", "INVENTORY_VALUE": 840.0},
                     {"WAREHOUSE": "SOUTH DEPOT", "INVENTORY_VALUE": 160.0}]

    def test_nor_by_the_fallback_cards(self):
        """"Leader share of returned total: 35.1%" of the inventory value at
        three month ends."""
        from core.response_builder import summarize_result_context

        rows = [{"SNAPSHOT": label, "INVENTORY_VALUE": value}
                for label, value in zip(("31/01/2025", "28/02/2025", "31/03/2025"), (900.0, 950.0, 1000.0))]
        context = summarize_result_context(rows, "Inventory value by snapshot", "SELECT SNAPSHOT, SUM(V) FROM F")
        assert context["comparison_stats"].get("leader_share_pct") is None
        assert context["distribution_stats"].get("top_3_share_pct") is None

    def test_a_store_code_is_no_half_year(self):
        rows = [{"STORE": name, "INVENTORY_VALUE": value}
                for name, value in (("S1", 400.0), ("S2", 300.0), ("S3", 200.0), ("S4", 100.0))]
        breakdown = _answer(rows, "Inventory value by store", "SELECT STORE, SUM(V) FROM F")["brief"][
            "category_breakdown"]
        assert breakdown["total"] == 1000.0

    def test_a_number_in_a_name_is_no_date(self):
        rows = [{"STORE": name, "INVENTORY_VALUE": value} for name, value in (("Store 12", 840.0), ("Store 7", 160.0))]
        breakdown = _answer(rows, "Inventory value by store", "SELECT STORE, SUM(V) FROM F")["brief"][
            "category_breakdown"]
        assert breakdown["total"] == 1000.0

    def test_shares_out_across_warehouses(self):
        answer = _answer(self._BY_WAREHOUSE, "Inventory value by warehouse", "SELECT WAREHOUSE, SUM(V) FROM F")
        breakdown = answer["brief"]["category_breakdown"]
        assert (breakdown["total"], breakdown["leader_share_pct"]) == (1000.0, pytest.approx(84.0))

    def test_the_contribution_chip_is_offered(self):
        from core.response_builder import compute_chip_eligibility, summarize_result_context

        answer = _answer(self._BY_WAREHOUSE, "Inventory value by warehouse", "SELECT WAREHOUSE, SUM(V) FROM F")
        ctx = summarize_result_context(self._BY_WAREHOUSE, "Inventory value by warehouse")
        chips = {chip["id"]: chip for chip in compute_chip_eligibility(ctx, brief=answer["brief"])}
        assert "84%" in chips["contribution"]["pre_context"]

    @pytest.mark.parametrize("column,labels,lang", [
        ("MONTH", ("2025-01", "2025-02", "2025-03"), "en"),
        # Time as a French reader's labels say it, or as a column is named.
        ("MOIS", ("janvier", "février", "mars"), "fr"),
        ("LIBELLE", ("janvier 2025", "février 2025", "mars 2025"), "fr"),
        ("JOUR", ("lundi", "mardi", "mercredi"), "fr"),
        ("PERIODE_FIN", ("P01", "P02", "P03"), "fr"),
        # A year in every label, or a column named for a snapshot.
        ("SNAPSHOT", ("31/01/2025", "28/02/2025", "31/03/2025"), "en"),
        ("FY", ("FY2023", "FY2024", "FY2025"), "en"),
        ("AS_OF", ("2025-W01", "2025-W02", "2025-W03"), "en"),
        ("LABEL", ("Stock at 31 Jan 2025", "Stock at 28 Feb 2025", "Stock at 31 Mar 2025"), "en"),
        # A year alone says it; so does a column named for a date, alone.
        ("LIBELLE", ("Inventaire 2023", "Inventaire 2024", "Inventaire 2025"), "fr"),
        ("AS_OF", ("W01", "W02", "W03"), "en"),
        # French quarters, halves and short months.
        ("LIBELLE", ("T1", "T2", "T3"), "fr"),
        ("LIBELLE", ("S1", "S2"), "fr"),
        ("LIBELLE", ("janv.", "févr.", "mars"), "fr"),
        ("LIBELLE", ("janv.", "févr.", "avr."), "fr"),
        ("LIBELLE", ("1er semestre", "2e semestre"), "fr"),
    ])
    def test_is_not_shared_out_across_time(self, column, labels, lang):
        rows = [{column: label, "INVENTORY_VALUE": value} for label, value in zip(labels, (900.0, 950.0, 1000.0))]
        question = "Inventory value by month" if lang == "en" else "Valeur du stock par mois"
        breakdown = _answer(rows, question, f"SELECT {column}, SUM(V) FROM F", lang)["brief"].get(
            "category_breakdown") or {}
        assert {"total", "leader_share_pct", "top_3_share_pct"}.isdisjoint(breakdown)


class TestRowsInDifferentUnits:

    def test_have_no_share_of_their_sum(self):
        answer = _answer(_ITEMS, "Units sold by item", "SELECT ITEM, UNT_OF_MSR, SUM(SLD_QTY) AS UNITS_SOLD FROM F")
        breakdown = answer["brief"]["category_breakdown"]
        assert {"total", "leader_share_pct", "top_3_share_pct"}.isdisjoint(breakdown)
        assert "%" not in answer["summary"]
        assert not answer["signal"]

    def test_rows_in_one_unit_keep_theirs(self):
        rows = [dict(row, UNT_OF_MSR="EA") for row in _ITEMS]
        breakdown = _answer(rows, "Units sold by item", "SELECT ITEM, UNT_OF_MSR, SUM(SLD_QTY) AS UNITS_SOLD FROM F")[
            "brief"]["category_breakdown"]
        assert breakdown["leader_share_pct"] == pytest.approx(81.8)

    def test_money_beside_them_keeps_its_share(self):
        rows = [{"ITEM": row["ITEM"], "UNT_OF_MSR": row["UNT_OF_MSR"], "SALES_VALUE": value}
                for row, value in zip(_ITEMS, (500.0, 300.0, 200.0))]
        breakdown = _answer(rows, "Sales value by item", "SELECT ITEM, UNT_OF_MSR, SUM(SLS_AMT) AS SALES_VALUE FROM F")[
            "brief"]["category_breakdown"]
        assert breakdown["leader_share_pct"] == pytest.approx(50.0)


class TestNothingElseSharesACutResult:
    """The second measure's clause, the context the chips and the fallback
    cards read, and the follow-ups offered: no share of a top five either."""

    _SQL = ("SELECT TOP 5 ITEM_GROUP, COUNT(*) AS NUMBER_OF_RECEIPTS, SUM(NET) AS NET_SALES FROM RECEIPTS "
            "GROUP BY ITEM_GROUP")
    _QUESTION = "Top 5 item groups by number of receipts in 2022"

    def test_the_second_measure(self):
        rows = [dict(group, NET_SALES=value)
                for group, value in zip(_GROUPS, (50000.0, 10000.0, 8000.0, 4000.0, 3000.0))]
        summary = _answer(rows, self._QUESTION, self._SQL)["summary"]
        assert "Net Sales" in summary and "%" not in summary

    @pytest.mark.parametrize("action", ["compare", "analyze"])
    def test_the_context_and_its_cards(self, action):
        from core.response_builder import build_analysis_response, summarize_result_context

        context = summarize_result_context(_GROUPS, self._QUESTION, _TOP_5)
        assert context["comparison_stats"].get("leader_share_pct") is None
        assert context["distribution_stats"].get("top_3_share_pct") is None
        card = build_analysis_response(action, context)
        assert "%" not in " ".join([card["body"], *card["bullets"]])

    def test_the_follow_ups(self):
        from core.result_renderer import _result_signals

        assert "group_imbalance" in {signal["type"] for signal in _result_signals(_GROUPS)}
        kinds = {signal["type"] for signal in _result_signals(_GROUPS, {"was_limited": True})}
        assert not kinds & {"group_imbalance", "pareto"}


class TestABalanceAcrossTimeElsewhere:
    """The regulated summary and the follow-up signals shared the same
    balance out: "T1 représente à lui seul 35,1 % du total de Inventory Value",
    "SNAPSHOT is dominated by one group (35% share)"."""

    @pytest.mark.parametrize("column,labels", [
        ("LIBELLE", ("T1", "T2", "T3")), ("LIBELLE", ("janv.", "févr.", "avr.")),
        ("LIBELLE", ("1er semestre", "2e semestre")), ("PERIODE", ("T1 2025", "T2 2025", "T3 2025")),
        ("SNAPSHOT", ("31/01/2025", "28/02/2025", "31/03/2025"))])
    def test_is_not_shared_out(self, column, labels):
        from core.response_builder import _regulated_analysis_fallback
        from core.stat_signals import compute_signals

        rows = [{column: label, "INVENTORY_VALUE": value} for label, value in zip(labels, (1000.0, 950.0, 900.0))]
        token = i18n.activate_language("fr")
        try:
            body = _regulated_analysis_fallback("why", rows)["body"]
        finally:
            i18n.deactivate_language(token)
        assert "%" not in body, body
        assert not [signal for signal in compute_signals(rows) if signal["type"] == "group_imbalance"]

    @pytest.mark.parametrize("labels", [("T1", "T2", "T3"), ("janv.", "févr.", "avr.")])
    def test_no_sentence_names_no_period(self, labels):
        """Read as a series by one classifier and not by the other, the
        summary said "Inventory Value est resté stable entre  et ."."""
        from core.response_builder import _build_insight_summary, summarize_result_context

        rows = [{"LIBELLE": label, "INVENTORY_VALUE": value} for label, value in zip(labels, (1000.0, 950.0, 900.0))]
        answer = _answer(rows, "Valeur du stock par période", "SELECT LIBELLE, SUM(V) FROM F", "fr")
        token = i18n.activate_language("fr")
        try:
            context = summarize_result_context(rows, "Valeur du stock par période", "SELECT LIBELLE, SUM(V) FROM F")
            summary = _build_insight_summary(rows, context, answer["brief"])
        finally:
            i18n.deactivate_language(token)
        assert "entre  et" not in summary and " ." not in summary, summary
