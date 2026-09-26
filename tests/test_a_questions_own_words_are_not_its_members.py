"""
A question's own words name its measure and its period, never a member.

Three English questions their French twins answered went wrong, each on a
word:

  * "Units sold by month" -- the value index matched "sold" to a product
    group whose name begins with it, so the question was taken to name that
    group and left to the model. A registered measure's own words are never a
    member.
  * "Unités vendues au premier semestre 2022" -- with the measure's words out
    of the way, "premier" matched a product group called "... Premier". A
    word of the stated period's own phrase is the period's; "le mois le plus
    récent" offered "plus" and "récent" the same way, and French question and
    time words are never members either.
  * "How many units did we sell" matched no metric -- "sell" is not "sold" --
    and was asked which dataset it meant. A verb is one word in any tense.

And "deliveries and physical inventory counts by month in 2022" was answered
with month-end inventory value, a metric it reached through the word "month":
a grain or a window says how the answer is cut, never which measure.

A synthetic tenant (tests/answer_harness.py) whose item groups include a
"SOLDER" and a "PREMIER FITTINGS" with no items; the warehouse and the model
are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("words")) as built:
        yield built


def _sold(first: int, last: int, by_month: bool = False) -> dict:
    """Units sold per unit (and month) over yyyymm first..last, month rows only."""
    totals: dict = {}
    for _whs, item, period, sold, _bought, _receipts, _cost in harness.MOVES:
        if first <= period <= last:
            key = (period, harness.ITEMS[item][3]) if by_month else harness.ITEMS[item][3]
            totals[key] = totals.get(key, 0) + sold
    return totals


def _rows(answer: dict, measure: str) -> list[dict]:
    (answered,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return answered["rows"]


class TestTheProductAnswersThem:

    def test_the_measures_word_is_not_a_product_group(self, warehouse):
        answer = harness.ask(warehouse, "Units sold by month in 2025")
        assert answer["model_wrote_sql"] is False
        assert {(int(str(row["PERIOD"])[:7].replace("-", "")), row["UNT_OF_MSR"]): row["UNITS_SOLD"]
                for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(_sold(202501, 202512, by_month=True))

    def test_the_periods_word_is_not_a_product_group(self, warehouse):
        answer = harness.ask(warehouse, "Unités vendues au premier semestre 2025", "fr")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(
            _sold(202501, 202506))

    def test_did_we_sell_is_units_sold(self, warehouse):
        answer = harness.ask(warehouse, "How many units did we sell in 2025?")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in _rows(answer, "UNITS_SOLD")} == pytest.approx(
            _sold(202501, 202512))


class TestWhatIsNeverAMember:

    @staticmethod
    def _candidates(question: str) -> set[str]:
        from core.value_resolver import build_known_terms, extract_candidate_phrases

        return {phrase.lower() for phrase in extract_candidate_phrases(
            question, build_known_terms(harness.ACCOUNT, None))}

    def test_a_registered_measures_words(self, warehouse):
        found = self._candidates("Units sold by month in 2025")
        assert not {"units", "sold"} & found and not any("sold" in phrase for phrase in found)

    @pytest.mark.parametrize("question", ["Stock le plus récent", "Ventes du mois le plus récent"])
    def test_french_time_words(self, warehouse, question):
        assert not any(set(phrase.split()) & {"plus", "récent", "mois"} for phrase in self._candidates(question))

    def test_a_member_is_still_one(self, warehouse):
        assert "brass elbow" in self._candidates("Units sold for BRASS ELBOW in 2025")


class TestTheStatedPeriodsWords:

    @staticmethod
    def _kept(phrases: list[str], question: str, reader_question: str = "") -> list[str]:
        from core.query_pipeline import _without_the_stated_period

        resolved = {"verified": [{"phrase": phrase} for phrase in phrases]}
        return [item["phrase"] for item in _without_the_stated_period(
            resolved, question, reader_question)["verified"]]

    def test_a_word_of_the_period_phrase_is_the_periods(self):
        assert self._kept(["premier", "Premier Fittings"], "unites vendues to premier semestre 2025",
                          "Unités vendues au premier semestre 2025") == ["Premier Fittings"]

    def test_read_on_the_readers_own_words_too(self):
        # "mars" is the reader's; the canonical question says "March".
        assert self._kept(["mars"], "unites vendues en March 2025", "Unités vendues en mars 2025") == []

    def test_a_question_with_no_stated_period_keeps_them(self):
        assert self._kept(["premier"], "units sold in the last 3 months") == ["premier"]


_METRICS = [
    {"name": "Units sold", "synonyms": "units sold, quantity sold", "base_table": "MART.PRD_FCT",
     "formula_type": "expression", "sql_template": "SUM(SLD_QTY)"},
    {"name": "Purchased quantity", "synonyms": "units purchased, quantity purchased", "base_table": "MART.PRD_FCT",
     "formula_type": "expression", "sql_template": "SUM(PCH_QTY)"},
    {"name": "Inventory value", "synonyms": "inventory value, stock value", "base_table": "MART.DLY_FCT",
     "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY * ITM_CST)"},
    {"name": "Month-end inventory value", "synonyms": "month-end inventory value, month end inventory",
     "base_table": "MART.PRD_FCT", "formula_type": "expression", "sql_template": "SUM(CUR_ON_HND_QTY * ITM_CST)"},
    {"name": "Month-to-date revenue", "synonyms": "revenue this month", "base_table": "MART.SLS_FCT",
     "formula_type": "expression", "sql_template": "SUM(NET_AMT)"},
    {"name": "Revenue", "synonyms": "revenue, sales amount", "base_table": "MART.SLS_FCT",
     "formula_type": "expression", "sql_template": "SUM(NET_AMT)"},
]


def _scoped(question: str) -> list[str]:
    from core.metric_scope import resolve_metric_scope

    return [metric["name"] for metric in resolve_metric_scope(_METRICS, question, None).metrics]


class TestWhichMetricTheQuestionNames:

    @pytest.mark.parametrize("question", [
        "How many units did we sell in 2025?", "units we are selling", "units the store sells"])
    def test_a_verb_in_any_tense(self, question):
        assert _scoped(question) == ["Units sold"]

    def test_bought_is_purchased(self):
        assert _scoped("units bought last quarter") == ["Purchased quantity"]

    def test_a_grain_is_not_evidence(self):
        # "month" cut the answer by month; it did not name month-end value.
        assert set(_scoped("deliveries and physical inventory counts by month in 2022")) == {
            "Inventory value", "Month-end inventory value"}

    def test_a_window_is_not_evidence(self):
        assert set(_scoped("inventory counts last month")) == {"Inventory value", "Month-end inventory value"}

    def test_a_metric_named_for_its_period_still_is(self):
        assert _scoped("what is the month-end inventory value?")[0] == "Month-end inventory value"

    def test_a_phrase_authored_with_its_window_still_matches(self):
        assert _scoped("revenue this month")[0] == "Month-to-date revenue"

    def test_the_prompts_candidate_list_reads_them_the_same_way(self):
        from store.config_store import _score_metric_for_question

        units_sold, _bought, _inventory, month_end, _mtd, _revenue = _METRICS
        assert _score_metric_for_question(units_sold, "How many units did we sell in 2025?") > (
            _score_metric_for_question(units_sold, "How many units in 2025?"))
        assert _score_metric_for_question(month_end, "physical inventory counts by month in 2022") == (
            _score_metric_for_question(month_end, "physical inventory counts in 2022"))
