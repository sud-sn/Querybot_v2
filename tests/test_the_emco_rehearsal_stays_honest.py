"""
tests/test_the_emco_rehearsal_stays_honest.py

The rehearsal harness must fail when the product regresses, and its own bugs
must not read as product failures.

evals/emco_rehearsal.py drives 56 EMCO questions through the deterministic
pipeline in both languages. It is only worth running if two things hold: it
reports a real regression, and it does not report one that is not there. The
first version of it did the latter twice --

  * YAML reads a bare `yes`/`no` as a BOOLEAN, so every expectation in the
    corpus read as its opposite and it announced 38 failures with none;
  * it handed the governed compiler a plan with no display fields, so "net sales
    by warehouse" legitimately compiled as a scalar and seven grouped questions
    read as product failures.

Both are fixed in the harness. These tests hold it to that, and to the gaps it
found on its first honest run -- an English spelled-out count, "l'an dernier",
match_metric answering a comparison from a fixed template, and a French reader
unable to name any of the four role-playing dates.
"""

from __future__ import annotations

import pytest
import yaml

from evals.emco_rehearsal import (
    CORPUS,
    DATE_ROLES,
    _role_for,
    _requested_fields,
    _wants,
    card_check,
    judge,
    observe,
    run,
)
from core.contextual_dates import detect_temporal_window
from core.question_normalizer import canonical_question


class TestTheHarnessReadsItsOwnCorpus:

    def test_a_bare_yaml_yes_is_a_yes(self):
        """The bug that made it report 38 failures with none."""
        assert _wants(True) is True
        assert _wants(False) is False
        assert _wants("yes") is True
        assert _wants("no") is False

    def test_the_corpus_parses_and_every_id_is_unique(self):
        cases = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["cases"]
        assert len(cases) >= 50
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids))
        for case in cases:
            assert case["english"] and case["french"], case["id"]
            assert case["english"] != case["french"], case["id"]
            assert case.get("expect"), case["id"]

    def test_a_grouped_question_gets_the_display_field_the_planner_would_add(self):
        """Without it the scalar compiler builds a scalar for "by warehouse" and
        the harness blames the product."""
        assert _requested_fields("net sales by warehouse")
        assert _requested_fields("net sales per branch")
        assert _requested_fields("breakdown of net sales by category")
        assert _requested_fields("total net sales for the last 6 months") == []
        assert _requested_fields("net sales") == []


class TestTheHarnessWouldNoticeARegression:

    def test_a_judge_reports_a_window_the_two_languages_disagree_on(self):
        case = {"expect": {"window": "last_n"}}
        en = observe("the last 6 months", "en", "")
        fr = observe("les 6 derniers mois", "fr", "")
        assert judge(case, en, fr) == []
        broken = observe("net sales", "en", "")
        assert judge(case, en, broken), "a lost window must be reported"

    def test_and_a_date_role_the_two_languages_disagree_on(self):
        en = observe("net sales by ship date", "en", "")
        fr = observe("ventes nettes par date de facturation", "fr", "")
        assert en.date_role != fr.date_role
        assert any("date role differs" in failure
                   for failure in judge({"expect": {}}, en, fr))

    def test_the_whole_corpus_is_as_expected_today(self):
        """The one assertion that makes this file worth running: every case the
        rehearsal covers behaves as the corpus says it should."""
        results = run()
        failures = {r.id: r.failures for r in results if not r.ok}
        assert not failures, failures

    def test_the_card_check_reports_both_shapes(self):
        cards = {row["shape"]: row for row in card_check()}
        assert len(cards) == 2
        ranking = next(row for name, row in cards.items() if "ranking" in name)
        assert ranking["measure"] == "NET_SLS_AMT"
        assert ranking["top3_share"] == 91.9
        assert "91.9" in ranking["callouts"]
        assert "diversified" not in ranking["signal_en"]
        series = next(row for name, row in cards.items() if "year" in name)
        assert series["mode"] == "time_series"
        assert series["measure"] == "NET_SLS_AMT"
        assert series["axis"] == "IVC_YR"


class TestTheGapsItFoundStayClosed:
    """Four defects the harness surfaced on its first honest run. Each is here
    because a rehearsal that finds something and does not pin it will find the
    same thing again."""

    @pytest.mark.parametrize("english,amount,unit", [
        ("the last six months", 6, "month"),
        ("the last thirty days", 30, "day"),
        ("the past twelve months", 12, "month"),
        ("trailing ninety days", 90, "day"),
        ("last two years", 2, "year"),
    ])
    def test_english_reads_a_count_spelled_as_a_word(self, english, amount, unit):
        """detect_top_n_intent has always read "top five"; the window detector
        read only digits, so "the last six months" carried NO window and the
        answer was measured against the server clock."""
        window = detect_temporal_window(english)
        assert window.get("kind") == "last_n", english
        assert (window.get("amount"), window.get("unit")) == (amount, unit)

    def test_a_digit_and_a_word_reach_the_same_window(self):
        assert detect_temporal_window("the last six months") == \
            detect_temporal_window("the last 6 months")

    @pytest.mark.parametrize("french,kind", [
        ("par rapport à l'an dernier", "previous_year"),
        ("l’an dernier", "previous_year"),
        ("l'an passé", "previous_year"),
    ])
    def test_an_is_a_year_in_french(self, french, kind):
        """"par rapport à l'an dernier" is how a French business writes a
        year-over-year comparison, and it carried no window at all."""
        assert detect_temporal_window(
            canonical_question(french, "fr")).get("kind") == kind

    @pytest.mark.parametrize("french,role", [
        ("ventes nettes par date de facturation", "invoice_date"),
        ("ventes nettes par date d'expédition", "delivery_date"),
        ("ventes nettes par date de livraison", "delivery_date"),
        ("ventes nettes par date de commande", "order_date"),
        ("ventes nettes par date d'annulation", "cancelled_order_date"),
    ])
    def test_a_french_reader_can_name_the_date_role(self, french, role):
        """EMCO's invoice fact carries four role-playing dates on one dimension,
        so which date a question means IS the answer. Not one French phrasing
        reached the role vocabulary, so every French question silently took the
        default role -- invoice-dated sales for a shipment-date question."""
        assert _role_for(canonical_question(french, "fr"))[0] == role
        assert role in DATE_ROLES

    @pytest.mark.parametrize("english,french,role", [
        ("net sales by invoice date", "ventes nettes par date de facturation",
         "invoice_date"),
        ("net sales by ship date", "ventes nettes par date d'expédition",
         "delivery_date"),
        ("net sales by order date", "ventes nettes par date de commande",
         "order_date"),
    ])
    def test_and_reaches_the_same_role_the_english_reader_does(
            self, english, french, role):
        assert _role_for(canonical_question(english, "en"))[0] == role
        assert _role_for(canonical_question(french, "fr"))[0] == role

    def test_the_longer_english_spelling_of_a_ship_date_works_too(self):
        """"ship date" was a synonym and "shipment date" was not, so the longer
        form reached no role and took the default."""
        assert _role_for("net sales by shipment date")[0] == "delivery_date"
        assert _role_for("net sales by ship date")[0] == "delivery_date"

    @pytest.mark.parametrize("question,lang", [
        ("what is the difference in net sales", "en"),
        ("quelle est la différence de ventes nettes", "fr"),
        ("why did net sales change", "en"),
        ("pourquoi les ventes nettes ont-elles changé", "fr"),
        ("net sales distribution", "en"),
        ("répartition des ventes nettes", "fr"),
        ("net sales forecast", "en"),
        ("net sales trend", "en"),
        ("tendance des ventes nettes", "fr"),
    ])
    def test_a_fixed_template_declines_a_question_one_number_cannot_answer(
            self, question, lang):
        """A stored SELECT SUM(...) is one number over all time. Asked for a
        difference, a cause, a distribution or a projection it answered with that
        number anyway -- the measure name matched and neither existing guard had
        anything to say, because the question names no breakdown and no window."""
        import os
        import shutil
        import tempfile
        import uuid

        import store

        workdir = tempfile.mkdtemp(prefix="qb-template-guard-")
        saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(workdir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        try:
            store.init_db()
            account_id = f"acct-{uuid.uuid4().hex[:8]}"
            store.upsert_client(account_id, "portal")
            store.save_metric(account_id, {
                "name": "Net Sales", "label": "Net Sales",
                "synonyms": "net sales, revenue, ventes nettes, chiffre d'affaires",
                "formula_type": "query",
                "sql_template": "SELECT SUM(NET_SLS_AMT) AS X FROM T",
                "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT",
            })
            assert store.match_metric(account_id, question, lang=lang) is None
            # ...and the bare measure question still uses it.
            bare = "net sales" if lang == "en" else "ventes nettes"
            assert store.match_metric(account_id, bare, lang=lang) is not None
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(workdir, ignore_errors=True)
