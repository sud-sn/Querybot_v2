# -*- coding: utf-8 -*-
"""A date is found by the words readers use for it, in English and French.

A question names a business date by its name ("ship date"), by the grain in
place of "date" ("ship month"), or by the event itself. Only the name, the
grain form and the stem with "ed", "d" or "ing" glued on were read, so:

  * "net sales shipped in March" and "by shipping month" named no date --
    "shiped" and "shiping" were what the matcher looked for;
  * "orders by month" and "deliveries by week" named none either;
  * "ventes expédiées en mars" reached it as "sales expediees", and "ventes
    mensuelles" as "sales mensuelles", with no month in it;
  * "mois de commande" -- French word order, "month of orders" once
    canonicalised -- was not the Order Date;
  * accented letters were separators, so a synonym written "date
    d'expédition" never met a question typed "date d'expedition".

A question naming no date falls back to a default or asks, so each of these
answered on a date the reader did not name.

Now the event's plural and verb forms count, French participles and period
adjectives are canonicalised, "month of X" is read like "X month", and
accents are folded on both sides. A verb outranks a noun: "orders delivered
last month" is the Delivery Date, not the Order Date as well.

The real matcher and resolver, on date roles built by the real model writer
from a synthetic schema.
"""

from __future__ import annotations

import json

import pytest

from core.contextual_dates import find_explicit_date_roles, requested_temporal_grain, resolve_contextual_date_binding
from core.question_normalizer import canonical_question
from core.semantic_model import load_semantic_model, patch_date_role, write_semantic_model

SALES = "MART.SLS_FCT"


def _date_roles(tmp_path, *date_keys, approved=True):
    columns = [("SLS_FCT_KEY", "bigint"), *((key, "int") for key in date_keys), ("NET_AMT", "decimal")]
    schema = {
        "WH.MART.SLS_FCT": {"database": "WH", "schema": "MART", "table": "SLS_FCT",
                            "columns": [{"name": n, "type": t} for n, t in columns]},
        "WH.MART.DT_DMS": {"database": "WH", "schema": "MART", "table": "DT_DMS",
                           "columns": [{"name": "DT_DMS_KEY", "type": "int"}, {"name": "CAL_DT", "type": "date"}]},
    }
    (tmp_path / "schema").mkdir()
    (tmp_path / "schema" / "_schema.json").write_text(json.dumps(schema))
    kb = str(tmp_path / "kb")
    write_semantic_model(schema_dir=str(tmp_path / "schema"), kb_dir=kb)
    for role in load_semantic_model(kb)["date_roles"] if approved else []:
        patch_date_role(kb_dir=kb, fact_table=role["fact_table"], fact_column=role["fact_column"], status="approved")
    return kb


@pytest.fixture
def order_invoice_ship(tmp_path):
    return _date_roles(tmp_path, "ORD_DT_DMS_KEY", "IVC_DT_DMS_KEY", "SHP_DT_DMS_KEY")


@pytest.fixture
def order_delivery(tmp_path):
    return _date_roles(tmp_path, "ORD_DT_DMS_KEY", "DLV_DT_DMS_KEY")


@pytest.fixture
def order_delivery_unreviewed(tmp_path):
    """The same dates as discovery leaves them: generated, not yet approved."""
    return _date_roles(tmp_path, "ORD_DT_DMS_KEY", "DLV_DT_DMS_KEY", approved=False)


def _named(kb, question, lang="en"):
    roles = load_semantic_model(kb)["date_roles"]
    return sorted(role["name"] for role in find_explicit_date_roles(canonical_question(question, lang), roles))


class TestTheEventsWords:

    @pytest.mark.parametrize("question, lang, date", [
        ("net sales shipped in march 2025", "en", "Shipment Date"),
        ("net sales by shipping month", "en", "Shipment Date"),
        ("orders by month", "en", "Order Date"),
        ("shipments by month of order", "en", "Order Date"),
        ("ventes expédiées en mars 2025", "fr", "Shipment Date"),
        ("expéditions par mois de commande", "fr", "Order Date"),
        ("commandes par semaine d’expédition", "fr", "Shipment Date"),
    ])
    def test_the_plural_verb_and_french_forms_name_the_date(self, order_invoice_ship, question, lang, date):
        assert _named(order_invoice_ship, question, lang) == [date]

    @pytest.mark.parametrize("question, lang", [
        ("deliveries by week", "en"),
        ("orders delivered last month", "en"),
        ("commandes livrées le mois dernier", "fr"),
    ])
    def test_the_events_verb_outranks_the_counted_noun(self, order_delivery, question, lang):
        assert _named(order_delivery, question, lang) == ["Delivery Date"]

    def test_a_verb_made_from_the_dates_name_outranks_the_counted_noun(self, order_invoice_ship):
        # "shipped" comes from "Shipment Date" itself; "orders" names the Order Date.
        assert _named(order_invoice_ship, "orders shipped last week") == ["Shipment Date"]

    def test_the_verb_is_read_from_the_name_when_no_synonym_spells_it(self, order_delivery):
        # An administrator's own synonym list replaces the discovered one,
        # which spelled "delivered date" out.
        patch_date_role(kb_dir=order_delivery, fact_table=SALES, fact_column="DLV_DT_DMS_KEY",
                        synonyms=["delivery date"], status="approved")
        assert _named(order_delivery, "orders delivered last month") == ["Delivery Date"]

    def test_a_named_date_still_outranks_the_event_words(self, order_invoice_ship):
        assert _named(order_invoice_ship, "orders shipped by invoice month") == ["Invoice Date"]

    def test_a_word_containing_the_event_is_not_the_event(self, order_invoice_ship):
        assert _named(order_invoice_ship, "revenue from reorders by month") == []


class TestAccents:

    def test_a_synonym_with_accents_meets_a_question_without_them(self, order_invoice_ship):
        patch_date_role(kb_dir=order_invoice_ship, fact_table=SALES, fact_column="SHP_DT_DMS_KEY",
                        synonyms=["shipment date", "date d'expédition"], status="approved")
        assert _named(order_invoice_ship, "net sales by date d'expedition") == ["Shipment Date"]

    def test_and_a_question_with_them_meets_a_synonym_without(self, order_invoice_ship):
        patch_date_role(kb_dir=order_invoice_ship, fact_table=SALES, fact_column="SHP_DT_DMS_KEY",
                        synonyms=["shipment date", "date d'envoi"], status="approved")
        assert _named(order_invoice_ship, "net sales by date d'énvoi") == ["Shipment Date"]


@pytest.mark.parametrize("question, grain", [
    ("ventes mensuelles", "month"),
    ("chiffre d’affaires hebdomadaire", "week"),
    ("ventes trimestrielles par région", "quarter"),
    ("quantités journalières du mois dernier", "day"),
])
def test_a_french_period_adjective_is_the_grain(question, grain):
    assert requested_temporal_grain(canonical_question(question, "fr")) == grain


def test_the_resolver_binds_the_date_the_verb_names(order_invoice_ship):
    resolution = resolve_contextual_date_binding(
        canonical_question("ventes expédiées en mars 2025", "fr"),
        matched_metrics=[{"name": "Net sales", "base_table": SALES}], bindings=[],
        date_roles=load_semantic_model(order_invoice_ship)["date_roles"], required_fact_tables={SALES})
    assert resolution["status"] == "selected"
    assert resolution["binding"]["fact_column"] == "SHP_DT_DMS_KEY"


def _resolved(kb, question, metrics):
    return resolve_contextual_date_binding(
        canonical_question(question, "en"), matched_metrics=metrics, bindings=[],
        date_roles=load_semantic_model(kb)["date_roles"], required_fact_tables={SALES})


class TestAMeasuresOwnNameIsNotItsEvent:
    """"Returns" summed beside revenue is the measure, not the Return Date's
    event: read as a date, it put a two-measure question on that one date and
    left the other measure's rows unfiltered (tests/test_one_date_per_fact.py
    has that case). Words in the measure's name say what is measured; the
    measure's own dates say when."""

    def test_the_measures_name_names_no_date(self, order_delivery):
        resolution = _resolved(order_delivery, "orders by month", [{"name": "Orders", "base_table": SALES}])
        assert resolution["status"] == "ambiguous"
        assert {option["fact_column"] for option in resolution["options"]} == {"ORD_DT_DMS_KEY", "DLV_DT_DMS_KEY"}

    def test_nor_does_its_synonym(self, order_delivery):
        measure = {"name": "Order Count", "synonyms": ["orders"], "base_table": SALES}
        assert _resolved(order_delivery, "orders by month", [measure])["status"] == "ambiguous"

    def test_without_that_measure_the_word_still_names_its_date(self, order_delivery):
        resolution = _resolved(order_delivery, "orders by month", [{"name": "Net sales", "base_table": SALES}])
        assert resolution["status"] == "selected"
        assert resolution["binding"]["fact_column"] == "ORD_DT_DMS_KEY"

    def test_an_event_word_outside_the_name_still_names_its_date(self, order_delivery):
        resolution = _resolved(order_delivery, "orders delivered last month", [{"name": "Orders", "base_table": SALES}])
        assert resolution["status"] == "selected"
        assert resolution["binding"]["fact_column"] == "DLV_DT_DMS_KEY"

    def test_nor_on_dates_not_yet_approved(self, order_delivery_unreviewed):
        orders = _resolved(order_delivery_unreviewed, "orders by month", [{"name": "Orders", "base_table": SALES}])
        sales = _resolved(order_delivery_unreviewed, "orders by month", [{"name": "Net sales", "base_table": SALES}])
        assert orders["status"] == "ambiguous"
        assert (sales["status"], sales["binding"]["fact_column"]) == ("selected", "ORD_DT_DMS_KEY")

    def test_a_date_named_outright_inside_the_name_still_counts(self, order_delivery):
        measure = {"name": "Order Date Count", "base_table": SALES}
        resolution = _resolved(order_delivery, "order date count for 2025", [measure])
        assert resolution["binding"]["fact_column"] == "ORD_DT_DMS_KEY"
