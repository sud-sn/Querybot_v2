"""
A question that names a period is never answered from the database clock.

The validator refused GETDATE() and its kin only when the question asked for
a period relative to now -- "last month", "this year". A period named on the
calendar was not watched: for "revenue since January" the model took the
year from GETDATE(), and for "revenue in 2025" it could write
YEAR(GETDATE()) - 1. The server's calendar is not the data's: on a warehouse
loaded to another day, or in another year, that is a different period, and
nothing said so.

Now the clock is refused whenever the question names a period -- relative,
or on the calendar -- read from the canonical question, so a French one is
seen too. And French period wording is recognised where it was not: "ventes
en 2025", "depuis 2024", "pendant 2025" and "jusqu'à 2025" (and "before" /
"after" a year) skipped date resolution altogether, where "sales in 2025"
entered it.

The real validator, canonicaliser and date resolver; the context is the one
the pipeline gives the validator.
"""

from __future__ import annotations

import json

import pytest

from core.contextual_dates import detect_temporal_window, resolve_contextual_date_binding
from core.question_normalizer import canonical_question
from core.semantic_model import load_semantic_model, patch_date_role, write_semantic_model
from core.validator import validate_sql_detailed

FACT = "DB.dbo.SALES_FCT"
COLUMNS = {FACT: {"INV_DT": "date", "SHP_DT": "date", "AMT": "decimal"}}
LAST_YEAR = f"SELECT SUM(AMT) FROM {FACT} WHERE YEAR(INV_DT) = YEAR(GETDATE()) - 1"
THIS_YEAR = f"SELECT SUM(AMT) FROM {FACT} WHERE INV_DT >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)"
CAPPED_AT_TODAY = (f"SELECT SUM(AMT) FROM {FACT} WHERE INV_DT >= '2026-01-01' "
                   "AND INV_DT < '2026-04-01' AND INV_DT <= CAST(GETDATE() AS DATE)")
AS_WRITTEN = f"SELECT SUM(AMT) FROM {FACT} WHERE INV_DT >= '2025-01-01' AND INV_DT < '2026-01-01'"


def _validate(question, sql, lang="en"):
    canonical = canonical_question(question, lang)
    return validate_sql_detailed(sql, {FACT}, "azure_sql", None, COLUMNS, {
        "question": question, "canonical_question": canonical,
        "semantic_plan": {"temporal_policies": []},
        "temporal_window": detect_temporal_window(canonical) or None,
    })


class TestTheClockIsRefused:

    @pytest.mark.parametrize("question, sql", [
        ("revenue in 2025", LAST_YEAR),
        ("revenue since January", THIS_YEAR),
        ("revenue for Q1 2026", CAPPED_AT_TODAY),
        ("revenue before 2026", LAST_YEAR),
    ])
    def test_for_a_period_named_on_the_calendar(self, question, sql):
        result = _validate(question, sql)
        assert (result.ok, result.code) == (False, "temporal_anchor_ungoverned")

    @pytest.mark.parametrize("question, sql", [
        ("chiffre d'affaires en 2025", LAST_YEAR),
        ("ventes pour le T1 2026", CAPPED_AT_TODAY),
        ("ventes depuis janvier", THIS_YEAR),
        ("ventes pendant 2025", LAST_YEAR),
        ("ventes durant 2025", LAST_YEAR),
        ("ventes jusqu'à 2025", LAST_YEAR),
        ("ventes après 2024", THIS_YEAR),
    ])
    def test_for_a_period_named_in_french(self, question, sql):
        result = _validate(question, sql, "fr")
        assert (result.ok, result.code) == (False, "temporal_anchor_ungoverned")

    def test_for_a_period_relative_to_now_as_before(self):
        result = _validate("revenue this year", THIS_YEAR)
        assert (result.ok, result.code) == (False, "temporal_anchor_ungoverned")


class TestWhatStillPasses:

    def test_the_named_period_written_as_it_stands(self):
        assert _validate("revenue in 2025", AS_WRITTEN).ok is True

    def test_a_question_that_names_no_period(self):
        assert _validate("list the open orders", THIS_YEAR).ok is True

    def test_an_amount_that_looks_like_a_year(self):
        assert _validate("orders over 2000 dollars", THIS_YEAR).ok is True


SALES = "MART.SLS_FCT"


@pytest.fixture
def two_approved_dates(tmp_path):
    """Order and ship dates on one sales fact, both approved, neither the default."""
    columns = [("ORD_DT_DMS_KEY", "int"), ("SHP_DT_DMS_KEY", "int"), ("NET_AMT", "decimal")]
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
    for role in load_semantic_model(kb)["date_roles"]:
        patch_date_role(kb_dir=kb, fact_table=role["fact_table"], fact_column=role["fact_column"], status="approved")
    return load_semantic_model(kb)["date_roles"]


def _resolved(question, roles, lang):
    return resolve_contextual_date_binding(
        canonical_question(question, lang), matched_metrics=[{"name": "Net sales", "base_table": SALES}],
        bindings=[], date_roles=roles, required_fact_tables={SALES})["status"]


class TestAYearNamedInFrenchEntersDateResolution:
    """With two approved dates and no default, a named year asks which date,
    as "net sales in 2025" does -- it does not skip date resolution as a
    question with no period in it."""

    def test_as_the_english_does(self, two_approved_dates):
        assert _resolved("net sales in 2025", two_approved_dates, "en") == "ambiguous"

    @pytest.mark.parametrize("question", [
        "ventes nettes en 2025", "ventes nettes depuis 2024", "ventes nettes pendant 2025",
        "ventes nettes avant 2026",
    ])
    def test_and_asks_which_date(self, two_approved_dates, question):
        assert _resolved(question, two_approved_dates, "fr") == "ambiguous"
