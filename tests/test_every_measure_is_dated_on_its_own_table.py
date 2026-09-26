"""
Every measure in a question is dated on its own table, however its date is settled.

"Net sales and returns for the last 6 months" sums two tables, and each
needs its own date. Only a question whose every table had a settled default
date got one each (tests/test_one_date_per_fact.py). A date settled any
other way settled one table and left the other unfiltered -- six months of
sales beside returns over all time, with nothing on the answer to show it:

- a date named for one measure: "... and returns by return date", or
  "returned amount", which names the Return Date by its verb;
- a metric's own default date (its Default time column);
- the reader's answer to "which date?" about one table, and a date
  remembered from the thread.

And two metrics each with a default date were offered to the reader as
alternatives: "which date should I use?".

Now each table gets its own date -- the one the reader confirmed for it, the
ones the question names on it, the one remembered for it, its measure's
default date, then its default date -- a real choice on one table is asked
about that table alone, and a table left with none is refused by name. A
question over one table resolves as before.

The real resolver, on date roles built by the real model writer from a
synthetic schema: a sales table with order and invoice dates, a returns
table with a return date.
"""

from __future__ import annotations

import json

import pytest

from core.contextual_dates import resolve_contextual_date_binding
from core.question_normalizer import canonical_question
from core.semantic_model import (
    load_semantic_model, patch_date_role, set_default_date_role, write_semantic_model,
)

SALES, RETURNS = "MART.SLS_FCT", "MART.RTN_FCT"


@pytest.fixture
def model(tmp_path):
    """Order and invoice dates on sales, a return date on returns, all approved;
    the invoice and return dates are their tables' defaults."""
    tables = {
        "SLS_FCT": [("ORD_DT_DMS_KEY", "int"), ("IVC_DT_DMS_KEY", "int"), ("NET_AMT", "decimal")],
        "RTN_FCT": [("RTN_DT_DMS_KEY", "int"), ("RTN_AMT", "decimal")],
        "DT_DMS": [("DT_DMS_KEY", "int"), ("CAL_DT", "date")],
    }
    schema = {f"WH.MART.{name}": {"database": "WH", "schema": "MART", "table": name,
                                  "columns": [{"name": c, "type": t} for c, t in columns]}
              for name, columns in tables.items()}
    (tmp_path / "schema").mkdir()
    (tmp_path / "schema" / "_schema.json").write_text(json.dumps(schema))
    kb = str(tmp_path / "kb")
    write_semantic_model(schema_dir=str(tmp_path / "schema"), kb_dir=kb)
    for role in load_semantic_model(kb)["date_roles"]:
        patch_date_role(kb_dir=kb, fact_table=role["fact_table"], fact_column=role["fact_column"], status="approved")
    # The writer reads RTN as "Rtn"; an administrator names it.
    patch_date_role(kb_dir=kb, fact_table=RETURNS, fact_column="RTN_DT_DMS_KEY", name="Return Date",
                    business_role="return_date", synonyms=["return date"], status="approved")
    set_default_date_role(kb, SALES, "IVC_DT_DMS_KEY")
    set_default_date_role(kb, RETURNS, "RTN_DT_DMS_KEY")
    return kb


def _roles(kb):
    return load_semantic_model(kb)["date_roles"]


def _role(kb, column):
    return next(role for role in _roles(kb) if role["fact_column"] == column)


def _without_default(kb, column):
    return [dict(role, is_default=False) if role["fact_column"] == column else role for role in _roles(kb)]


NET_SALES = {"name": "Net sales", "base_table": SALES}
RETURNED = {"name": "Returns", "base_table": RETURNS}


def _resolve(question, roles, metrics=(NET_SALES, RETURNED), **kw):
    return resolve_contextual_date_binding(
        canonical_question(question, "en"), matched_metrics=[dict(m) for m in metrics], bindings=[],
        date_roles=roles, required_fact_tables={SALES, RETURNS}, **kw)


def _dates(resolution):
    return sorted((binding["fact_table"], binding["fact_column"], binding["resolution_source"])
                  for binding in resolution["bindings"])


class TestADateNamedForOneMeasure:

    def test_leaves_the_other_on_its_own_default(self, model):
        resolution = _resolve("net sales and returns by return date for the last 6 months", _roles(model))
        assert resolution["status"] == "selected_many"
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "explicit_date_role"),
                                      (SALES, "IVC_DT_DMS_KEY", "fact_default_date_role")]

    def test_by_its_verb_too(self, model):
        resolution = _resolve("net sales and returned amount for the last 6 months", _roles(model))
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "explicit_date_role"),
                                      (SALES, "IVC_DT_DMS_KEY", "fact_default_date_role")]

    def test_is_refused_when_the_other_has_no_date(self, model):
        resolution = _resolve("net sales and returns by return date for the last 6 months",
                              _without_default(model, "IVC_DT_DMS_KEY"))
        assert resolution["status"] == "undated_fact_in_scope"
        assert resolution["undated_facts"] == [SALES]
        assert resolution["dated_facts"] == [RETURNS]


class TestAMetricsOwnDefaultDate:

    def test_dates_its_table_and_the_other_keeps_its_default(self, model):
        sales = dict(NET_SALES, default_time_column="ORD_DT_DMS_KEY")
        resolution = _resolve("net sales and returns for the last 6 months", _roles(model), (sales, RETURNED))
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "fact_default_date_role"),
                                      (SALES, "ORD_DT_DMS_KEY", "metric_default_time_column")]

    def test_two_metrics_with_their_own_dates_are_not_alternatives(self, model):
        roles = [dict(role, is_default=False) for role in _roles(model)]
        sales = dict(NET_SALES, default_time_column="ORD_DT_DMS_KEY")
        returned = dict(RETURNED, default_time_column="RTN_DT_DMS_KEY")
        resolution = _resolve("net sales and returns for the last 6 months", roles, (sales, returned))
        assert resolution["status"] == "selected_many"
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "metric_default_time_column"),
                                      (SALES, "ORD_DT_DMS_KEY", "metric_default_time_column")]

    def test_does_not_date_the_other_table(self, model):
        sales = dict(NET_SALES, default_time_column="ORD_DT_DMS_KEY")
        resolution = _resolve("net sales and returns for the last 6 months",
                              _without_default(model, "RTN_DT_DMS_KEY"), (sales, RETURNED))
        assert resolution["status"] == "undated_fact_in_scope"
        assert resolution["undated_facts"] == [RETURNS]


class TestTheReadersChoice:

    def test_is_asked_about_the_one_table_with_a_choice(self, model):
        roles = [dict(role, is_default=True) if role["fact_table"] == SALES else role for role in _roles(model)]
        resolution = _resolve("net sales and returns for the last 6 months", roles)
        assert resolution["status"] == "ambiguous"
        assert SALES in resolution["reason"]
        assert {option["fact_column"] for option in resolution["options"]} == {"ORD_DT_DMS_KEY", "IVC_DT_DMS_KEY"}

    def test_once_made_dates_that_table_and_the_other_keeps_its_own(self, model):
        roles = [dict(role, is_default=True) if role["fact_table"] == SALES else role for role in _roles(model)]
        resolution = _resolve("net sales and returns for the last 6 months", roles,
                              confirmed_date_role=_role(model, "ORD_DT_DMS_KEY"))
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "fact_default_date_role"),
                                      (SALES, "ORD_DT_DMS_KEY", "user_confirmed_date_role")]

    def test_remembered_from_the_thread_dates_only_its_table(self, model):
        remembered = dict(_role(model, "ORD_DT_DMS_KEY"), context_name="Order Date")
        resolution = _resolve("net sales and returns for the last 6 months", _roles(model),
                              remembered_date_role=remembered)
        assert _dates(resolution) == [(RETURNS, "RTN_DT_DMS_KEY", "fact_default_date_role"),
                                      (SALES, "ORD_DT_DMS_KEY", "thread_date_preference")]


class TestOneTableAsBefore:

    def test_a_named_date_on_the_only_table(self, model):
        resolution = _resolve("net sales by order date for the last 6 months", _roles(model), (NET_SALES,))
        assert resolution["status"] == "selected"
        assert resolution["binding"]["fact_column"] == "ORD_DT_DMS_KEY"

    def test_a_confirmed_date_on_the_only_table(self, model):
        resolution = _resolve("net sales for the last 6 months", _roles(model), (NET_SALES,),
                              confirmed_date_role=_role(model, "ORD_DT_DMS_KEY"))
        assert (resolution["status"], resolution["binding"]["resolution_source"]) == (
            "selected", "user_confirmed_date_role")


class TestWhatStillHoldsAcrossTables:

    def test_the_available_dates_take_no_date(self, model):
        resolution = _resolve("net sales and returns for the available dates", _roles(model))
        assert resolution["status"] == "none"

    def test_a_table_whose_date_is_coarser_than_the_question_is_said_to_be(self, model):
        month_grain = [dict(role, temporal_grain="month") if role["fact_table"] == RETURNS else role
                       for role in _roles(model)]
        resolution = _resolve("net sales and returns by day for the last 30 days", month_grain)
        assert resolution["status"] == "unsupported_grain"
        assert (resolution["requested_grain"], resolution["available_grain"]) == ("day", "month")
        assert [option["fact_table"] for option in resolution["options"]] == [RETURNS]
