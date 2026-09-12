"""
tests/test_the_date_role_form_reads_the_servers_answer.py

The Add-Date-Role form typed every Infor M3 surrogate date key as a packed
integer, then erased the date dimension it needed.

Two defects, one root. The root is that the form inferred a column's physical
storage type in JavaScript, with its own copy of the naming rules -- and that
copy was missing the rule ``classify_date_key`` states in as many words:

    A relationship to a date dimension always wins over name-based inference:
    integer IDs such as 4067 are surrogate keys, not YYYYMMDD values.

Measured, with EMDW_DMART.DT_DMS sitting in the same schema and the server's own
discovery resolving every one of these to ``surrogate_fk`` at confidence 92:

    column                  browser form        server
    CUS_IVC_DT_DMS_KEY      yyyymmdd_integer    surrogate_fk
    RTN_DT_DMS_KEY          yyyymmdd_integer    surrogate_fk
    BAL_DT_DMS_KEY          yyyymmdd_integer    surrogate_fk
    CNL_ORD_DT_DMS_KEY      yyyymmdd_integer    surrogate_fk
    SHP_DT_DMS_KEY          yyyymmdd_integer    surrogate_fk

Then the form hid the dimension pickers as unnecessary and DISABLED them, so the
POST carried empty strings, and ``patch_date_role`` -- which calls
``normalize_date_key_type`` without the flag that enforces the rule above --
cleared the validated mapping and left the surrogate key itself standing as the
calendar date. Status approved, confidence 100.

What that compiles to on Azure is the part that makes it a demo blocker rather
than an annoyance. ``format_date_value_expression`` emits

    TRY_CONVERT(date, CONVERT(varchar(8), CUS_IVC_DT_DMS_KEY), 112)

and TRY_CONVERT returns NULL rather than raising. A surrogate key of 4067 is
'4067', which is not a date, so every row's date is NULL, every date predicate
filters everything out, and every dated question on the fact returns zero rows
-- with no error, no caveat and nothing in the log to look at.

The fix removes the second copy of the rule instead of correcting it:
``suggest_date_key_bindings`` computes the answer on the server with the same
functions ``_date_roles`` uses, the form looks it up and prefills the mapping,
and ``patch_date_role`` refuses to trade a validated mapping for a name-inferred
integer encoding and says so.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from admin import routes
from core.semantic_model import (
    MODEL_JSON,
    _date_roles,
    load_semantic_model,
    patch_date_role,
    suggest_date_key_bindings,
)
from tests.js_lift import function as lift

TEMPLATE = (Path(__file__).resolve().parents[1] / "admin" / "templates"
            / "client_date_roles.html").read_text(encoding="utf-8")

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DIM = "EMDW_DMART.DT_DMS"

# EMCO's shape: surrogate date keys on the fact, one shared date dimension.
SCHEMA = {
    FACT: {"columns": [
        {"name": "CUS_IVC_DT_DMS_KEY", "type": "int"},
        {"name": "CNL_ORD_DT_DMS_KEY", "type": "int"},
        {"name": "SHP_DT_DMS_KEY", "type": "bigint"},
        {"name": "IVC_PRD_DMS_KEY", "type": "int"},
        {"name": "ORD_ENT_DT", "type": "date"},
        {"name": "NET_SLS_AMT", "type": "decimal(18,2)"},
        {"name": "CUS_NBR", "type": "int"},
    ]},
    DIM: {"columns": [
        {"name": "DT_DMS_KEY", "type": "int"},
        {"name": "DMS_DT", "type": "date"},
        {"name": "DMS_YR", "type": "int"},
    ]},
}

SURROGATE_KEYS = ["CUS_IVC_DT_DMS_KEY", "CNL_ORD_DT_DMS_KEY", "SHP_DT_DMS_KEY"]


def model_from(schema: dict) -> dict:
    """The semantic model as the KB build writes it, from a raw schema."""
    tables = []
    for fqn, meta in schema.items():
        table = {
            "qualified_name": fqn,
            "table": fqn.split(".")[-1],
            "fields": [{"column": c["name"], "data_type": c["type"]}
                       for c in meta["columns"]],
        }
        table["date_roles"] = _date_roles(schema, fqn, meta)
        tables.append(table)
    return {
        "tables": tables,
        "date_roles": [dict(role) for table in tables
                       for role in table["date_roles"]],
    }


MODEL = model_from(SCHEMA)


class TestTheServerResolvesEmcosKeysToTheDimension:
    """Not the fix under test -- the ground truth the fix has to agree with."""

    @pytest.mark.parametrize("column", SURROGATE_KEYS)
    def test_discovery_calls_it_a_surrogate_key(self, column):
        role = next(r for r in MODEL["date_roles"] if r["fact_column"] == column)
        assert role["date_key_type"] == "surrogate_fk"
        assert (role["dimension_table"], role["dimension_key"],
                role["date_value_column"]) == (DIM, "DT_DMS_KEY", "DMS_DT")


class TestTheSuggestionCannotDriftFromDiscovery:
    """One rule, one implementation. The form's old copy disagreed with
    discovery on every column that mattered, which is the whole defect."""

    @pytest.mark.parametrize("column", [
        "CUS_IVC_DT_DMS_KEY", "CNL_ORD_DT_DMS_KEY", "SHP_DT_DMS_KEY",
        "IVC_PRD_DMS_KEY", "ORD_ENT_DT",
    ])
    def test_it_says_what_discovery_says(self, column):
        suggestion = suggest_date_key_bindings(MODEL)["bindings"][f"{FACT}|{column}"]
        role = next(r for r in MODEL["date_roles"] if r["fact_column"] == column)
        assert suggestion["date_key_type"] == role["date_key_type"]
        assert suggestion["dimension_table"] == role["dimension_table"]
        assert suggestion["dimension_key"] == role["dimension_key"]
        assert suggestion["date_value_column"] == role["date_value_column"]

    def test_a_measure_gets_no_date_suggestion_at_all(self):
        bindings = suggest_date_key_bindings(MODEL)["bindings"]
        assert f"{FACT}|NET_SLS_AMT" not in bindings
        assert f"{FACT}|CUS_NBR" not in bindings

    def test_the_date_dimension_is_named_so_the_form_can_offer_it(self):
        """The dimension select used to list only tables the model TYPED as
        dimensions, so an untyped DT_DMS could not be selected at all."""
        assert suggest_date_key_bindings(MODEL)["date_dimensions"] == [DIM]

    def test_the_dimension_is_not_a_role_on_itself(self):
        bindings = suggest_date_key_bindings(MODEL)["bindings"]
        assert not [key for key in bindings if key.startswith(f"{DIM}|")]


class TestTheBrowserNowAgrees:
    """The form's JavaScript is EXECUTED here against the same model the route
    hands it. Reading the two side by side is exactly what let them drift."""

    @staticmethod
    def _options(table_name: str) -> dict:
        dukpy = pytest.importorskip(
            "dukpy",
            reason="a JavaScript engine is required to EXECUTE the form's own "
                   "inference; reading it is how it drifted from the server",
        )
        bindings = suggest_date_key_bindings(MODEL)
        harness = f"""
var document = {{
  createElement: function(){{ return {{dataset: {{}}, value: '', textContent: ''}}; }}
}};
var tables = {json.dumps(MODEL["tables"])};
var byName = {{}};
tables.forEach(function(t){{
  byName[String(t.qualified_name || t.table || '').toUpperCase()] = t;
}});
var BINDINGS = {json.dumps(bindings["bindings"])};
{lift(TEMPLATE, "function columnName(field)")}
{lift(TEMPLATE, "function bindingFor(tableName, column)")}
{lift(TEMPLATE, "function fill(select, tableName, predicate, placeholder)")}
var select = {{
  innerHTML: '', options: [],
  appendChild: function(option){{ this.options.push(option); }}
}};
fill(select, {json.dumps(table_name)}, null, 'Select any source column');
var out = {{}};
select.options.forEach(function(option){{ out[option.value] = option.dataset; }});
out;
"""
        return dukpy.evaljs(harness)

    @pytest.mark.parametrize("column", SURROGATE_KEYS)
    def test_the_form_types_it_as_a_surrogate_key(self, column):
        assert self._options(FACT)[column]["dateType"] == "surrogate_fk"

    @pytest.mark.parametrize("column", SURROGATE_KEYS)
    def test_and_carries_the_mapping_so_it_can_be_prefilled(self, column):
        dataset = self._options(FACT)[column]
        assert dataset["dimensionTable"] == DIM
        assert dataset["dimensionKey"] == "DT_DMS_KEY"
        assert dataset["dateValueColumn"] == "DMS_DT"

    def test_a_month_grain_period_key_is_still_an_encoding(self):
        """_date_roles deliberately does not attach a monthly *_PRD_DMS_KEY to a
        day dimension. The form must not either."""
        dataset = self._options(FACT)["IVC_PRD_DMS_KEY"]
        assert dataset["dateType"] == "yyyymm_integer"
        assert dataset["dimensionTable"] == ""

    def test_a_native_date_column_needs_no_dimension(self):
        dataset = self._options(FACT)["ORD_ENT_DT"]
        assert dataset["dateType"] == "native_date"
        assert dataset["dimensionTable"] == ""

    def test_a_measure_is_offered_with_no_date_type(self):
        dataset = self._options(FACT)["NET_SLS_AMT"]
        assert dataset["dateType"] == ""

    def test_every_column_the_form_stamps_matches_the_server(self):
        bindings = suggest_date_key_bindings(MODEL)["bindings"]
        for column, dataset in self._options(FACT).items():
            expected = bindings.get(f"{FACT}|{column}") or {}
            assert dataset["dateType"] == (expected.get("date_key_type") or "")
            assert dataset["dimensionTable"] == (expected.get("dimension_table") or "")


class _OnDisk(unittest.TestCase):
    """A real model.json, because patch_date_role reads and writes one."""

    def setUp(self):
        self.kb_dir = tempfile.mkdtemp(prefix="qb-date-role-")
        (Path(self.kb_dir) / MODEL_JSON).write_text(json.dumps(MODEL))

    def tearDown(self):
        shutil.rmtree(self.kb_dir, ignore_errors=True)

    def role(self, column="CUS_IVC_DT_DMS_KEY"):
        return next(r for r in load_semantic_model(self.kb_dir)["date_roles"]
                    if r["fact_column"] == column)

    def per_table_role(self, column="CUS_IVC_DT_DMS_KEY"):
        table = next(t for t in load_semantic_model(self.kb_dir)["tables"]
                     if t["qualified_name"] == FACT)
        return next(r for r in table["date_roles"] if r["fact_column"] == column)


class TestTheMappingSurvivesTheFormsOwnPost(_OnDisk):

    def _post_as_the_form_did(self, column="CUS_IVC_DT_DMS_KEY",
                              date_key_type="yyyymmdd_integer"):
        """What the browser actually submitted: the auto-typed integer, and the
        dimension fields empty because they were hidden and disabled."""
        notes: list[str] = []
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT, fact_column=column,
            dimension_table="", dimension_key="", date_value_column="",
            date_key_type=date_key_type, business_role="invoice_date",
            status="approved", notes=notes,
        )
        return notes

    def test_the_dimension_mapping_is_not_erased(self):
        self._post_as_the_form_did()
        role = self.role()
        self.assertEqual(role["dimension_table"], DIM)
        self.assertEqual(role["dimension_key"], "DT_DMS_KEY")
        self.assertEqual(role["date_value_column"], "DMS_DT")

    def test_the_surrogate_key_is_not_left_standing_as_the_date(self):
        """This is the state that returns zero rows for every dated question:
        the key itself declared to be the calendar date value."""
        self._post_as_the_form_did()
        self.assertNotEqual(self.role()["date_value_column"], "CUS_IVC_DT_DMS_KEY")
        self.assertEqual(self.role()["date_key_type"], "surrogate_fk")

    def test_the_per_table_copy_is_kept_too(self):
        """The compiler reads the per-table entry; the page reads the top-level
        one. Fixing one and not the other is invisible until a query runs."""
        self._post_as_the_form_did()
        self.assertEqual(self.per_table_role()["date_key_type"], "surrogate_fk")
        self.assertEqual(self.per_table_role()["dimension_table"], DIM)

    def test_the_month_grain_key_is_overruled_the_same_way(self):
        notes = self._post_as_the_form_did(date_key_type="yyyymm_integer")
        self.assertEqual(self.role()["date_key_type"], "surrogate_fk")
        self.assertTrue(notes)

    def test_the_admin_is_told_rather_than_quietly_overruled(self):
        notes = self._post_as_the_form_did()
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("CUS_IVC_DT_DMS_KEY", notes[0])
        self.assertIn(DIM, notes[0])
        self.assertIn("DMS_DT", notes[0])

    def test_the_approval_still_takes_effect(self):
        """Overruling the storage type must not cost the approval itself."""
        self._post_as_the_form_did()
        self.assertEqual(self.role()["status"], "approved")
        self.assertEqual(self.role()["confidence"], 100)

    def test_synonyms_and_the_role_name_still_save(self):
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT,
            fact_column="CUS_IVC_DT_DMS_KEY", date_key_type="yyyymmdd_integer",
            name="Date de facturation", synonyms=["date de facturation"],
            status="approved",
        )
        self.assertEqual(self.role()["name"], "Date de facturation")
        self.assertEqual(self.role()["synonyms"], ["date de facturation"])


class TestTheAdminCanStillSayWhatTheyMean(_OnDisk):
    """The guard overrules exactly two auto-inferred integer encodings. It must
    not become a lock on the admin's own deliberate choices."""

    def test_a_native_date_column_is_honoured(self):
        notes: list[str] = []
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT,
            fact_column="CUS_IVC_DT_DMS_KEY", date_key_type="native_date",
            date_value_column="ORD_ENT_DT", status="approved", notes=notes,
        )
        self.assertEqual(self.role()["date_key_type"], "native_date")
        self.assertEqual(self.role()["dimension_table"], "")
        self.assertEqual(self.role()["date_value_column"], "ORD_ENT_DT")
        self.assertEqual(notes, [])

    def test_a_remap_to_another_dimension_is_honoured(self):
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT,
            fact_column="CUS_IVC_DT_DMS_KEY", date_key_type="surrogate_fk",
            dimension_table=DIM, dimension_key="DT_DMS_KEY",
            date_value_column="DMS_YR", status="approved",
        )
        self.assertEqual(self.role()["date_value_column"], "DMS_YR")

    def test_an_integer_encoding_stands_where_no_mapping_exists(self):
        """A real packed-integer column with no date dimension behind it must
        still be declarable -- that is what this form is for."""
        model = json.loads(
            (Path(self.kb_dir) / MODEL_JSON).read_text())
        for role in model["date_roles"] + model["tables"][0]["date_roles"]:
            if role["fact_column"] == "CUS_IVC_DT_DMS_KEY":
                role.update(dimension_table="", dimension_key="",
                            date_value_column="")
        (Path(self.kb_dir) / MODEL_JSON).write_text(json.dumps(model))
        notes: list[str] = []
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT,
            fact_column="CUS_IVC_DT_DMS_KEY", date_key_type="yyyymmdd_integer",
            status="approved", notes=notes,
        )
        self.assertEqual(self.role()["date_key_type"], "yyyymmdd_integer")
        self.assertEqual(notes, [])

    def test_a_mapping_naming_a_column_the_model_lacks_is_not_validated(self):
        """The guard trusts a mapping only when the model has the table and both
        columns. A stale mapping to a dropped column must not block a correction."""
        model = json.loads((Path(self.kb_dir) / MODEL_JSON).read_text())
        for role in model["date_roles"] + model["tables"][0]["date_roles"]:
            if role["fact_column"] == "CUS_IVC_DT_DMS_KEY":
                role["date_value_column"] = "COLUMN_THAT_WAS_DROPPED"
        (Path(self.kb_dir) / MODEL_JSON).write_text(json.dumps(model))
        patch_date_role(
            kb_dir=self.kb_dir, fact_table=FACT,
            fact_column="CUS_IVC_DT_DMS_KEY", date_key_type="yyyymmdd_integer",
            status="approved",
        )
        self.assertEqual(self.role()["date_key_type"], "yyyymmdd_integer")


def _request(query: dict | None = None):
    class _Req:
        pass
    request = _Req()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    return request


class TestThePageHandsTheFormTheAnswer(_OnDisk):

    def _context(self, query=None):
        client = {"account_id": "acct-dr", "client_name": "EMCO",
                  "state": "READY", "state_data": "{}"}
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=client), \
                patch.object(routes.store, "get_client_state",
                             return_value={"kb_dir": self.kb_dir}), \
                patch.object(routes.store, "list_metrics", return_value=[]), \
                patch.object(routes.store, "list_metric_date_contexts",
                             return_value=[]), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: (t, c)):
            return asyncio.run(routes.date_roles_page(
                _request(query), "acct-dr"))

    def test_the_suggestions_reach_the_template(self):
        template, context = self._context()
        self.assertEqual(template, "client_date_roles.html")
        bindings = context["date_key_bindings"]["bindings"]
        self.assertEqual(
            bindings[f"{FACT}|CUS_IVC_DT_DMS_KEY"]["date_key_type"],
            "surrogate_fk")
        self.assertEqual(
            bindings[f"{FACT}|CUS_IVC_DT_DMS_KEY"]["dimension_table"], DIM)

    def test_the_date_dimension_is_offered(self):
        _template, context = self._context()
        self.assertIn(DIM, context["date_key_bindings"]["date_dimensions"])

    def test_a_notice_is_carried_to_the_page(self):
        _template, context = self._context({"notice": "kept the mapping"})
        self.assertEqual(context["notice"], "kept the mapping")

    def test_a_broken_suggestion_does_not_take_the_page_down(self):
        """Fail-open, but the page must still render a usable form."""
        client = {"account_id": "acct-dr", "client_name": "EMCO",
                  "state": "READY", "state_data": "{}"}
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=client), \
                patch.object(routes.store, "get_client_state",
                             return_value={"kb_dir": self.kb_dir}), \
                patch.object(routes.store, "list_metrics", return_value=[]), \
                patch.object(routes.store, "list_metric_date_contexts",
                             return_value=[]), \
                patch("core.semantic_model.suggest_date_key_bindings",
                      side_effect=RuntimeError("model unreadable")), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: (t, c)):
            _template, context = asyncio.run(
                routes.date_roles_page(_request(), "acct-dr"))
        self.assertEqual(context["date_key_bindings"]["bindings"], {})
        self.assertTrue(context["has_model"])


class TestTheRouteTellsTheAdmin(_OnDisk):

    def _add(self, **form):
        payload = dict(
            fact_table=FACT, fact_column="CUS_IVC_DT_DMS_KEY",
            dimension_table="", dimension_key="", date_value_column="",
            business_role="invoice_date", name="", synonyms="",
            date_key_type="yyyymmdd_integer",
        )
        payload.update(form)
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client_state",
                             return_value={"kb_dir": self.kb_dir}), \
                patch.object(routes, "_after_semantic_approval",
                             return_value=None):
            return asyncio.run(routes.date_role_add(
                _request(), "acct-dr", **payload))

    def test_the_redirect_carries_the_override_notice(self):
        response = self._add()
        location = response.headers["location"]
        self.assertIn("saved=role-added", location)
        self.assertIn("notice=", location)

    def test_and_says_nothing_when_nothing_was_overruled(self):
        response = self._add(date_key_type="surrogate_fk",
                            dimension_table=DIM, dimension_key="DT_DMS_KEY",
                            date_value_column="DMS_DT")
        self.assertNotIn("notice=", response.headers["location"])

    def _approve(self, **form):
        payload = dict(
            fact_table=FACT, fact_column="CUS_IVC_DT_DMS_KEY",
            dimension_table="", dimension_key="", date_value_column="",
            business_role="invoice_date", name="", synonyms="",
            date_key_type="yyyymmdd_integer",
        )
        payload.update(form)
        client = {"account_id": "acct-dr", "client_name": "EMCO",
                  "state": "READY", "state_data": "{}"}
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=client), \
                patch.object(routes.store, "get_client_state",
                             return_value={"kb_dir": self.kb_dir}), \
                patch.object(routes, "_after_semantic_approval",
                             return_value=None):
            return asyncio.run(routes.date_role_approve(
                _request(), "acct-dr", **payload))

    def test_the_approve_route_reaches_its_own_success_path(self):
        """The per-row Approve form goes through a different route, and that one
        imported `quote` inside its error branches only -- so building the
        notice on the way out was an UnboundLocalError that only fired when
        something was actually overruled."""
        response = self._approve()
        location = response.headers["location"]
        self.assertIn("saved=1", location)
        self.assertIn("notice=", location)

    def test_approving_also_keeps_the_mapping(self):
        self._approve()
        self.assertEqual(self.role()["date_key_type"], "surrogate_fk")
        self.assertEqual(self.role()["dimension_table"], DIM)

    def test_approving_says_nothing_when_nothing_was_overruled(self):
        response = self._approve(date_key_type="surrogate_fk",
                                 dimension_table=DIM,
                                 dimension_key="DT_DMS_KEY",
                                 date_value_column="DMS_DT")
        self.assertNotIn("notice=", response.headers["location"])


class _FakeUrl:
    path = "/admin/clients/acct-dr/date-roles"

    def __str__(self):
        return self.path


class _FakeRequest:
    url = _FakeUrl()
    query_params: dict = {}
    session = {"admin_id": "admin_user_1"}
    scope = {"type": "http"}
    cookies: dict = {}
    headers: dict = {}


def render(**overrides) -> str:
    """The page as Jinja actually emits it."""
    context = {
        "client": {"account_id": "acct-dr", "client_name": "EMCO",
                   "state": "READY", "state_data": "{}"},
        "date_roles": [], "has_model": True,
        "semantic_tables": [
            {"qualified_name": FACT, "table": "CUS_ORD_IVC_FCT",
             "type": "fact", "fields": []},
            # Deliberately NOT typed as a dimension: that is the case the form
            # could not offer, and DT_DMS is exactly the table it happens to.
            {"qualified_name": DIM, "table": "DT_DMS", "type": "", "fields": []},
        ],
        "metrics": [], "context_bindings": [], "date_role_coverage": {},
        "date_key_bindings": {"bindings": {}, "date_dimensions": [DIM]},
        "saved": None, "error": None, "notice": None,
        "request": _FakeRequest(),
    }
    context.update(overrides)
    return routes.templates.get_template("client_date_roles.html").render(**context)


def dimension_options(html: str) -> str:
    """Just the Date dimension select. DT_DMS also appears in the Source table
    select, correctly, so asserting on the whole page proves nothing."""
    start = html.index('id="drDimensionTable"')
    return html[start:html.index("</select>", start)]


class TestThePageOffersTheDimensionAndExplainsItself:

    def test_an_untyped_date_dimension_can_still_be_selected(self):
        """The select listed only tables the model typed as dimensions. A mart
        whose DT_DMS came back untyped could not be chosen at all -- which is
        the one selection this form exists to make."""
        assert f'<option value="{DIM}">' in dimension_options(render())

    def test_it_is_absent_when_the_server_does_not_call_it_a_date_dimension(self):
        html = render(date_key_bindings={"bindings": {}, "date_dimensions": []})
        assert f'<option value="{DIM}">' not in dimension_options(html)

    def test_a_table_the_model_typed_as_a_dimension_is_still_listed(self):
        """The original filter still applies -- this widened it, not replaced it."""
        html = render(
            semantic_tables=[
                {"qualified_name": FACT, "table": "CUS_ORD_IVC_FCT",
                 "type": "fact", "fields": []},
                {"qualified_name": "EMDW_DMART.CUS_DMS", "table": "CUS_DMS",
                 "type": "dimension", "fields": []},
            ],
            date_key_bindings={"bindings": {}, "date_dimensions": []},
        )
        assert '<option value="EMDW_DMART.CUS_DMS">' in dimension_options(html)

    def test_an_override_notice_is_shown_to_the_admin(self):
        html = render(notice="its date-dimension mapping was kept")
        assert "its date-dimension mapping was kept" in html

    def test_and_no_empty_banner_when_there_is_nothing_to_say(self):
        assert "alert-amber" not in render()
