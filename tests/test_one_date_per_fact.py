"""
tests/test_one_date_per_fact.py

Six months of revenue was compared against returns over all time.

A question can read several facts -- "revenue and returns last 6 months" -- and
each of them has its own business date. resolve_contextual_date_binding returned
ONE binding either way, and both outcomes were wrong. Executed against the real
resolver at HEAD~1:

    two facts, each with a settled default
        -> status "ambiguous", offering [Invoice Date] [Return Date]

    Those are not alternatives. Each is the right date for its own fact, the
    chips do not say which fact they belong to, so the question cannot be
    answered as asked -- and whichever the reader picks, the compiled plan
    carries ONE temporal policy and the other fact is left unfiltered.

    two facts, only one settled
        -> status "selected", quietly, no question asked at all, the undated
           fact unfiltered. The same wrong number with nothing to notice it by,
           and the commoner shape on a partly curated model.

Everything downstream was already multi-capable: selected_many,
build_contextual_date_plan_many, the pipeline's own branch for it, and the
validator's approved set, which unions every policy's fact_column precisely so a
second governed role is not read as a substitution. The resolver was the only
single-valued link in the chain, which is why this is a small change to one
function rather than a new mechanism.

Three outcomes now, and the distinction between them is the point:

  * every aggregating fact has a settled date  -> selected_many, one policy each
  * one fact carries two defaults              -> ambiguous, about THAT fact
  * an aggregating fact has no settled date    -> undated_fact_in_scope, refused

The refusal is scoped to the measures' own base tables, not to the whole fact
scope. fact_scope carries dimensions and whatever the graph anchored on, and a
date policy on those is meaningless -- using it fired on ordinary single-fact
questions whose graph had pulled in a neighbour.
"""

from __future__ import annotations

import pytest

from core.contextual_dates import (
    build_contextual_date_plan_many, resolve_contextual_date_binding,
    same_date_fact,
)
from core.query_pipeline import (
    business_date_window_kind, measures_without_a_business_date,
    resolve_common_business_anchor,
)

DIM = "EMDW_DMART.DT_DMS"
INVOICE_FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
RETURN_FACT = "EMDW_DMART.CUS_RTN_FCT"
STOCK_FACT = "EMDW_DMART.INV_BAL_FCT"
QUESTION = "net revenue and returns last 6 months"
WINDOW = {"kind": "last_n", "amount": 6, "unit": "month",
          "anchor_policy": "latest_available"}


def role(fact, column, key, label, *, default=1, status="approved"):
    return {
        "fact_table": fact, "fact_column": column, "role_key": key,
        "label": label, "name": label, "business_role": key,
        "status": status, "is_default": default,
        "date_key_type": "surrogate_fk", "dimension_table": DIM,
        "dimension_key": "DT_DMS_KEY", "date_value_column": "DMS_DT",
        "_matched_phrase": label.lower(),
    }


INVOICE_DATE = role(INVOICE_FACT, "CUS_IVC_DT_DMS_KEY", "invoice_date", "Invoice Date")
ORDER_DATE = role(INVOICE_FACT, "CUS_ORD_DT_DMS_KEY", "order_date", "Order Date")
RETURN_DATE = role(RETURN_FACT, "RTN_DT_DMS_KEY", "return_date", "Return Date")
STOCK_DATE = role(STOCK_FACT, "BAL_DT_DMS_KEY", "balance_date", "Balance Date")

REVENUE = {"id": 1, "name": "Net Revenue", "base_table": INVOICE_FACT}
RETURNS = {"id": 2, "name": "Returns", "base_table": RETURN_FACT}
STOCK = {"id": 3, "name": "Stock Value", "base_table": STOCK_FACT}


def resolve(metrics, roles, scope, question=QUESTION, **kw):
    return resolve_contextual_date_binding(
        question,
        matched_metrics=[dict(m) for m in metrics],
        bindings=[],
        date_roles=[dict(r) for r in roles],
        required_fact_tables=set(scope),
        **kw,
    )


class TestEachFactGetsItsOwnDate:

    def test_two_facts_each_resolve(self):
        out = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                      {INVOICE_FACT, RETURN_FACT})
        assert out["status"] == "selected_many", out
        bound = {(b["fact_table"], b["fact_column"]) for b in out["bindings"]}
        assert bound == {(INVOICE_FACT, "CUS_IVC_DT_DMS_KEY"),
                         (RETURN_FACT, "RTN_DT_DMS_KEY")}

    def test_three_facts_too(self):
        out = resolve([REVENUE, RETURNS, STOCK],
                      [INVOICE_DATE, RETURN_DATE, STOCK_DATE],
                      {INVOICE_FACT, RETURN_FACT, STOCK_FACT})
        assert out["status"] == "selected_many"
        assert len(out["bindings"]) == 3

    def test_the_bindings_are_ordered_stably(self):
        """Two calls must not produce two different plans, or the same question
        caches under two keys and the trace reads differently each time."""
        first = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                        {INVOICE_FACT, RETURN_FACT})
        second = resolve([RETURNS, REVENUE], [RETURN_DATE, INVOICE_DATE],
                         {RETURN_FACT, INVOICE_FACT})
        assert [b["fact_table"] for b in first["bindings"]] == \
               [b["fact_table"] for b in second["bindings"]]

    def test_every_binding_carries_its_provenance(self):
        out = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                      {INVOICE_FACT, RETURN_FACT})
        for binding in out["bindings"]:
            assert binding["resolution_source"] == "fact_default_date_role"


class TestThePlanFiltersEveryFact:
    """The defect was never in the plan builder -- it already handled many
    bindings. It was that the resolver never gave it any."""

    def test_one_temporal_policy_per_fact(self):
        out = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                      {INVOICE_FACT, RETURN_FACT})
        plan = build_contextual_date_plan_many(
            out["bindings"], QUESTION, temporal_window=dict(WINDOW))
        filtered = {(p["fact_table"], p["fact_column"])
                    for p in plan["temporal_policies"]}
        assert filtered == {(INVOICE_FACT, "CUS_IVC_DT_DMS_KEY"),
                            (RETURN_FACT, "RTN_DT_DMS_KEY")}, (
            "a fact with no temporal policy is unfiltered, so its figures cover "
            "all time beside figures covering the requested window")

    def test_each_fact_joins_the_dimension_under_its_own_alias(self):
        """The same date dimension, once per role. Two joins sharing an alias
        would be invalid SQL; one join would silently date only one fact."""
        out = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                      {INVOICE_FACT, RETURN_FACT})
        plan = build_contextual_date_plan_many(
            out["bindings"], QUESTION, temporal_window=dict(WINDOW))
        joins = plan["joins"]
        assert len(joins) == 2
        assert len({j["role_alias"] for j in joins}) == 2
        assert {j["from"] for j in joins} == {INVOICE_FACT, RETURN_FACT}
        assert {j["to"] for j in joins} == {DIM}
        for join in joins:
            assert join["role_playing"] is True

    def test_both_facts_and_the_dimension_are_required(self):
        out = resolve([REVENUE, RETURNS], [INVOICE_DATE, RETURN_DATE],
                      {INVOICE_FACT, RETURN_FACT})
        plan = build_contextual_date_plan_many(
            out["bindings"], QUESTION, temporal_window=dict(WINDOW))
        assert set(plan["required_tables"]) == {INVOICE_FACT, RETURN_FACT, DIM}


class TestAFactWithNoDateIsRefusedNotAnswered:
    """The quieter half, and the commoner one."""

    def test_it_refuses_rather_than_leaving_the_fact_unfiltered(self):
        out = resolve([REVENUE, RETURNS],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        assert out["status"] == "undated_fact_in_scope", out

    def test_it_names_the_fact_that_is_missing_one(self):
        out = resolve([REVENUE, RETURNS],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        assert out["undated_facts"] == [RETURN_FACT]
        assert out["dated_facts"] == [INVOICE_FACT]

    def test_a_generated_role_does_not_count_as_settled(self):
        """The resolver requires approved AND default everywhere else; a guess
        must not quietly satisfy this check either."""
        out = resolve([REVENUE, RETURNS],
                      [INVOICE_DATE, dict(RETURN_DATE, status="generated")],
                      {INVOICE_FACT, RETURN_FACT})
        assert out["status"] == "undated_fact_in_scope"

    def test_the_caller_can_map_the_fact_back_to_its_measure(self):
        """The reader is told which MEASURE, never the warehouse table.

        Nothing is handed between the two halves: `undated_facts` comes out of
        the resolver and goes straight into the pipeline's own helper. If the two
        sides ever disagree about what names a fact, this is where it shows.
        """
        out = resolve([REVENUE, RETURNS],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        assert measures_without_a_business_date(
            [REVENUE, RETURNS], out["undated_facts"]) == ["Returns"]

    def test_it_maps_back_through_the_spelling_an_admin_typed(self):
        """base_table is whatever was typed into the registry; undated_facts is
        canonicalized (brackets stripped, upper-cased, trimmed to SCHEMA.TABLE).
        An equality comparison between the two names NO measure, and the refusal
        degrades to "these measures" while the resolver knew exactly which one.
        """
        returns = {"id": 2, "name": "Returns",
                   "base_table": "[emdw_dmart].[cus_rtn_fct]"}
        out = resolve([REVENUE, returns],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        assert out["status"] == "undated_fact_in_scope", out
        assert returns["base_table"] not in out["undated_facts"], (
            "the fixture no longer tests a spelling difference")
        assert measures_without_a_business_date(
            [REVENUE, returns], out["undated_facts"]) == ["Returns"]

    def test_a_dated_measure_is_never_named_in_the_refusal(self):
        out = resolve([REVENUE, RETURNS],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        named = measures_without_a_business_date([REVENUE, RETURNS],
                                                 out["undated_facts"])
        assert "Net Revenue" not in named

    def test_two_measures_on_the_same_undated_fact_are_both_named(self):
        gross = {"id": 4, "name": "Gross Returns", "base_table": RETURN_FACT}
        out = resolve([REVENUE, RETURNS, gross],
                      [INVOICE_DATE, dict(RETURN_DATE, is_default=0)],
                      {INVOICE_FACT, RETURN_FACT})
        assert measures_without_a_business_date(
            [REVENUE, RETURNS, gross], out["undated_facts"]) == [
                "Gross Returns", "Returns"]

    def test_nothing_undated_names_nothing(self):
        assert measures_without_a_business_date([REVENUE, RETURNS], []) == []
        assert measures_without_a_business_date([REVENUE, RETURNS], None) == []


class TestItStillAsksWhenThereIsAChoiceToMake:

    def test_one_fact_with_two_defaults_is_a_real_question(self):
        out = resolve([REVENUE], [INVOICE_DATE, ORDER_DATE], {INVOICE_FACT},
                      question="net revenue last 6 months")
        assert out["status"] == "ambiguous"
        assert {o["context_name"] for o in out["options"]} == \
            {"Invoice Date", "Order Date"}

    def test_the_reason_names_the_fact_it_is_asking_about(self):
        """Two facts each offering their own date read as alternatives and
        cannot be answered. A question about ONE fact's two dates can."""
        out = resolve([REVENUE], [INVOICE_DATE, ORDER_DATE], {INVOICE_FACT},
                      question="net revenue last 6 months")
        assert INVOICE_FACT in out["reason"]

    def test_the_readers_own_choice_still_wins(self):
        out = resolve([REVENUE], [INVOICE_DATE, ORDER_DATE], {INVOICE_FACT},
                      question="net revenue last 6 months",
                      confirmed_date_role=dict(ORDER_DATE, context_name="Order Date"))
        assert out["status"] == "selected"
        assert out["binding"]["fact_column"] == "CUS_ORD_DT_DMS_KEY"


class TestTheSingleFactPathIsUntouched:
    """Most questions. A regression here is a regression for everybody."""

    def test_one_fact_one_default_still_selects(self):
        out = resolve([REVENUE], [INVOICE_DATE], {INVOICE_FACT},
                      question="net revenue last 6 months")
        assert out["status"] == "selected"
        assert out["binding"]["resolution_source"] == "fact_default_date_role"

    def test_a_date_named_in_the_question_still_wins(self):
        out = resolve([REVENUE], [INVOICE_DATE, ORDER_DATE], {INVOICE_FACT},
                      question="net revenue by order date last 6 months")
        assert out["status"] == "selected"
        assert out["binding"]["fact_column"] == "CUS_ORD_DT_DMS_KEY"

    def test_no_matched_metric_makes_no_claim_about_missing_facts(self):
        """fact_scope is not "facts that aggregate": it carries dimensions and
        graph anchors. With no metric there is no knowable set, so an ad-hoc
        question binds what it found and refuses nothing."""
        out = resolve([], [INVOICE_DATE], {INVOICE_FACT, RETURN_FACT},
                      question="revenue trend last 7 days")
        assert out["status"] == "selected"

    def test_a_scope_wider_than_the_measures_does_not_refuse(self):
        """The live regression this caused: a single-fact question whose graph
        anchored on a neighbour was refused for want of a date on the
        neighbour."""
        out = resolve([REVENUE], [INVOICE_DATE],
                      {INVOICE_FACT, "EMDW_DMART.CUS_DMS"},
                      question="net revenue last 6 months")
        assert out["status"] == "selected"


class TestTheRefusalReachesTheReader:
    """_handle_query_impl needs a warehouse, a websocket and a model, so the
    branch is read from the tree. It is read for REACHABILITY and for what it
    NAMES -- a status the pipeline does not handle falls through every branch
    and the question proceeds ungoverned, which is worse than the defect."""

    @staticmethod
    def _branch():
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            literals = {
                inner.value for inner in ast.walk(node.test)
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            }
            if "undated_fact_in_scope" in literals:
                return node
        return None

    def test_the_pipeline_handles_the_new_status(self):
        assert self._branch() is not None, (
            "resolve_contextual_date_binding can return undated_fact_in_scope "
            "and the pipeline does not branch on it, so it falls through every "
            "other check and the question is answered with no governed date at "
            "all -- worse than the defect this status exists to report")

    def test_it_returns_rather_than_carrying_on(self):
        """Fail closed. Sending a message and then answering anyway is the
        shape that produced the original wrong number."""
        import ast

        branch = self._branch()
        assert any(isinstance(node, ast.Return) for node in ast.walk(branch))

    def test_it_tells_the_reader_and_records_it_for_an_operator(self):
        import ast

        branch = self._branch()
        calls = {
            node.func.attr for node in ast.walk(branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        } | {
            node.func.id for node in ast.walk(branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "send_message" in calls, "the reader is told nothing"
        assert "_trace_step" in calls, "an operator can retrieve nothing"
        assert "_trace_finish" in calls, "the turn is never closed"

    def test_it_names_the_measure_through_the_catalogue(self):
        """Not the warehouse table, and not an English literal. Both were
        defects fixed earlier on this branch; a new refusal must not reintroduce
        either."""
        import ast

        branch = self._branch()
        message_ids = {
            arg.value
            for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", "") in {"t", "_t"}
                 or getattr(node.func, "attr", "") in {"t", "_t"})
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
        assert "clar.date.measure_has_no_business_date" in message_ids, message_ids

    def test_the_message_says_which_measure_and_why(self):
        from core import i18n

        for lang in ("en", "fr"):
            rendered = i18n.t(
                "clar.date.measure_has_no_business_date", lang=lang,
                measures="Returns",
                window=i18n.t("clar.date.the_requested_period", lang=lang))
            assert "Returns" in rendered
            assert "{" not in rendered
        assert i18n.t("clar.date.measure_has_no_business_date", lang="en",
                      measures="R", window="w") != \
            i18n.t("clar.date.measure_has_no_business_date", lang="fr",
                   measures="R", window="w")

    def test_no_warehouse_identifier_is_in_the_reader_facing_text(self):
        import re

        from core import i18n

        for message_id in ("clar.date.measure_has_no_business_date",
                           "clar.date.the_requested_period",
                           "clar.date.these_measures"):
            for lang in ("en", "fr"):
                rendered = i18n.t(message_id, lang=lang, measures="Returns",
                                  window="the period")
                assert not re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", rendered), \
                    (message_id, lang, rendered)


class TestOneAnchorPerFactReducedToTheEarliest:
    """Each fact anchoring on its OWN newest date makes revenue cover Mar-Aug
    and returns Feb-Jul, and the two beside each other are not a comparison."""

    @staticmethod
    def _resolve(values, *, fail=()):
        def resolve_one(policy):
            fact = policy["fact_table"]
            if fact in fail:
                return {}
            return {"value": values[fact], "fact_table": fact, "cached": True}

        return resolve_common_business_anchor(
            [{"fact_table": fact} for fact in values], resolve_one=resolve_one)

    def test_the_earliest_maximum_wins(self):
        out = self._resolve({INVOICE_FACT: "2026-09-10",
                             RETURN_FACT: "2026-09-08"})
        assert out["value"] == "2026-09-08", (
            "the window reaches past the end of one fact's data")
        assert out["limited_by_fact"] == RETURN_FACT

    def test_whichever_order_they_arrive_in(self):
        first = self._resolve({INVOICE_FACT: "2026-09-10", RETURN_FACT: "2026-09-08"})
        second = self._resolve({RETURN_FACT: "2026-09-08", INVOICE_FACT: "2026-09-10"})
        assert first["value"] == second["value"] == "2026-09-08"

    def test_it_records_every_facts_own_date(self):
        """So an operator can see WHY the window stopped where it did, and an
        admin knows which load is behind."""
        out = self._resolve({INVOICE_FACT: "2026-09-10", RETURN_FACT: "2026-09-08"})
        assert {item["fact_table"]: item["value"] for item in out["per_fact"]} == {
            INVOICE_FACT: "2026-09-10", RETURN_FACT: "2026-09-08"}
        assert out["common_across_facts"] is True

    def test_a_single_fact_is_unchanged(self):
        """Most questions. One anchor in, the same anchor out, and none of the
        multi-fact bookkeeping attached to it."""
        out = self._resolve({INVOICE_FACT: "2026-09-10"})
        assert out["value"] == "2026-09-10"
        assert "common_across_facts" not in out
        assert "per_fact" not in out

    def test_one_unresolvable_fact_abandons_the_common_anchor(self):
        """Fails open rather than pinning the answer to a window derived only
        from the facts that happened to answer. The compiled SQL then keeps its
        in-query anchor, which is correct if slower."""
        out = self._resolve({INVOICE_FACT: "2026-09-10", RETURN_FACT: "2026-09-08"},
                            fail={RETURN_FACT})
        assert out == {}

    def test_no_policies_resolves_nothing(self):
        assert resolve_common_business_anchor([], resolve_one=lambda p: {}) == {}
        assert resolve_common_business_anchor(None, resolve_one=lambda p: {}) == {}

    def test_it_is_cached_only_when_every_probe_was(self):
        """The trace reports `cached` to explain latency. Claiming a cache hit
        when one fact was probed makes a slow answer look inexplicable."""
        def resolve_one(policy):
            return {"value": "2026-09-08", "fact_table": policy["fact_table"],
                    "cached": policy["fact_table"] == INVOICE_FACT}

        out = resolve_common_business_anchor(
            [{"fact_table": INVOICE_FACT}, {"fact_table": RETURN_FACT}],
            resolve_one=resolve_one)
        assert out["cached"] is False

    def test_the_pipeline_no_longer_requires_exactly_one_policy(self):
        """The gate that made all of this unreachable: a two-fact question
        resolved no anchor at all."""
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "resolve_common_business_anchor"
        ]
        assert calls, "the pipeline does not reduce an anchor across facts"
        source = inspect.getsource(qp._handle_query_impl)
        assert "len(_anchor_policies) == 1" not in source, (
            "the anchor is still gated on there being exactly one policy, so a "
            "question resolving one date per fact gets no anchor at all")


class TestTheFreshnessNoteReadsEveryFactsWindow:
    """On a warehouse loaded to the 8th, "this month's revenue and returns"
    must still say so. The kind was read off policies[0], so whether the reader
    was told came down to which fact's table name sorted first."""

    def test_a_data_relative_kind_anywhere_in_the_list_earns_the_note(self):
        from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS

        kind = business_date_window_kind([
            {"fact_table": INVOICE_FACT, "kind": "last_n"},
            {"fact_table": RETURN_FACT, "kind": "this_month"},
        ])
        assert kind == "this_month"
        assert kind in BUSINESS_DATE_WINDOW_KINDS, (
            "the freshness banner is gated on membership, so a kind the gate "
            "rejects is the same as telling the reader nothing")

    def test_it_does_not_depend_on_the_order_the_facts_arrive_in(self):
        forward = business_date_window_kind([
            {"fact_table": INVOICE_FACT, "kind": "this_month"},
            {"fact_table": RETURN_FACT, "kind": "last_n"},
        ])
        backward = business_date_window_kind([
            {"fact_table": RETURN_FACT, "kind": "last_n"},
            {"fact_table": INVOICE_FACT, "kind": "this_month"},
        ])
        assert forward == backward == "this_month"

    def test_a_single_policy_is_unchanged(self):
        assert business_date_window_kind([{"kind": "today"}]) == "today"
        assert business_date_window_kind([{"kind": "last_n"}]) == "last_n"

    def test_no_data_relative_kind_earns_no_note(self):
        from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS

        kind = business_date_window_kind([
            {"kind": "last_n"}, {"kind": "previous_month"},
        ])
        assert kind not in BUSINESS_DATE_WINDOW_KINDS

    def test_nothing_resolved_is_not_a_window(self):
        from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS

        for empty in ([], None, [{}], ["not a policy"]):
            assert business_date_window_kind(empty) not in BUSINESS_DATE_WINDOW_KINDS

    def test_the_pipeline_asks_this_function_rather_than_indexing(self):
        """The banner block needs a websocket to execute, so the wiring is read
        from the tree: the disclosure must not go back to reading one policy."""
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        assert any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "business_date_window_kind"
            for node in ast.walk(tree)
        ), "the freshness disclosure no longer reduces the kind across facts"
