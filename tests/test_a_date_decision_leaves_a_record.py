"""
tests/test_a_date_decision_leaves_a_record.py

The three date outcomes a dispute is actually about left nothing behind.

The answer trace recorded a `date_context_resolution` step only where a date was
successfully bound, and that call sat inside the selected branch, nested under
`if _date_plan.get("enabled")`. So:

  * status "none" -- no governed date, the question answered by free-form SQL
    over whatever column the model picked -- recorded nothing, and was
    indistinguishable in the trace from a question that was never about a date;
  * status "ambiguous" -- the reader was asked which date to use -- recorded
    nothing, so which dates were offered, and why none could be chosen, was not
    retrievable at all;
  * and the whole block is wrapped in a fail-open handler, so a date that
    failed to resolve for a reason nobody has seen looked exactly like a
    question with no dates in it.

That is the wrong way round. A resolved date is the case needing least
explanation; "why did this answer use that date, or no date" is the question an
operator gets, and answering it meant re-deriving the resolution by hand.

date_resolution_trace summarises any resolution, and the pipeline emits it on
every path -- the normal one, the unsupported-grain refusal that returns early,
and the exception handler.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from core.contextual_dates import date_resolution_trace

FACT = "DBO.F_SALES"


class TestEveryStatusProducesSomething:
    """A summary that is empty for the interesting cases would satisfy a
    "the step is emitted" test while recording nothing."""

    @pytest.mark.parametrize("status", [
        "none", "ambiguous", "selected", "selected_many", "unsupported_grain",
        "error",
    ])
    def test_the_status_and_a_reason_are_always_there(self, status):
        summary = date_resolution_trace({"status": status, "reason": "because"})
        assert summary["status"] == status
        assert summary["reason"] == "because"

    def test_an_empty_resolution_still_records_a_status(self):
        assert date_resolution_trace({})["status"] == "none"
        assert date_resolution_trace(None)["status"] == "none"


class TestTheInputsThatDroveItAreRecorded:
    """A "none" outcome is almost always one of these being empty, and which
    one says whose afternoon it is: an empty fact scope is a retrieval problem,
    no date roles at all is a modelling problem."""

    def test_the_fact_scope_and_metrics_are_carried(self):
        summary = date_resolution_trace(
            {"status": "none", "reason": "no governed date context"},
            fact_scope={FACT, "DBO.F_RETURNS"},
            metric_names=["Net Revenue", None, ""],
            date_binding_count=0,
            date_role_count=3,
        )
        assert summary["fact_scope"] == ["DBO.F_RETURNS", FACT]
        assert summary["matched_metrics"] == ["Net Revenue"]
        assert summary["date_roles_considered"] == 3
        assert summary["date_bindings_considered"] == 0

    def test_an_empty_fact_scope_is_visible_as_empty(self):
        """Rather than absent, which reads as "not recorded"."""
        summary = date_resolution_trace({"status": "none"}, fact_scope=set())
        assert summary["fact_scope"] == []
        assert "date_roles_considered" in summary


class TestAResolvedDateSaysWhichOne:

    BINDING = {
        "metric_name": "Net Revenue", "context_name": "Invoice Date",
        "date_role": "invoice_date", "fact_table": FACT,
        "fact_column": "INVOICE_DT", "resolution_source": "user_confirmed_date_role",
    }

    def test_the_binding_is_summarised(self):
        summary = date_resolution_trace(
            {"status": "selected", "binding": dict(self.BINDING)})
        assert summary["context"] == "Invoice Date"
        assert summary["fact_column"] == "INVOICE_DT"
        assert summary["fact_table"] == FACT
        assert summary["source"] == "user_confirmed_date_role"

    def test_selected_many_summarises_the_first_of_them(self):
        summary = date_resolution_trace(
            {"status": "selected_many", "bindings": [dict(self.BINDING)]})
        assert summary["fact_column"] == "INVOICE_DT"

    def test_the_provenance_token_survives_into_the_trace(self):
        """The operator's whole question is often "was this the reader's choice
        or a machine default", and resolution_source is the only field that
        answers it."""
        for source in ("user_confirmed_date_role", "fact_default_date_role",
                       "discovered_date_role", "inferred_encoded_fact_date"):
            summary = date_resolution_trace({
                "status": "selected",
                "binding": dict(self.BINDING, resolution_source=source)})
            assert summary["source"] == source


class TestAClarificationRecordsWhatItOffered:
    """Without this a clarification is a dead end in the trace: an operator can
    see that one happened and not what it said."""

    OPTIONS = [
        {"context_name": "Invoice Date", "fact_table": FACT,
         "fact_column": "IVC_DT", "resolution_source": "discovered_date_role"},
        {"context_name": "Delivery Date", "fact_table": FACT,
         "fact_column": "DLV_DT", "resolution_source": "discovered_date_role"},
    ]

    def test_the_offered_dates_are_listed(self):
        summary = date_resolution_trace(
            {"status": "ambiguous", "reason": "no default",
             "options": [dict(o) for o in self.OPTIONS]})
        assert summary["offered_count"] == 2
        assert [o["context"] for o in summary["offered"]] == [
            "Invoice Date", "Delivery Date"]
        assert [o["fact_column"] for o in summary["offered"]] == ["IVC_DT", "DLV_DT"]

    def test_a_long_list_is_capped_but_the_count_is_not(self):
        """A trace row is not the place for fifty dates, and "we offered 4 of
        23" is itself the finding on a fact nobody has curated."""
        many = [dict(self.OPTIONS[0], context_name=f"Date {i}") for i in range(23)]
        summary = date_resolution_trace({"status": "ambiguous", "options": many})
        assert len(summary["offered"]) == 8
        assert summary["offered_count"] == 23

    def test_junk_options_do_not_raise(self):
        summary = date_resolution_trace(
            {"status": "ambiguous", "options": [None, "x", {"context_name": "A"}]})
        assert summary["offered_count"] == 1

    def test_an_unsupported_grain_records_both_grains(self):
        summary = date_resolution_trace({
            "status": "unsupported_grain", "reason": "too coarse",
            "requested_grain": "day", "available_grain": "month"})
        assert summary["requested_grain"] == "day"
        assert summary["available_grain"] == "month"


class TestThePipelineEmitsItOnEveryPath:
    """_handle_query_impl needs a warehouse, a websocket and a model, so the
    wiring is read from the tree -- and read for REACHABILITY, which is the
    whole point: the defect was a step that existed and was nested three
    branches deep inside the one outcome that least needed it."""

    @staticmethod
    def _sites():
        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_trace_step"
            and any(isinstance(arg, ast.Constant)
                    and arg.value == "date_context_resolution"
                    for arg in node.args)
        ]

    def test_there_is_a_site_for_the_normal_path_the_early_return_and_the_error(self):
        assert len(self._sites()) == 3, (
            "expected the date resolution to be traced on the normal path, on "
            "the unsupported-grain return, and in the fail-open handler; found "
            f"{len(self._sites())} site(s)")

    def test_every_site_summarises_through_the_helper(self):
        """A hand-built dict at one site is how the "none" case lost its
        inputs in the first place."""
        for site in self._sites():
            calls = {
                inner.func.id for inner in ast.walk(site)
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            }
            assert "date_resolution_trace" in calls, ast.dump(site)[:200]

    # The statuses resolve_contextual_date_binding can return. A guard that
    # mentions one of these narrows its site to that single outcome; so does
    # the date plan's `enabled`.
    NARROWING = {
        "selected", "selected_many", "ambiguous", "none", "unsupported_grain",
        "error", "enabled",
    }

    @classmethod
    def _unconditional_sites(cls):
        """Sites reached whatever the resolution said.

        Walks down from the function carrying (a) the narrowing literals of
        every enclosing `if`, and (b) whether we are inside an except handler --
        a site in there only fires on an exception, so it cannot stand in for
        the normal path.
        """
        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        found = []

        def walk(node, guards, in_handler):
            for child in ast.iter_child_nodes(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id == "_trace_step"
                        and any(isinstance(arg, ast.Constant)
                                and arg.value == "date_context_resolution"
                                for arg in child.args)
                        and not guards and not in_handler):
                    found.append(child.lineno)
                if isinstance(child, ast.If):
                    literals = {
                        inner.value for inner in ast.walk(child.test)
                        if isinstance(inner, ast.Constant)
                        and isinstance(inner.value, str)
                    } & cls.NARROWING
                    attrs = {
                        inner.attr for inner in ast.walk(child.test)
                        if isinstance(inner, ast.Attribute)
                    } & cls.NARROWING
                    for statement in child.body:
                        walk(statement, guards | literals | attrs, in_handler)
                    for statement in child.orelse:
                        walk(statement, guards, in_handler)
                    continue
                if isinstance(child, ast.ExceptHandler):
                    for statement in child.body:
                        walk(statement, guards, True)
                    continue
                walk(child, guards, in_handler)

        walk(tree, frozenset(), False)
        return found

    def test_one_site_is_reached_whatever_the_resolution_said(self):
        """The defect, stated as the property that excludes it.

        A step guarded by `status == "selected"`, or nested under
        `_date_plan.get("enabled")`, cannot record the outcomes an operator
        asks about. Neither can one that only fires on an exception. So at
        least one site has to sit on the path every resolution takes.
        """
        sites = self._unconditional_sites()
        assert sites, (
            "every date trace site is either gated on a particular resolution "
            "status, nested under the date plan being enabled, or inside an "
            "except handler -- so a 'none' or 'ambiguous' outcome records "
            "nothing an operator can retrieve, which is the defect this file "
            "exists to prevent")

    def test_the_narrow_sites_are_still_there_too(self):
        """The unsupported-grain branch returns before the unconditional site,
        and the handler runs when the block raised. Both need their own."""
        assert len(self._sites()) - len(self._unconditional_sites()) == 2
