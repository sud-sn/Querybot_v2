"""
tests/test_model_health_sees_the_undecided_date.py

Two date states that interrupt every reader, reported to nobody.

1. The metric's "Default time column" does nothing unless a separately APPROVED
   Date Role exists on that same column. core.contextual_dates
   ._metric_default_time_role_bindings requires it, and that strictness is
   right -- a column name matching an unreviewed discovery candidate must not
   become governed merely because somebody picked it on a metric form.

   But nothing told the admin. They set "Default time column = INVOICE_DT",
   saved, got a success message, and the setting had no effect at all --
   indistinguishable, from the outside, from one that worked.

2. A fact carrying several business dates with none marked as the default is the
   state resolve_contextual_date_binding calls "ambiguous": it cannot choose,
   so it asks the reader. Every reader, every time, because the answer is
   remembered for the thread and not for the model.

   core.semantic_model._date_role_coverage returns coverage_pct 100.0 for a fact
   with four approved roles and no default -- which is what the Date Roles page
   prints -- and core.kb_quality carries an item only for ABSENCE. Presence
   without a decision is a different afternoon with a different remedy: one
   click, on a page that currently says the work is done.

Both are now findings from detect_ambiguous_date_roles, which is where the other
date-governance conflicts live.

The fixtures carry canonical_id on metrics and tables because the contract
compiler always stamps it, and _metric_canonical_ids skips anything without one.
My first version of these fixtures omitted it, and check 3 silently never ran --
the wrong-layer fixture, which is the failure mode that produces a test proving
nothing about code that works.
"""

from __future__ import annotations

import pytest

from core.semantic_conflicts import detect_ambiguous_date_roles

FACT = "EMDW.CUS_ORD_IVC_FCT"


def _role(column, label, *, status="approved", is_default=0):
    return {
        "fact_table": FACT, "fact_column": column, "name": label,
        "business_role": label.lower().replace(" ", "_"),
        "status": status, "is_default": is_default,
        "canonical_id": f"dr:{column.lower()}",
    }


def _contract(roles, *, default_time_column="", date_contexts=None,
              metric_table=FACT):
    return {
        "model": {"tables": [{
            "qualified_name": FACT, "schema_name": "EMDW",
            "canonical_id": "tbl:ivc_fct", "date_roles": list(roles),
        }]},
        "metrics": [{
            "id": 7, "name": "Net Revenue", "canonical_id": "met:7",
            "base_table": metric_table,
            "default_time_column": default_time_column,
        }],
        "date_contexts": list(date_contexts or []),
    }


def _codes(contract):
    return {finding["code"] for finding in detect_ambiguous_date_roles(contract)}


def _finding(contract, code):
    return next(f for f in detect_ambiguous_date_roles(contract)
                if f["code"] == code)


SETTLED = [_role("IVC_DT", "Invoice Date", is_default=1),
           _role("DLV_DT", "Delivery Date")]
UNDECIDED = [_role("IVC_DT", "Invoice Date"),
             _role("DLV_DT", "Delivery Date")]
TWO_DEFAULTS = [_role("IVC_DT", "Invoice Date", is_default=1),
                _role("DLV_DT", "Delivery Date", is_default=1)]


class TestAFactNobodyHasSettled:

    CODE = "fact_date_roles_without_default"

    def test_several_dates_and_no_default_is_reported(self):
        assert self.CODE in _codes(_contract(UNDECIDED))

    def test_it_names_the_table_and_the_candidates(self):
        finding = _finding(_contract(UNDECIDED), self.CODE)
        assert finding["table_name"] == FACT
        assert "Invoice Date" in finding["message"]
        assert "Delivery Date" in finding["message"]
        assert finding["object_type"] == "table"

    def test_the_remedy_is_the_one_click_that_ends_it(self):
        finding = _finding(_contract(UNDECIDED), self.CODE)
        assert finding["suggestions"]
        assert "default" in finding["suggestions"][0].lower()

    def test_one_settled_default_is_not_a_finding(self):
        assert self.CODE not in _codes(_contract(SETTLED))

    def test_two_defaults_is_a_finding_with_its_own_wording(self):
        """The resolver cannot choose between two defaults either, and the
        remedy is the opposite one: remove, not add."""
        finding = _finding(_contract(TWO_DEFAULTS), self.CODE)
        assert "2 of them are marked default" in finding["message"]
        assert "exactly one" in finding["suggestions"][0].lower()

    def test_a_single_date_is_never_undecided(self):
        """There is nothing to choose between. Whether to APPROVE it is a
        different question and a different item."""
        for status in ("approved", "generated"):
            roles = [_role("IVC_DT", "Invoice Date", status=status)]
            assert self.CODE not in _codes(_contract(roles))

    def test_a_generated_role_counts_as_a_candidate(self):
        """The reader is offered it, so it is part of the choice they face."""
        roles = [_role("IVC_DT", "Invoice Date", status="generated"),
                 _role("DLV_DT", "Delivery Date", status="generated")]
        assert self.CODE in _codes(_contract(roles))

    def test_but_a_generated_default_does_not_settle_it(self):
        """The resolver requires approved AND default. A guessed default is
        not a decision anybody made."""
        roles = [_role("IVC_DT", "Invoice Date", status="generated", is_default=1),
                 _role("DLV_DT", "Delivery Date")]
        assert self.CODE in _codes(_contract(roles))


class TestASettingThatDoesNothing:

    CODE = "metric_default_time_column_not_approved"

    def test_an_unapproved_target_column_is_reported(self):
        roles = [_role("IVC_DT", "Invoice Date", status="generated"),
                 _role("DLV_DT", "Delivery Date", is_default=1)]
        assert self.CODE in _codes(
            _contract(roles, default_time_column="IVC_DT"))

    def test_an_approved_target_column_is_not(self):
        assert self.CODE not in _codes(
            _contract(SETTLED, default_time_column="IVC_DT"))

    def test_a_column_with_no_role_at_all_is_reported_differently(self):
        """Two states, two consequences: an unapproved role means the question
        is steered by something nobody reviewed; no role at all means there is
        no governed date whatsoever."""
        no_role = _finding(
            _contract(SETTLED, default_time_column="SOME_OTHER_DT"), self.CODE)
        assert "no governed date at all" in no_role["message"]

        unapproved = _finding(
            _contract([_role("IVC_DT", "Invoice Date", status="generated"),
                       _role("DLV_DT", "Delivery Date", is_default=1)],
                      default_time_column="IVC_DT"), self.CODE)
        assert "nobody approved" in unapproved["message"]

    def test_no_setting_means_no_finding(self):
        assert self.CODE not in _codes(_contract(SETTLED, default_time_column=""))

    def test_the_remedy_names_the_page_that_makes_it_work(self):
        finding = _finding(
            _contract(SETTLED, default_time_column="SOME_OTHER_DT"), self.CODE)
        assert "Date Roles" in finding["suggestions"][0]
        assert "SOME_OTHER_DT" in finding["suggestions"][0]

    @pytest.mark.parametrize("spelling", [
        "IVC_DT", "ivc_dt", "[IVC_DT]", "`IVC_DT`", "EMDW.F.IVC_DT",
    ])
    def test_the_column_is_matched_however_it_was_typed(self, spelling):
        """The admin types this into a form. A quoted or qualified spelling
        must not read as a different column and produce a false finding."""
        assert self.CODE not in _codes(
            _contract(SETTLED, default_time_column=spelling))

    def test_a_metric_on_another_table_is_not_judged_by_this_one(self):
        assert self.CODE not in _codes(
            _contract(SETTLED, default_time_column="IVC_DT",
                      metric_table="EMDW.SOME_OTHER_FCT"))


class TestTheFindingsAreWellFormed:
    """These rows go into the Model Health table and are keyed for
    deduplication across runs."""

    ALL = [
        ("fact_date_roles_without_default", lambda: _contract(UNDECIDED)),
        ("metric_default_time_column_not_approved",
         lambda: _contract(SETTLED, default_time_column="SOME_OTHER_DT")),
    ]

    @pytest.mark.parametrize("code,build", ALL, ids=[c for c, _ in ALL])
    def test_every_required_field_is_present(self, code, build):
        finding = _finding(build(), code)
        for field in ("conflict_key", "code", "severity", "object_type",
                      "object_id", "origin", "message", "evidence",
                      "suggestions"):
            assert finding.get(field) not in (None, "", [], {}), field
        assert finding["origin"] == "date_governance"
        assert finding["severity"] in {"ERROR", "WARNING", "INFO"}

    @pytest.mark.parametrize("code,build", ALL, ids=[c for c, _ in ALL])
    def test_the_key_is_stable_across_runs(self, code, build):
        assert _finding(build(), code)["conflict_key"] == \
            _finding(build(), code)["conflict_key"]

    def test_the_two_codes_do_not_share_a_key(self):
        contract = _contract(UNDECIDED, default_time_column="SOME_OTHER_DT")
        keys = {f["conflict_key"] for f in detect_ambiguous_date_roles(contract)}
        assert len(keys) == len(detect_ambiguous_date_roles(contract))


class TestItNeverRaisesOnAThinContract:
    """Model Health is a report. A detector that throws takes the whole page
    down, including the findings that were fine."""

    @pytest.mark.parametrize("contract", [
        {}, {"model": {}}, {"model": {"tables": []}},
        {"model": {"tables": [None, "x"]}, "metrics": [None]},
        {"model": {"tables": [{"date_roles": [None, "x"]}]}},
        {"metrics": [{"id": 7, "default_time_column": "X"}]},
        {"model": {"tables": [{"qualified_name": FACT, "date_roles": UNDECIDED}]}},
    ])
    def test_it_returns_a_list(self, contract):
        assert isinstance(detect_ambiguous_date_roles(contract), list)

    def test_a_table_with_no_name_is_skipped_rather_than_reported_blank(self):
        contract = {"model": {"tables": [{"date_roles": UNDECIDED}]}}
        assert "fact_date_roles_without_default" not in _codes(contract)
