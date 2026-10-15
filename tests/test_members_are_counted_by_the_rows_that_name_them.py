"""A count of a list's members by something the list has no link to is counted on the rows that name both.

"What is the patient count by each doctor" was refused: the patient count counts the patient list, and a
patient has no doctor of their own; the planner then offered measures by their slugs ("number_of_doctors").
Each prescription names its patient and its prescriber: a doctor's patients are the distinct patients on that
doctor's prescriptions, as "patients in Q2" are the distinct patients on Q2's prescriptions. The answer says it
counted them there. A grouping the list itself reaches (patients by state) still counts the list; one no table of
rows reaches both ways is still refused. Options of a question back are shown by name.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, _named_options, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def pharmacy():
    built, model = learn(domains.build("compounding_pharmacy"), "descriptive")
    return built.con, model


def _run(pharmacy, plan: dict):
    con, model = pharmacy
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    compiled = compile_query(logical, model, "duckdb")
    return logical, compiled, DuckDBWarehouse(con).query(compiled.sql).rows


def test_patients_by_doctor_are_the_distinct_patients_on_each_doctors_prescriptions(pharmacy):
    con, _ = pharmacy
    logical, compiled, rows = _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_patients"],
                                              "group_by": ["prescriber"]})
    assert "COUNT(DISTINCT prescription.patient_id)" in compiled.sql
    by_hand = con.execute("""SELECT d.prescriber_name, COUNT(DISTINCT p.patient_id) FROM prescriptions p
                             JOIN prescribers d ON p.prescriber_npi = d.prescriber_npi GROUP BY 1""").fetchall()
    assert sorted((r[0], r[-1]) for r in rows if r[0] is not None) == sorted(by_hand)
    assert any("counts the patients on prescription rows" in n for n in logical.notes)


def test_by_a_doctors_own_field_and_over_a_period_too(pharmacy):
    con, _ = pharmacy
    _, _, rows = _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_patients"],
                                 "group_by": ["prescriber.specialty"],
                                 "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}})
    by_hand = dict(con.execute("""SELECT d.specialty, COUNT(DISTINCT p.patient_id) FROM prescriptions p
                                  JOIN prescribers d ON p.prescriber_npi = d.prescriber_npi
                                  WHERE p.written_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
                                  GROUP BY 1""").fetchall())
    assert {r[0]: r[-1] for r in rows if r[0] is not None} == by_hand


def test_a_link_by_another_value_each_member_has_one_of_counts_the_same(pharmacy):
    """Prescriptions name their prescriber by NPI; the prescriber count counts prescriber ids: one of each per
    prescriber, so the distinct NPIs on a formula's prescriptions are its prescribers."""
    con, _ = pharmacy
    _, compiled, rows = _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_prescribers"],
                                        "group_by": ["formula"]})
    assert "COUNT(DISTINCT prescription.prescriber_npi)" in compiled.sql
    by_hand = dict(con.execute("""SELECT f.formula_name, COUNT(DISTINCT p.prescriber_npi) FROM prescriptions p
                                  JOIN formulas f ON p.formula_id = f.formula_id GROUP BY 1""").fetchall())
    assert {r[0]: r[-1] for r in rows if r[0] is not None} == by_hand


def test_what_the_list_reaches_itself_still_counts_the_list(pharmacy):
    con, _ = pharmacy
    logical, compiled, rows = _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_patients"],
                                              "group_by": ["patient.state"]})
    assert "prescription" not in compiled.sql
    assert sum(r[-1] for r in rows) == con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    _, compiled, rows = _run(pharmacy, {"intent": "value", "measures": ["number_of_patients"]})
    assert "prescription" not in compiled.sql and rows[0][-1] == con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]


def test_rows_a_link_or_two_away_count_them_too(pharmacy):
    """A fill names its pharmacist and its prescription, the prescription its patient: a pharmacist's patients
    are the distinct patients of the prescriptions they filled."""
    con, _ = pharmacy
    logical, _, rows = _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_patients"],
                                       "group_by": ["pharmacist"]})
    by_hand = con.execute("""SELECT ph.pharmacist_name, COUNT(DISTINCT p.patient_id) FROM fills f
                             JOIN prescriptions p ON f.rx_number = p.rx_number
                             JOIN pharmacists ph ON CAST(f.pharmacist_code AS INTEGER) = ph.pharmacist_id
                             GROUP BY 1""").fetchall()
    assert sorted((r[0], r[-1]) for r in rows if r[0] is not None) == sorted(by_hand)
    assert any("on fill rows" in n for n in logical.notes)


def test_what_no_rows_name_together_is_still_refused(pharmacy):
    """A quality test names its batch, and nothing names a test and a prescriber both."""
    with pytest.raises(ResolveError, match="is not linked to"):
        _run(pharmacy, {"intent": "breakdown", "measures": ["number_of_quality_tests"], "group_by": ["prescriber"]})


def test_in_the_chat_the_answer_says_where_it_counted(pharmacy):
    con, model = pharmacy

    def ai(stable: str, tail: str) -> str:
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["number_of_patients"],
                           "group_by": ["prescriber"]})

    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=ai, index=MemberIndex(), today=TODAY)
    payload = answer_question("what is the patient count by each doctor", services, Session())
    assert payload.get("clarify") is None and payload["data"]["rows"]
    assert "with a prescription" in payload["answer"]["headline"]


def test_options_of_a_question_back_are_shown_by_name(pharmacy):
    _, model = pharmacy
    assert _named_options(model, ["number_of_patients", "number_of_prescribers", "Something else",
                                  "Number of patients"]) == ["Number of patients", "Number of prescribers",
                                                             "Something else"]
