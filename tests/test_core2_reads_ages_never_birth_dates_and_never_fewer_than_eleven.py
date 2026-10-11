"""People's ages, from a birth date an admin allowed: in bands or 5-year ranges, never a group of fewer than 11.

The invented pharmacy lists its patients with their birth dates. Learn proposes the birth date as usable through
age; until an admin confirms it, no question can use it. Once confirmed, a question can count fills or patients by
age band, or keep only some ages, and each answer is checked here against the patients' ages worked out by hand.
No answer shows a group of fewer than 11 patients, nor lets one be worked out from the rest; an age limit where
the 5 years just inside it hold fewer than 11 is refused; the birth date never reaches the AI; and the age is the
same, birthday or not, on every warehouse.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest
import sqlglot
from sqlglot import exp

from core2.compile.compiler import _Compiler
from core2.model.overrides import apply_overrides
from core2.plan.ir import Plan
from core2.plan.planner import stable_prompt
from core2.plan.values import MemberIndex
from core2.resolve.age import MIN_PEOPLE, age_fields, age_range
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


def _age(born: dt.date, on: dt.date = TODAY) -> int | None:
    if born < dt.date(1901, 1, 1) or born > on:
        return None
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


def _band(age: int | None) -> str:
    if age is None:
        return "Unknown"
    edges = [(20, "Under 20"), (35, "20–34"), (50, "35–49"), (65, "50–64"), (80, "65–79")]
    return next((label for edge, label in edges if age < edge), "80 and over")


@pytest.fixture(scope="module")
def pharmacy():
    built, model = learn(domains.build("compounding_pharmacy"), "descriptive")
    return built, model


@pytest.fixture(scope="module")
def allowed(pharmacy):
    built, model = pharmacy
    confirmed = model.model_copy(deep=True)
    birth = next(c for c in confirmed.columns.values() if c.age == "proposed")
    assert apply_overrides(confirmed, [{"object_key": f"column:{birth.key}", "field": "age",
                                        "value": "confirmed"}]) == []
    return built, confirmed


def _ask(learned, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _patients(built) -> list[tuple]:
    return built.con.execute("SELECT patient_id, birth_date, state FROM patients").fetchall()


def test_learn_proposes_the_birth_date_and_no_question_reads_it_until_an_admin_confirms(pharmacy):
    built, model = pharmacy
    proposed = [c for c in model.columns.values() if c.age == "proposed"]
    assert [c.name for c in proposed] == ["birth_date"]
    assert proposed[0].personal == "detail" and proposed[0].sensitivity == "pii"
    assert age_fields(model) == {}
    with pytest.raises(ResolveError):
        resolve(Plan.model_validate({"kind": "query", "measures": ["number_of_fills"], "group_by": ["patient.age"]}),
                model, Context(today=TODAY))


def test_the_birth_date_never_reaches_the_ai_only_that_an_age_exists(allowed):
    built, model = allowed
    prompt = stable_prompt(model)
    assert "patient.age" in prompt and "birth" not in prompt.lower()
    assert all(str(d) not in prompt for (_, d, _) in _patients(built)[:50])


def test_fills_by_age_band_are_the_patients_ages_worked_out_by_hand(allowed):
    built, model = allowed
    ages = {pid: _age(born) for pid, born, _ in _patients(built)}
    fills = built.con.execute("SELECT r.patient_id FROM fills f JOIN prescriptions r ON f.rx_number = r.rx_number"
                              ).fetchall()
    expected: dict[str, int] = {}
    for (pid,) in fills:
        expected[_band(ages.get(pid))] = expected.get(_band(ages.get(pid)), 0) + 1
    got = _ask(allowed, {"measures": ["number_of_fills"], "group_by": ["patient.age"]})
    assert got["personal"] is True                                  # never summarised by the AI
    shown = {r["age_band"]: r["number_of_fills"] for r in got["export_rows"]}
    assert [r["age_band"] for r in got["export_rows"]] == [b for b in ["Under 20", "20–34", "35–49", "50–64", "65–79",
                                                                       "80 and over"] if b in shown]
    assert shown == {k: v for k, v in expected.items() if k != "Unknown"}
    assert "Ages are whole years on 15 Jun 2026" in " ".join(got["trust"]["date_context"])
    assert "birth" not in json.dumps(got["export_rows"]).lower()


def _block(age: int) -> str:
    return "Under 30" if age < 30 else ("85 and over" if age >= 85 else f"{5 * (age // 5)}–{5 * (age // 5) + 4}")


def test_no_group_of_fewer_than_eleven_patients_is_shown_or_can_be_worked_out(allowed):
    built, model = allowed
    groups: dict[tuple, int] = {}
    for _, born, state in _patients(built):
        if _age(born) is not None:
            key = (state, _block(_age(born)))
            groups[key] = groups.get(key, 0) + 1
    small = {k for k, n in groups.items() if n < MIN_PEOPLE}
    assert small, "the invented patients must hold some small groups for this test to mean anything"
    got = _ask(allowed, {"measures": ["number_of_patients"], "group_by": ["patient.state"],
                         "bands": {"of": "patient.age", "edges": list(range(30, 90, 5))}})
    shown = {(r["patient_state"], r["age_band"]): r["number_of_patients"] for r in got["export_rows"]}
    assert all(n >= MIN_PEOPLE for n in shown.values())
    assert not set(shown) & small
    for key, n in shown.items():
        assert groups[key] == n                                     # what is shown is right
    hidden = set(groups) - set(shown)
    assert len(hidden) >= 2 and sum(groups[k] for k in hidden) >= MIN_PEOPLE    # and cannot be subtracted out
    assert any("not shown" in c for c in got["coverage_caveats"])


def test_an_age_limit_is_taken_in_steps_of_five_and_refused_where_too_few_are_just_inside(allowed):
    built, model = allowed
    assert (age_range("gt", [63]).low, age_range("lt", [18]).high) == (65, 20)
    assert (age_range("between", [41, 52]).low, age_range("between", [41, 52]).high) == (40, 55)
    ages = {pid: _age(born) for pid, born, _ in _patients(built)}
    over = sum(1 for a in ages.values() if a is not None and a >= 65)
    got = _ask(allowed, {"measures": ["number_of_patients"],
                         "filters": [{"field": "patient.age", "op": "gte", "values": [63]}]})
    assert got["export_rows"][0]["number_of_patients"] == over
    assert "for patients aged 65 and over" in got["answer"]["headline"]
    oldest = max(a for a in ages.values() if a is not None)
    edge = 5 * (oldest // 5) + 5                 # the 5 years from here hold no one
    assert sum(1 for a in ages.values() if a is not None and edge <= a < edge + 5) < MIN_PEOPLE
    refused = _ask(allowed, {"measures": ["number_of_patients"],
                             "filters": [{"field": "patient.age", "op": "gte", "values": [edge]}]})
    assert "fewer than 11 people are aged" in refused["answer"]["headline"] and "export_rows" not in refused


@pytest.mark.parametrize("plan, words", [
    ({"intent": "list", "group_by": ["patient.name", "patient.age"]}, "never listed with their age"),
    ({"measures": ["number_of_fills"], "group_by": ["patient.age"], "sort": [{"by": "patient.age"}]}, "never sorted"),
    ({"measures": ["number_of_fills"], "group_by": ["patient.age"], "time": {
        "window": {"kind": "this", "unit": "year"}, "compare": {"kind": "previous_period"}}}, "not compared"),
])
def test_what_could_point_at_someone_is_refused(allowed, plan, words):
    with pytest.raises(ResolveError) as refused:
        resolve(Plan.model_validate({"kind": "query", **plan}), allowed[1], Context(today=TODAY))
    assert words in refused.value.message


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "tsql", "oracle"])
def test_every_warehouse_counts_a_birthday_today_and_not_tomorrow(allowed, dialect):
    import duckdb

    births = [dt.date(2000, 6, 14), dt.date(2000, 6, 15), dt.date(2000, 6, 16), dt.date(2004, 2, 29),
              dt.date(1900, 1, 1), dt.date(2030, 1, 1)]
    con = duckdb.connect()
    con.execute("CREATE TABLE b (born DATE)")
    con.executemany("INSERT INTO b VALUES (?)", [(d,) for d in births])
    writer = _Compiler(None, allowed[1], dialect)  # type: ignore[arg-type]
    sql = exp.select(writer.age(exp.column("born"), TODAY).as_("age")).from_("b").sql(dialect=dialect)
    sqlglot.parse_one(sql, read=dialect)
    try:
        back = sqlglot.transpile(sql, read=dialect, write="duckdb")[0]
        got = [r[0] for r in con.execute(back).fetchall()]
    except Exception:  # noqa: BLE001 - an idiom DuckDB has no word for: it read back in its own dialect above
        pytest.skip(f"{dialect} does not carry back to DuckDB")
    assert got == [_age(d) for d in births] == [26, 26, 25, 22, None, None]


def test_each_use_of_an_age_is_logged(allowed, caplog):
    with caplog.at_level(logging.INFO, logger="core2.age"):
        _ask(allowed, {"measures": ["number_of_fills"], "group_by": ["patient.age"]})
    assert any("core2 age used: patient.age" in r.getMessage() for r in caplog.records)


def test_the_admin_confirms_it_on_the_knowledge_base_page_and_only_a_birth_date_can_be(pharmacy):
    from core2.model.knowledge import KnowledgeError, field_changes, table_view

    built, model = pharmacy
    birth = next(c for c in model.columns.values() if c.age == "proposed")
    field = next(f for f in table_view(model, birth.table)["fields"] if f["key"] == birth.key)
    assert field["age"] == "proposed" and field["ageable"] is True
    assert field_changes(model, birth.key, {"age": "confirmed"}) == {"age": "confirmed"}
    other = next(c for c in model.columns.values() if c.table == birth.table and c.key != birth.key)
    with pytest.raises(KnowledgeError):
        field_changes(model, other.key, {"age": "confirmed"})


def test_one_small_group_takes_the_next_smallest_with_it_and_shares_are_of_what_is_shown():
    from core2.resolve.age import protect

    columns = ["age_band", "fills", "fills_share", "people_counted"]
    rows = [("20–34", 200, 0.4, 40), ("35–49", 30, 0.06, 5), ("50–64", 120, 0.24, 14), ("65–79", 150, 0.3, 25),
            (None, 9, None, 3)]
    kept = protect(columns, rows, people="people_counted", band="age_band", shares=["fills_share"])
    # 35-49 alone would be the total less the rest: 50-64 (the next smallest) goes too, so 19 people are hidden.
    assert [r[0] for r in kept.rows] == ["20–34", "65–79"] and kept.hidden == 2 and kept.unknown == 3
    assert [round(r[2], 4) for r in kept.rows] == [round(200 / 350, 4), round(150 / 350, 4)]
    alone = protect(["fills", "people_counted"], [(4, 7)], people="people_counted", band=None, shares=[])
    assert alone.rows == [] and alone.hidden == 1
