"""A name written without quotes narrows the answer by the field it names, when the question says which.

"How many schools are in Harbor Valley Unified?" names a district. The district is a school's, and the tables of
what happened at a school (its test scores, its meals) keep it again: three fields hold the name, so it was read
as a name two groupings share and narrowed nothing, and the count was of every school. The school's own field is
meant whenever each table keeping a copy is linked to the school. Likewise the field written beside the name
("the city of Northdale", "Northdale County"), the plain field over one that qualifies it (the city, not the
mailing city), and the field named elsewhere in the question ("which county ..."). A name still unclear (a city
and a county of one name, nothing to tell them apart) narrows nothing, as before.

A member written as a field's value ("Virtual = F", "DOC = 31") is read as that field's, exactly as written.

And a condition the AI planned as the grouping of a top-N answer ("between Westmere and Southbay, which county has
the most") holds only when it narrows the rows: the top county of all shown is a part left out.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import build_index
from core2.service import Services, Session, _assigned, _one_reading, _own_names, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
# District, county, city. Northdale is a county and a city; Linden is a city and some schools' mailing city.
DISTRICTS = [("Harbor Valley Unified", "Westmere", "Port Avalon"), ("Cedar Ridge Unified", "Westmere", "Linden"),
             ("Granite Falls Unified", "Northdale", "Granite Falls"), ("Lakeshore Elementary", "Northdale", "Northdale"),
             ("Pinecrest Unified", "Southbay", "Pinecrest"), ("Mesa Verde Unified", "Southbay", "Ocotillo")]


@pytest.fixture(scope="module")
def schools():
    rng = random.Random(5)
    con = duckdb.connect()
    con.execute("""CREATE TABLE schools (cds_code VARCHAR PRIMARY KEY, school_name VARCHAR, district VARCHAR,
                   county VARCHAR, city VARCHAR, mail_city VARCHAR, virtual VARCHAR, doc VARCHAR, open_date DATE)""")
    con.execute("""CREATE TABLE test_scores (cds VARCHAR, district VARCHAR, county VARCHAR, num_tested INTEGER,
                   avg_math INTEGER, test_year INTEGER)""")
    con.execute("""CREATE TABLE meals (cds_code VARCHAR, district VARCHAR, county VARCHAR, enrollment INTEGER,
                   free_meals INTEGER, school_year VARCHAR)""")
    cities = [c for _, _, c in DISTRICTS]
    for i in range(240):
        district, county, city = DISTRICTS[i % len(DISTRICTS)]
        code = f"{10000000 + i * 37:08d}"
        con.execute("INSERT INTO schools VALUES (?,?,?,?,?,?,?,?,?)",
                    [code, f"{['North', 'South', 'East', 'West', 'Central', 'Hill'][i % 6]} School {i}", district,
                     county, city, city if rng.random() < 0.7 else rng.choice(cities), rng.choice("FPN"),
                     rng.choice(["52", "54", "31", "00"]), f"{1960 + i % 60}-0{1 + i % 9}-1{i % 9}"])
        for year in (2019, 2020):
            con.execute("INSERT INTO test_scores VALUES (?,?,?,?,?,?)",
                        [code, district, county, rng.randint(20, 400), rng.randint(350, 650), year])
        con.execute("INSERT INTO meals VALUES (?,?,?,?,?,?)",
                    [code, district, county, rng.randint(100, 2000), rng.randint(10, 900), "2019-2020"])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id="eval-schools",
                        options=BuildOptions(workers=1))

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _names(schools, question):
    model, _, index = schools
    return [(m.text, m.attribute) for m in index.written_names(question, not_names=_own_names(model),
                                                               prefer=lambda f: _one_reading(model, question, f))]


def test_the_learned_shape_is_the_one_these_tests_are_about(schools):
    # Each name is held by the school and again by the two tables of what happened at a school, all linked.
    model, _, index = schools
    held = sorted(m.attribute for m in index.match("in Harbor Valley Unified"))
    assert len(held) == 3 and "school.district" in held, held
    assert {model.attributes[a].business_name for a in held} == {"District"}
    assert sorted(m.attribute for m in index.match("in Linden")) == ["school.city", "school.mail_city"]


@pytest.mark.parametrize("question, read", [
    ("How many schools are in Harbor Valley Unified?", [("Harbor Valley Unified", "school.district")]),
    ("Average math score of the schools in Cedar Ridge Unified", [("Cedar Ridge Unified", "school.district")]),
    ("How many schools are in Westmere County?", [("Westmere", "school.county")]),
    ("How many schools are in the city of Northdale?", [("Northdale", "school.city")]),
    ("Free meals in Northdale County", [("Northdale", "school.county")]),
    ("How many schools are in Linden?", [("Linden", "school.city")]),
    ("Which county has the most schools, Northdale or Southbay?", [("Northdale", "school.county"),
                                                                    ("Southbay", "school.county")]),
])
def test_a_name_reads_as_the_field_the_question_means(schools, question, read):
    assert _names(schools, question) == read


def test_a_name_the_question_leaves_unclear_narrows_nothing(schools):
    # A county and a city of one name, with no word for either: which is meant is unclear.
    assert _names(schools, "How many schools are in Northdale?") == []


def test_copies_on_tables_not_linked_to_the_thing_stay_unclear(schools):
    model, _, index = schools
    question = "How many schools are in Harbor Valley Unified?"
    found = index.match(question)
    assert _one_reading(model, question, found).attribute == "school.district"
    unlinked = model.model_copy(update={"joins": {}})
    assert _one_reading(unlinked, question, found) is None


def test_a_member_written_as_a_fields_value_is_that_fields(schools):
    model, _, index = schools
    read = lambda q: [(m.text, m.attribute, m.value) for m in _assigned(model, index, q)]  # noqa: E731
    assert read("How many schools have Virtual = F?") == [("F", "school.virtual", "F")]
    assert read("Schools with DOC = 31 opened after 2000") == [("31", "school.document", "31")]
    assert read("Schools where virtual = 'P'") == [("P", "school.virtual", "P")]
    # Exactly as written: "3" is no code of the field, and is never read as part of "31".
    assert read("Schools with DOC = 3") == []
    # A field the question does not name holds nothing it says.
    assert read("Schools with code = 31") == []


class Recorded:
    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        return self.plans.pop(0)


def _count(warehouse, where):
    return warehouse.query(f"SELECT COUNT(*) FROM schools WHERE {where}").rows[0][0]


def test_the_planner_is_told_the_field_and_the_answer_counts_its_schools(schools):
    model, warehouse, index = schools
    plan = {"intent": "value", "measures": ["number_of_schools"],
            "filters": [{"field": "school.district", "op": "eq", "values": ["Harbor Valley Unified"]}]}
    ai = Recorded(plan)
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    payload = answer_question("How many schools are in Harbor Valley Unified?", services, Session())
    assert 'school.district = "Harbor Valley Unified"' in ai.sent[0]
    assert round(float(payload["kpi"]["value"])) == _count(warehouse, "district = 'Harbor Valley Unified'")


def test_a_value_written_after_its_field_reaches_the_planner(schools):
    model, warehouse, index = schools
    plan = {"intent": "value", "measures": ["number_of_schools"],
            "filters": [{"field": "school.virtual", "op": "eq", "values": ["F"]}]}
    ai = Recorded(plan)
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    payload = answer_question("How many schools have Virtual = F?", services, Session())
    assert 'school.virtual = "F"' in ai.sent[0]
    assert round(float(payload["kpi"]["value"])) == _count(warehouse, "virtual = 'F'")


# ── a condition planned as the grouping of a top-N answer (core2/agent/parts.py) ──

def _missing(schools, plan, headers, question, named=()):
    from core2.agent import parts as P
    from core2.plan.ir import Plan

    payload = {"data": {"headers": headers, "rows": [{h: 1 for h in headers}]}}
    return [(g.words, g.why) for g in P.missing(question, Plan.model_validate({"kind": "query", **plan}), payload,
                                                schools[0], named=list(named))]


def test_a_condition_planned_as_the_grouping_of_the_top_one_is_left_out(schools):
    question = "Between Westmere and Southbay, which county has the most schools?"
    asked = [{"words": "between Westmere and Southbay", "kind": "condition", "plan": "school.county"}]
    top = {"asked": asked, "intent": "rank", "measures": ["number_of_schools"], "group_by": ["school.county"],
           "limit": 1, "sort": [{"by": "number_of_schools", "desc": True}]}
    assert _missing(schools, top, ["county", "number_of_schools"], question) == [
        ("between Westmere and Southbay", "not_applied")]
    narrowed = {**top, "filters": [{"field": "school.county", "op": "in", "values": ["Westmere", "Southbay"]}]}
    assert _missing(schools, narrowed, ["county", "number_of_schools"], question) == []
    every = {k: v for k, v in top.items() if k not in ("limit", "sort")}
    assert _missing(schools, every, ["county", "number_of_schools"], question) == [], \
        "a row each of the grouping shown holds every member it names"


def test_a_named_member_of_the_grouping_of_the_top_one_must_be_held(schools):
    model, _, index = schools
    question = "Which county has the most schools, Northdale or Southbay?"
    named = index.written_names(question, not_names=_own_names(model),
                                prefer=lambda f: _one_reading(model, question, f))
    top = {"intent": "rank", "measures": ["number_of_schools"], "group_by": ["school.county"], "limit": 1}
    assert sorted(w for w, _ in _missing(schools, top, ["county", "number_of_schools"], question, named)) == [
        "Northdale", "Southbay"]
    every = {"intent": "breakdown", "measures": ["number_of_schools"], "group_by": ["school.county"]}
    assert _missing(schools, every, ["county", "number_of_schools"], question, named) == []
