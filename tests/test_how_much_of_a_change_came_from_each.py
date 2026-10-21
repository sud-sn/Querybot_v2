"""How much of a change came from each member: its change as a share of the whole change.

"Total originations fell from 2024 to 2025 - how much of the drop came from each product family?" was answered
with each family's own % change, never its part of the drop. An analyst's table has the two periods, the change
and each member's share of the total change: the shares add up to 100%, a member that went the other way has a
negative share, and one that fell more than the whole fell has more than 100%.

The plan asks for it (``share_of_change``), the question's own words ask for it when the plan missed it, and a
"why did X change" question keeps its drivers answer. A figure whose changes do not add up (an average, a ratio)
has no share of the change, and the answer says why. Every number is compared with SQL written by hand against
the invented retail warehouse.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question

THIS = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
LAST = {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}
BOTH = {"window": THIS, "compare": {"kind": "window", "window": LAST}}


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")


def _ask(retail, plan: dict, question: str = "q") -> dict:
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 15))
    return answer_question(question, services, Session())


def _truth(retail, sql: str) -> dict[str, float]:
    built, _ = retail
    return {row[0]: float(row[1]) for row in built.con.execute(sql).fetchall()}


BY_REGION = """
    with y as (select r.region_name, extract(year from c.full_date) as y, sum(o.net_amount) as v
               from order_lines o join calendar c on c.date_key = o.order_date_key
               join stores s using (store_id) join regions r using (region_id)
               where c.full_date between date '2024-01-01' and date '2025-12-31' group by 1, 2),
         d as (select region_name, sum(case when y = 2025 then v else -v end) as change from y group by 1)
    select region_name, change * 100.0 / sum(change) over () from d"""


def _column(payload: dict, role_words: str) -> str:
    labels = payload["data"]["header_labels"]
    return next(k for k, v in labels.items() if role_words in v)


def test_each_regions_share_of_the_total_change(retail):
    payload = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"], "time": BOTH,
                            "share_of_change": True})
    column = _column(payload, "share of the total change")
    got = {r["region_name"]: float(r[column]) for r in payload["data"]["rows"]}
    assert got == pytest.approx(_truth(retail, BY_REGION), abs=1e-3)
    assert sum(got.values()) == pytest.approx(100.0, abs=0.01)
    # The two periods, the change and its % change stay beside it, as an analyst's table has them.
    assert {"net_amount", "net_amount_prior", "net_amount_change", "net_amount_pct_change"} <= set(
        payload["data"]["rows"][0])


def test_the_headline_says_the_leading_members_share(retail):
    payload = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"], "time": BOTH,
                            "share_of_change": True})
    truth = _truth(retail, BY_REGION)
    lead = max(truth, key=truth.get)
    assert f"{round(truth[lead])}% of the total change" in payload["answer"]["headline"], payload["answer"]


def test_a_share_of_a_change_is_only_given_when_asked(retail):
    payload = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"], "time": BOTH})
    assert not any("share of the total change" in v for v in payload["data"]["header_labels"].values())


@pytest.mark.parametrize("question", [
    "Net sales went up from 2024 to 2025 - how much of that change came from each region?",
    "How much of the growth in net sales from 2024 to 2025 came from each region?",
    "Net sales 2024 vs 2025 by region, with each region's % of the total change",
    "each region's share of the increase in net sales, 2025 against 2024",
])
def test_the_questions_own_words_ask_for_it(retail, question):
    payload = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"], "time": BOTH},
                   question)
    column = _column(payload, "share of the total change")
    got = {r["region_name"]: float(r[column]) for r in payload["data"]["rows"]}
    assert got == pytest.approx(_truth(retail, BY_REGION), abs=1e-3)


def test_a_drivers_plan_for_how_much_came_from_each_is_the_table_asked_for(retail):
    # The AI planned it as "why did it change", looking only at regions: the reader asked for each region's part.
    plan = {"intent": "drivers", "measures": ["net_amount"], "time": BOTH, "drivers": {"dimensions": ["region"]}}
    payload = _ask(retail, plan, "How much of the change in net sales from 2024 to 2025 came from each region?")
    column = _column(payload, "share of the total change")
    got = {r["region_name"]: float(r[column]) for r in payload["data"]["rows"]}
    assert got == pytest.approx(_truth(retail, BY_REGION), abs=1e-3)


def test_why_did_it_change_keeps_its_drivers_answer():
    from core2.plan import analyst_math

    plan = Plan.model_validate({"kind": "query", "intent": "drivers", "measures": ["net_amount"], "time": BOTH,
                                "drivers": {"dimensions": ["region"]}})
    for question in ("Why did net sales rise from 2024 to 2025, by region?", "What drove the change in net sales?"):
        assert analyst_math.share_of_change(plan, None, question) is plan


def test_a_ratio_has_no_share_of_the_change_and_says_why(retail):
    payload = _ask(retail, {"intent": "compare", "measures": ["margin_percent"], "group_by": ["region"],
                            "time": BOTH, "share_of_change": True})
    assert payload["data"]["rows"]
    assert not any("share of the total change" in v for v in payload["data"]["header_labels"].values())
    said = json.dumps(payload)
    assert "No share of the change is worked out" in said, said[:2000]


def test_a_figure_of_two_tables_that_adds_up_has_one(retail):
    # Net sales less refunds: refunds are on the returns table. Its changes add up like a total's.
    from evals.core2 import domains  # noqa: F401 - the warehouse is the fixture's

    built, model = retail
    model = model.model_copy(deep=True)
    from core2.model import authoring
    from core2.model.schema import Measure

    read = authoring.read(model, "SUM([Order line · Net amount]) - SUM([Return · Refund amount])")
    model.measures["kept"] = Measure(key="kept", slug="kept", business_name="Kept sales", table=read.table,
                                     expr=read.expr, format="currency", status="approved")
    payload = _ask((built, model), {"intent": "compare", "measures": ["kept"], "group_by": ["region"],
                                    "time": BOTH, "share_of_change": True})
    column = _column(payload, "share of the total change")
    got = [float(r[column]) for r in payload["data"]["rows"] if r[column] is not None]
    assert len(got) == len(payload["data"]["rows"]) and sum(got) == pytest.approx(100.0, abs=0.01)


def test_the_request_part_for_it_is_held_by_the_answer(retail):
    from core2.agent import parts

    question = "Net sales 2024 vs 2025 by region, with each region's % of the total change"
    plan = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                "group_by": ["region"], "time": BOTH, "share_of_change": True,
                                "asked": [{"words": "% of the total change", "kind": "figure", "plan": "compare"}]})
    payload = _ask(retail, plan.model_dump(mode="json", exclude={"kind"}, exclude_defaults=True), question)
    assert not parts.missing(question, plan, payload, retail[1])
