"""Analytical answers are asked for in the reader's words, and offer chips that change one setting with no AI.

The planner is told how to ask for each shape the workspace's data can give, in the workspace's own names, and
every example it is shown resolves. An answer offers chips ("Median instead", "6-month moving average", "Top 5
in each store"): a click runs the answer's plan with that setting changed, without the AI, and the chips of an
earlier answer mean nothing once another question is asked. The conditions on who had none follow the rules
every filter does: a name narrows only when it is quoted, and none reaches the AI where names are withheld.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import _analyses, _plan_shown, _unmask_plan, stable_prompt
from core2.plan.values import Masked, MemberIndex, build_index
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _chat(learned, *plans: dict, index: MemberIndex | None = None):
    """A chat whose AI answers with ``plans`` in turn; it records each time it is asked."""
    built, model = learned
    waiting = [json.dumps({"kind": "query", **p}) for p in plans]
    asked: list[str] = []

    def complete(stable: str, tail: str) -> str:
        asked.append(tail)
        return waiting.pop(0)

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete,
                        index=index or MemberIndex(), today=TODAY)
    session = Session()
    return (lambda question: answer_question(question, services, session)), asked


def _chips(payload: dict) -> list[str]:
    return [s["label"] for s in payload.get("follow_up_suggestions") or [] if s.get("kind") == "setting"]


# ── chips ──────────────────────────────────────────────────────────────────


def test_a_chip_changes_one_setting_and_asks_no_ai(retail):
    ask, asked = _chat(retail, {"measures": ["net_amount"], "per": {"of": "customer"}, "time": Q1})
    first = ask("average spend per customer in Q1")
    assert _chips(first) == ["Median instead", "Counting every customer listed"]
    median = ask("Median instead")
    assert len(asked) == 1                                       # the chip ran the plan: no AI
    assert median["answer"]["headline"].startswith("Median net amount per customer in Q1 2026: $")
    assert median["plan"]["per"] == {"of": "customer", "summary": "median"}
    assert "following" not in median                             # the chip says what it changed
    assert _chips(median) == ["Average instead", "75th percentile instead", "90th percentile instead"]
    p90 = ask("90th percentile instead")
    assert len(asked) == 1 and p90["plan"]["per"] == {"of": "customer", "summary": "percentile", "p": 90.0}
    # No why or forecast is offered for a figure per member: it is not worked out yet.
    assert not any(s["label"].startswith(("Why did", "Forecast")) for s in p90["follow_up_suggestions"])


@pytest.mark.parametrize("plan, chips", [
    ({"intent": "trend", "measures": ["net_amount"], "rolling": {"kind": "moving_average", "periods": 3},
      "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-05-31"}}},
     ["6-month moving average", "12-month moving average", "Running total instead"]),
    ({"intent": "trend", "measures": ["net_amount"], "rolling": {"kind": "running", "reset": "year"},
      "time": {"grain": "month", "window": {"kind": "this", "unit": "year"}}},
     ["Running total since the data begins"]),
    ({"intent": "rank", "measures": ["net_amount"], "group_by": ["store.name", "product.name"],
      "within": {"group": "store.name", "top": 3}, "time": Q1},
     ["Top 5 in each store", "Top 10 in each store"]),
    ({"intent": "list", "absent": {"of": "customer", "measure": "number_of_orders"},
      "time": {"window": {"kind": "last", "unit": "day", "n": 90}}},
     ["No orders in the last 60 days", "No orders in the last 180 days", "Only those with orders before"]),
    ({"measures": ["number_of_orders"], "per": {"of": "customer", "summary": "count", "op": "gte", "values": [3]},
      "time": Q1},
     ["2 or more orders instead", "4 or more orders instead", "Counting every customer listed"]),
])
def test_each_shape_offers_its_settings_and_each_runs(retail, plan, chips):
    ask, asked = _chat(retail, plan)
    first = ask("q")
    assert _chips(first) == chips
    for chip in chips:          # every chip runs, each from the first answer's offers
        ask, asked = _chat(retail, plan)
        ask("q")
        changed = ask(chip)
        assert len(asked) == 1 and changed.get("type") == "assistant_response" and "export_rows" in changed, chip


def test_a_plain_answer_offers_no_settings_and_old_chips_mean_nothing_after_a_new_question(retail):
    ask, asked = _chat(retail, {"measures": ["net_amount"], "per": {"of": "customer"}, "time": Q1},
                       {"measures": ["net_amount"], "group_by": ["store.name"], "time": Q1},
                       {"measures": ["net_amount"], "time": Q1})
    ask("average spend per customer")
    plain = ask("sales by store")
    assert _chips(plain) == []
    ask("Median instead")                    # offered two answers ago: read as a question of its own
    assert len(asked) == 3


def test_why_or_forecast_of_a_shaped_figure_is_refused_with_why(retail):
    ask, _ = _chat(retail, {"intent": "drivers", "measures": ["net_amount"], "per": {"of": "customer"},
                            "time": Q1})
    said = ask("why did spend per customer change")
    assert "not worked out yet" in str(said)


# ── the planner is told how to ask ─────────────────────────────────────────


@pytest.mark.parametrize("name", [*domains.available(), "compounding_pharmacy"])
def test_every_example_the_planner_is_shown_resolves(name):
    built, model = learn(domains.build(name), "descriptive")
    text = _analyses(model)
    assert text.startswith("ANALYSES") and text in stable_prompt(model)
    plans = [json.loads(line[3:]) for line in text.splitlines() if line.startswith("A: ")]
    assert len(plans) >= 3 and any(p.get("rolling") for p in plans)
    for plan in plans:
        resolve(Plan.model_validate(plan), model, Context(today=TODAY))


def test_ranges_and_per_order_are_taught_only_where_the_data_has_them(retail):
    from core2.model.schema import SemanticModel

    built, model = retail
    taught = _analyses(model)
    assert '"durations"' in taught and '"per": {"of": "number_of_orders"}' in taught
    one_date = SemanticModel.model_validate(model.model_dump())
    for role in list(one_date.date_roles.values()):
        if not role.is_default:
            del one_date.date_roles[role.key]
    for key in [k for k, m in one_date.measures.items() if getattr(m.expr, "agg", "") == "count_distinct"]:
        del one_date.measures[key]
    fewer = _analyses(one_date)
    assert '"durations"' not in fewer and "number_of_orders" not in fewer and '"rolling"' in fewer


# ── conditions on who had none ─────────────────────────────────────────────


def test_a_name_narrows_who_had_none_only_when_quoted(retail):
    built, model = retail
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    index = build_index(model, fetch)
    absent = {"intent": "list", "time": {"window": {"kind": "last", "unit": "day", "n": 90}},
              "absent": {"of": "customer", "measure": "number_of_orders",
                         "without": [{"field": "store.name", "op": "eq", "values": ["Airport Store"]}]}}
    ask, _ = _chat(retail, absent, absent, index=index)
    unquoted = ask("customers with no orders at the airport store in the last 90 days")
    assert unquoted["plan"]["absent"].get("without", []) == []
    assert any('Not narrowed to store name "Airport Store"' in n for n in unquoted["trust"]["date_context"])
    quoted = ask('customers with no orders at the "Airport Store" in the last 90 days')
    assert quoted["plan"]["absent"]["without"][0]["values"] == ["Airport Store"]
    assert len(quoted["export_rows"]) > len(unquoted["export_rows"])


def test_no_name_in_who_had_none_reaches_the_ai_where_names_are_withheld(retail):
    plan = Plan.model_validate({"kind": "query", "intent": "list", "absent": {
        "of": "customer", "measure": "number_of_orders",
        "without": [{"field": "store.name", "op": "eq", "values": ["Airport Store"]}],
        "had": [{"field": "customer.segment", "op": "eq", "values": ["Online"]}]}})
    masked = Masked(question="q", values={})
    shown = _plan_shown(plan, retail[1], masked)
    assert "Airport Store" not in shown and "Online" not in shown and "⟨v1⟩" in shown and "⟨v2⟩" in shown
    back = _unmask_plan(Plan.model_validate(json.loads(shown)), masked)
    assert back.absent.without[0].values == ["Airport Store"] and back.absent.had[0].values == ["Online"]
