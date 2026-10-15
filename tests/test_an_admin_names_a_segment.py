"""An admin names a segment in the metric editor; readers find it in the Data guide and ask for it by name.

"Regular shoppers": customers with orders on 2 or more days of the period asked. Saved under Metrics → Segments as a
decision, so every Learn keeps it. Readers see it in the Data guide with what it means and a question to try, and
ask for it by name: "net sales from Regular shoppers by store" narrows a figure to them, "how many Regular shoppers in
Q2 2023" counts them. Each answer equals SQL written by hand and calls them by their name. Invented retail data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.model import saved_segments
from core2.model.overrides import apply_overrides, target
from core2.model.schema import SavedSegment
from core2.plan.catalog import catalog_text
from core2.plan.ir import Plan
from core2.plan.planner import saved_segment_named
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_readers_suggest_and_admins_decide import retail, site  # noqa: F401 - the fixtures

TODAY = dt.date(2026, 6, 15)
REGULAR = {"name": "Regular shoppers", "kind": "repeat", "of": "customer", "measure": "number_of_orders",
           "also": "regulars, frequent buyers", "description": "Our loyalty programme's target group."}
Q2_2023 = {"kind": "between", "start": "2023-04-01", "end": "2023-06-30"}
LINES = """SELECT o.*, d.full_date AS day, s.store_name FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id
           WHERE o.customer_id IN (SELECT customer_id FROM customers)"""
REPEAT_Q2 = f"""SELECT customer_id FROM ({LINES}) WHERE day >= DATE '2023-04-01' AND day < DATE '2023-07-01'
                GROUP BY customer_id HAVING COUNT(DISTINCT day) >= 2"""


def _saved(model) -> SavedSegment:
    return saved_segments.build(model, REGULAR)


def _with(model, segment: SavedSegment):
    copy = model.model_copy(deep=True)
    apply_overrides(copy, [{"object_key": target("segment", segment.key), "field": "define",
                            "value": segment.model_dump(mode="json")}])
    return copy


# ── kept as a decision ─────────────────────────────────────────────────────


def test_a_saved_segment_is_a_decision_every_learn_keeps(retail):
    _, model = retail
    segment = _saved(model)
    assert (segment.key, segment.slug, segment.kind) == ("admin:regular_shoppers", "regular_shoppers", "repeat")
    assert segment.synonyms == {"en": ["regulars", "frequent buyers"]}
    learned_again = _with(model, segment)
    assert learned_again.segments[segment.key].business_name == "Regular shoppers"
    assert learned_again.segments[segment.key].kind == "repeat"        # never taken for a decision's status
    # A segment whose members are no longer in the data is said, not applied.
    stray = segment.model_copy(update={"key": "admin:gone", "of": "nobody"})
    lost = model.model_copy(deep=True)
    notes = apply_overrides(lost, [{"object_key": target("segment", stray.key), "field": "define",
                                    "value": stray.model_dump(mode="json")}])
    assert not lost.segments and notes and "not in the data any more" in notes[0]


@pytest.mark.parametrize("change, said", [
    ({"name": ""}, "Give the segment the name"),
    ({"name": "Net amount"}, "already the name of a metric"),
    ({"kind": "best"}, "Choose which members it holds"),
    ({"of": "store", "measure": "number_of_orders"}, "Choose whose activity it follows"),
])
def test_what_cannot_be_saved_is_said(retail, change, said):
    with pytest.raises(saved_segments.SegmentError, match=said):
        saved_segments.build(retail[1], {**REGULAR, **change})


def test_its_meaning_in_a_sentence(retail):
    model = _with(retail[1], _saved(retail[1]))
    row = saved_segments.listing(model)[0]
    assert row["meaning"] == "Customers with orders on 2 or more days of the period asked"
    assert row["example"] == "Net amount from Regular shoppers last quarter"


# ── asked for by name ──────────────────────────────────────────────────────


@pytest.mark.parametrize("question", ["net sales from Regular shoppers by store", "what did our regular shopper spend",
                                      "how many regulars came in Q2", "Frequent buyers by month"])
def test_named_by_its_name_or_another_word(retail, question):
    model = _with(retail[1], _saved(retail[1]))
    assert saved_segments.named(question, model).business_name == "Regular shoppers"


def test_the_planner_is_told_about_it(retail):
    model = _with(retail[1], _saved(retail[1]))
    text = catalog_text(model)
    assert "SAVED SEGMENTS" in text
    assert "- Regular shoppers | customer | number_of_orders | repeat | Customers with orders on 2 or more days" in text


def test_a_figure_for_them_and_a_count_of_them(retail):
    model = _with(retail[1], _saved(retail[1]))
    figure = saved_segment_named(Plan(kind="query", measures=["net_amount"]), model, "net sales from Regular shoppers")
    assert figure.segment is not None and (figure.segment.status, figure.segment.name) == ("repeat", "Regular shoppers")
    counted = saved_segment_named(Plan(kind="query", measures=["number_of_customers"]), model,
                                  "how many Regular shoppers in Q2 2023")
    assert counted.lifecycle is not None and (counted.lifecycle.status, counted.lifecycle.name) == \
        ("repeat", "Regular shoppers") and counted.measures == []


def _ask(retail, question: str, plan: dict) -> dict:
    built, model = retail
    model = _with(model, _saved(model))
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def test_net_sales_from_regular_shoppers_by_store(retail):
    # The AI read the name as nothing: code knows it.
    want = dict(retail[0].con.execute(f"""SELECT store_name, SUM(net_amount) FROM ({LINES})
        WHERE day >= DATE '2023-04-01' AND day < DATE '2023-07-01' AND customer_id IN ({REPEAT_Q2})
        GROUP BY 1""").fetchall())
    got = _ask(retail, "Net sales from Regular shoppers by store in Q2 2023",
               {"measures": ["net_amount"], "group_by": ["store.name"], "time": {"window": Q2_2023}})
    rows = got["export_rows"]
    key = next(k for k in rows[0] if "store" in k)
    value = next(k for k in rows[0] if k != key)
    assert {r[key]: float(r[value]) for r in rows} == pytest.approx({k: float(v) for k, v in want.items()})
    assert "for Regular shoppers" in got["answer"]["headline"]


def test_how_many_regular_shoppers(retail):
    want = retail[0].con.execute(f"SELECT COUNT(*) FROM ({REPEAT_Q2})").fetchone()[0]
    got = _ask(retail, "How many Regular shoppers were there in Q2 2023?",
               {"measures": ["number_of_customers"], "time": {"window": Q2_2023}})
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    assert "Regular shoppers" in json.dumps(got["export_rows"][:1]) or "Regular shoppers" in got["answer"]["headline"]


# ── the metric editor and the Data guide ──────────────────────────────────


def test_an_admin_saves_edits_and_removes_it_and_readers_see_it(site):  # noqa: F811
    base = f"/admin/clients/{site.account}/measures"
    page = site.client.get(base).text
    assert 'id="segments"' in page and "Customers, by their orders" in page
    saved = site.client.post(f"{base}/api/segment/save", json=REGULAR).json()
    assert saved == {"ok": True, "key": "admin:regular_shoppers"}
    assert site.model().segments["admin:regular_shoppers"].kind == "repeat"
    page = site.client.get(base).text
    assert "Regular shoppers" in page and "Customers with orders on 2 or more days of the period asked" in page
    # The same name twice is refused; an edit keeps its key.
    assert "already a segment" in site.client.post(f"{base}/api/segment/save", json=REGULAR).json()["error"]
    edited = site.client.post(f"{base}/api/segment/save",
                              json={**REGULAR, "key": "admin:regular_shoppers", "kind": "one_time"}).json()
    assert edited["ok"] and site.model().segments["admin:regular_shoppers"].kind == "one_time"
    # Readers see it in the Data guide, with a question to try.
    guide = site.client.get("/portal/kb").text
    assert 'class="card ask-segments"' in guide and "Regular shoppers" in guide
    assert "Customers with orders on one day only of the period asked" in guide
    removed = site.client.post(f"{base}/api/segment/remove", json={"key": "admin:regular_shoppers"}).json()
    assert removed == {"ok": True} and not site.model().segments
    assert 'class="card ask-segments"' not in site.client.get("/portal/kb").text


def test_said_in_french_and_kept_by_its_name(retail):
    from core2.answer.builder import _condition_fr
    from core2.resolve.resolver import Condition

    assert _condition_fr(Condition("segment", "Regular shoppers", op="repeat", values=["customers"]), False) == \
        "pour Regular shoppers"
    assert _condition_fr(Condition("segment", "repeat customers", op="repeat", values=["customers"]), False) == \
        "pour les customers réguliers"
    from core2.plan.ir import Segment
    from core2.service import _kept_said

    before = Plan(kind="query", measures=["net_amount"], segment=Segment(of="customer", measure="number_of_orders",
                                                                         status="repeat", name="Regular shoppers"))
    after = before.model_copy(update={"follow_up": "refine", "group_by": ["store.name"]})
    assert "Regular shoppers" in _kept_said(after, before, "and by store?", retail[1])
