# -*- coding: utf-8 -*-
"""Why a total moved: a bridge, and price, volume and mix.

"Why did revenue move?" had one answer an investigation could compute: the
`compare` tool, a list of each member's change between two results. A list of
changes does not add up to anything the reader can check, and it says nothing
about whether revenue moved because more was sold, because the mix shifted
towards dearer lines, or because prices changed.

core/variance_bridge.py: a bridge from the first total to the second -- each
member's move, the rest together -- that lands exactly on it; and, when both
results carry a quantity beside the value, the move split into volume, mix,
price, new and lost. Both are investigation tools (core.investigation_tools),
computed on rows already released to the reader: no SQL, no model.

Synthetic rows; no customer data.
"""

from __future__ import annotations

import asyncio
import os
import random

import pytest

from core.variance_bridge import BridgeRefused, price_volume_mix, variance_bridge

LAST_YEAR = [{"REGION": "North", "REVENUE": 400.0}, {"REGION": "South", "REVENUE": 350.0},
             {"REGION": "East", "REVENUE": 150.0}, {"REGION": "West", "REVENUE": 100.0}]
THIS_YEAR = [{"REGION": "North", "REVENUE": 500.0}, {"REGION": "South", "REVENUE": 300.0},
             {"REGION": "East", "REVENUE": 150.0}, {"REGION": "West", "REVENUE": 50.0},
             {"REGION": "Central", "REVENUE": 20.0}]

# Item A: 10 units at 10 -> 12 units at 11. Item B: 10 at 20 -> 8 at 20.
# C was sold last year only, D this year only.
BEFORE = [{"ITEM": "A", "QTY": 10.0, "NET_AMT": 100.0}, {"ITEM": "B", "QTY": 10.0, "NET_AMT": 200.0},
          {"ITEM": "C", "QTY": 5.0, "NET_AMT": 50.0}]
AFTER = [{"ITEM": "A", "QTY": 12.0, "NET_AMT": 132.0}, {"ITEM": "B", "QTY": 8.0, "NET_AMT": 160.0},
         {"ITEM": "D", "QTY": 3.0, "NET_AMT": 45.0}]


class TestTheBridge:

    def test_it_lands_exactly_on_the_second_total(self):
        bridge = variance_bridge(LAST_YEAR, THIS_YEAR, member="REGION", value="REVENUE", items=2)
        assert (bridge.start, bridge.end, bridge.change) == (1000.0, 1020.0, 20.0)
        assert bridge.steps == (("North", 100.0), ("South", -50.0))
        assert (bridge.rest, bridge.rest_members) == (-30.0, 2)
        assert bridge.start + sum(move for _, move in bridge.steps) + bridge.rest == bridge.end

    def test_it_says_how_much_of_the_movement_the_named_members_explain(self):
        bridge = variance_bridge(LAST_YEAR, THIS_YEAR, member="REGION", value="REVENUE", items=2)
        # |100| + |-50| of |100| + |-50| + |-50| + |+20| gross.
        assert bridge.explained_share == pytest.approx(150 / 220)

    def test_a_member_on_one_side_moves_from_or_to_nothing(self):
        bridge = variance_bridge(LAST_YEAR, THIS_YEAR, member="REGION", value="REVENUE")
        assert dict(bridge.steps)["Central"] == 20.0

    @pytest.mark.parametrize("measure", ["MARGIN_PCT", "AVG_SELL_PRICE", "UNIT_CST"])
    def test_a_measure_that_does_not_add_up_is_refused(self, measure):
        rows = [{"REGION": "North", measure: 1.0}]
        with pytest.raises(BridgeRefused) as refused:
            variance_bridge(rows, rows, member="REGION", value=measure)
        assert refused.value.reason == "non_additive"

    def test_a_result_with_two_rows_for_a_member_is_refused(self):
        twice = LAST_YEAR + [{"REGION": "North", "REVENUE": 1.0}]
        with pytest.raises(BridgeRefused) as refused:
            variance_bridge(twice, THIS_YEAR, member="REGION", value="REVENUE")
        assert refused.value.reason == "not_one_row_per_member"


class TestPriceVolumeMix:

    def test_the_move_is_split_exactly(self):
        split = price_volume_mix(BEFORE, AFTER, member="ITEM", quantity="QTY", value="NET_AMT")
        assert split.parts() == {"volume": 0.0, "mix": -20.0, "price": 12.0,
                                 "new": 45.0, "lost": -50.0, "other": 0.0}
        assert sum(split.parts().values()) == split.change == -13.0
        # B lost weight, A gained it; only A's own price moved.
        assert split.mix_drivers == (("B", -40.0), ("A", 20.0))
        assert split.price_drivers == (("A", 12.0),)

    def test_twice_the_units_at_the_same_prices_is_all_volume(self):
        doubled = [{**row, "QTY": row["QTY"] * 2, "NET_AMT": row["NET_AMT"] * 2} for row in BEFORE]
        split = price_volume_mix(BEFORE, doubled, member="ITEM", quantity="QTY", value="NET_AMT")
        assert split.volume == pytest.approx(350.0)
        assert split.mix == pytest.approx(0.0) and split.price == pytest.approx(0.0)

    def test_the_same_units_ten_percent_dearer_is_all_price(self):
        dearer = [{**row, "NET_AMT": row["NET_AMT"] * 1.1} for row in BEFORE]
        split = price_volume_mix(BEFORE, dearer, member="ITEM", quantity="QTY", value="NET_AMT")
        assert split.price == pytest.approx(35.0)
        assert split.volume == pytest.approx(0.0) and split.mix == pytest.approx(0.0)

    def test_the_same_units_moved_to_the_dearer_item_is_all_mix(self):
        shifted = [{"ITEM": "A", "QTY": 5.0, "NET_AMT": 50.0}, {"ITEM": "B", "QTY": 15.0, "NET_AMT": 300.0},
                   {"ITEM": "C", "QTY": 5.0, "NET_AMT": 50.0}]
        split = price_volume_mix(BEFORE, shifted, member="ITEM", quantity="QTY", value="NET_AMT")
        assert split.mix == pytest.approx(50.0)
        assert split.volume == pytest.approx(0.0) and split.price == pytest.approx(0.0)

    def test_a_member_with_no_units_on_one_side_has_no_price(self):
        before = BEFORE + [{"ITEM": "E", "QTY": 0.0, "NET_AMT": 5.0}]
        after = AFTER + [{"ITEM": "E", "QTY": 2.0, "NET_AMT": 9.0}]
        split = price_volume_mix(before, after, member="ITEM", quantity="QTY", value="NET_AMT")
        assert split.other == 4.0
        assert sum(split.parts().values()) == pytest.approx(split.change)

    def test_the_parts_always_sum_to_the_move(self):
        rng = random.Random(20260925)
        for _ in range(300):
            members = [f"M{n}" for n in range(rng.randint(1, 8))]

            def side(present: float) -> list[dict]:
                return [{"ITEM": m, "QTY": float(rng.randint(0, 50)), "NET_AMT": round(rng.uniform(0, 900), 2)}
                        for m in members if rng.random() < present]

            before, after = side(0.85), side(0.85)
            try:
                split = price_volume_mix(before, after, member="ITEM", quantity="QTY", value="NET_AMT")
            except BridgeRefused:
                continue
            assert sum(split.parts().values()) == pytest.approx(split.change, abs=1e-6)

    def test_a_stock_value_splits_like_revenue(self):
        """A period-end balance adds up across items at one point in time."""
        before = [{"ITM_CD": r["ITEM"], "ON_HND_QTY": r["QTY"], "STK_VAL_AMT": r["NET_AMT"]} for r in BEFORE]
        after = [{"ITM_CD": r["ITEM"], "ON_HND_QTY": r["QTY"], "STK_VAL_AMT": r["NET_AMT"]} for r in AFTER]
        split = price_volume_mix(before, after, member="ITM_CD", quantity="ON_HND_QTY", value="STK_VAL_AMT")
        assert split.change == -13.0 and split.price == 12.0


# ── The tools, through the investigation's own entry point ──────────────────


def _run(results: dict[str, list[dict]]):
    """An investigation whose steps already hold these results, in the cache
    the tools read."""
    import store
    from core.investigation import ToolResult, investigation_session_id
    from core.investigation_planner import InvestigationStep
    from core.result_cache import result_cache

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    user = {"id": 7, "name": "Ada", "role": "viewer"}
    run_id = f"run-{os.urandom(3).hex()}"
    session = investigation_session_id(account, user, run_id)
    steps = []
    for index, (question, rows) in enumerate(results.items(), start=1):
        result_id = result_cache.store(session, rows, question, "SELECT 1")
        steps.append(InvestigationStep(index=index, question=question, result=ToolResult(
            ok=True, kind="query", question=question, result_id=result_id, row_count=len(rows),
            columns=tuple(rows[0]))))

    def tool(name: str, inputs: dict, lang: str = "en"):
        from core.investigation_tools import run_tool

        result = asyncio.run(run_tool(name, inputs, account_id=account, portal_user=user, run_id=run_id,
                                      steps=steps, lang=lang))
        kept = result_cache.get_snapshot(session, result.result_id) if result.result_id else {}
        return result, kept.get("rows") or []

    return tool


class TestTheTools:

    def test_the_bridge_tool_explains_the_move_in_figures(self):
        tool = _run({"revenue by region last year": LAST_YEAR, "revenue by region this year": THIS_YEAR})
        result, rows = tool("bridge", {"step": 1, "other_step": 2})
        assert result.ok, result.error
        assert "REVENUE went from 1,000 at step 1 to 1,020 at step 2, up 20" in result.brief
        assert "North up 100; South down 50; West down 50; Central up 20" in result.brief
        assert [row["kind"] for row in rows] == ["start", "member", "member", "member", "member", "end"]
        assert sum(row["amount"] for row in rows[:-1]) == rows[-1]["amount"]

    def test_a_member_column_named_differently_is_still_one_member(self):
        renamed = [{"REGION_NM": row["REGION"], "REVENUE": row["REVENUE"]} for row in THIS_YEAR]
        tool = _run({"last year": LAST_YEAR, "this year": renamed})
        result, _rows = tool("bridge", {"step": 1, "other_step": 2})
        assert result.ok and "North up 100" in result.brief

    def test_a_percentage_gets_no_bridge_and_says_why(self):
        margins = [{"REGION": "North", "MARGIN_PCT": 12.0}, {"REGION": "South", "MARGIN_PCT": 9.0}]
        tool = _run({"margin last year": margins, "margin this year": margins})
        result, _rows = tool("bridge", {"step": 1, "other_step": 2})
        assert not result.ok and "MARGIN_PCT does not add up across members" in result.error

    def test_the_price_volume_mix_tool_names_each_part(self):
        tool = _run({"sales by item last year": BEFORE, "sales by item this year": AFTER})
        result, rows = tool("price_volume_mix", {"step": 1, "other_step": 2, "quantity": "QTY"})
        assert result.ok, result.error
        assert ("NET_AMT went from 350 at step 1 to 337 at step 2, down 13. Volume unchanged, mix down 20, "
                "price up 12, new members up 45, lost members down 50, other unchanged.") in result.brief
        assert ("The price moved most at A (up 12); the mix shifted most at B (down 40), A (up 20)."
                in result.brief)
        assert {row["kind"]: row["amount"] for row in rows} == {
            "volume": 0.0, "mix": -20.0, "price": 12.0, "new": 45.0, "lost": -50.0}

    def test_without_a_quantity_there_is_no_split(self):
        tool = _run({"a": BEFORE, "b": AFTER})
        result, _rows = tool("price_volume_mix", {"step": 1, "other_step": 2})
        assert not result.ok and "quantity" in result.error

    def test_a_french_reader_gets_a_french_brief(self):
        tool = _run({"a": BEFORE, "b": AFTER})
        result, _rows = tool("price_volume_mix", {"step": 1, "other_step": 2, "quantity": "QTY"}, lang="fr")
        assert "NET_AMT est passé de 350 à l'étape 1 à 337 à l'étape 2, en baisse de 13" in result.brief
        assert "prix en hausse de 12" in result.brief

    def test_the_planner_is_offered_both(self):
        from core.investigation_tools import describe_tools

        catalogue = describe_tools("en")
        assert "- bridge: Bridge one step's total to another's" in catalogue
        assert "- price_volume_mix: Split the move between two steps into volume, mix and price" in catalogue
        assert "quantity (the quantity column" in catalogue


# ── A whole investigation: asked, split, summarised, and the summary checked ─


class _Warehouse:
    """dispatch, replaced at its boundary: each known question's rows go out
    through the real result renderer, which caches them as a real answer is."""

    ANSWERS = {"sales by item last year": BEFORE, "sales by item this year": AFTER}

    async def dispatch(self, account_id, event, adapter, _background, portal_user=None):
        from core.result_renderer import _send_results

        rows = self.ANSWERS.get(event.text)
        if rows is None:
            await adapter.send_message(event, "No records matched the filters.")
            return
        await _send_results(event, adapter, event.text, [dict(row) for row in rows],
                            "SELECT ITEM, SUM(QTY) AS QTY, SUM(NET_AMT) AS NET_AMT FROM SALES GROUP BY ITEM",
                            3, portal_user, account_id, {"id": 0, "db_type": "azure_sql"})


def _investigate(synthesis: str):
    from unittest.mock import patch

    import core.compliance.policy_engine as pe
    import core.dispatcher as dispatcher
    import store
    from core.investigation_planner import run_investigation

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    replies = [
        '{"action": "query", "question": "sales by item this year"}',
        '{"action": "tool", "tool": "price_volume_mix", "inputs": {"step": 1, "other_step": 2, "quantity": "QTY"}}',
        '{"action": "finish", "synthesis": %s}' % __import__("json").dumps(synthesis),
    ]

    async def planner(system="", user="", *args, **kwargs):
        return replies.pop(0), 10, 10

    with patch.object(dispatcher, "dispatch", _Warehouse().dispatch), \
            patch.object(pe, "result_llm_features_allowed", lambda account_id: True):
        return asyncio.run(run_investigation(
            objective="sales by item last year", account_id=account,
            portal_user={"id": 7, "name": "Ada", "role": "viewer"}, run_id=f"run-{os.urandom(3).hex()}",
            max_steps=4, complete=planner))


class TestAnInvestigationExplainsTheMove:

    def test_a_summary_citing_the_split_is_checked_against_it_and_kept(self):
        outcome = _investigate("Sales fell 13, from 350 to 337: prices added 12 and the shift towards "
                               "the cheaper item cost 20; a new item brought 45 and a lost one took 50.")
        assert [step.result.kind for step in outcome.steps] == ["query", "query", "price_volume_mix"]
        assert outcome.phrasing == "llm", outcome.synthesis

    def test_a_figure_the_split_did_not_compute_is_refused(self):
        outcome = _investigate("Sales fell 13: prices added 18.")
        assert outcome.phrasing == "template"
