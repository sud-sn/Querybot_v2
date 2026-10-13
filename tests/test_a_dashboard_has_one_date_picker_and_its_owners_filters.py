"""A dashboard has one date picker over every tile, and its owner adds the filters it offers.

Asked for: a calendar picker on every dashboard, with all the calendars, on the current month unless chosen
otherwise; each chart may count by a different date, and the dates picked go to each chart's own date. And: more
filters on a dashboard, which its creator adds.

* the picker: presets by kind (days, weeks, months, quarters, years, and the fiscal ones where the fiscal year does
  not start in January), two days picked on its calendar, or each tile's own dates; this month unless the address
  says otherwise, and in the address, so a dashboard is shared as it is seen;
* each tile counts the dates picked on its own date (orders by order date, shipments by ship date); a chart over
  time shows them when they make a line of four points or more, else keeps its range with them shaded -- where
  the chart has them, counted back from where the other tiles counted theirs;
* where the data ends long ago, the picker counts its dates back from there as the tiles do, and says so;
* a field takes several members; the bar holds the fields the tiles offer and those the owner added (eight at
  most): fields the owner may read whose members may be listed, members a reader picks as they may see them;
* a schedule keeps the view it was saved with; one saved before the picker draws each tile in its own window.

Invented data only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

import pytest
from starlette.datastructures import QueryParams
from starlette.requests import Request

from portal import dashboard_view as dv
from tests.test_a_dashboard_reads_as_the_plan import APRIL, BY_REGION, TOTAL, _pin, _request
from tests.test_a_new_core_answer_can_be_pinned import _reader, _replay, fresh_store, retail  # noqa: F401

OCTOBER = dt.date(2026, 10, 13)
JUNE = dt.date(2026, 6, 15)
MONTHLY = {"intent": "trend", "measures": ["net_amount"],
           "time": {"grain": "month", "window": {"kind": "between", "start": "2025-07-01", "end": "2026-06-30"}}}


def _read(*pairs, fiscal_start=None) -> dict:
    return dv.read_view(QueryParams(urlencode(pairs)), fiscal_start=fiscal_start)


def _spans(picker: dict) -> dict:
    return {p["key"]: (p["start"], p["end"]) for g in picker["groups"] for p in g["presets"]}


# ── What the address asks for ───────────────────────────────────────────────

def test_a_dashboard_opens_on_this_month_and_its_address_says_otherwise():
    assert _read() == {"period": "this_month", "start": "", "end": "", "members": {}}
    assert _read(("period", "last_quarter"))["period"] == "last_quarter"
    assert _read(("period", "own"))["period"] == "own", "each tile's own window, as pinned"
    assert _read(("period", "next_decade"))["period"] == "this_month"
    assert _read(("period", "fiscal_year_to_date"))["period"] == "this_month", "no fiscal year known"
    assert _read(("period", "fiscal_year_to_date"), fiscal_start=1)["period"] == "this_month", "it is the calendar's"
    assert _read(("period", "fiscal_year_to_date"), fiscal_start=7)["period"] == "fiscal_year_to_date"


def test_two_days_picked_on_the_calendar_are_one_span_in_order():
    view = _read(("period", "custom"), ("from", "2026-03-15"), ("to", "2026-02-02"))
    assert (view["period"], view["start"], view["end"]) == ("custom", "2026-02-02", "2026-03-15")
    assert dv.period_window(view) == {"kind": "between", "start": "2026-02-02", "end": "2026-03-15"}
    assert _read(("period", "custom"), ("from", "2026-02-30"), ("to", "2026-03-01"))["period"] == "this_month"
    assert _read(("period", "custom"), ("from", "2026-02-01"))["period"] == "this_month", "both days or none"


def test_a_field_takes_several_members_and_the_bar_holds_twelve_fields():
    view = _read(("f.store", "Downtown Store"), ("f.store", "Eastview Store"), ("f.store", "Downtown Store"),
                 ("f.region.name", "North"), ("f.empty", " "))
    assert view["members"] == {"store": ["Downtown Store", "Eastview Store"], "region.name": ["North"]}
    assert len(_read(*[(f"f.field{i}", "x") for i in range(20)])["members"]) == dv.FIELD_FILTERS == 12
    assert len(_read(*[("f.store", f"s{i}") for i in range(80)])["members"]["store"]) == dv.MEMBER_VALUES


# ── On each tile ────────────────────────────────────────────────────────────

def test_the_dates_picked_go_on_each_tiles_own_date(retail):  # noqa: F811
    by_order = {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}}
    by_ship = {"intent": "value", "measures": ["net_amount"], "time": {"date": "ship_date", "window": APRIL}}
    view = {"period": "custom", "start": "2026-02-02", "end": "2026-03-15", "members": {}}
    replay = _replay(retail)
    drawn = []
    for plan in (by_order, by_ship):
        viewed, narrowed = dv.view_plan(plan, view)
        assert viewed["time"]["window"] == {"kind": "between", "start": "2026-02-02", "end": "2026-03-15"}
        assert viewed["time"].get("date") == plan["time"].get("date") and narrowed == []
        assert plan["time"]["window"] == APRIL, "the stored plan is left as it was"
        drawn.append(replay("acct", viewed, None))
    assert [d["trust"]["window"]["start"] for d in drawn] == ["2026-02-02"] * 2
    assert [d["trust"]["window"]["end"] for d in drawn] == ["2026-03-15"] * 2
    order_sql, ship_sql = (d["trust"]["sql"] for d in drawn)
    assert "ship_date" not in order_sql and "ship_date" in ship_sql, "each on its own date"
    assert drawn[0]["kpi"]["value"] != drawn[1]["kpi"]["value"], "orders placed then are not orders shipped then"


def test_a_chart_over_time_shows_the_dates_picked_when_they_make_a_line():
    assert dv.points(dv.DASH_PERIODS["last_12_months"], "month", today=JUNE) == 12
    assert dv.points(dv.DASH_PERIODS["last_quarter"], "month", today=JUNE) == 3
    assert dv.points(dv.DASH_PERIODS["this_month"], "day", today=JUNE) == 15
    assert dv.points({"kind": "between", "start": "2026-01-03", "end": "2026-02-20"}, "week", today=JUNE) == 8
    assert dv.view_plan(MONTHLY, {"period": "last_12_months", "today": JUNE})[0]["time"]["window"] == \
        dv.DASH_PERIODS["last_12_months"]
    assert dv.view_plan(MONTHLY, {"period": "last_quarter", "today": JUNE})[0] == MONTHLY, "three points: shaded"
    daily = {"intent": "trend", "measures": ["net_amount"],
             "time": {"grain": "day", "window": {"kind": "last", "unit": "day", "n": 90}}}
    assert dv.view_plan(daily, {"period": "this_month", "today": JUNE})[0]["time"]["window"] == \
        dv.DASH_PERIODS["this_month"]
    assert dv.view_plan(MONTHLY, {"period": "own"})[0] == MONTHLY
    assert dv.view_plan(MONTHLY, {"period": ""})[0] == MONTHLY


def test_a_member_filter_takes_several_members():
    viewed, narrowed = dv.view_plan(TOTAL, {"members": {"region.name": ["North", "West"], "store": "Downtown Store"}})
    assert viewed["filters"] == [{"field": "region.name", "op": "in", "values": ["North", "West"]},
                                 {"field": "store", "op": "in", "values": ["Downtown Store"]}]
    assert narrowed == ["region.name", "store"]
    assert dv.view_plan(BY_REGION, {"members": {"region.name": ["North"]}})[1] == [], "it offers the choices"


def test_a_band_is_shaded_where_the_chart_has_it_counted_as_the_other_tiles_count():
    year_2025 = {"x_key": "period", "chart_spec": {"x": {"role": "temporal"}},
                 "rows": [{"period": f"2025-{m:02d}-01"} for m in range(1, 13)]}
    view = {"period": "last_quarter", "today": OCTOBER}
    # From the chart's own last point (December 2025), when nothing else is known: July to September 2025.
    assert dv.shaded_period(view, year_2025) == {"start": "2025-07-01", "end": "2025-10-01"}
    # As the tiles counted theirs, from data that ends in June 2026: the first quarter of 2026, not on this chart.
    assert dv.shaded_period(view, year_2025, last_data=dt.date(2026, 6, 14)) is None
    to_september = {**year_2025, "rows": [{"period": f"{2025 + (m > 3)}-{(m + 8) % 12 + 1:02d}-01"}
                                          for m in range(1, 13)]}
    assert to_september["rows"][-1]["period"] == "2026-09-01"
    assert dv.shaded_period(view, to_september, last_data=None) == {"start": "2026-07-01", "end": "2026-10-01"}
    assert dv.shaded_period({"period": "own"}, to_september) is None


# ── The picker ─────────────────────────────────────────────────────────────

def test_the_picker_lists_presets_by_kind_and_the_fiscal_ones_where_the_year_is_not_the_calendars():
    plain = dv.picker(_read(), fiscal_start=None, lang="en", today=OCTOBER)
    assert [g["key"] for g in plain["groups"]] == ["days", "weeks", "months", "quarters", "years"]
    assert [g["key"] for g in dv.picker(_read(), fiscal_start=1, lang="en", today=OCTOBER)["groups"]][-1] == "years"
    july = dv.picker(_read(), fiscal_start=7, lang="en", today=OCTOBER)
    assert july["groups"][-1]["key"] == "fiscal" and july["groups"][-1]["label"] == "Fiscal year"
    spans = _spans(july)
    assert spans["today"] == ("2026-10-13", "2026-10-13") and spans["yesterday"] == ("2026-10-12", "2026-10-12")
    assert spans["last_7_days"] == ("2026-10-07", "2026-10-13")
    assert spans["last_week"] == ("2026-10-05", "2026-10-11"), "weeks start on Monday"
    assert spans["this_month"] == ("2026-10-01", "2026-10-13")
    assert spans["last_quarter"] == ("2026-07-01", "2026-09-30")
    assert spans["fiscal_year_to_date"] == ("2026-07-01", "2026-10-13")
    assert spans["last_fiscal_year"] == ("2025-07-01", "2026-06-30")
    assert (plain["period"], plain["label"], plain["span"], plain["note"]) == ("this_month", "This month",
                                                                              "1–13 Oct 2026", "")
    custom = _read(("period", "custom"), ("from", "2026-02-02"), ("to", "2026-03-15"))
    french = dv.picker(custom, fiscal_start=None, lang="fr", today=OCTOBER)
    assert (french["label"], french["span"]) == ("2 févr. – 15 mars 2026", "")
    assert (french["from"], french["to"]) == ("2026-02-02", "2026-03-15")
    assert dv.picker(_read(("period", "own")), fiscal_start=None, lang="en", today=OCTOBER)["label"] == \
        "Each chart's own dates"


def test_where_the_data_ends_long_ago_the_picker_counts_back_from_there_and_says_so():
    shown = dv.picker(_read(), fiscal_start=None, lang="en", today=OCTOBER, data_end="2026-06-14")
    assert (shown["span"], shown["from"], shown["to"]) == ("1–14 Jun 2026", "2026-06-01", "2026-06-14")
    assert shown["note"] and _spans(shown)["last_quarter"] == ("2026-01-01", "2026-03-31")
    custom = _read(("period", "custom"), ("from", "2026-02-02"), ("to", "2026-03-15"))
    assert not dv.picker(custom, fiscal_start=None, lang="en", today=OCTOBER, data_end="2026-06-14")["note"]


def test_a_span_is_written_as_short_as_it_reads():
    assert dv.span_label(dt.date(2026, 10, 1), dt.date(2026, 10, 13), "en") == "1–13 Oct 2026"
    assert dv.span_label(dt.date(2026, 1, 1), dt.date(2026, 3, 31), "en") == "1 Jan – 31 Mar 2026"
    assert dv.span_label(dt.date(2025, 12, 1), dt.date(2026, 1, 31), "en") == "1 Dec 2025 – 31 Jan 2026"
    assert dv.span_label(dt.date(2026, 5, 4), dt.date(2026, 5, 4), "en") == "4 May 2026"
    parts = dv.filter_parts({"period": "last_quarter", "members": {"region.name": ["North", "West"]}},
                            {"region.name": "Region"}, "en")
    assert dv.filters_line(parts) == "Period: Last quarter · Region: North, West"
    assert dv.filter_parts({"period": "own", "members": {}}, {}, "en") == []


# ── Kept with a schedule ────────────────────────────────────────────────────

def test_a_schedule_keeps_its_view_and_an_older_one_draws_each_tile_in_its_own_window():
    from core.dashboard_schedule import dashboard_link
    from core.dashboard_snapshot import _request as snapshot_request

    older = {"period": "", "members": {"region.name": "North"}}
    assert dv.clean_view(older) == {"period": "own", "start": "", "end": "", "members": {"region.name": ["North"]}}
    kept = {"period": "custom", "start": "2026-02-02", "end": "2026-03-15", "members": {"store": ["A", "B"]}}
    assert dv.clean_view(kept) == kept
    assert dv.clean_view({"period": "last_fiscal_year"})["period"] == "last_fiscal_year", "the admin's setting"
    assert dv.read_view(QueryParams(urlencode(dv.address(kept)))) == kept, "read back as it was kept"
    link = dashboard_link("https://bi.example.com", {"dashboard_id": 5, "view": kept})
    assert "period=custom&from=2026-02-02&to=2026-03-15&f.store=A&f.store=B" in link
    assert snapshot_request(5, "Overview", older, "en").query_params.get("period") == "own"


# ── The page ────────────────────────────────────────────────────────────────

def _page(store, user: dict, board: int, retail, pairs=(), snapshot=None) -> tuple[str, dict]:  # noqa: F811
    from portal import routes

    query = urlencode([("dashboard_id", str(board)), *pairs]).encode()
    request = Request({"type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
                       "server": ("portal.test", 80), "client": ("127.0.0.1", 0), "root_path": "",
                       "path": "/portal/dashboard", "raw_path": b"/portal/dashboard", "query_string": query,
                       "headers": [(b"host", b"portal.test")]})
    reader = {**store.get_user(user["id"]), **user}
    with patch("core2.service.portal_replay", _replay(retail)):
        response = asyncio.run(routes._render_dashboard(request, reader, snapshot=snapshot))
    return response.body.decode("utf-8"), dict(response.context)


def test_the_page_counts_this_month_unless_asked_and_hands_the_bar_its_state(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    tiles = _pin(fresh_store, retail, user, TOTAL, "net amount in april", new_dashboard_name="Mine")
    board = tiles[0]["dashboard_id"]
    _pin(fresh_store, retail, user, BY_REGION, "net amount by region in april", dashboard_id=board)

    html, page = _page(fresh_store, user, board, retail)
    own = _page(fresh_store, user, board, retail, [("period", "own")])[1]
    assert page["dash_view"]["period"] == "this_month" and own["dash_view"]["period"] == "own"
    assert page["charts"][0]["kpi"]["value"] != own["charts"][0]["kpi"]["value"], "June, not the April it was pinned"
    state = page["dash_filter_state"]
    assert state["picker"]["period"] == "this_month" and state["dashboard_id"] == board
    assert [(f["field"], f["source"]) for f in state["filters"]] == [("region.name", "tile")]
    assert 'id="dashFilterState"' in html and "dist/dashboard.js" in html and 'id="dashFilterApp"' in html

    two = _page(fresh_store, user, board, retail, [("period", "own"), ("f.region.name", "North"),
                                                    ("f.region.name", "West")])[1]
    regions = {r["region_name"]: r for r in json.loads(own["charts"][1]["chart_json"])["rows"]}
    assert two["charts"][0]["kpi"]["value"] == pytest.approx(
        regions["North"]["net_amount"] + regions["West"]["net_amount"])
    assert two["dash_filters_line"] == "Region: North, West"
    assert two["dash_filter_state"]["filters"][0]["selected"] == ["North", "West"]

    emailed, drawn = _page(fresh_store, user, board, retail, [("period", "last_month")], snapshot={"drawn": "Drawn"})
    assert "dist/dashboard.js" not in emailed and "Period: <strong>Last month</strong>" in emailed
    assert drawn["dash_filters_line"] == "Period: Last month"


def _owner(store) -> dict:
    """A creator of a workspace of their own, who keeps dashboards."""
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    db_id = store.save_db_config("azure_sql", "invented", {"server": "invented", "user": "u", "password": "p"})
    store.update_client_meta(account_id, db_config_id=db_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Owner", f"o{os.urandom(3).hex()}@example.com", role="analyst",
                                   password="owner-pass-123", portal_role="creator")
    return {**store.get_user(user_id), "id": user_id, "account_id": account_id}


def test_the_owner_adds_a_field_they_may_read_and_a_reader_picks_its_members(fresh_store, retail):  # noqa: F811
    from portal import routes

    built, model = retail
    owner = _owner(fresh_store)
    board = _pin(fresh_store, retail, owner, TOTAL, "net amount in april", new_dashboard_name="Mine")[0]["dashboard_id"]

    # The tables the owner may read: not the returns.
    readable = {"main.order_lines", "main.regions", "main.stores", "main.products", "main.categories"}

    def call(handler, *args, user=owner, **kwargs):
        with patch("core2.bootstrap.service.answering_model", return_value=model), \
                patch("core2.service.portal_replay", _replay(retail)), \
                patch.object(routes.store, "get_allowed_tables", return_value=readable), \
                patch.object(routes, "_get_portal_user", return_value=user):
            response = asyncio.run(handler(*args, **kwargs))
        return response.status_code, json.loads(response.body)

    status, offered = call(routes.dashboard_filter_fields_api, MagicMock(), board)
    assert status == 200 and offered["ok"]
    fields = {f["field"]: f for f in offered["fields"]}
    assert fields["region.name"]["label"] == "Region", "as the reader thinks of it"
    assert {"store.name", "product.name", "category.name"} <= set(fields)
    assert "return.reason_code" not in fields and "customer.name" not in fields, "only tables the owner may read"
    assert call(routes.dashboard_filters_api, _request({"action": "add", "field": "return.reason_code"}), board)[0] \
        == 400
    status, added = call(routes.dashboard_filters_api, _request({"action": "add", "field": "region.name",
                                                                 "label": "Sales region"}), board)
    assert status == 200 and added["filters"] == [{"kind": "member", "field": "region.name", "label": "Sales region"}]
    assert "region.name" not in {f["field"] for f in call(routes.dashboard_filter_fields_api, MagicMock(), board)[1]
                                 ["fields"]}, "not offered twice"
    assert call(routes.dashboard_filters_api, _request({"action": "add", "field": "no.such.field"}), board)[0] == 400
    with patch.object(dv, "CREATOR_FILTERS", 1):
        assert call(routes.dashboard_filters_api, _request({"action": "add", "field": "store.name"}), board)[0] == 400

    status, values = call(routes.dashboard_filter_values_api, MagicMock(), board, field="region.name", q="")
    assert status == 200 and values["values"] == sorted(values["values"]) and "North" in values["values"]
    status, found = call(routes.dashboard_filter_values_api, MagicMock(), board, field="region.name", q="nor")
    assert found["values"] == ["North"], "those holding what was typed, in any case"
    assert call(routes.dashboard_filter_values_api, MagicMock(), board, field="store.name")[0] == 404, \
        "only the dashboard's own filters"

    _, page = _page(fresh_store, owner, board, retail)
    assert {"field": "region.name", "label": "Sales region", "values": None, "source": "owner", "selected": []} in \
        page["dash_filter_state"]["filters"]
    assert page["dashboard_filters"] == [], "never the old filter form's"
    versions = fresh_store.list_dashboard_versions(board, owner["id"], owner["account_id"])
    assert any(v["change_summary"] == "Filter added: Sales region" for v in versions)

    stranger = _owner(fresh_store)
    assert call(routes.dashboard_filters_api, _request({"action": "remove", "field": "region.name"}), board,
                user=stranger)[0] == 404, "its owner alone"
    assert call(routes.dashboard_filter_values_api, MagicMock(), board, field="region.name", user=stranger)[0] == 404
    status, removed = call(routes.dashboard_filters_api, _request({"action": "remove", "field": "region.name"}), board)
    assert status == 200 and removed["filters"] == []


def test_the_owners_filter_routes_need_a_reader_who_keeps_dashboards():
    from portal.access import action_for

    assert action_for("GET", "/api/dashboard/5/filter-fields") == "own_dashboards"
    assert action_for("POST", "/api/dashboard/5/filters") == "own_dashboards"
    assert action_for("GET", "/api/dashboard/5/filter-values") is None, "whoever may open the dashboard"
