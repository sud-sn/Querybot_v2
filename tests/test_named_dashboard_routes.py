import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import portal.routes as routes


USER = {"id": 7, "account_id": "tenant-a", "name": "Analyst"}
PIN = {
    "user_id": 7,
    "account_id": "tenant-a",
    "question": "Revenue by state",
    "sql_query": "SELECT state, SUM(revenue) AS revenue FROM sales GROUP BY state",
    "chart_type": "pie",
    "db_config_id": 4,
}


def _request(payload):
    request = MagicMock()
    request.json = AsyncMock(return_value=payload)
    return request


def _json(response):
    return json.loads(response.body.decode("utf-8"))


def test_add_result_requires_named_dashboard_without_consuming_token():
    with (
        patch.object(routes, "_get_portal_user", return_value=USER),
        patch.object(routes, "_peek_pin_token", return_value=PIN),
        patch.object(routes, "_consume_pin_token") as consume,
    ):
        response = asyncio.run(routes.pin_chart_api(_request({"token": "opaque"})))
    assert response.status_code == 400
    assert "Choose an existing dashboard" in _json(response)["error"]
    consume.assert_not_called()


def test_add_result_targets_owned_dashboard_and_reuses_governed_sql():
    dashboard = {"id": 31, "name": "Pharmacy performance"}
    with (
        patch.object(routes, "_get_portal_user", return_value=USER),
        patch.object(routes, "_peek_pin_token", return_value=PIN),
        patch.object(routes, "_consume_pin_token", return_value=PIN),
        patch.object(routes.store, "get_dashboard", return_value=dashboard),
        patch.object(routes.store, "create_data_source", return_value={"id": 51}) as source,
        patch.object(routes.store, "pin_chart", return_value=71),
        patch.object(routes.store, "add_chart_to_dashboard", return_value=True) as add,
    ):
        response = asyncio.run(routes.pin_chart_api(_request({
            "token": "opaque",
            "dashboard_id": 31,
            "title": "Revenue distribution",
            "chart_type": "pie",
        })))
    data = _json(response)
    assert response.status_code == 200
    assert data["dashboard"]["id"] == 31
    assert source.call_args.kwargs["sql_query"] == PIN["sql_query"]
    assert "rows" not in source.call_args.kwargs
    add.assert_called_once_with(
        31, 71, 7, "tenant-a", data_source_id=51, tab="Overview"
    )


def test_add_result_can_create_dashboard_and_add_in_one_action():
    created = {"id": 44, "name": "Executive overview"}
    with (
        patch.object(routes, "_get_portal_user", return_value=USER),
        patch.object(routes, "_peek_pin_token", return_value=PIN),
        patch.object(routes, "_consume_pin_token", return_value=PIN),
        patch.object(routes.store, "create_dashboard", return_value=created) as create,
        patch.object(routes.store, "create_data_source", return_value={"id": 52}),
        patch.object(routes.store, "pin_chart", return_value=72),
        patch.object(routes.store, "add_chart_to_dashboard", return_value=True),
    ):
        response = asyncio.run(routes.pin_chart_api(_request({
            "token": "opaque",
            "new_dashboard_name": "Executive overview",
            "visibility": "team",
        })))
    assert response.status_code == 200
    assert _json(response)["dashboard"]["url"].endswith("dashboard_id=44")
    assert create.call_args.kwargs["visibility"] == "team"


def test_cross_workspace_result_is_rejected_before_consumption():
    foreign = {**PIN, "account_id": "tenant-b"}
    with (
        patch.object(routes, "_get_portal_user", return_value=USER),
        patch.object(routes, "_peek_pin_token", return_value=foreign),
        patch.object(routes, "_consume_pin_token") as consume,
    ):
        response = asyncio.run(routes.pin_chart_api(_request({
            "token": "opaque", "dashboard_id": 31,
        })))
    assert response.status_code == 403
    consume.assert_not_called()
