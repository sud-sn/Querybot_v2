"""A link's role is named, renamed and taken away on the link itself (Admin → Data → Relationships).

Taking a learned link's role away saved a link of the admin's own on the same columns, whose key was the learned
link's own (a learned key holds no role), and then turned the learned link off: the turning off took every decision
on that key with it, the one just saved included. The link kept its role, and lost the condition an admin had set on
it. Now a role named, changed or taken away on the same tables and columns is a decision on that link: it keeps its
key, its conditions and its confirmation, holds through the next Learn, and a question follows it as the plain link
when its role is gone. Another link on columns already linked is refused (pick that link to change it).

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import os
from unittest.mock import patch

import pytest

from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.fixture
def page(retail, tmp_path, monkeypatch):
    built, model = retail
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_relationships, core2_routes, routes
    from core2.warehouse.runner import DuckDBWarehouse

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Retail (invented)")
    store.save_core2_model(account, None, model.model_dump_json())
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_routes, "_is_auth", return_value=True), \
            patch.object(core2_relationships, "_is_auth", return_value=True), \
            patch.object(core2_relationships, "_warehouse", lambda *a: DuckDBWarehouse(built.con)):
        yield TestClient(app), account, store


def _model(account):
    from core2.bootstrap.service import load_model

    return load_model(account, None)


def _link(model, source: str, target: str):
    return next(j for j in model.joins.values()
                if j.from_table.endswith(f".{source}") and j.to_table.endswith(f".{target}") and not j.to_calendar)


def _column(model, table: str, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(f".{table}") and c.name == name)


def _body(link, **change) -> dict:
    return {"key": link.key, "from": link.from_table, "to": link.to_table, "role": link.role or "",
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)],
            "conditions": [c.model_dump(mode="json") for c in link.conditions], **change}


def _save(client, account, body) -> dict:
    return client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json()


def test_a_learned_links_role_is_taken_away_and_the_link_kept_with_its_condition(page):
    client, account, _ = page
    model = _model(account)
    home = _link(model, "customers", "stores")
    assert home.role == "Home store"
    # The admin had kept the link to open stores only, then takes its role away.
    code = _column(model, "stores", "store_code")
    kept = [{"column": code, "op": "ne", "values": ["NONE"]}]
    assert _save(client, account, _body(home, conditions=kept))["ok"]
    saved = _save(client, account, _body(_model(account).joins[home.key], role=""))
    assert saved == {"ok": True, "key": home.key}
    after = _model(account)
    link = after.joins[home.key]
    assert link.role is None and link.trust == "admin"
    assert [c.model_dump(mode="json")["values"] for c in link.conditions] == [["NONE"]]
    assert [j.key for j in after.joins.values() if j.from_table == home.from_table and j.to_table == home.to_table
            and j.trust != "rejected"] == [home.key]


def test_a_plain_links_role_is_named_renamed_and_taken_away(page):
    client, account, _ = page
    link = _link(_model(account), "order_lines", "customers")
    assert link.role is None
    for role in ("Buyer", "Sold to customer", ""):
        assert _save(client, account, _body(_model(account).joins[link.key], role=role)) == {"ok": True,
                                                                                            "key": link.key}
        after = _model(account).joins[link.key]
        assert after.role == (role or None) and after.trust == "admin"


def test_a_role_taken_away_holds_through_the_next_learn(page):
    client, account, store = page
    model = _model(account)
    home = _link(model, "customers", "stores")
    _save(client, account, _body(home, role=""))
    # Learn again: the model it learns is saved over the old one, and the admin's decisions are read on top.
    store.save_core2_model(account, None, model.model_dump_json())
    assert _model(account).joins[home.key].role is None


def test_another_link_on_columns_already_linked_is_refused(page):
    client, account, _ = page
    link = _link(_model(account), "order_lines", "customers")
    r = client.post(f"/admin/clients/{account}/relationships/save-link", json=_body(link, key="", role="Buyer"))
    assert r.status_code == 400 and "already linked" in r.json()["error"]
