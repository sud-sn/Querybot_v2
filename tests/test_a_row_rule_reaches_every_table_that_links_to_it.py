"""A row rule written on one table narrows every table that points at it, for the people it is written for.

Row access was written per table: a rule on the customers table ("Gold customers only") narrowed a question only
where it named that table. "Total order amount" reads the orders table alone, never naming customers, and
answered with every customer's orders. A rule now reaches each table linked to its own (orders -> customers),
through the learned model's links. Rules are kept by the workspace whatever compliance policy version is active,
can be for everyone, a role, a group or one person, can read a person's own value ("their segment"), and can lift
a field for some ("managers see every segment").

Invented data only: twenty customers in three segments, six hundred orders.
"""

from __future__ import annotations

import types
import uuid

import pytest
import sqlglot

TOTAL = "SELECT SUM(order_amount) AS total FROM orders"
SEGMENTS = ("Gold", "Silver", "Bronze")


@pytest.fixture
def shop(tmp_path, monkeypatch):
    import duckdb
    import numpy as np
    import pandas as pd

    import store
    import store.crypto
    from core.compliance import governed_query as gq
    from core.schema import load_known_tables
    from core2.bootstrap.service import build_workspace
    from core2.warehouse import querybot
    from core2.warehouse.governed import GovernedWarehouse
    from tests.test_core2_learned_page_shows_what_was_learned import _Connection, _schema_file

    rng = np.random.default_rng(7)
    con = duckdb.connect()
    n = 600
    frames = {
        "customers": pd.DataFrame({"customer_id": np.arange(1, 21),
                                   "customer_name": [f"Customer {i:02d}" for i in range(1, 21)],
                                   "segment": [SEGMENTS[i % 3] for i in range(20)]}),
        "orders": pd.DataFrame({"order_id": np.arange(1, n + 1), "customer_id": rng.integers(1, 21, n),
                                "region": np.where(rng.random(n) < 0.5, "North", "South"),
                                "order_amount": np.round(rng.uniform(10, 900, n), 2)}),
    }
    for name, frame in frames.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    built = types.SimpleNamespace(con=con, tables={t: t for t in frames}, declared_pks={}, declared_fks=[])

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = f"acct-rows-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "web")
    store.save_compliance_profile(account, mode="standard")
    db_id = store.save_db_config("azure_sql", "shop", {"server": "s", "database": "d", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    schema_dir = tmp_path / "clients" / account / "schema"
    store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(con))
    build_workspace(account)

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    known = load_known_tables(str(schema_dir))

    def person(name: str, *, group: int | None = None, role: str = "analyst") -> dict:
        uid = store.create_user(account, name, f"{uuid.uuid4().hex[:8]}@test.com", group_id=group, role=role,
                                password="a-password-they-chose")[0]
        return store.get_user(uid)

    def total(reader: dict, sql: str = TOTAL) -> float:
        governed = GovernedWarehouse(account, reader, store.get_db_config(db_id), known_tables=known)
        rows = governed.query(sql).rows
        return round(float(rows[0][0] or 0), 2)

    def truth(where: str = "") -> float:
        sql = ("SELECT SUM(o.order_amount) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
               + (f"WHERE {where}" if where else ""))
        return round(float(con.execute(sql).fetchone()[0] or 0), 2)

    return types.SimpleNamespace(account=account, person=person, total=total, truth=truth, store=store, con=con)


def _rule(who: str, whom="", *, field="segment", table="CUSTOMERS", values=None, own=None, every=False) -> dict:
    condition = {"field": field, "operator": "ALL" if every else "IN"}
    if own:
        condition["value_source"] = f"user.attributes.{own}"
    elif not every:
        condition["value_source"], condition["value"] = "static", values
    return {"subject_type": who, "subject_id": str(whom), "table_fqn": table, "condition": condition}


def test_a_rule_on_customers_narrows_a_question_on_orders_alone(shop):
    ada = shop.person("Ada")
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    assert shop.total(ada) == shop.truth("c.segment = 'Gold'") < shop.truth()
    assert shop.total(shop.person("Ben")) == shop.truth(), "the rule is Ada's alone"


def test_the_rules_on_one_field_are_ways_in_and_the_fields_each_a_limit(shop):
    west = shop.store.create_group(shop.account, "Silver team")
    ada = shop.person("Ada", group=west)
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    shop.store.save_row_rule(shop.account, _rule("group", west, values=["Silver"]))
    assert shop.total(ada) == shop.truth("c.segment IN ('Gold', 'Silver')")
    shop.store.save_row_rule(shop.account, _rule("everyone", field="region", table="ORDERS", values=["North"]))
    assert shop.total(ada) == shop.truth("c.segment IN ('Gold', 'Silver') AND o.region = 'North'")


def test_a_rule_that_sees_every_row_lifts_its_field_for_the_people_it_names(shop):
    managers = shop.store.create_group(shop.account, "Managers")
    shop.store.save_row_rule(shop.account, _rule("everyone", values=["Bronze"]))
    shop.store.save_row_rule(shop.account, _rule("group", managers, every=True))
    assert shop.total(shop.person("Cy", group=managers)) == shop.truth()
    assert shop.total(shop.person("Di")) == shop.truth("c.segment = 'Bronze'")


def test_a_rule_reads_the_persons_own_value_and_no_value_is_no_row(shop):
    shop.store.save_row_rule(shop.account, _rule("role", "analyst", own="segment"))
    ada, ben, cy = shop.person("Ada"), shop.person("Ben"), shop.person("Cy")
    shop.store.set_user_attributes(ada["id"], {"segment": "Gold"})
    shop.store.set_user_attributes(ben["id"], {"segment": "Silver, Bronze"})
    assert shop.total(shop.store.get_user(ada["id"])) == shop.truth("c.segment = 'Gold'")
    assert shop.total(shop.store.get_user(ben["id"])) == shop.truth("c.segment IN ('Silver', 'Bronze')")
    assert shop.total(cy) == 0, "a person with no segment of their own sees none of the rows, never all of them"


def test_a_rule_on_orders_never_hides_a_customer_from_the_list_of_customers(shop):
    ada = shop.person("Ada")
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], field="region", table="ORDERS", values=["North"]))
    assert shop.total(ada, "SELECT COUNT(*) AS n FROM customers") == 20


def test_the_workspaces_rules_hold_whatever_policy_version_is_active(shop):
    """Saving the compliance page's rules or a pack makes a new policy version: rules kept with the old one stopped
    applying. The workspace's own rules hold under every version, and a version's rules go on into the next."""
    ada = shop.person("Ada")
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    first = shop.store.create_policy_version(shop.account, {"rules": []}, change_summary="first")
    shop.store.activate_policy_version(shop.account, first)
    shop.store.replace_row_policies(shop.account, first, [_rule("user", ada["id"], field="region", table="ORDERS",
                                                                values=["North"])])
    assert shop.total(ada) == shop.truth("c.segment = 'Gold' AND o.region = 'North'")
    second = shop.store.create_policy_version(shop.account, {"rules": []}, change_summary="second")
    shop.store.activate_policy_version(shop.account, second)
    assert shop.total(ada) == shop.truth("c.segment = 'Gold' AND o.region = 'North'")


def test_two_readers_under_one_rule_never_share_a_kept_figure(shop):
    """A figure kept for one reader is kept under their rows' fingerprint: one that reaches orders through
    customers counts, though the question names orders alone."""
    from core.compliance.policy_engine import resolve_context
    from core.compliance.sql_guard import row_policy_scope

    shop.store.save_row_rule(shop.account, _rule("everyone", own="segment"))
    ada, ben = shop.person("Ada"), shop.person("Ben")
    shop.store.set_user_attributes(ada["id"], {"segment": "Gold"})
    shop.store.set_user_attributes(ben["id"], {"segment": "Silver"})
    scope = {who["id"]: row_policy_scope(resolve_context(shop.account, shop.store.get_user(who["id"]),
                                                         action="query_execution"), ["ORDERS"])
             for who in (ada, ben)}
    assert scope[ada["id"]] and scope[ben["id"]] and scope[ada["id"]] != scope[ben["id"]]


def test_a_rule_that_does_not_say_who_where_or_what_is_refused(shop):
    for bad in (_rule("nobody", 1, values=["Gold"]), _rule("user", "", values=["Gold"]),
                _rule("everyone", values=[]), _rule("everyone", field="", values=["Gold"])):
        with pytest.raises(ValueError):
            shop.store.save_row_rule(shop.account, bad)
    assert shop.store.list_row_rules(shop.account) == []


def test_a_rule_is_changed_and_removed_in_place(shop):
    ada = shop.person("Ada")
    rule_id = shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Silver"]), rule_id)
    assert shop.total(ada) == shop.truth("c.segment = 'Silver'")
    assert shop.store.delete_row_rule(shop.account, rule_id)
    assert shop.total(ada) == shop.truth() and shop.store.list_row_rules(shop.account) == []


# ── the admin's page ────────────────────────────────────────────────────────

def _call(handler, path: str, *args, body: dict | None = None, method: str = "POST", query: str = "", **kw):
    import asyncio
    import json
    from unittest.mock import patch

    from starlette.requests import Request

    from admin import row_access

    raw = json.dumps(body or {}).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request({"type": "http", "method": method, "path": path, "root_path": "", "scheme": "http",
                       "query_string": query.encode(), "server": ("testserver", 80), "client": ("127.0.0.1", 1),
                       "headers": [(b"content-type", b"application/json"),
                                   (b"content-length", str(len(raw)).encode())]}, receive)
    with patch.object(row_access, "_is_auth", return_value=True):
        response = asyncio.run(handler(request, *args, **kw))
    if response.media_type == "application/json":
        return json.loads(response.body)
    import html as _html
    return _html.unescape(response.body.decode())


def _segment_field(shop) -> dict:
    from admin import row_access
    from core2.bootstrap.service import load_model

    model = load_model(shop.account, shop.store.get_client(shop.account)["db_config_id"])
    return next(f for f in row_access.fields_of(model) if f["field"] == "segment")


def test_an_admin_writes_a_rule_and_reads_what_each_person_sees(shop):
    from admin import row_access

    team = shop.store.create_group(shop.account, "Gold team")
    ada, ben = shop.person("Ada", group=team), shop.person("Ben")
    field = _segment_field(shop)
    saved = _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
                  body={"who": "group", "whom": str(team), "table": field["table"], "field": "segment",
                        "sees": "values", "values": "Gold"})
    assert saved["ok"]
    own = _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
                body={"who": "user", "whom": str(ben["id"]), "table": field["table"], "field": "segment",
                      "sees": "own", "key": "segment"})
    assert own["ok"]
    assert shop.total(ada) == shop.truth("c.segment = 'Gold'")
    assert shop.total(ben) == 0, "Ben has no segment of his own yet"
    kept = _call(row_access.row_person_values, f"/admin/clients/{shop.account}/row-access/api/people/{ben['id']}",
                 shop.account, ben["id"], body={"values": {"segment": "Silver, Bronze"}})
    assert kept["ok"] and kept["values"] == {"segment": ["Silver", "Bronze"]}
    assert shop.total(shop.store.get_user(ben["id"])) == shop.truth("c.segment IN ('Silver', 'Bronze')")

    page = _call(row_access.row_access_page, f"/admin/clients/{shop.account}/row-access", shop.account,
                 method="GET")
    assert "Group Gold team" in page and field["label"] in page and "only Gold" in page
    assert "their own segment" in page and "People's own values" in page
    assert f"{field['label']}: only Silver, Bronze" in page


def test_the_page_warns_of_a_person_a_rule_reads_a_value_they_have_not(shop):
    from admin import row_access

    shop.person("Cy")
    field = _segment_field(shop)
    _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
          body={"who": "everyone", "table": field["table"], "field": "segment", "sees": "own", "key": "segment"})
    page = _call(row_access.row_access_page, f"/admin/clients/{shop.account}/row-access", shop.account,
                 method="GET")
    assert "No segment of their own: they see none of the rows that rule covers." in page


def test_the_page_offers_a_fields_values_and_removes_a_rule(shop):
    from admin import row_access

    field = _segment_field(shop)
    values = _call(row_access.row_field_values, f"/admin/clients/{shop.account}/row-access/api/values",
                   shop.account, method="GET", table=field["table"], field="segment")
    assert values["ok"] and sorted(values["values"]) == ["Bronze", "Gold", "Silver"]
    saved = _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
                  body={"who": "everyone", "table": field["table"], "field": "segment", "sees": "every"})
    gone = _call(row_access.row_rule_delete, f"/admin/clients/{shop.account}/row-access/api/rules/x",
                 shop.account, saved["id"])
    assert gone["ok"] and shop.store.list_row_rules(shop.account) == []
    refused = _call(row_access.row_rule_save, f"/admin/clients/{shop.account}/row-access/api/rules", shop.account,
                    body={"who": "everyone", "table": "NOWHERE", "field": "segment", "sees": "every"})
    assert not refused["ok"] and "field" in refused["error"]


def test_row_access_is_a_workspace_admins_page():
    from admin import access

    role, scoped, account = access.rule_for("/clients/acct-x/row-access")
    assert role == access.WORKSPACE and scoped and account == "acct-x"
    assert access.rule_for("/clients/acct-x/row-access/api/rules")[0] == access.WORKSPACE


# ── every other way rows reach a person ─────────────────────────────────────

def test_the_question_box_offers_a_reader_only_their_own_past_questions(shop):
    """Another reader's question can name what this one may not see ("net amount for Customer 07")."""
    ada, ben = shop.person("Ada"), shop.person("Ben")
    shop.store.log_query(shop.account, "Net amount for Customer 07", "SELECT 1", portal_user_id=ben["id"])
    shop.store.log_query(shop.account, "Net amount by month", "SELECT 1", portal_user_id=ada["id"])
    offered = [s["text"] for s in shop.store.get_suggestions(shop.account, "net amount", user_id=ada["id"])]
    assert "Net amount by month" in offered and "Net amount for Customer 07" not in offered
    assert shop.store.get_suggestions(shop.account, "net amount for", user_id=None) == []


def test_an_answer_kept_from_before_a_rule_narrowed_the_reader_is_asked_again(shop):
    """History and export serve the rows kept with an answer: kept before a rule that binds the reader was added,
    or before their own values changed, they may hold rows the reader may no longer see."""
    from core.compliance import kept_answers

    ada, ben = shop.person("Ada"), shop.person("Ben")
    old = {"id": 1, "route": "core2", "created_at": "2000-01-01 00:00:00"}
    assert not kept_answers.narrowed_since(shop.account, ada, old)
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    assert kept_answers.narrowed_since(shop.account, shop.store.get_user(ada["id"]), old)
    assert not kept_answers.narrowed_since(shop.account, ben, old), "Ada's rule leaves Ben's answers as they were"
    shop.store.set_user_attributes(ben["id"], {"segment": "Silver"})
    assert kept_answers.narrowed_since(shop.account, shop.store.get_user(ben["id"]), old)
    later = {**old, "created_at": "2999-01-01 00:00:00"}
    assert not kept_answers.narrowed_since(shop.account, shop.store.get_user(ada["id"]), later)


def test_homes_kept_figures_are_kept_under_the_rules_and_the_readers_own_values(shop):
    """Home keeps a reader's figures for minutes: a rule added or a value changed meanwhile is a new key, not the
    figures counted before it."""
    import types as _types

    from portal import home

    ada = shop.person("Ada")
    model = _types.SimpleNamespace(version=1, built_at="x")
    today = __import__("datetime").date(2026, 6, 15)
    first = home._kept_key(shop.store.get_user(ada["id"]), model, None, "en", today)
    shop.store.save_row_rule(shop.account, _rule("user", ada["id"], values=["Gold"]))
    second = home._kept_key(shop.store.get_user(ada["id"]), model, None, "en", today)
    shop.store.set_user_attributes(ada["id"], {"segment": "Gold"})
    third = home._kept_key(shop.store.get_user(ada["id"]), model, None, "en", today)
    assert len({first, second, third}) == 3


def test_everyone_means_every_person_not_the_workspaces_own_service_work(shop):
    """An admin's check of a link, or a field's values on the Row access page, runs as no one: a rule for
    everyone is the people's, and leaves that work every row."""
    from core.compliance.models import PolicyContext
    from core.compliance.sql_guard import _policy_subject_matches

    rule = {"subject_type": "everyone", "subject_id": ""}
    assert _policy_subject_matches(rule, PolicyContext(account_id=shop.account, user_id="7", role="analyst"))
    assert not _policy_subject_matches(rule, PolicyContext(account_id=shop.account, role="system"))
