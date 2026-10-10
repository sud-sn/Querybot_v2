"""A bridge table is crossed: each row counts under each of its members, and a filter keeps it once.

A warehouse links two kinds of things through a table of pairs: a product has several tags (PRODUCT_TAGS),
a film several actors (FILM_ACTOR). Questions only followed links from many to one, so "sales by tag"
could not be answered at all. Now a question crosses a bridge that holds each pair once:

- grouped by the member ("sales by tag"), a sale counts under each tag of its product: each tag's figure
  is right, and the answer says they add up to more than the total, so no share, no total and no "Other"
  is worked out of them;
- grouped by something several members share ("sales by tag group"), a sale would count twice under one
  group: the question is refused, and the answer says to ask by the member;
- filtered by members ("sales of products tagged A or B"), each sale with a match is kept once (EXISTS),
  however many of them its product has;
- a bridge holding one row for each product (a product's one category) is exact: shares as usual.

A bridge whose pairs could repeat (its key is its own row number) is not crossed.
A small warehouse in DuckDB, invented data; every figure is checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)
TAGS = [(1, "Organic", "Diet", "ORG"), (2, "Vegan", "Diet", "VEG"), (3, "Gluten free", "Diet", "GLF"),
        (4, "Imported", "Origin", "IMP"), (5, "Local", "Origin", "LOC")]


def _warehouse(*, pair_key: bool = True, own_tag: bool = False) -> duckdb.DuckDBPyConnection:
    """``pair_key`` False: the bridge keeps each pair once per day it was tagged, some pairs on two days.
    ``own_tag``: each sale also names a tag of its own (the promotion it was sold under)."""
    """Thirty products, none to three tags each, one category each; four hundred sales."""
    con = duckdb.connect()
    con.execute("CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name VARCHAR)")
    con.execute("INSERT INTO categories VALUES (1, 'Bakery'), (2, 'Dairy'), (3, 'Produce')")
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name VARCHAR)")
    con.execute("INSERT INTO products SELECT i, 'Product ' || lpad(CAST(i AS VARCHAR), 2, '0') FROM range(1, 31) t(i)")
    con.execute("CREATE TABLE tags (tag_id INTEGER PRIMARY KEY, tag_name VARCHAR, tag_group VARCHAR, "
                "tag_code VARCHAR)")
    con.execute("INSERT INTO tags VALUES " + ", ".join(f"({i}, '{n}', '{g}', '{c}')" for i, n, g, c in TAGS))
    con.execute("CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, supplier_name VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO suppliers SELECT 100 + i, 'Supplier ' || lpad(CAST(i AS VARCHAR), 2, '0'), "
                "['Norway', 'Spain', 'Chile', 'Kenya'][1 + i % 4] FROM range(1, 13) t(i)")
    con.execute("CREATE TABLE product_suppliers (product_id INTEGER, supplier_id INTEGER, "
                "PRIMARY KEY (product_id, supplier_id))")
    con.execute("INSERT INTO product_suppliers VALUES " + ", ".join(
        f"({p}, {s})" for p in range(1, 31) for s in sorted({101 + p % 12, 101 + (p * 5) % 12})))
    pairs = sorted({(p, 1 + (p + 2 * k) % 5) for p in range(1, 31) for k in range(p % 4)})
    if pair_key:
        con.execute("CREATE TABLE product_tags (product_id INTEGER, tag_id INTEGER, PRIMARY KEY (product_id, tag_id))")
        con.execute("INSERT INTO product_tags VALUES " + ", ".join(f"({p}, {t})" for p, t in pairs))
    else:
        con.execute("CREATE TABLE product_tags (product_id INTEGER, tag_id INTEGER, tagged_on DATE, "
                    "PRIMARY KEY (product_id, tag_id, tagged_on))")
        dated = [(p, t, "2025-01-01") for p, t in pairs] + [(p, t, "2025-06-01") for p, t in pairs if p % 3 == 0]
        con.execute("INSERT INTO product_tags VALUES " + ", ".join(f"({p}, {t}, DATE '{d}')" for p, t, d in dated))
    con.execute("CREATE TABLE product_categories (product_id INTEGER PRIMARY KEY, category_id INTEGER)")
    con.execute("INSERT INTO product_categories SELECT i, 1 + i % 3 FROM range(1, 31) t(i)")
    own = ", tag_id INTEGER" if own_tag else ""
    con.execute("CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, sale_date DATE, product_id INTEGER, "
                f"sale_amount DECIMAL(12, 2){own})")
    con.execute("INSERT INTO sales SELECT i, DATE '2026-01-01' + CAST(i % 170 AS INTEGER), 1 + (i * 7) % 30, "
                f"round(5 + (i * 13) % 90, 2){', 1 + i % 5' if own_tag else ''} FROM range(1, 401) t(i)")
    return con


def _learn(con):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="bridges", options=BuildOptions(workers=1))


@pytest.fixture(scope="module")
def shop():
    con = _warehouse()
    return con, _learn(con)


def _table(model, name):
    return next(t for t in model.tables.values() if t.name == name)


def _attribute(model, table, name):
    return next(a.slug for a in model.attributes.values()
                if model.tables[model.columns[a.column].table].name == table and model.columns[a.column].name == name)


def _entity(model, table):
    return next(e.slug for e in model.entities.values() if model.tables[e.table].name == table)


def _ask(con, model, plan):
    answer = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answer,
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _rows(payload) -> dict:
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    assert headers, payload["answer"]["headline"] + " / " + str(payload.get("trust", {}).get("stopped"))
    # Rows with no member (a product with no tag) are one group, as in every breakdown: not a member.
    return {r[headers[0]]: float(r[headers[-1]]) for r in data.get("rows") or []
            if r[headers[0]] not in (None, "Unknown")}


def _notes(payload) -> str:
    return " ".join((payload.get("trust") or {}).get("date_context") or [])


def _total(con) -> float:
    return float(con.execute("SELECT SUM(sale_amount) FROM sales").fetchone()[0])


# ── what Learn sees ───────────────────────────────────────────────────────


def test_the_tables_of_pairs_are_bridges(shop):
    _, model = shop
    assert _table(model, "product_tags").kind == "bridge"
    assert _table(model, "product_categories").kind == "bridge"


# ── grouped by the member ─────────────────────────────────────────────────


def test_sales_by_tag_count_each_sale_under_each_tag_of_its_product(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")]})
    want = {n: float(v) for n, v in con.execute(
        "SELECT t.tag_name, SUM(s.sale_amount) FROM sales s JOIN product_tags pt ON pt.product_id = s.product_id "
        "JOIN tags t ON t.tag_id = pt.tag_id GROUP BY 1").fetchall()}
    got = _rows(payload)
    assert got == pytest.approx(want)
    assert sum(want.values()) > _total(con), "the data must have products with several tags"
    assert "counted under each tag of its product (a product with three tags" in _notes(payload)
    assert "add up to more than the total" in _notes(payload)


def test_no_share_of_a_total_is_worked_out_of_overlapping_members(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "share", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")]})
    headers = (payload.get("data") or {}).get("headers") or []
    assert not any("share" in h or "pct" in h for h in headers), headers
    assert "its groups overlap" in _notes(payload)
    assert "% of the total" not in payload["answer"]["headline"]


def test_the_top_tags_leave_out_no_other_and_name_no_share(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "rank", "measures": ["sale_amount"], "group_by": [_entity(model, "tags")],
                                "sort": [{"by": "sale_amount", "desc": True}], "limit": 2})
    text = json.dumps(payload.get("chart") or {}) + payload["answer"]["headline"]
    assert "Other" not in text and "% of" not in text, text


# ── grouped by what members share ─────────────────────────────────────────


def test_a_grouping_several_members_share_is_refused_with_the_reason(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_attribute(model, "tags", "tag_group")]})
    stopped = str((payload.get("trust") or {}).get("stopped"))
    assert "cannot split sales" in stopped and "Ask by tag instead" in stopped, stopped
    assert not (payload.get("data") or {}).get("rows")


# ── filtered by members ───────────────────────────────────────────────────


def test_a_sale_whose_product_has_both_tags_asked_for_is_counted_once(shop):
    con, model = shop
    tag = _attribute(model, "tags", "tag_name")
    payload = _ask(con, model, {"intent": "value", "measures": ["sale_amount"],
                                "filters": [{"field": tag, "op": "in", "values": ["Organic", "Vegan"]}]})
    right = float(con.execute(
        "SELECT SUM(sale_amount) FROM sales s WHERE EXISTS (SELECT 1 FROM product_tags pt JOIN tags t "
        "ON t.tag_id = pt.tag_id WHERE pt.product_id = s.product_id AND t.tag_name IN ('Organic', 'Vegan'))"
    ).fetchone()[0])
    joined = float(con.execute(
        "SELECT SUM(sale_amount) FROM sales s JOIN product_tags pt ON pt.product_id = s.product_id "
        "JOIN tags t ON t.tag_id = pt.tag_id WHERE t.tag_name IN ('Organic', 'Vegan')").fetchone()[0])
    assert joined > right, "the data must have products with both tags"
    assert payload["kpi"]["value"] == pytest.approx(right)
    assert "counted once, however many tags it matches" in _notes(payload)


def test_a_tag_filter_limits_a_breakdown_by_product_without_repeating_a_sale(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "products")],
                                "filters": [{"field": _attribute(model, "tags", "tag_group"), "op": "eq",
                                             "values": ["Diet"]}]})
    want = {n: float(v) for n, v in con.execute(
        "SELECT p.product_name, SUM(s.sale_amount) FROM sales s JOIN products p ON p.product_id = s.product_id "
        "WHERE EXISTS (SELECT 1 FROM product_tags pt JOIN tags t ON t.tag_id = pt.tag_id "
        "WHERE pt.product_id = s.product_id AND t.tag_group = 'Diet') GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)
    assert "add up to more than" not in _notes(payload)


def test_grouped_and_filtered_by_tag_the_filter_keeps_only_the_tags_asked_for(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")],
                                "filters": [{"field": _attribute(model, "tags", "tag_name"), "op": "in",
                                             "values": ["Organic", "Local"]}]})
    want = {n: float(v) for n, v in con.execute(
        "SELECT t.tag_name, SUM(s.sale_amount) FROM sales s JOIN product_tags pt ON pt.product_id = s.product_id "
        "JOIN tags t ON t.tag_id = pt.tag_id WHERE t.tag_name IN ('Organic', 'Local') GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


# ── a bridge holding one row for each ─────────────────────────────────────


def test_a_bridge_with_one_row_for_each_product_is_exact_and_shares_are_worked_out(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "share", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "categories")]})
    want = {n: float(v) for n, v in con.execute(
        "SELECT c.category_name, SUM(s.sale_amount) FROM sales s JOIN product_categories pc "
        "ON pc.product_id = s.product_id JOIN categories c ON c.category_id = pc.category_id GROUP BY 1").fetchall()}
    got = _rows(payload)
    shares = {k: v for k, v in got.items()}
    assert sum(want.values()) == pytest.approx(_total(con))
    assert "add up to more than" not in _notes(payload) and "overlap" not in _notes(payload)
    headers = payload["data"]["headers"]
    assert any("share" in h or "pct" in h for h in headers), headers
    assert set(shares) == set(want)


# ── a bridge whose pairs could repeat ─────────────────────────────────────


def test_a_bridge_that_can_hold_a_pair_twice_is_not_crossed():
    """Tagged again on another day, a pair is held twice: a sale would count twice under its tag."""
    con = _warehouse(pair_key=False)
    model = _learn(con)
    assert _table(model, "product_tags").kind == "bridge"
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")]})
    assert (payload.get("trust") or {}).get("stopped"), payload["answer"]["headline"]
    assert not (payload.get("data") or {}).get("rows")


# ── why it changed ────────────────────────────────────────────────────────


def test_why_sales_changed_by_tag_shows_each_tag_but_no_share_of_the_change(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "drivers", "measures": ["sale_amount"],
                                "drivers": {"dimensions": [_entity(model, "tags")]},
                                "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}})
    groupings = (payload.get("drivers") or {}).get("groupings") or []
    assert groupings, payload["answer"]["headline"] + str((payload.get("trust") or {}).get("stopped"))
    assert "share_of_change" not in ((payload.get("data") or {}).get("headers") or [])


def test_why_sales_changed_never_picks_an_overlapping_grouping_by_itself(shop):
    con, model = shop
    payload = _ask(con, model, {"intent": "drivers", "measures": ["sale_amount"],
                                "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}})
    picked = [g.get("slug") or g.get("grouping") or g.get("label") for g in
              (payload.get("drivers") or {}).get("groupings") or []]
    assert _entity(model, "tags") not in picked and "Tag" not in picked, picked


# ── every warehouse ───────────────────────────────────────────────────────


@pytest.mark.parametrize("dialect", ["snowflake", "oracle", "tsql"])
def test_the_filter_across_a_bridge_compiles_for_each_warehouse(shop, dialect):
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    _, model = shop
    plan = Plan.model_validate({"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "products")],
                                "filters": [{"field": _attribute(model, "tags", "tag_group"), "op": "eq",
                                             "values": ["Diet"]}]})
    plan_filter = Plan.model_validate({"intent": "value", "measures": ["sale_amount"],
                                       "filters": [{"field": _attribute(model, "tags", "tag_name"), "op": "in",
                                                    "values": ["Organic", "Vegan"]}]})
    ctx = Context(today=TODAY)
    with pytest.raises(Exception, match="cannot split"):
        compile_query(resolve(plan.model_copy(update={"group_by": [_attribute(model, "tags", "tag_group")]}),
                              model, ctx), model, dialect)
    grouped = compile_query(resolve(plan, model, ctx), model, dialect).sql
    filtered = compile_query(resolve(plan_filter, model, ctx), model, dialect).sql
    assert "EXISTS" in grouped.upper() and "EXISTS" in filtered.upper()


# ── two bridges, a direct link, a code ────────────────────────────────────


def test_two_bridges_in_one_question_each_need_groups_of_one_member(shop):
    con, model = shop
    both = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                             "group_by": [_entity(model, "tags"), _entity(model, "suppliers")]})
    data = both.get("data") or {}
    headers = data.get("headers") or []
    assert headers, str((both.get("trust") or {}).get("stopped"))
    got = {(r[headers[0]], r[headers[1]]): float(r[headers[-1]]) for r in data["rows"]
           if None not in (r[headers[0]], r[headers[1]]) and "Unknown" not in (r[headers[0]], r[headers[1]])}
    want = {(t, s_): float(v) for t, s_, v in con.execute(
        "SELECT t.tag_name, su.supplier_name, SUM(s.sale_amount) FROM sales s "
        "JOIN product_tags pt ON pt.product_id = s.product_id JOIN tags t ON t.tag_id = pt.tag_id "
        "JOIN product_suppliers ps ON ps.product_id = s.product_id JOIN suppliers su ON su.supplier_id = ps.supplier_id "
        "GROUP BY 1, 2").fetchall()}
    assert got == pytest.approx(want)
    country = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags"), _attribute(model, "suppliers", "country")]})
    stopped = str((country.get("trust") or {}).get("stopped"))
    assert "cannot split sales" in stopped and "Ask by supplier instead" in stopped, stopped


def test_a_sales_own_tag_is_followed_before_any_bridge():
    con = _warehouse(own_tag=True)
    model = _learn(con)
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")]})
    want = {n: float(v) for n, v in con.execute(
        "SELECT t.tag_name, SUM(s.sale_amount) FROM sales s JOIN tags t ON t.tag_id = s.tag_id GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)
    assert "add up to more than" not in _notes(payload)


def test_a_direct_link_waiting_for_an_admin_is_asked_about_not_crossed_round():
    con = _warehouse(own_tag=True)
    model = _learn(con)
    direct = next(j for j in model.joins.values() if model.tables[j.from_table].name == "sales"
                  and model.tables[j.to_table].name == "tags")
    from core2.model.schema import Evidence
    model.joins[direct.key] = direct.model_copy(update={
        "trust": "proposed", "status": "proposed",
        "evidence": [*direct.evidence, Evidence(kind="ambiguous", weight=-0.5, detail="two tables equally")]})
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["sale_amount"],
                                "group_by": [_entity(model, "tags")]})
    stopped = str((payload.get("trust") or {}).get("stopped"))
    assert "not confirmed" in stopped, stopped


def test_a_members_own_code_tells_them_apart_though_its_count_was_estimated(shop):
    con, model = shop
    code = next(c for c in model.columns.values() if model.tables[c.table].name == "tags" and c.name == "tag_code")
    entity = next(e for e in model.entities.values() if model.tables[e.table].name == "tags")
    assert entity.code_column == code.key
    estimated = model.model_copy(deep=True)
    estimated.columns[code.key].profile.distinct -= 1      # an approximate count on a large table
    payload = _ask(con, estimated, {"intent": "breakdown", "measures": ["sale_amount"],
                                    "group_by": [_attribute(estimated, "tags", "tag_code")]})
    want = {c: float(v) for c, v in con.execute(
        "SELECT t.tag_code, SUM(s.sale_amount) FROM sales s JOIN product_tags pt ON pt.product_id = s.product_id "
        "JOIN tags t ON t.tag_id = pt.tag_id GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


# ── access ────────────────────────────────────────────────────────────────


def test_a_reader_without_the_bridge_table_is_refused(shop):
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, ResolveError, resolve

    _, model = shop
    granted = {t.key for t in model.tables.values() if t.name != "product_tags"}
    for plan in ({"intent": "breakdown", "measures": ["sale_amount"], "group_by": [_entity(model, "tags")]},
                 {"intent": "value", "measures": ["sale_amount"],
                  "filters": [{"field": _attribute(model, "tags", "tag_name"), "op": "eq", "values": ["Vegan"]}]}):
        with pytest.raises(ResolveError) as refused:
            resolve(Plan.model_validate(plan), model, Context(today=TODAY, allowed_tables=granted))
        assert refused.value.kind == "denied", refused.value


def test_a_row_rule_on_the_tags_reaches_inside_the_filter_across_the_bridge(tmp_path, monkeypatch):
    """The governed warehouse narrows every table a query reads, the one inside the EXISTS too: a reader
    kept to the diet tags asks for sales tagged Organic or Local, and only Organic can match."""
    import types

    import sqlglot

    import store
    import store.crypto
    from core.compliance import governed_query as gq
    from core.compliance import sql_guard
    from core.compliance.models import PolicyContext
    from core.schema import load_known_tables
    from core2.bootstrap.service import build_workspace, load_model
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse import querybot
    from core2.warehouse.governed import GovernedWarehouse
    from tests.test_core2_learned_page_shows_what_was_learned import _Connection, _schema_file

    con = _warehouse()
    names = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()]
    built = types.SimpleNamespace(con=con, tables={t: t for t in names}, declared_pks={}, declared_fks=[])
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = "acct-core2-bridge-policy"
    store.upsert_client(account, "web")
    store.save_compliance_profile(account, mode="standard")
    db_id = store.save_db_config("azure_sql", "shop", {"server": "s", "database": "d", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    schema_dir = tmp_path / "clients" / account / "schema"
    store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(con))
    build_workspace(account)
    store.replace_row_policies(account, 0, [{
        "name": "diet tags only", "subject_type": "user", "subject_id": "7", "table_fqn": "TAGS",
        "condition": {"field": "tag_group", "operator": "=", "value": "Diet"}}])

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    model = load_model(account, db_id)
    tag = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "tag_name")
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": ["sale_amount"],
                                           "filters": [{"field": tag, "op": "in", "values": ["Organic", "Local"]}]}),
                      model, Context(today=TODAY))
    compiled = compile_query(logical, model, "tsql")
    assert "EXISTS" in compiled.sql.upper()
    governed = GovernedWarehouse(account, {"id": 7, "role": "analyst"}, store.get_db_config(db_id),
                                 known_tables=load_known_tables(str(schema_dir)))
    got = float(governed.query(compiled.sql, max_rows=compiled.row_cap).rows[0][0])

    def matching(tags: str) -> float:
        return float(con.execute(
            "SELECT SUM(sale_amount) FROM sales s WHERE EXISTS (SELECT 1 FROM product_tags pt JOIN tags t "
            f"ON t.tag_id = pt.tag_id WHERE pt.product_id = s.product_id AND t.tag_name IN ({tags}))").fetchone()[0])

    assert got == pytest.approx(matching("'Organic'"))
    assert matching("'Organic'") < matching("'Organic', 'Local'"), "the rule must narrow something here"
    rewritten, applied = sql_guard.inject_row_policies(compiled.sql, "azure_sql",
                                                       PolicyContext(account_id=account, user_id="7"))
    assert applied and "tag_group" in rewritten
