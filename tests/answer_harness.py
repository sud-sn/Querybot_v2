"""
A small warehouse to ask questions of, through the real pipeline.

Synthetic tables in a distribution mart's naming convention -- a daily stock
snapshot, a monthly movement fact keyed yyyymm with its whole-year row, item,
item group, warehouse and party dimensions, a day calendar and a period
calendar -- with invented rows. The tenant is set up the way discovery and an
admin set one up: the schema files written by the Azure discovery writer, the
value index, the join graph, every join checked against the rows, the
dimensions' placeholder members, the admin accepting the joins the data
vouches for, the date roles approved with their defaults and the starter
metrics accepted.

A question goes through core.query_pipeline._handle_query_impl. Two
boundaries are replaced: the warehouse (DuckDB over the rows below, the
product's T-SQL translated by sqlglot) and the model, which answers the SQL
writer with a marker query so an answer that needed the model to write its SQL
shows as such. Everything between is the product's own code.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
from unittest.mock import patch

# One tenant per process: the suite shares a store.
ACCOUNT = f"acct-answers-{os.urandom(3).hex()}"
MARKER = "NEEDS_THE_MODEL"
SQL_WRITER = "SQL expert. Convert the user"
LATEST = 20260331          # the newest daily snapshot
EARLIER = 20260330         # an older one, which a stock total must not add in
CONNECTION = {"db_type": "azure_sql", "name": "harness",
              "credentials": {"server": "harness", "user": "u", "password": "p"}}
_saved: dict = {}


def _table(key: str, *columns: tuple[str, str]) -> dict:
    return {"columns": [{"name": n, "type": t, "nullable": True, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 0, "comment": "", "schema": "MART", "database": "WH"}


SCHEMA = {
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("WHS_DMS_KEY", "int"), ("ITM_DMS_KEY", "int"),
        ("BYR_PTY_DMS_KEY", "int"), ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ITM_WHS_CRN_DT_DMS_KEY", "int"),
        ("ON_HND_QTY", "decimal"), ("ALC_ON_HND_QTY", "decimal"), ("ITM_CST", "decimal"), ("UNT_OF_MSR", "nvarchar"),
        ("RSV_QTY", "decimal"), ("RSV_BCK_ORD_QTY", "decimal")),
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY", ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("WHS_DMS_KEY", "int"), ("ITM_DMS_KEY", "int"),
        ("PRD_DMS_KEY", "int"), ("SLD_QTY", "decimal"), ("PCH_QTY", "decimal"), ("NUM_OF_RCT", "int"),
        ("CUR_ON_HND_QTY", "decimal"), ("ITM_CST", "decimal"), ("UNT_OF_MSR", "nvarchar")),
    "WH.MART.WHS_DMS": _table("WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "nvarchar"), ("WHS_DSC", "nvarchar")),
    "WH.MART.ITM_DMS": _table("ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "nvarchar"), ("ITM_NM", "nvarchar"),
                              ("ITM_GRP_DMS_KEY", "int"), ("UNT_OF_MSR", "nvarchar"), ("ITM_FR_NM", "nvarchar")),
    "WH.MART.ITM_GRP_DMS": _table("ITM_GRP_DMS_KEY", ("ITM_GRP_DMS_KEY", "int"), ("ITM_GRP_CD", "nvarchar"),
                                  ("ITM_GRP_DSC", "nvarchar")),
    "WH.MART.PTY_DMS": _table("PTY_DMS_KEY", ("PTY_DMS_KEY", "int"), ("PTY_CD", "nvarchar"), ("PTY_NM", "nvarchar")),
    "WH.MART.DT_DMS": _table("DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("QR", "int"),
                             ("MTH", "int"), ("MTH_NM", "nvarchar"), ("MTH_FR_NM", "nvarchar"), ("WK_OF_YR", "int"),
                             ("DAY_OF_WK", "int"), ("DAY_NM", "nvarchar"), ("DAY_FR_NM", "nvarchar")),
    "WH.MART.PRD_DMS": _table("PRD_DMS_KEY", ("PRD_DMS_KEY", "int"), ("PRD_DSC", "nvarchar"),
                              ("PRD_FR_DSC", "nvarchar"), ("PRD_YR", "int"), ("PRD_MTH", "int")),
}

_PLACEHOLDER = "NULL value provided"
# Two warehouses with no stock yet, coded like French words a question uses:
# "par pays", "par entrepôt".
WAREHOUSES = {0: ("0", _PLACEHOLDER), 1: ("N01", "NORTH DEPOT"), 2: ("S02", "SOUTH DEPOT"), 3: ("PAY", "PAYETTE YARD"),
              4: ("ENTREPOT", "ENTREPOT LAVAL")}
# Two groups with no items yet, named like words a question uses for its
# measure ("sold") and its period ("premier semestre").
GROUPS = {0: ("0", _PLACEHOLDER), 10: ("FIT", "FITTINGS"), 20: ("PIP", "PIPE"), 30: ("SLD", "SOLDER"),
          40: ("PRM", "PREMIER FITTINGS")}
# key: (code, name, group, unit)
ITEMS = {0: ("0", _PLACEHOLDER, 0, ""), 101: ("BE-1", "BRASS ELBOW", 10, "EA"), 102: ("ST-2", "STEEL TEE", 10, "EA"),
         201: ("CP-9", "COPPER PIPE", 20, "FT"), 202: ("PX-4", "PEX PIPE", 20, "FT")}
# An item's French name, where the warehouse keeps one: blank for two of them,
# as a French twin often is.
ITEM_FR_NAMES = {101: "COUDE EN LAITON", 102: "", 201: "TUYAU DE CUIVRE", 202: ""}
PARTIES = {0: ("0", _PLACEHOLDER), 1: ("ALO", "ANA LOPEZ"), 2: ("BOK", "BEN OKAFOR")}
_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
              "novembre", "décembre"]

# The daily snapshot: (warehouse, item, buyer, created yyyymmdd, on hand, allocated, cost)
STOCK = [
    (1, 101, 1, 20250114, 120, 20, 2.50),
    (1, 102, 1, 20250203, 40, 0, 7.25),
    (1, 201, 2, 20250411, 900, 100, 3.10),
    (2, 101, 2, 20250722, 60, 60, 2.50),
    (2, 202, 2, 20251105, 350, 0, 1.40),
]
# Reserved and back-ordered stock at the newest snapshot, row for row with
# STOCK: parts of the stock of their own, beside the allocated part.
RESERVED = [5, 0, 30, 10, 0]
BACK_ORDERED = [0, 4, 0, 10, 0]
# The monthly facts: (warehouse, item, yyyymm, sold, purchased, receipts, cost)
MOVES = [
    (1, 101, 202501, 10, 30, 2, 2.50), (1, 101, 202502, 15, 0, 0, 2.50), (1, 201, 202502, 200, 500, 1, 3.10),
    (1, 102, 202503, 4, 10, 1, 7.25), (2, 101, 202503, 8, 20, 3, 2.50), (2, 202, 202504, 120, 300, 2, 1.40),
    (2, 202, 202506, 60, 0, 0, 1.40),
]
YEAR_ROW = (1, 101, 202500, 999, 999, 99, 2.50)   # the whole-year row, never added to its months


def rows() -> dict[str, list[tuple]]:
    data: dict[str, list[tuple]] = {
        "WHS_DMS": [(k, c, n) for k, (c, n) in WAREHOUSES.items()],
        "ITM_GRP_DMS": [(k, c, n) for k, (c, n) in GROUPS.items()],
        "ITM_DMS": [(k, c, n, g, u, ITEM_FR_NAMES.get(k, "")) for k, (c, n, g, u) in ITEMS.items()],
        "PTY_DMS": [(k, c, n) for k, (c, n) in PARTIES.items()],
    }
    days = []
    day = dt.date(2025, 1, 1)
    while day <= dt.date(2026, 3, 31):
        key = int(day.strftime("%Y%m%d"))
        days.append((key, day.isoformat(), day.year, (day.month - 1) // 3 + 1, day.month,
                     _MONTHS_EN[day.month - 1], _MONTHS_FR[day.month - 1], day.isocalendar()[1],
                     day.isoweekday(), _DAYS_EN[day.weekday()], _DAYS_FR[day.weekday()]))
        day += dt.timedelta(days=1)
    data["DT_DMS"] = days
    periods = [(202500, "Year 2025", "Année 2025", 2025, 0)]
    periods += [(202500 + m, f"{_MONTHS_EN[m - 1]} 2025", f"{_MONTHS_FR[m - 1]} 2025", 2025, m) for m in range(1, 13)]
    data["PRD_DMS"] = periods
    daily = []
    for snapshot in (EARLIER, LATEST):
        for n, (whs, itm, byr, created, on_hand, allocated, cost) in enumerate(STOCK):
            # The earlier snapshot holds different quantities: adding it in
            # would change every total.
            qty = on_hand if snapshot == LATEST else on_hand + 1000
            reserved, back_ordered = RESERVED[n], BACK_ORDERED[n]
            if snapshot != LATEST:
                reserved, back_ordered = reserved + 100, back_ordered + 100
            daily.append((len(daily) + 1, whs, itm, byr, snapshot, created, qty, allocated, cost, ITEMS[itm][3],
                          reserved, back_ordered))
    data["ITM_BAL_DLY_FCT"] = daily
    data["ITM_BAL_PRD_FCT"] = [
        (n + 1, whs, itm, prd, sold, bought, receipts, None, cost, "")
        for n, (whs, itm, prd, sold, bought, receipts, cost) in enumerate(MOVES + [YEAR_ROW])
    ]
    return data


# ── The warehouse ────────────────────────────────────────────────────────────

_DUCK_TYPE = {"int": "INTEGER", "bigint": "BIGINT", "decimal": "DOUBLE", "date": "DATE", "nvarchar": "VARCHAR"}


class Warehouse:
    def __init__(self, path: Path):
        import duckdb

        self.path = path
        con = duckdb.connect(str(path))
        for fqn, meta in SCHEMA.items():
            name = fqn.split(".")[-1]
            cols = ", ".join(f'"{c["name"]}" {_DUCK_TYPE[c["type"]]}' for c in meta["columns"])
            con.execute(f'CREATE TABLE "{name}" ({cols})')
            table_rows = rows()[name]
            if table_rows:
                marks = ", ".join("?" for _ in meta["columns"])
                con.executemany(f'INSERT INTO "{name}" VALUES ({marks})', table_rows)
            meta["row_count"] = len(table_rows)
        con.close()
        self.con = duckdb.connect(str(path), read_only=True)

    def query(self, sql: str, max_rows: int = 200) -> list[dict]:
        import sqlglot
        from sqlglot import exp

        statements = []
        for tree in sqlglot.parse(sql, read="tsql"):
            if tree is None:
                continue
            for table in tree.find_all(exp.Table):
                table.set("db", None)
                table.set("catalog", None)
            # Three T-SQL meanings sqlglot does not carry over: `+` beside text
            # joins the text; TRY_CONVERT(date, text, 112) is NULL for a string
            # that is not a date -- a yyyymm key's whole-year row decodes to
            # month 00 -- where DuckDB's STRPTIME raises; and the date 0 in
            # DATEADD(month, DATEDIFF(month, 0, d), 0) is 1900-01-01.
            tree = tree.transform(_tsql_concatenation).transform(_tsql_try_date).transform(_tsql_day_zero)
            statements.append(tree.sql(dialect="duckdb"))
        cursor = self.con.execute(";\n".join(statements))
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]


def _is_text(node) -> bool:
    from sqlglot import exp

    if isinstance(node, exp.Literal):
        return node.is_string
    if isinstance(node, exp.Convert):
        return node.this.is_type(*exp.DataType.TEXT_TYPES)
    if isinstance(node, exp.Cast):
        return node.to.is_type(*exp.DataType.TEXT_TYPES)
    return False


def _tsql_concatenation(node):
    from sqlglot import exp

    if isinstance(node, exp.Add) and (_is_text(node.this) or _is_text(node.expression)):
        return exp.DPipe(this=node.this, expression=node.expression)
    return node


def _tsql_try_date(node):
    from sqlglot import exp

    if (
        isinstance(node, exp.Convert) and node.args.get("safe") and node.this.is_type("date")
        and str(node.args.get("style") or "") == "112"
    ):
        parsed = exp.Anonymous(this="TRY_STRPTIME", expressions=[node.expression, exp.Literal.string("%Y%m%d")])
        return exp.TryCast(this=parsed, to=exp.DataType.build("date"))
    return node


def _tsql_day_zero(node):
    from sqlglot import exp

    if isinstance(node, exp.DateAdd) and isinstance(node.this, exp.Literal) and not node.this.is_string \
            and str(node.this.this) == "0":
        node.set("this", exp.cast(exp.Literal.string("1900-01-01"), "date"))
    return node


# ── The tenant ───────────────────────────────────────────────────────────────

def _schema_files(schema_dir: Path, warehouse: Warehouse) -> None:
    from core.schema import _MAX_DISTINCT, _az_md, _is_categorical

    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    for fqn, meta in SCHEMA.items():
        database, schema, name = fqn.split(".")
        columns = [{"COLUMN_NAME": c["name"], "DATA_TYPE": c["type"], "IS_NULLABLE": "YES",
                    "CHARACTER_MAXIMUM_LENGTH": None, "NUMERIC_PRECISION": None, "COMMENT": ""}
                   for c in meta["columns"]]
        distinct = {}
        for column in columns:
            if _is_categorical(column["COLUMN_NAME"], column["DATA_TYPE"]):
                found = warehouse.query(f'SELECT DISTINCT "{column["COLUMN_NAME"]}" AS v FROM "{name}" '
                                        f'WHERE "{column["COLUMN_NAME"]}" IS NOT NULL', _MAX_DISTINCT + 1)
                if 0 < len(found) <= _MAX_DISTINCT:
                    distinct[column["COLUMN_NAME"]] = sorted(str(r["v"]) for r in found)
        sample = warehouse.query(f'SELECT * FROM "{name}" LIMIT 5', 5)
        (schema_dir / f"{name}.md").write_text(
            _az_md(name, {"TABLE_SCHEMA": schema, "TABLE_NAME": name, "TABLE_TYPE": "BASE TABLE",
                          "TABLE_CATALOG": database, "COMMENT": ""},
                   columns, sample, schema, distinct, database=database, stats_map={},
                   row_count=meta["row_count"], pk_columns=meta["pk_columns"]),
            encoding="utf-8")


def build_tenant(root: Path) -> Warehouse:
    """The warehouse and a tenant over it, set up by discovery and an admin.
    Call with the working directory at ``root``: the tenant's files live
    under clients/ there."""
    import store
    from core.graph_autopopulate import auto_populate_from_schema
    from core.relationship_validator import profile_suggested_relationships
    from core.semantic_contract import write_contract
    from core.semantic_model import load_semantic_model, patch_date_role, set_default_date_role, write_semantic_model
    from core.starter_metrics import starter_metrics
    from core.unknown_members import detect_unknown_members
    from core.value_index import build_value_index
    from core.vocab_packs import forget_account_vocab, vocab_for_account

    warehouse = Warehouse(root / "warehouse.duckdb")
    store.init_db()
    store.upsert_client(ACCOUNT, "Answer Harness Ltd")
    store.save_compliance_profile(ACCOUNT, mode="standard", industry="standard")
    db_id = store.save_db_config(CONNECTION["db_type"], CONNECTION["name"], CONNECTION["credentials"])
    _saved["id"] = db_id
    store.update_client_meta(ACCOUNT, db_config_id=db_id, chat_ui_enabled=1)
    schema_dir = root / "schema"
    _schema_files(schema_dir, warehouse)
    store.update_client_state(ACCOUNT, "READY", {"schema_dir": str(schema_dir), "kb_dir": str(schema_dir)})
    forget_account_vocab(ACCOUNT)

    def probe(db_type, raw_cfg, sql, *, timeout_seconds=20, max_rows=200):
        return [tuple(row.values()) for row in warehouse.query(sql, max_rows)]

    with patch("core.relationship_validator.run_probe", probe):
        build_value_index(ACCOUNT, {}, "azure_sql", str(schema_dir), vocab=vocab_for_account(ACCOUNT),
                          run_query_fn=lambda creds, db_type, sql, max_rows=200: warehouse.query(sql, max_rows))
        auto_populate_from_schema(ACCOUNT, str(schema_dir))
        profile_suggested_relationships(ACCOUNT)
        detect_unknown_members(ACCOUNT)
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(schema_dir), account_id=ACCOUNT)
    _accept_what_the_data_vouches_for()
    model = load_semantic_model(str(schema_dir))
    for table, column, default in (("ITM_BAL_DLY_FCT", "ITM_BAL_EFC_DT_DMS_KEY", True),
                                   ("ITM_BAL_DLY_FCT", "ITM_WHS_CRN_DT_DMS_KEY", False),
                                   ("ITM_BAL_PRD_FCT", "PRD_DMS_KEY", True)):
        role = next(r for r in model.get("date_roles") or []
                    if str(r.get("fact_table") or "").upper().endswith(table) and r.get("fact_column") == column)
        patch_date_role(kb_dir=str(schema_dir), fact_table=role["fact_table"], fact_column=column,
                        dimension_table=role.get("dimension_table") or "", dimension_key=role.get("dimension_key") or "",
                        business_role=role.get("business_role") or "", name=role.get("name") or "",
                        date_value_column=role.get("date_value_column") or "",
                        date_key_type=role.get("date_key_type") or "", status="approved")
        if default:
            set_default_date_role(str(schema_dir), role["fact_table"], column)
    for proposal in starter_metrics(load_semantic_model(str(schema_dir))):
        store.save_metric(ACCOUNT, proposal.as_metric())
    write_contract(ACCOUNT, str(schema_dir))
    # Discovery's own steps above needed the saved connection; nothing after
    # them does -- the pipeline reads it through saved_connection(). It is not
    # left in the shared store: modules later in a full run repoint the key
    # file, and a page that lists every connection, the admin dashboard, could
    # not read it.
    store.delete_db_config(db_id)
    return warehouse


def saved_connection() -> dict:
    """The saved connection as the pipeline reads it, without decrypting it.

    By the time a full-suite run gets here other modules have repointed
    QUERYBOT_KEY_FILE and re-imported store, so the store the pipeline bound
    at import can hold another key than the one the connection was saved
    with (tests/test_client_sources.py documents the condition). The
    connection goes nowhere: the warehouse is DuckDB.
    """
    return {"id": _saved.get("id"), **CONNECTION}


def _accept_what_the_data_vouches_for() -> None:
    """The admin's bulk review: the joins the rows confirmed, and the tables."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import admin.routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    with patch.object(routes, "_is_auth", return_value=True):
        for body in ({"action": "accept", "kinds": ["rel"], "evidence": "data"},
                     {"action": "accept", "kinds": ["entity"]}):
            response = client.post(f"/admin/clients/{ACCOUNT}/graph/api/review/bulk", json=body)
            assert response.status_code == 200, response.text


# ── Asking ───────────────────────────────────────────────────────────────────

class _Channel:
    """The portal adapter as the pipeline sees it; every reply is kept."""

    platform = "portal"

    def __init__(self, thread: str):
        self.session_id = f"{ACCOUNT}:portal:harness"
        self.thread_id = thread
        self.last_result_id = None
        self.replies: list[tuple[str, object]] = []

    async def send_message(self, event, text, **kwargs):
        self.replies.append(("message", str(text)))

    async def send_clarification_prompt(self, event, question, options, **kwargs):
        self.replies.append(("clarify", {"question": question, "options": options}))

    async def send_assistant_response(self, event, payload, *args, **kwargs):
        self.replies.append(("answer", payload))

    async def send_analysis_response(self, event, insight, **kwargs):
        self.replies.append(("analysis", insight))

    async def send_status(self, *args, **kwargs):
        return None

    async def send_typing(self, *args, **kwargs):
        return None

    async def send_chart(self, *args, **kwargs):
        return None

    async def send_suggested_questions(self, *args, **kwargs):
        return None

    async def upload_file(self, *args, **kwargs):
        return None

    def add_to_history(self, **kwargs):
        return None

    def cache_result(self, *args, **kwargs):
        return None


class _Retriever:
    last_retrieval_weak = False
    last_retrieval_unscored = False

    def retrieve(self, question, n=8, allowed_tables=None):
        return []

    def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
        return []

    def _is_global(self, doc):
        return False


_asked = 0


def ask(warehouse: Warehouse, question: str, lang: str = "en") -> dict:
    """One question through the real pipeline. Returns what the warehouse
    ran (``executed``: sql and rows), whether the model was asked to write
    SQL (``model_wrote_sql``) and the prompts it was given (``prompts``),
    and every reply."""
    global _asked
    import core.llm as llm
    import core.query_pipeline as qp
    import core.schema as schema_module
    import store
    from gateway.base import PlatformEvent

    _asked += 1
    executed: list[dict] = []
    model_calls: list[str] = []

    def run_azure_sql(cfg, sql, max_rows=200):
        found = warehouse.query(sql, max_rows)
        executed.append({"sql": sql, "rows": found})
        return found

    async def model(system, user, *args, **kwargs):
        model_calls.append(str(system))
        return (f"SELECT '{MARKER}' AS marker" if SQL_WRITER in str(system)[:400] else ""), 1, 1

    reader = store.get_user_by_email(ACCOUNT, "reader@harness.example")
    if reader is None:
        store.create_user(ACCOUNT, "Reader", "reader@harness.example", password="a-password-they-chose", role="admin")
        reader = store.get_user_by_email(ACCOUNT, "reader@harness.example")
    user = {"id": reader["id"], "role": "admin", "email": reader["email"], "name": "Reader", "group_name": None,
            "lang": lang, "account_id": ACCOUNT}
    channel = _Channel(f"harness-{_asked}")
    event = PlatformEvent(ACCOUNT, f"harness-{lang}", f"c{_asked}", question, "portal", raw={})
    original = llm.llm_complete
    with contextlib.ExitStack() as stack:
        for module in list(__import__("sys").modules.values()):
            if module is not None and getattr(module, "llm_complete", None) is original:
                stack.enter_context(patch.object(module, "llm_complete", model))
        stack.enter_context(patch.object(qp, "resolve_provider", return_value=("azure_openai", "gpt-4o", "k", {})))
        stack.enter_context(patch.object(qp, "load_retriever", return_value=_Retriever()))
        stack.enter_context(patch.object(qp, "retrieve_similar_examples", return_value=[]))
        stack.enter_context(patch.object(schema_module, "_run_azure_sql", run_azure_sql))
        stack.enter_context(patch.object(qp, "get_client_db", lambda *args, **kwargs: saved_connection()))
        asyncio.run(qp._handle_query_impl(ACCOUNT, event, channel, question, user))
    answers = [e for e in executed if MARKER not in e["sql"]]
    return {
        "executed": answers,
        "rows": answers[-1]["rows"] if answers else [],
        "sql": answers[-1]["sql"] if answers else "",
        "model_wrote_sql": any(SQL_WRITER in call[:400] for call in model_calls),
        "prompts": [call for call in model_calls if SQL_WRITER in call[:400]],
        "replies": channel.replies,
    }


_built: dict = {}


@contextlib.contextmanager
def tenant_in(root: Path):
    """The tenant, with the working directory where its files live while the
    tests ask their questions. It is built once per store, at the first
    ``root``: its account is process-wide and the suite shares one store, so a
    second build would lay a second graph and metric set over the first. A
    module that points the process at another store -- as
    tests/test_metric_authoring.py does when it is imported -- leaves the
    tenant in the old one, and it is built again in the new."""
    import store

    previous = Path.cwd()
    if "warehouse" not in _built or store.get_client(ACCOUNT) is None:
        os.chdir(root)
        try:
            _built["warehouse"] = build_tenant(root)
        except BaseException:
            os.chdir(previous)
            raise
        _built["root"] = root
    os.chdir(_built["root"])
    try:
        yield _built["warehouse"]
    finally:
        os.chdir(previous)
