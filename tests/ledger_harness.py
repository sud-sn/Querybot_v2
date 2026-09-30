"""
A third small warehouse to ask questions of: procurement and finance in the
lowercase snake_case of a dbt project.

tests/answer_harness.py keeps a distribution mart in one ERP family's
abbreviated uppercase (ITM_BAL_DLY_FCT); tests/star_harness.py a Kimball star
in PascalCase (FactInternetSales). This one is spelled the third common way --
dim_ and fct_ models in lowercase, full words, under one schema -- and its facts
carry dates, not date keys: a purchase order line is ordered_on one date and
received_on another, a journal line posted_on a third, each joined to the
calendar on its date_day. Two business areas: purchase orders to suppliers,
for materials kept in two units of measure, and expenses booked to accounts
and cost centres. Rows are invented.

The tenant is set up as the other two are, by discovery and an admin: the
schema files, the value index, the join graph checked against the rows, the
star-schema vocabulary, the joins the data vouches for, the date roles approved
with each fact's everyday date, and the business's metrics.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from pathlib import Path

from tests import answer_harness as harness

ACCOUNT = f"acct-ledger-{os.urandom(3).hex()}"
CONNECTION = {"db_type": "azure_sql", "name": "ledger",
              "credentials": {"server": "ledger", "user": "u", "password": "p"}}
_saved: dict = {}


def _table(key: str, *columns: tuple[str, str]) -> dict:
    return {"columns": [{"name": n, "type": t, "nullable": True, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 0, "comment": "", "schema": "analytics", "database": "warehouse"}


SCHEMA = {
    "warehouse.analytics.dim_date": _table(
        "date_day", ("date_day", "date"), ("month_name", "nvarchar"), ("month_name_fr", "nvarchar"),
        ("month_of_year", "int"), ("quarter_of_year", "int"), ("calendar_year", "int"),
        ("day_of_week_name", "nvarchar"), ("day_of_week_name_fr", "nvarchar")),
    "warehouse.analytics.dim_supplier": _table(
        "supplier_id", ("supplier_id", "int"), ("supplier_code", "nvarchar"), ("supplier_name", "nvarchar"),
        ("country", "nvarchar")),
    "warehouse.analytics.dim_material": _table(
        "material_id", ("material_id", "int"), ("material_code", "nvarchar"), ("material_name", "nvarchar"),
        ("material_family", "nvarchar"), ("unit_of_measure", "nvarchar")),
    "warehouse.analytics.fct_purchase_order_lines": _table(
        "purchase_order_line_id", ("purchase_order_line_id", "bigint"), ("purchase_order_number", "nvarchar"),
        ("supplier_id", "int"), ("material_id", "int"), ("ordered_on", "date"), ("received_on", "date"),
        ("ordered_quantity", "decimal"), ("received_quantity", "decimal"), ("unit_price", "decimal"),
        ("line_amount", "decimal")),
    "warehouse.analytics.dim_cost_center": _table(
        "cost_center_id", ("cost_center_id", "int"), ("cost_center_code", "nvarchar"),
        ("cost_center_name", "nvarchar"), ("department", "nvarchar")),
    "warehouse.analytics.dim_gl_account": _table(
        "gl_account_id", ("gl_account_id", "int"), ("account_number", "nvarchar"), ("account_name", "nvarchar"),
        ("account_type", "nvarchar")),
    "warehouse.analytics.fct_journal_entries": _table(
        "journal_entry_line_id", ("journal_entry_line_id", "bigint"), ("journal_number", "nvarchar"),
        ("posted_on", "date"), ("gl_account_id", "int"), ("cost_center_id", "int"), ("amount", "decimal")),
}

_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
              "novembre", "décembre"]
_DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# key: (code, name, country)
SUPPLIERS = {1: ("NM", "Nordic Metals", "Sweden"), 2: ("AV", "Atlas Valves", "Canada"),
             3: ("PS", "Prairie Seals", "Canada"), 4: ("GP", "Gulf Pumps", "Mexico")}
# key: (code, name, family, unit)
MATERIALS = {1: ("ST-SH", "Steel sheet", "Metals", "KG"), 2: ("CU-WR", "Copper wire", "Metals", "KG"),
             3: ("VL-50", "Ball valve", "Valves", "EA"), 4: ("GK-10", "Gasket", "Seals", "EA"),
             5: ("PM-20", "Centrifugal pump", "Pumps", "EA")}
UNIT_PRICES = {1: 2.40, 2: 9.10, 3: 38.0, 4: 1.25, 5: 640.0}
# key: (code, name, department)
COST_CENTERS = {1: ("CC-100", "Plant maintenance", "Operations"), 2: ("CC-200", "Head office", "Administration"),
                3: ("CC-300", "Field service", "Operations")}
# key: (number, name, type)
ACCOUNTS = {1: ("6100", "Salaries", "Payroll"), 2: ("6200", "Repairs", "Operations"),
            3: ("6300", "Travel", "Travel"), 4: ("6400", "Utilities", "Operations")}
FIRST_DAY, LAST_DAY = dt.date(2025, 1, 1), dt.date(2026, 3, 31)


def _months() -> list[tuple[int, int]]:
    return [(year, month) for year in (2025, 2026) for month in range(1, 13) if (year, month) <= (2026, 3)]


def purchase_order_lines() -> list[tuple]:
    """Three lines a month on two orders, January 2025 to March 2026. Each is
    received some days after it was ordered -- across the month's end for
    some -- and the last month's second order is not received yet. (id, order
    number, supplier, material, ordered on, received on, ordered, received,
    unit price, amount)"""
    lines = []
    for n, (year, month) in enumerate(_months()):
        for k in range(3):
            order = f"PO-{year}{month:02d}-{1 + k // 2}"
            material = 1 + (n + 2 * k) % len(MATERIALS)
            supplier = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}[material] if k != 2 else 1 + (n + k) % len(SUPPLIERS)
            ordered_on = dt.date(year, month, 4 + 9 * k)
            quantity = 10 * (1 + (n + k) % 4) if MATERIALS[material][3] == "KG" else 1 + (n + k) % 3
            received_on = ordered_on + dt.timedelta(days=6 + 5 * k)
            received = quantity - (1 if k == 1 and n % 3 == 0 else 0)
            if (year, month) == (2026, 3) and k == 2:
                received_on, received = None, 0
            price = UNIT_PRICES[material]
            lines.append((len(lines) + 1, order, supplier, material, ordered_on, received_on, quantity, received,
                          price, round(quantity * price, 2)))
    return lines


def journal_entries() -> list[tuple]:
    """Each month's expenses: salaries and repairs for every cost centre, travel
    for field service, and utilities for the plant. (id, journal, posted on,
    account, cost centre, amount)"""
    lines = []
    for n, (year, month) in enumerate(_months()):
        posted_on = dt.date(year, month, 28)
        journal = f"JE-{year}{month:02d}"
        for centre in COST_CENTERS:
            lines.append((len(lines) + 1, journal, posted_on, 1, centre, 10000.0 + 1500 * centre + 50 * n))
            lines.append((len(lines) + 1, journal, posted_on, 2, centre, 400.0 + 75 * ((n + centre) % 5)))
        lines.append((len(lines) + 1, journal, posted_on, 3, 3, 900.0 + 120 * (n % 4)))
        lines.append((len(lines) + 1, journal, posted_on, 4, 1, 1800.0 + 60 * (month % 6)))
    return lines


def rows() -> dict[str, list[tuple]]:
    days = []
    day = dt.date(2024, 1, 1)
    while day <= dt.date(2026, 12, 31):
        days.append((day, _MONTHS_EN[day.month - 1], _MONTHS_FR[day.month - 1], day.month,
                     (day.month - 1) // 3 + 1, day.year, _DAYS_EN[day.weekday()], _DAYS_FR[day.weekday()]))
        day += dt.timedelta(days=1)
    return {
        "dim_date": days,
        "dim_supplier": [(k, *v) for k, v in SUPPLIERS.items()],
        "dim_material": [(k, *v) for k, v in MATERIALS.items()],
        "fct_purchase_order_lines": purchase_order_lines(),
        "dim_cost_center": [(k, *v) for k, v in COST_CENTERS.items()],
        "dim_gl_account": [(k, *v) for k, v in ACCOUNTS.items()],
        "fct_journal_entries": journal_entries(),
    }


# What the business reports, as its administrator registers it.
METRICS = [
    {"name": "Purchase Spend", "sql_template": "SUM(line_amount)", "base_table": "analytics.fct_purchase_order_lines",
     "formula_type": "expression", "category": "procurement",
     "synonyms": "purchase spend, spend, purchases, dépenses d'achat, achats",
     "description": "What was ordered from suppliers, at the order's unit price."},
    {"name": "Quantity Ordered", "sql_template": "SUM(ordered_quantity)",
     "base_table": "analytics.fct_purchase_order_lines", "formula_type": "expression", "category": "procurement",
     "synonyms": "quantity ordered, ordered quantity, quantité commandée",
     "description": "Units of material ordered, each in its own unit of measure."},
    {"name": "Quantity Received", "sql_template": "SUM(received_quantity)",
     "base_table": "analytics.fct_purchase_order_lines", "formula_type": "expression", "category": "procurement",
     "synonyms": "quantity received, received quantity, quantité reçue",
     "description": "Units of material received, each in its own unit of measure."},
    {"name": "Number of Purchase Orders", "sql_template": "COUNT(DISTINCT purchase_order_number)",
     "base_table": "analytics.fct_purchase_order_lines", "formula_type": "expression", "category": "procurement",
     "synonyms": "number of purchase orders, purchase orders, nombre de bons de commande, bons de commande",
     "description": "Purchase orders placed; an order has one or more lines."},
    {"name": "Expenses", "sql_template": "SUM(amount)", "base_table": "analytics.fct_journal_entries",
     "formula_type": "expression", "category": "finance",
     "synonyms": "expenses, expense amount, costs booked, charges, montant des charges",
     "description": "Expenses booked to the general ledger."},
]


def build_tenant(root: Path) -> harness.Warehouse:
    """The warehouse and a tenant over it, set up by discovery and an admin."""
    from unittest.mock import patch

    import store
    from core.graph_autopopulate import auto_populate_from_schema
    from core.relationship_validator import profile_suggested_relationships
    from core.semantic_contract import write_contract
    from core.semantic_model import load_semantic_model, patch_date_role, set_default_date_role, write_semantic_model
    from core.unknown_members import detect_unknown_members
    from core.value_index import build_value_index
    from core.vocab_packs import forget_account_vocab, vocab_for_account

    warehouse = harness.Warehouse(root / "warehouse.duckdb", SCHEMA, rows())
    store.init_db()
    store.upsert_client(ACCOUNT, "Ledger Harness Ltd")
    store.save_compliance_profile(ACCOUNT, mode="standard", industry="standard")
    db_id = store.save_db_config(CONNECTION["db_type"], CONNECTION["name"], CONNECTION["credentials"])
    _saved["id"] = db_id
    store.update_client_meta(ACCOUNT, db_config_id=db_id, chat_ui_enabled=1,
                             erp_packs=json.dumps(["generic_star_schema"]))
    schema_dir = root / "schema"
    harness._schema_files(schema_dir, warehouse, SCHEMA)
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
    harness._accept_what_the_data_vouches_for(ACCOUNT)
    # Each fact's everyday date: the order's, and the journal's posting date.
    everyday = {"FCT_PURCHASE_ORDER_LINES": "ordered_on", "FCT_JOURNAL_ENTRIES": "posted_on"}
    for role in load_semantic_model(str(schema_dir)).get("date_roles") or []:
        table = str(role.get("fact_table") or "").split(".")[-1].upper()
        if table not in everyday:
            continue
        patch_date_role(kb_dir=str(schema_dir), fact_table=role["fact_table"], fact_column=role["fact_column"],
                        dimension_table=role.get("dimension_table") or "",
                        dimension_key=role.get("dimension_key") or "",
                        business_role=role.get("business_role") or "", name=role.get("name") or "",
                        date_value_column=role.get("date_value_column") or "",
                        date_key_type=role.get("date_key_type") or "", status="approved")
        if str(role["fact_column"]).lower() == everyday[table]:
            set_default_date_role(str(schema_dir), role["fact_table"], role["fact_column"])
    for metric in METRICS:
        store.save_metric(ACCOUNT, dict(metric))
    write_contract(ACCOUNT, str(schema_dir))
    store.delete_db_config(db_id)
    return warehouse


def saved_connection() -> dict:
    return {"id": _saved.get("id"), **CONNECTION}


def ask(warehouse: harness.Warehouse, question: str, lang: str = "en", *, choose: str | None = None) -> dict:
    return harness.ask(warehouse, question, lang, account=ACCOUNT, connection=saved_connection(), choose=choose)


_built: dict = {}


@contextlib.contextmanager
def tenant_in(root: Path):
    """The tenant, built once per store as the other harnesses build theirs."""
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
