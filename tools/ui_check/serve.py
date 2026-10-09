"""The product, served locally on invented data, for the UI check (tools/ui_check/check.py).

    python tools/ui_check/serve.py <run dir> [port]

Two workspaces. An inventory-style workspace answered by today's pipeline, set up the way discovery
and an admin would (tests/answer_harness.py), so the admin pages have content. A retail workspace
answered by the new core (evals/core2/domains/retail.py), so the chat has real answers and charts.
The warehouse is DuckDB behind the product's own SQL and the AI planner is replaced by scripted plans
for the questions below: nothing here reaches a real database or a real AI. Everything the server
writes (its database, key, schema files) lives under <run dir>.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
RUN.mkdir(parents=True, exist_ok=True)
os.environ["QUERYBOT_DB_PATH"] = str(RUN / "qb.db")
os.environ["DB_PATH"] = str(RUN / "qb.db")
os.environ["QUERYBOT_KEY_FILE"] = str(RUN / ".key")
sys.path.insert(0, str(REPO))
os.chdir(RUN)

RETAIL = "acct-retail"
STATE = RUN / "accounts.json"
ADMIN_PASSWORD = "ui-check-admin-pass-1"
READER = ("reader@example.com", "reader-pass-123")

import store  # noqa: E402
import store.crypto  # noqa: E402

store.crypto.KEY_FILE = RUN / ".key"
store.init_db()
fresh = not STATE.exists()

from tests import answer_harness as H  # noqa: E402

# ── the inventory-style tenant (today's pipeline) ─────────────────────────────
if fresh:
    inventory = H.build_tenant(RUN)
    inv_account = H.ACCOUNT
    db_id = store.save_db_config("azure_sql", "Inventory warehouse", {"server": "harness", "user": "u", "password": "p"})
    store.update_client_meta(inv_account, db_config_id=db_id, chat_ui_enabled=1)
else:
    inv_account = json.loads(STATE.read_text())["inventory"]
    inventory = H.Warehouse.__new__(H.Warehouse)
    import duckdb
    inventory.path = RUN / "warehouse.duckdb"
    inventory.con = duckdb.connect(str(inventory.path), read_only=True)

# ── the retail workspace (the new core) ───────────────────────────────────────
from evals.core2.domains import retail  # noqa: E402
from evals.core2.framework import materialize  # noqa: E402

built = materialize(retail.build(), "descriptive")


class RetailWarehouse(H.Warehouse):
    def __init__(self, con):
        self.con = con


retail_wh = RetailWarehouse(built.con)

from core2.warehouse import querybot as qbw  # noqa: E402
from core2.warehouse.runner import DuckDBWarehouse  # noqa: E402


class _Connection(DuckDBWarehouse):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


qbw.QueryBotWarehouse = lambda db_type, credentials: _Connection(built.con)

import core.schema as schema_module  # noqa: E402


def run_azure_sql(cfg, sql, max_rows=200):
    target = retail_wh if (cfg or {}).get("server") == "retail" else inventory
    return target.query(sql, max_rows)


schema_module._run_azure_sql = run_azure_sql

if fresh:
    from tests.test_core2_learned_page_shows_what_was_learned import _schema_file

    store.upsert_client(RETAIL, "portal")
    store.save_compliance_profile(RETAIL, mode="standard")
    rid = store.save_db_config("azure_sql", "Retail warehouse", {"server": "retail", "database": "d", "user": "u",
                                                                  "password": "p"})
    # Feedback on: the check sees the whole action row a workspace can have.
    store.update_client_meta(RETAIL, client_name="Retail (invented)", db_config_id=rid, chat_ui_enabled=1,
                             enable_feedback_collection=1)
    schema_dir = RUN / "clients" / RETAIL / "schema"
    store.update_client_state(RETAIL, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    from admin import core2_routes

    core2_routes._run_build(RETAIL)
    store.set_query_engine(RETAIL, "core2")
    from admin import credentials as admin_credentials

    admin_credentials.claim_first(ADMIN_PASSWORD)
    # A reader sees only the tables their group is given: the check reads as one, with the whole warehouse.
    sales = store.create_group(RETAIL, "Sales team", "Every table of the retail warehouse")
    store.set_group_tables(sales, RETAIL, [f"MEMORY.MAIN.{physical}" for physical in built.tables.values()])
    store.create_user(RETAIL, "Riley Reader", READER[0], group_id=sales, role="analyst", password=READER[1])
    store.create_user(inv_account, "Ines Inventory", "ines@example.com", role="admin", password=READER[1])
    store.create_user(RETAIL, "Adi Admin", "adi@example.com", role="admin", password=READER[1])
    # An Azure deployment with no model or price on file yet: System asks for its price.
    from core import llm
    from core.llm_audit import llm_audit_scope
    from core.llm_prices import Usage

    store.log_query(inv_account, "stock on hand by warehouse", "SELECT 1", row_count=4, question_id="ui-check-azure",
                    llm_provider="azure_openai", llm_model="prod-4o")
    for component in ("sql_generation", "analysis"):
        with llm_audit_scope(account_id=inv_account, question="stock on hand by warehouse", enabled=False,
                             question_id="ui-check-azure", component=component):
            llm._record_usage("azure_openai", "prod-4o", Usage(input=2400, cached_input=6144, output=420), "success")
    STATE.write_text(json.dumps({"inventory": inv_account, "retail": RETAIL}))

# ── scripted plans in place of the AI ─────────────────────────────────────────
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
SINCE = {"kind": "between", "start": "2025-01-01", "end": "2026-06-30"}
PLANS = {
    "what was net sales in april 2026": {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}},
    "net sales by store in the first half of 2026": {"intent": "breakdown", "measures": ["net_amount"],
                                                     "group_by": ["store"], "time": {"window": H1}},
    "monthly net sales since january 2025": {"intent": "trend", "measures": ["net_amount"],
                                             "time": {"grain": "month", "window": SINCE}},
    "net sales by region by month in 2026": {"intent": "trend", "measures": ["net_amount"], "group_by": ["region.name"],
                                             "time": {"grain": "month", "window": H1}},
    "top 5 products by net sales in 2026": {"intent": "rank", "measures": ["net_amount"], "group_by": ["product.name"],
                                            "sort": [{"by": "net_amount", "desc": True}], "limit": 5,
                                            "time": {"window": H1}},
    "who were our top 10 customers in the first half of 2026": {
        "intent": "share", "measures": ["net_amount"], "group_by": ["customer"],
        "sort": [{"by": "net_amount", "desc": True}], "limit": 10, "time": {"window": H1}},
    "share of net sales by customer segment": {"intent": "share", "measures": ["net_amount"],
                                               "group_by": ["customer.segment"], "time": {"window": H1}},
    "compare net sales by store in april against march": {
        "intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
        "time": {"window": APRIL, "compare": {"kind": "previous_period"}}},
    "net sales, cost and margin by category": {"intent": "breakdown",
                                               "measures": ["net_amount", "cost_amount", "margin_percent"],
                                               "group_by": ["category.name"], "time": {"window": H1}},
    "why did net sales drop in april": {"intent": "drivers", "measures": ["net_amount"], "time": {"window": APRIL}},
    "forecast net sales for the next 3 months": {"intent": "forecast", "measures": ["net_amount"],
                                                 "time": {"grain": "month"}, "forecast": {"periods": 3}},
    "net sales, cost and gross amount in 2025": {"intent": "value",
                                                 "measures": ["net_amount", "cost_amount", "gross_amount"],
                                                 "time": {"window": {"kind": "between", "start": "2025-01-01",
                                                                     "end": "2025-12-31"}}},
    "list the stores": {"intent": "list", "group_by": ["store"]},
    "net sales by product in 2026": {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["product.name"],
                                     "time": {"window": H1}},
    "how many customers ordered in april": {"intent": "count", "measures": ["number_of_customers"],
                                            "time": {"window": APRIL}},
    "what data do you have": {"kind": "describe_data"},
    # the first question "What you can ask" offers, followed from that page
    "net amount by store in 2026": {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                                    "time": {"window": H1}},
}


def _scripted(question_text: str) -> str:
    text = question_text.lower()
    best = max((k for k in PLANS if k in text), key=lambda k: text.rfind(k), default=None)
    if best is None:
        return json.dumps({"kind": "smalltalk", "notes": ["This local copy answers only its scripted questions."]})
    plan = PLANS[best]
    return json.dumps({"kind": plan.get("kind", "query"), **{k: v for k, v in plan.items() if k != "kind"}})


import core2.bootstrap.ai as ai  # noqa: E402


def workspace_planner(account_id, client, *, question="", question_id=""):
    from core import llm
    from core.llm_audit import llm_audit_scope
    from core.llm_prices import Usage

    def complete(stable, tail):
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        asked = next((ln for ln in reversed(lines) if re.search("[A-Za-z]", ln) and ln.lower().strip(" :").startswith(
            ("question", "user", "q"))), "") or tail
        # Recorded and priced as a real planner call is: only the provider's token counts are invented.
        with llm_audit_scope(account_id=account_id, question=question, enabled=False,
                             question_id=question_id, component="core2_planner"):
            llm._record_usage("anthropic", "claude-sonnet-4-6",
                              Usage(input=len(tail) // 4, cached_input=len(stable) // 4, output=180), "success")
        return _scripted(question or asked or tail)
    return complete


ai.workspace_planner = workspace_planner

if __name__ == "__main__":
    import uvicorn

    print(json.dumps({"inventory": inv_account, "retail": RETAIL, "admin_password": ADMIN_PASSWORD,
                      "reader": READER}), flush=True)
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, log_level="warning")
