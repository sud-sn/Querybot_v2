"""
One-time fix script — run via paramiko on the server.
1. Fix entity_graph table_name for 5 confirmed entities with empty table_name
2. Read pharmacy schema to find column names
3. Add metric_registry entries for pharmacy revenue, gross profit, on-hand inventory
"""
import sys, json, sqlite3, datetime, os

sys.stdout.reconfigure(encoding="utf-8")

DB = "/home/chatbotadmin/Querybot_v2/data/querybot.db"
ACCOUNT = "Web_UI"
NOW = datetime.datetime.utcnow().isoformat()

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

# ── 1. Check pharmacy schema for column names ─────────────────────────────────
schema_file = "/home/chatbotadmin/Querybot_v2/clients/Web_UI/schema/_schema.json"
schema = {}
if os.path.exists(schema_file):
    with open(schema_file, encoding="utf-8") as f:
        schema = json.load(f)
print(f"Schema tables ({len(schema)}): {list(schema.keys())}\n")

for tname, tdata in schema.items():
    cols = [c.get("column_name") or c.get("name", "") for c in (tdata.get("columns") or [])]
    if any(x in tname for x in ["Prescription", "ITM_BAL"]):
        print(f"{tname}: {cols}")

# ── 2. Fix entity_graph table_names ──────────────────────────────────────────
print("\n--- Entity graph table_name fixes ---")
fixes = [
    ("DIM_Patient",         "DIM_Patient"),
    ("DIM_Prescriber",      "DIM_Prescriber"),
    ("DIM_Staff",           "DIM_Staff"),
    ("DIM_Supplier",        "DIM_Supplier"),
    ("Valid Delivery Date", "DT_DMS"),
]
for entity_name, table_name in fixes:
    cur.execute(
        "UPDATE entity_graph SET table_name=? "
        "WHERE account_id=? AND entity_name=? AND (table_name IS NULL OR table_name='')",
        (table_name, ACCOUNT, entity_name),
    )
    print(f"  {entity_name} -> '{table_name}': {cur.rowcount} row(s) updated")
db.commit()

cur.execute(
    "SELECT entity_name, table_name, status FROM entity_graph "
    "WHERE account_id=? AND entity_name IN ('DIM_Patient','DIM_Prescriber','DIM_Staff','DIM_Supplier','Valid Delivery Date') "
    "ORDER BY entity_name",
    (ACCOUNT,),
)
for r in cur.fetchall():
    print(f"  verified: {r['entity_name']} -> table='{r['table_name']}' [{r['status']}]")

# ── 3. Existing metric_registry entries ───────────────────────────────────────
cur.execute("SELECT name, base_table, sql_template FROM metric_registry WHERE account_id=? AND is_active=1", (ACCOUNT,))
existing = cur.fetchall()
print(f"\n--- metric_registry: {len(existing)} existing active ---")
for r in existing:
    print(f"  {r['name']} | table={r['base_table']} | sql={r['sql_template'][:70]}")

# ── 4. Add pharmacy metrics ───────────────────────────────────────────────────
print("\n--- Adding pharmacy metric registry entries ---")

def upsert_metric(name, synonyms, sql_template, description, base_table, required_columns, example_questions, result_format="currency"):
    cur.execute("SELECT id FROM metric_registry WHERE account_id=? AND name=?", (ACCOUNT, name))
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE metric_registry SET synonyms=?, sql_template=?, description=?, base_table=?, "
            "required_columns=?, example_questions=?, result_format=?, is_active=1, updated_at=? "
            "WHERE account_id=? AND name=?",
            (synonyms, sql_template, description, base_table, required_columns, example_questions, result_format, NOW, ACCOUNT, name),
        )
        print(f"  Updated: {name}")
    else:
        cur.execute(
            "INSERT INTO metric_registry (account_id, name, synonyms, sql_template, description, "
            "formula_type, base_table, required_columns, example_questions, result_format, grain, "
            "is_active, category, created_at, updated_at) "
            "VALUES (?,?,?,?,?, 'expression',?,?,?, ?,NULL, 1,'pharmacy',?,?)",
            (ACCOUNT, name, synonyms, sql_template, description,
             base_table, required_columns, example_questions, result_format, NOW, NOW),
        )
        print(f"  Inserted: {name}")

upsert_metric(
    name="Pharmacy Revenue",
    synonyms="total revenue, total charges, revenue, charges, billing, prescription revenue, sales",
    sql_template="SUM(Total_Charge_USD)",
    description="Total billed amount charged to customers for pharmacy prescription fills. Use column Total_Charge_USD from FACT_Prescription_Fill.",
    base_table="CHATBOT_DB.PHARMACY.FACT_Prescription_Fill",
    required_columns="Total_Charge_USD",
    example_questions="What is total pharmacy revenue? What are total charges this month? Show revenue by prescriber",
    result_format="currency",
)

upsert_metric(
    name="Pharmacy Gross Profit",
    synonyms="gross profit, profit, margin, net margin, pharmacy profit",
    sql_template="SUM(Gross_Profit_USD)",
    description="Total gross profit from pharmacy prescription fills. Use column Gross_Profit_USD from FACT_Prescription_Fill.",
    base_table="CHATBOT_DB.PHARMACY.FACT_Prescription_Fill",
    required_columns="Gross_Profit_USD",
    example_questions="What is gross profit? Show profit by product. Total pharmacy margin",
    result_format="currency",
)

upsert_metric(
    name="On-Hand Inventory Quantity",
    synonyms="on hand inventory, current inventory, stock on hand, inventory quantity, warehouse stock, current stock",
    sql_template="SUM(TRY_CAST(CUR_ON_HND_QTY AS DECIMAL(18,4)))",
    description="Total on-hand inventory quantity. CUR_ON_HND_QTY is stored as nvarchar — TRY_CAST to DECIMAL is required to avoid numeric conversion error.",
    base_table="CHATBOT_DB.PROFITABILITY.ITM_BAL_PRD_FCT",
    required_columns="CUR_ON_HND_QTY",
    example_questions="What is on-hand inventory by warehouse? Total stock on hand? Current inventory levels",
    result_format="number",
)

db.commit()
print("\nAll changes committed.")
db.close()
print("Done.")
