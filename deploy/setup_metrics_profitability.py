"""
Insert profitability metric registry entries for the Web_UI account.
Also fix the CUR_ON_HND_QTY nvarchar issue via TRY_CAST.
"""
import sys, sqlite3, datetime
sys.path.insert(0, '/home/chatbotadmin/Querybot_v2')
sys.stdout.reconfigure(encoding='utf-8')

DB = '/home/chatbotadmin/Querybot_v2/data/querybot.db'
ACCOUNT = 'Web_UI'
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

def upsert(name, synonyms, sql_template, description, base_table, required_columns,
           example_questions, result_format='currency', formula_type='expression'):
    cur.execute("SELECT id FROM metric_registry WHERE account_id=? AND name=?", (ACCOUNT, name))
    if cur.fetchone():
        cur.execute(
            "UPDATE metric_registry SET synonyms=?, sql_template=?, description=?, base_table=?, "
            "required_columns=?, example_questions=?, result_format=?, formula_type=?, "
            "is_active=1, updated_at=? WHERE account_id=? AND name=?",
            (synonyms, sql_template, description, base_table, required_columns,
             example_questions, result_format, formula_type, NOW, ACCOUNT, name),
        )
        print(f"  Updated: {name}")
    else:
        cur.execute(
            "INSERT INTO metric_registry (account_id, name, synonyms, sql_template, description, "
            "formula_type, base_table, required_columns, example_questions, result_format, "
            "is_active, category, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,1,'profitability',?,?)",
            (ACCOUNT, name, synonyms, sql_template, description, formula_type,
             base_table, required_columns, example_questions, result_format, NOW, NOW),
        )
        print(f"  Inserted: {name}")

print("--- Profitability Metrics ---")

upsert(
    name="Revenue",
    synonyms="revenue, total revenue, sales, invoice amount, billing, turnover, net sales",
    sql_template="SUM(CASE WHEN DEL_IVC_REC_IND = 0 THEN SOP_CUS_IVC_LIN_AMT ELSE 0 END)",
    description="Total customer invoice revenue excluding credit/debit notes. Use SOP_CUS_IVC_LIN_AMT from CUS_ORD_IVC_FCT.",
    base_table="CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    required_columns="SOP_CUS_IVC_LIN_AMT",
    example_questions="What is total revenue? Show revenue by customer. Revenue this year by profit center",
)

upsert(
    name="COGS",
    synonyms="cost of goods sold, cost of sales, cogs, cost, product cost, item cost",
    sql_template="SUM(SOP_CUS_IVC_LIN_CST_AMT)",
    description="Total cost of goods sold from customer invoice lines.",
    base_table="CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    required_columns="SOP_CUS_IVC_LIN_CST_AMT",
    example_questions="What is total cost? Show COGS by customer type. Cost of goods by profit center",
)

upsert(
    name="Gross Margin",
    synonyms="gross margin, gross profit, margin, profit, gp, profitability",
    sql_template="SUM(SOP_CUS_IVC_LIN_AMT) - SUM(SOP_CUS_IVC_LIN_CST_AMT)",
    description="Gross margin = Revenue minus COGS. Use SOP_CUS_IVC_LIN_AMT - SOP_CUS_IVC_LIN_CST_AMT.",
    base_table="CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    required_columns="SOP_CUS_IVC_LIN_AMT, SOP_CUS_IVC_LIN_CST_AMT",
    example_questions="What is gross margin? Show profit by customer. Gross profit by profit center this year",
)

upsert(
    name="Gross Margin Percentage",
    synonyms="gross margin percentage, margin percentage, margin pct, profit margin, gm%, margin %",
    sql_template="CASE WHEN SUM(SOP_CUS_IVC_LIN_AMT) = 0 THEN 0 ELSE (SUM(SOP_CUS_IVC_LIN_AMT) - SUM(SOP_CUS_IVC_LIN_CST_AMT)) / SUM(SOP_CUS_IVC_LIN_AMT) * 100 END",
    description="Gross margin as a percentage of revenue.",
    base_table="CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    required_columns="SOP_CUS_IVC_LIN_AMT, SOP_CUS_IVC_LIN_CST_AMT",
    example_questions="What is margin percentage? Show gross margin % by customer type. Which profit center has highest margin",
    result_format="percentage",
)

upsert(
    name="On-Hand Inventory Quantity",
    synonyms="on hand inventory, current inventory, stock on hand, inventory quantity, warehouse stock, current stock, stock level",
    sql_template="SUM(TRY_CAST(CUR_ON_HND_QTY AS DECIMAL(18,4)))",
    description="Total on-hand inventory quantity. CUR_ON_HND_QTY is nvarchar — TRY_CAST to DECIMAL is required.",
    base_table="CHATBOT_DB.PROFITABILITY.ITM_BAL_PRD_FCT",
    required_columns="CUR_ON_HND_QTY",
    example_questions="What is on-hand inventory by warehouse? Total stock on hand? Current inventory levels by warehouse",
    result_format="number",
)

upsert(
    name="Purchase Order Count",
    synonyms="purchase orders, number of purchase orders, po count, orders received, receipts",
    sql_template="COUNT(DISTINCT PCH_ORD_RCT_FCT_KEY)",
    description="Total number of purchase order receipts.",
    base_table="CHATBOT_DB.PROFITABILITY.PCH_ORD_RCT_FCT",
    required_columns="PCH_ORD_RCT_FCT_KEY",
    example_questions="How many purchase orders were received? Count of POs this month. Total purchase orders by warehouse",
    result_format="number",
)

db.commit()

print("\n--- All active metrics ---")
cur.execute("SELECT name, base_table, result_format FROM metric_registry WHERE account_id=? AND is_active=1 ORDER BY name", (ACCOUNT,))
for r in cur.fetchall():
    print(f"  {r['name']} | {r['base_table']} | {r['result_format']}")

db.close()
print("\nDone.")
