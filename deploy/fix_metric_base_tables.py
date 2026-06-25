"""
Set base_table on existing profitability metrics that have no base_table.
This lets metric_scope.py filter them out when the pharmacy schema is active,
preventing cross-schema metric injection.
"""
import sqlite3, datetime

DB = "/home/chatbotadmin/Querybot_v2/data/querybot.db"
ACCOUNT = "Web_UI"

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

fixes = [
    ("Revenue",         "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT"),
    ("COGS",            "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT"),
    ("Gross Margin",    "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT"),
    ("Vendor Rebates",  "CHATBOT_DB.PROFITABILITY.FIFO_BI_SAL_MGP_EXT"),
    ("General Expense", "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT"),
]
print("Setting base_table on existing profitability metrics:")
for name, base_table in fixes:
    cur.execute(
        "UPDATE metric_registry SET base_table=? WHERE account_id=? AND name=? AND (base_table IS NULL OR base_table='')",
        (base_table, ACCOUNT, name),
    )
    print(f"  {name} -> {base_table}: {cur.rowcount} row(s)")
db.commit()

cur.execute(
    "SELECT name, base_table, formula_type FROM metric_registry WHERE account_id=? AND is_active=1 ORDER BY name",
    (ACCOUNT,),
)
print("\nAll active metrics:")
for r in cur.fetchall():
    print(f"  [{r['formula_type'] or 'query'}] {r['name']} | table={r['base_table']}")
db.close()
print("Done.")
