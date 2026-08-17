import sys, json, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

DB = '/home/chatbotadmin/Querybot_v2/data/querybot.db'
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

# Find the real account
cur.execute("SELECT account_id, client_name, db_config_id, state, state_data FROM client WHERE db_config_id IS NOT NULL")
rows = cur.fetchall()
print("=== CLIENTS WITH DB ===")
for r in rows:
    sd = r['state_data']
    if sd:
        try: sd = json.loads(sd)
        except: pass
    schema_dir = sd.get('schema_dir','') if isinstance(sd,dict) else ''
    kb_dir = sd.get('kb_dir','') if isinstance(sd,dict) else ''
    print(f"  account_id={r['account_id']} name={r['client_name']} state={r['state']}")
    print(f"    schema_dir={schema_dir}")
    print(f"    kb_dir={kb_dir}")

# All clients regardless of DB
print("\n=== ALL CLIENTS ===")
cur.execute("SELECT account_id, client_name, db_config_id, state FROM client ORDER BY account_id")
for r in cur.fetchall():
    print(f"  {dict(r)}")

# Entity graph
print("\n=== ENTITY GRAPH ===")
cur.execute("SELECT account_id, COUNT(*) as n FROM entity_graph GROUP BY account_id ORDER BY n DESC")
for r in cur.fetchall():
    print(f"  {dict(r)}")

# Relationships
print("\n=== RELATIONSHIPS ===")
cur.execute("SELECT account_id, COUNT(*) as n FROM entity_relationships WHERE is_active=1 GROUP BY account_id")
for r in cur.fetchall():
    print(f"  {dict(r)}")

# For the accounts that have entities, show them
print("\n=== ENTITY GRAPH DETAIL ===")
cur.execute("""SELECT account_id, entity_name, table_name, entity_type, status
               FROM entity_graph
               WHERE account_id NOT LIKE 'test%'
               ORDER BY account_id, entity_type DESC, entity_name""")
for r in cur.fetchall():
    print(f"  [{r['account_id']}] {r['entity_name']} | table={r['table_name']} | type={r['entity_type']} | status={r['status']}")

db.close()
