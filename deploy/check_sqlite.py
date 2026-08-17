import sys, json, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

DB = '/home/chatbotadmin/Querybot_v2/data/querybot.db'
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

print("=== CLIENTS ===")
cur.execute("SELECT account_id, client_name, db_config_id, state, llm_provider, llm_model FROM client ORDER BY account_id")
for r in cur.fetchall():
    print(f"  {dict(r)}")

print("\n=== DB CONFIGS ===")
cur.execute("SELECT id, name, db_type FROM db_config LIMIT 5")
for r in cur.fetchall():
    print(f"  {dict(r)}")

print("\n=== ENTITY GRAPH (by account) ===")
cur.execute("SELECT account_id, COUNT(*) as n FROM entity_graph GROUP BY account_id ORDER BY n DESC")
for r in cur.fetchall():
    print(f"  {dict(r)}")

print("\n=== METRIC REGISTRY (by account) ===")
cur.execute("SELECT account_id, COUNT(*) as n FROM metric_registry WHERE is_active=1 GROUP BY account_id")
for r in cur.fetchall():
    print(f"  {dict(r)}")

print("\n=== STATE DATA (accounts with DB) ===")
cur.execute("SELECT account_id, state_data FROM client WHERE db_config_id IS NOT NULL")
for r in cur.fetchall():
    sd = r['state_data']
    if sd:
        try: sd = json.loads(sd)
        except: pass
    if isinstance(sd, dict):
        print(f"  {r['account_id']}: schema_dir={sd.get('schema_dir')} kb_dir={sd.get('kb_dir')}")
    else:
        print(f"  {r['account_id']}: state_data={sd}")

print("\n=== PORTAL USERS ===")
cur.execute("SELECT user_id, account_id, role FROM portal_user LIMIT 5")
for r in cur.fetchall():
    print(f"  {dict(r)}")

db.close()
