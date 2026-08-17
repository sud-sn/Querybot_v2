import sys, os
sys.path.insert(0, '/home/chatbotadmin/Querybot_v2')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir('/home/chatbotadmin/Querybot_v2')
from dotenv import load_dotenv
load_dotenv('/home/chatbotadmin/Querybot_v2/.env')
import store

checks = [
    ("clients",        "SELECT COUNT(*) FROM client"),
    ("db_configs",     "SELECT COUNT(*) FROM db_config"),
    ("portal_users",   "SELECT COUNT(*) FROM portal_user"),
    ("entities",       "SELECT COUNT(*) FROM entity_graph"),
    ("metrics",        "SELECT COUNT(*) FROM metric_registry"),
    ("relationships",  "SELECT COUNT(*) FROM entity_relationships"),
]
print("=== Current DB state ===")
for label, sql in checks:
    try:
        with store.get_db() as conn:
            row = conn.execute(sql).fetchone()
            n = row[0] if row else 0
            print(f"  {label}: {n}")
    except Exception as e:
        print(f"  {label}: ERROR - {e}")

# List clients with full detail
print("\n=== Client detail ===")
try:
    import json
    with store.get_db() as conn:
        rows = conn.execute("SELECT account_id, client_name, db_config_id, state, llm_provider, llm_model FROM client").fetchall()
        for r in rows:
            print(f"  {dict(r)}")
        if not rows:
            print("  (none)")
except Exception as e:
    print(f"  ERROR: {e}")

# State data for accounts with DB
print("\n=== Accounts with DB config ===")
try:
    with store.get_db() as conn:
        rows = conn.execute("SELECT account_id, state_data FROM client WHERE db_config_id IS NOT NULL").fetchall()
        for r in rows:
            sd = r['state_data']
            if isinstance(sd, str):
                try: sd = json.loads(sd)
                except: pass
            print(f"  {r['account_id']}: schema_dir={sd.get('schema_dir') if isinstance(sd,dict) else sd}")
        if not rows:
            print("  (none)")
except Exception as e:
    print(f"  ERROR: {e}")
