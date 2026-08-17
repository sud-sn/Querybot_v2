"""Check PostgreSQL state — accounts, DB configs, entity graph, metrics."""
import sys, json, os
sys.path.insert(0, '/home/chatbotadmin/Querybot_v2')
sys.stdout.reconfigure(encoding='utf-8')

os.chdir('/home/chatbotadmin/Querybot_v2')
from dotenv import load_dotenv
load_dotenv('/home/chatbotadmin/Querybot_v2/.env')

import store

def pg_query(sql, params=()):
    with store.get_db() as conn:
        result = conn.execute(sql, params)
        try:
            return result.fetchall()
        except Exception:
            return []

# Clients
print("=== CLIENTS ===")
try:
    rows = pg_query("SELECT account_id, db_config_id, state, llm_provider, llm_model FROM client ORDER BY account_id")
    for r in rows:
        print(f"  {dict(r)}")
except Exception as e:
    print(f"  ERROR: {e}")

# DB Configs
print("\n=== DB CONFIGS ===")
try:
    rows = pg_query("SELECT id, name, db_type FROM db_config")
    for r in rows:
        print(f"  {dict(r)}")
except Exception as e:
    print(f"  ERROR: {e}")

# Entity graph
print("\n=== ENTITY GRAPH ===")
try:
    rows = pg_query("SELECT account_id, COUNT(*) as n FROM entity_graph GROUP BY account_id ORDER BY n DESC LIMIT 10")
    for r in rows:
        print(f"  {dict(r)}")
except Exception as e:
    print(f"  ERROR: {e}")

# Metric registry
print("\n=== METRIC REGISTRY ===")
try:
    rows = pg_query("SELECT account_id, COUNT(*) as n FROM metric_registry WHERE is_active=true GROUP BY account_id")
    for r in rows:
        print(f"  {dict(r)}")
except Exception as e:
    print(f"  ERROR2: {e}")
    # try without true
    try:
        rows = pg_query("SELECT account_id, COUNT(*) as n FROM metric_registry WHERE is_active=1 GROUP BY account_id")
        for r in rows:
            print(f"  {dict(r)}")
    except Exception as e2:
        print(f"  ERROR3: {e2}")

# State data
print("\n=== CLIENT STATE DATA ===")
try:
    rows = pg_query("SELECT account_id, state_data FROM client WHERE db_config_id IS NOT NULL")
    for r in rows:
        sd = r['state_data']
        if isinstance(sd, str):
            try: sd = json.loads(sd)
            except: pass
        if isinstance(sd, dict):
            print(f"  {r['account_id']}: schema_dir={sd.get('schema_dir')} kb_dir={sd.get('kb_dir')}")
        else:
            print(f"  {r['account_id']}: state_data={sd}")
except Exception as e:
    print(f"  ERROR: {e}")
