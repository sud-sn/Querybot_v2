import sys, sqlite3, json
sys.path.insert(0, '/home/chatbotadmin/Querybot_v2')
sys.stdout.reconfigure(encoding='utf-8')

DB = '/home/chatbotadmin/Querybot_v2/data/querybot.db'
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
cur = db.cursor()

# All tables schema
cur.execute("PRAGMA table_info(client)")
client_cols = [r['name'] for r in cur.fetchall()]
print('Client columns:', client_cols)

# All accounts anywhere
print('\nAll accounts with entity_graph entries:')
cur.execute("SELECT DISTINCT account_id, COUNT(*) as n FROM entity_graph GROUP BY account_id")
for r in cur.fetchall():
    print(f'  {r["account_id"]} ({r["n"]} entities)')

print('\nAll accounts with metric_registry entries:')
cur.execute("SELECT DISTINCT account_id, COUNT(*) as n FROM metric_registry GROUP BY account_id")
for r in cur.fetchall():
    print(f'  {r["account_id"]} ({r["n"]} metrics)')

print('\nAll accounts with db_config entries:')
cur.execute("PRAGMA table_info(db_config)")
dc_cols = [r['name'] for r in cur.fetchall()]
print('  db_config columns:', dc_cols)
cur.execute("SELECT * FROM db_config LIMIT 5")
for r in cur.fetchall():
    print(f'  {dict(r)}')

# Client table
print('\nClient entries:')
cur.execute("SELECT * FROM client")
for r in cur.fetchall():
    d = dict(r)
    if 'state_data' in d and d['state_data']:
        try:
            sd = json.loads(d['state_data'])
            d['state_data'] = {'schema_dir': sd.get('schema_dir'), 'kb_dir': sd.get('kb_dir')}
        except: pass
    print(f'  {d}')

db.close()
