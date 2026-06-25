"""
One-time migration: delete all heuristic/LLM suggested relationships
and rebuild them with the new star-schema-aware algorithm.

Run from the project root:
    python3 migrate_clean_rels.py [ACCOUNT_ID]
"""
import sys, os
sys.path.insert(0, "/home/chatbotadmin/Querybot_v2")
os.chdir("/home/chatbotadmin/Querybot_v2")

import sqlite3
import store
from store.config_store import upsert_relationship_by_pair
from core.schema import build_entity_graph_from_schema

DB      = "/home/chatbotadmin/Querybot_v2/data/querybot.db"
ACCOUNT = sys.argv[1] if len(sys.argv) > 1 else "Web_UI"

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

# ── Step 1: count before ─────────────────────────────────────────────────────
cur.execute(
    "SELECT COUNT(*) as n FROM entity_relationships "
    "WHERE account_id=? AND status='suggested' AND is_active=1",
    (ACCOUNT,),
)
before = cur.fetchone()["n"]
print(f"Before: {before} suggested relationships for account '{ACCOUNT}'")

# ── Step 2: delete all auto-generated suggested relationships ─────────────────
cur.execute(
    """DELETE FROM entity_relationships
       WHERE account_id=?
         AND status='suggested'
         AND is_active=1""",
    (ACCOUNT,),
)
deleted = cur.rowcount
con.commit()
print(f"Deleted {deleted} stale suggested relationships")

# ── Step 3: find schema_dir for this account ──────────────────────────────────
import json as _json
schema_dir = None
cli = store.get_client(ACCOUNT)
if cli:
    try:
        state_data = _json.loads(cli.get("state_data") or "{}")
        schema_dir = state_data.get("schema_dir", "")
        # Make absolute if relative
        if schema_dir and not os.path.isabs(schema_dir):
            schema_dir = os.path.join("/home/chatbotadmin/Querybot_v2", schema_dir)
    except Exception as e:
        print(f"Warning: could not parse state_data: {e}")
print(f"Schema dir: {schema_dir}")

if not schema_dir or not os.path.exists(schema_dir):
    print("ERROR: schema_dir not found. Re-run Suggest from the UI after deploying the code fix.")
    sys.exit(1)

# ── Step 4: rebuild with new algorithm ───────────────────────────────────────
graph        = build_entity_graph_from_schema(schema_dir)
all_entities = {e["entity_name"] for e in store.list_entities(ACCOUNT, active_only=False)}

saved = 0
skipped = 0
for rel in graph["relationships"]:
    if rel["from_entity"] not in all_entities or rel["to_entity"] not in all_entities:
        skipped += 1
        continue
    upsert_relationship_by_pair(
        account_id        = ACCOUNT,
        from_entity       = rel["from_entity"],
        to_entity         = rel["to_entity"],
        from_column       = rel["from_column"],
        to_column         = rel["to_column"],
        relationship_type = rel.get("relationship_type", "many_to_one"),
        join_type         = rel.get("join_type", "INNER"),
        label             = rel.get("label", ""),
        confidence_score  = rel.get("confidence_score", 70),
        status            = "suggested",
        generated_by      = "heuristic",
        reason            = rel.get("reason", ""),
    )
    saved += 1

print(f"Saved {saved} new clean relationships  ({skipped} skipped — entities not in graph)")

# ── Step 5: show final state ──────────────────────────────────────────────────
cur.execute(
    """SELECT from_entity, to_entity, from_column, to_column, confidence_score, status
       FROM entity_relationships
       WHERE account_id=? AND is_active=1
       ORDER BY from_entity, to_entity""",
    (ACCOUNT,),
)
rows = cur.fetchall()
print(f"\nFinal: {len(rows)} total active relationships")
for r in rows:
    flag = "✓" if r["status"] == "confirmed" else "~"
    print(f"  {flag} {r['from_entity']:30} --[{r['from_column']}]--> "
          f"{r['to_entity']:30}  conf={r['confidence_score']}")

con.close()
print("\nDone.")
