"""
migrate_env.py

One-time migration: moves credentials from your old .env file into the
new encrypted SQLite config store.

Usage:
    python migrate_env.py --env /path/to/old/.env

After running, the .env file is renamed to .env.bak and is no longer needed.
"""

import argparse
import sys
from pathlib import Path


def parse_env(path: str) -> dict:
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def run(env_path: str) -> None:
    print(f"\nReading {env_path}...")
    env = parse_env(env_path)

    sys.path.insert(0, str(Path(__file__).parent))
    from store.db import init_db
    from store.config_store import set_system, save_platform, save_db_config

    init_db()
    print("Database initialised.")

    ok, warn = [], []

    # ── System config ──────────────────────────────────────────────────────────
    if env.get("ANTHROPIC_API_KEY"):
        set_system("anthropic_api_key", env["ANTHROPIC_API_KEY"])
        ok.append("Anthropic API key")
    else:
        warn.append("ANTHROPIC_API_KEY not found")

    if env.get("OPENAI_API_KEY"):
        set_system("openai_api_key", env["OPENAI_API_KEY"])
        ok.append("OpenAI API key")

    set_system("default_llm_provider", env.get("DEFAULT_LLM_PROVIDER", "anthropic"))
    set_system("default_llm_model",    env.get("LLM_MODEL", "claude-sonnet-4-6"))
    set_system("kb_llm_model",         env.get("KB_LLM_MODEL", "claude-opus-4-5"))
    ok.append("LLM settings")

    # ── Zoom platform ──────────────────────────────────────────────────────────
    zoom = {
        "client_id":      env.get("ZOOM_CLIENT_ID", ""),
        "client_secret":  env.get("ZOOM_CLIENT_SECRET", ""),
        "bot_jid":        env.get("ZOOM_BOT_JID", ""),
        "webhook_secret": env.get("ZOOM_WEBHOOK_SECRET_TOKEN", ""),
    }
    if all(zoom.values()):
        save_platform("zoom", "Zoom (migrated from .env)", zoom)
        ok.append("Zoom platform credentials")
    else:
        missing = [k for k, v in zoom.items() if not v]
        warn.append(f"Zoom credentials incomplete — missing: {missing}")

    # ── Database ───────────────────────────────────────────────────────────────
    db_type = env.get("DB_TYPE", "snowflake").lower()

    if db_type == "snowflake":
        creds = {
            "account":   env.get("SF_ACCOUNT", ""),
            "user":      env.get("SF_USER", ""),
            "password":  env.get("SF_PASSWORD", ""),
            "warehouse": env.get("SF_WAREHOUSE", ""),
            "database":  env.get("SF_DATABASE", ""),
            "schema":    env.get("SF_SCHEMA", "PUBLIC"),
            "role":      env.get("SF_ROLE", ""),
        }
        name = f"Snowflake {creds['database']} (migrated)"
        required = ["account", "user", "password", "warehouse", "database"]
    elif db_type == "oracle":
        creds = {
            "user":     env.get("ORA_USER", ""),
            "password": env.get("ORA_PASSWORD", ""),
            "dsn":      env.get("ORA_DSN", ""),
            "schema":   env.get("ORA_SCHEMA", ""),
        }
        name = f"Oracle {creds['dsn']} (migrated)"
        required = ["user", "password", "dsn"]
    else:
        creds = {
            "server":   env.get("AZ_SERVER", ""),
            "database": env.get("AZ_DATABASE", ""),
            "user":     env.get("AZ_USER", ""),
            "password": env.get("AZ_PASSWORD", ""),
            "schema":   env.get("AZ_SCHEMA", "dbo"),
            "driver":   env.get("AZ_DRIVER", "ODBC Driver 18 for SQL Server"),
        }
        name = f"Azure SQL {creds['database']} (migrated)"
        required = ["server", "database", "user", "password"]

    missing_db = [f for f in required if not creds.get(f)]
    if not missing_db:
        save_db_config(db_type, name, creds)
        ok.append(f"{db_type} database credentials")
    else:
        warn.append(f"DB credentials incomplete — missing: {missing_db}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\nMigration results:")
    for item in ok:
        print(f"  ✓ {item}")
    for item in warn:
        print(f"  ✗ {item}")

    if ok:
        bak = env_path + ".bak"
        Path(env_path).rename(bak)
        print(f"\n.env renamed to {bak}")
        print("It is no longer needed. Keep it as a backup in a secure location.")

    print("\nDone. Start the bot with:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")
    print("Then open http://YOUR-VM-IP:8000/admin to complete setup.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate .env to QueryBot encrypted store")
    parser.add_argument("--env", default=".env", help="Path to old .env file")
    args = parser.parse_args()
    if not Path(args.env).exists():
        print(f"Error: {args.env} not found")
        sys.exit(1)
    run(args.env)
