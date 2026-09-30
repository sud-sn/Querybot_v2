"""
QueryBot v2 — main entry point

Per-user access control flow:
  1. User messages bot → Zoom sends accountId + userId
  2. Bot looks up portal_user by zoom_user_id
  3. Unknown user → one-time registration link sent
  4. Registered user → group tables loaded → enforced in RAG + validator
  5. Admin role → unrestricted all-table access

Module layout (post-split):
  core/pipeline_context.py   — state, DB config, rate limits
  core/pipeline_helpers.py   — stateless SQL + formatting utilities
  core/pipeline_trace.py     — observability: trace, log_q, learning candidates
  core/result_renderer.py    — result formatting and _send_results
  core/dispatcher.py         — message routing (dispatch / handle_unregistered_user)
  core/query_pipeline.py     — full query pipeline (handle_query)
  gateway/webhooks.py        — Zoom / Teams / Slack webhooks + WebSocket chat
"""

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import store
from store import crypto
from store.db import init_db
from admin import router as admin_router
from portal import router as portal_router
from gateway.webhooks import router as webhooks_router
from core.release import KB_FORMAT, code_release, kb_rebuild_needed, product_version
from core.web_security import RefuseCrossSiteRequests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("querybot")

app = FastAPI(title="QueryBot", version=product_version())
app.add_middleware(RefuseCrossSiteRequests)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
app.include_router(admin_router)
app.include_router(portal_router)
app.include_router(webhooks_router)


def _upgrade_checks() -> None:
    """What an upgrade can leave behind without a word: saved secrets this
    server's key cannot read, and knowledge bases a different release built.
    Each otherwise shows up only when a reader's question fails, or is
    answered from what the old release wrote."""
    log.info("QueryBot %s (release %s, knowledge-base format %d)",
             product_version(), code_release(), KB_FORMAT)
    unreadable = store.unreadable_credentials()
    if unreadable:
        named = ", ".join(f"{item['kind']} {item['name']!r}" for item in unreadable)
        key_file = unreadable[0]["key_file"]
        if key_file == "missing":
            log.critical(
                "The key file %s does not exist, and %d saved credentials were encrypted with the "
                "one it held: %s. The first save writes a new key, under which none of them can be "
                "read. Stop the service and restore the key file (docs/UPGRADE_RUNBOOK.md, sections "
                "2 and 3), or enter every one of them again.",
                crypto.KEY_FILE, len(unreadable), named)
        elif key_file == "unreadable":
            log.critical(
                "The key file %s cannot be opened by this service's user, so %d saved credentials "
                "cannot be read: %s. Let that user read the key file and its directory, or point "
                "QUERYBOT_KEY_FILE at a copy it can read, and restart.",
                crypto.KEY_FILE, len(unreadable), named)
        else:
            log.error(
                "%d saved credentials cannot be read with the key at %s: %s. The key file changed "
                "since they were saved: restore it and restart, or enter them again under Admin > "
                "Databases, Platforms or System.",
                len(unreadable), crypto.KEY_FILE, named)
    for client in store.list_clients():
        stale = kb_rebuild_needed(client)
        if stale:
            log.warning(
                "Workspace %s: its knowledge base was built by %s (format %d); this release builds "
                "format %d. Rebuild it: Admin > the workspace > Setup > Rebuild Knowledge Base.",
                client.get("account_id"), stale["built_version"] or "a release before 2.1.0",
                stale["built_format"], stale["format"])


@app.on_event("startup")
async def startup() -> None:
    init_db()

    try:
        _upgrade_checks()
    except Exception:
        log.error("The upgrade checks could not run at startup", exc_info=True)

    # Warn if session secrets are using insecure defaults
    if not os.getenv("SESSION_SECRET") and not os.getenv("PORTAL_SESSION_SECRET"):
        log.warning(
            "⚠️  SESSION_SECRET / PORTAL_SESSION_SECRET not set — "
            "using insecure default. Set these environment variables "
            "before deploying to production."
        )
    if not os.getenv("ADMIN_SESSION_SECRET") and not os.getenv("SESSION_SECRET"):
        log.warning(
            "⚠️  ADMIN_SESSION_SECRET not set — "
            "admin sessions use an insecure default."
        )
    if not any(os.getenv(name) for name in ("PII_PSEUDONYM_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET")):
        log.warning(
            "PII_PSEUDONYM_SECRET not set - safe display aliases use a development key. "
            "Set a separate random secret before deploying regulated workloads."
        )

    # LLM audit log retention — default 30 days; override via LLM_AUDIT_RETENTION_DAYS
    try:
        retention = int(os.getenv("LLM_AUDIT_RETENTION_DAYS", "30"))
        deleted = store.purge_old_llm_calls(retention)
        if deleted:
            log.info("Purged %d llm_call_log rows older than %d days", deleted, retention)
    except Exception as e:
        log.warning("LLM audit purge failed at startup: %s", e)

    # KB egress log retention — much longer than the LLM-call window: these rows
    # are the record of what schema/sample data ever left the database.
    try:
        egress_retention = int(os.getenv("KB_EGRESS_RETENTION_DAYS", "365"))
        deleted = store.purge_old_kb_egress(egress_retention)
        if deleted:
            log.info(
                "Purged %d kb_data_egress_log rows older than %d days",
                deleted, egress_retention,
            )
    except Exception as e:
        log.warning("KB egress purge failed at startup: %s", e)

    try:
        from core.log_export import scheduled_log_export_loop
        app.state.log_export_task = asyncio.create_task(scheduled_log_export_loop())
        log.info("External log export scheduler started")
    except Exception as e:
        log.warning("External log export scheduler failed to start: %s", e)

    try:
        from core.notification_scheduler import scheduled_notification_loop
        app.state.notification_task = asyncio.create_task(scheduled_notification_loop())
        log.info("Notification scheduler started")
    except Exception as e:
        log.warning("Notification scheduler failed to start: %s", e)

    # Pre-warm vector store singletons so the first user query is not slow.
    # SentenceTransformer (~90 MB) + CrossEncoder (~22 MB) + Qdrant TCP connect
    # all lazy-load on the first query — moving them here eliminates that spike.
    async def _warmup():
        try:
            from core.vector_store import _qdrant, _embedder, _get_cross_encoder
            await asyncio.to_thread(_embedder)           # ~4–6 s: loads all-MiniLM-L6-v2
            await asyncio.to_thread(_get_cross_encoder)  # ~1–2 s: loads ms-marco cross-encoder
            await asyncio.to_thread(_qdrant)             # ~0.3 s: TCP connect + collection check
            log.info("Vector store warm-up complete")
        except Exception as exc:
            log.warning("Vector store warm-up failed (non-fatal): %s", exc)

    from core.background_tasks import spawn
    spawn(_warmup(), name="vector-store-warmup")

    # Notify active Teams users the service is back up — symmetric to the
    # "signing off" notification in the shutdown handler below.
    try:
        import json

        with store.db.get_db() as conn:
            startup_users = conn.execute(
                "SELECT p.account_id, p.platform_user_id, p.display_name, p.conversation_ref "
                "FROM pending_platform_user p "
                "LEFT JOIN portal_user pu ON p.portal_user_id = pu.id "
                "WHERE p.status='approved' AND p.platform_type='teams' "
                "AND COALESCE(pu.is_active, 1) = 1"
            ).fetchall()

        if startup_users:
            teams_platforms = store.list_platforms("teams")
            active_teams = [p for p in teams_platforms if p.get("is_active")]
            if active_teams:
                from gateway.teams_adapter import TeamsAdapter
                from gateway.base import PlatformEvent
                adapter = TeamsAdapter(active_teams[0]["credentials"])

                async def notify_teams_user_startup(user):
                    try:
                        conv_ref = json.loads(user["conversation_ref"] or "{}")
                        if not conv_ref.get("service_url"):
                            return
                        name = (user["display_name"] or "").split(" ")[0]
                        greeting = f"Hey {name}" if name else "Hello"

                        event = PlatformEvent(
                            account_id = user["account_id"],
                            user_id    = user["platform_user_id"],
                            channel_id = user["conversation_ref"],
                            text       = "",
                            platform   = "teams",
                        )
                        await adapter.send_message(
                            event,
                            f"👋 {greeting}, I'm up and running — ready to analyze your data!"
                        )
                    except Exception as e:
                        log.debug("Failed to notify Teams user on startup: %s", e)

                # Run concurrently — don't let one slow/broken conversation_ref
                # delay the rest or block server startup.
                await asyncio.gather(*(notify_teams_user_startup(u) for u in startup_users), return_exceptions=True)
    except Exception as exc:
        log.warning("Failed to send startup notifications: %s", exc)

    log.info("QueryBot v2 started — database ready")


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "log_export_task", None)
    if task:
        task.cancel()

    notification_task = getattr(app.state, "notification_task", None)
    if notification_task:
        notification_task.cancel()

    try:
        from core.portal_notifications import portal_notification_hub
        import json
        
        # 1. Notify portal users via WebSocket
        targets = list(portal_notification_hub._meta.keys())
        if targets:
            payload = {
                "type": "system_message", 
                "message": "👋 QueryBot is signing off for now! Service has been temporarily stopped."
            }
            await portal_notification_hub._broadcast(targets, payload)
            
        # 2. Notify active Teams users proactively
        with store.db.get_db() as conn:
            users = conn.execute(
                "SELECT p.account_id, p.platform_user_id, p.display_name, p.conversation_ref "
                "FROM pending_platform_user p "
                "LEFT JOIN portal_user pu ON p.portal_user_id = pu.id "
                "WHERE p.status='approved' AND p.platform_type='teams' "
                "AND COALESCE(pu.is_active, 1) = 1"
            ).fetchall()
            
        if users:
            teams_platforms = store.list_platforms("teams")
            active_teams = [p for p in teams_platforms if p.get("is_active")]
            if active_teams:
                from gateway.teams_adapter import TeamsAdapter
                from gateway.base import PlatformEvent
                adapter = TeamsAdapter(active_teams[0]["credentials"])
                
                async def notify_teams_user(user):
                    try:
                        conv_ref = json.loads(user["conversation_ref"] or "{}")
                        if not conv_ref.get("service_url"):
                            return
                        name = (user["display_name"] or "").split(" ")[0]
                        greeting = f"Hey {name}" if name else "Hello"
                        
                        event = PlatformEvent(
                            account_id = user["account_id"],
                            user_id    = user["platform_user_id"],
                            channel_id = user["conversation_ref"],
                            text       = "",
                            platform   = "teams",
                        )
                        await adapter.send_message(
                            event,
                            f"👋 {greeting}, I'm signing off for now! The service has been temporarily stopped."
                        )
                    except Exception as e:
                        log.debug("Failed to notify Teams user on shutdown: %s", e)
                
                # Run concurrently to avoid delaying shutdown timeout
                await asyncio.gather(*(notify_teams_user(u) for u in users), return_exceptions=True)
                
    except Exception as exc:
        log.warning("Failed to send shutdown notifications: %s", exc)



@app.get("/health")
async def health():
    clients = store.list_clients()
    return {
        "status":  "ok",
        "version": product_version(),
        "release": code_release(),
        "kb_format": KB_FORMAT,
        "clients": len(clients),
        "ready":   sum(1 for c in clients if c["state"] == "READY"),
        # Counts, not names: this page answers without signing in.
        "kb_rebuild_needed": sum(1 for c in clients if kb_rebuild_needed(c)),
        "unreadable_credentials": len(store.unreadable_credentials()),
    }


@app.get("/")
async def root():
    return RedirectResponse("/admin")
