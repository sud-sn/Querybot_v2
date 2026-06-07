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
from store.db import init_db
from admin import router as admin_router
from portal import router as portal_router
from gateway.webhooks import router as webhooks_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("querybot")

app = FastAPI(title="QueryBot", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
app.include_router(admin_router)
app.include_router(portal_router)
app.include_router(webhooks_router)


@app.on_event("startup")
async def startup() -> None:
    init_db()

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

    # LLM audit log retention — default 30 days; override via LLM_AUDIT_RETENTION_DAYS
    try:
        retention = int(os.getenv("LLM_AUDIT_RETENTION_DAYS", "30"))
        deleted = store.purge_old_llm_calls(retention)
        if deleted:
            log.info("Purged %d llm_call_log rows older than %d days", deleted, retention)
    except Exception as e:
        log.warning("LLM audit purge failed at startup: %s", e)

    try:
        from core.log_export import scheduled_log_export_loop
        app.state.log_export_task = asyncio.create_task(scheduled_log_export_loop())
        log.info("External log export scheduler started")
    except Exception as e:
        log.warning("External log export scheduler failed to start: %s", e)

    log.info("QueryBot v2 started — database ready")


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "log_export_task", None)
    if task:
        task.cancel()


@app.get("/health")
async def health():
    clients = store.list_clients()
    return {
        "status":  "ok",
        "version": "2.0.0",
        "clients": len(clients),
        "ready":   sum(1 for c in clients if c["state"] == "READY"),
    }


@app.get("/")
async def root():
    return RedirectResponse("/admin")
