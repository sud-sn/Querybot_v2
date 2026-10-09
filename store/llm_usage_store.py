"""Every AI call's tokens and cost, and the admin's own prices (core/llm_prices.py prices them).

One row per call to the model, written by ``core.llm.llm_complete`` whatever asked for
it: a question's planner and its repair, a narrative, a clarification, Learn, an admin
tool. A question's cost is the sum of the rows carrying its ``question_id``.

``query_log`` rows written before this table existed keep their own estimate
(``cost_source = 'estimate'``); rows written since are marked ``'usage'`` and their
cost lives here, so totals add the two without counting anything twice.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from store.db import get_db

log = logging.getLogger("querybot.store")

_TOKEN_COLUMNS = ("input_tokens", "cached_input_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens",
                  "output_tokens")


def record_llm_usage(*, account_id: str, question_id: str, component: str, provider: str, model: str,
                     priced_model: str, deployment_type: str, usage: Any, cost_usd: Optional[float],
                     status: str = "success") -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO llm_usage
                   (account_id, question_id, component, provider, model, priced_model, deployment_type,
                    input_tokens, cached_input_tokens, cache_write_5m_tokens, cache_write_1h_tokens,
                    output_tokens, reasoning_tokens, cost_usd, priced, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (account_id or "", question_id or "", component or "", provider or "", model or "",
             priced_model or "", deployment_type or "", usage.input, usage.cached_input, usage.cache_write_5m,
             usage.cache_write_1h, usage.output, usage.reasoning,
             float(cost_usd) if cost_usd is not None else 0.0, 1 if cost_usd is not None else 0, status))


def _sum_row(row: Any) -> dict[str, Any]:
    data = dict(row) if row else {}
    out = {k: int(data.get(k) or 0) for k in _TOKEN_COLUMNS}
    out["reasoning_tokens"] = int(data.get("reasoning_tokens") or 0)
    out["calls"] = int(data.get("calls") or 0)
    out["unpriced_calls"] = int(data.get("unpriced_calls") or 0)
    out["cost_usd"] = float(data.get("cost_usd") or 0.0)
    out["prompt_tokens"] = sum(out[k] for k in _TOKEN_COLUMNS if k != "output_tokens")
    out["total_tokens"] = out["prompt_tokens"] + out["output_tokens"]
    return out


_SUMS = """COUNT(*) AS calls,
           COALESCE(SUM(input_tokens), 0) AS input_tokens,
           COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
           COALESCE(SUM(cache_write_5m_tokens), 0) AS cache_write_5m_tokens,
           COALESCE(SUM(cache_write_1h_tokens), 0) AS cache_write_1h_tokens,
           COALESCE(SUM(output_tokens), 0) AS output_tokens,
           COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
           COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
           COALESCE(SUM(CASE WHEN priced = 0 THEN 1 ELSE 0 END), 0) AS unpriced_calls"""


def question_usage(question_id: str) -> dict[str, Any]:
    """Everything one question asked of the model so far."""
    if not question_id:
        return _sum_row(None)
    with get_db() as conn:
        row = conn.execute(f"SELECT {_SUMS} FROM llm_usage WHERE question_id = ?", (question_id,)).fetchone()
    return _sum_row(row)


def usage_by_question(question_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per question: its calls, tokens, cost, and the models that answered it."""
    ids = sorted({q for q in question_ids if q})
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT question_id, {_SUMS} FROM llm_usage WHERE question_id IN ({marks}) GROUP BY question_id",
            ids).fetchall()
        models = conn.execute(
            f"SELECT question_id, component, priced_model, model, COUNT(*) AS calls FROM llm_usage "
            f"WHERE question_id IN ({marks}) GROUP BY question_id, component, priced_model, model", ids).fetchall()
    out = {str(dict(r)["question_id"]): _sum_row(r) for r in rows}
    for r in models:
        d = dict(r)
        entry = out.setdefault(str(d["question_id"]), _sum_row(None))
        entry.setdefault("steps", []).append({"component": d["component"] or "general",
                                              "model": d["priced_model"] or d["model"], "calls": int(d["calls"])})
    return out


def _where(account_id: Optional[str], month: Optional[str], since: str = "") -> tuple[str, list[Any]]:
    clauses, params = [], []
    if account_id:
        clauses.append("account_id = ?")
        params.append(account_id)
    if month:
        clauses.append("SUBSTRING(created_at, 1, 7) = ?")
        params.append(month)
    if since:
        clauses.append("created_at >= ?")
        params.append(since)
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


def usage_totals(account_id: Optional[str] = None, month: Optional[str] = None, *,
                 component_prefix: str = "", since: str = "", until: str = "") -> dict[str, Any]:
    where, params = _where(account_id, month, since)
    if component_prefix:
        where += (" AND " if where else "WHERE ") + "component LIKE ?"
        params.append(component_prefix + "%")
    if until:
        where += (" AND " if where else "WHERE ") + "created_at <= ?"
        params.append(until)
    with get_db() as conn:
        row = conn.execute(f"SELECT {_SUMS} FROM llm_usage {where}", params).fetchone()
    return _sum_row(row)


def usage_daily(account_id: str, month: str) -> dict[str, dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT SUBSTRING(created_at, 1, 10) AS day, {_SUMS} FROM llm_usage "
            "WHERE account_id = ? AND SUBSTRING(created_at, 1, 7) = ? GROUP BY day ORDER BY day",
            (account_id, month)).fetchall()
    return {str(dict(r)["day"]): _sum_row(r) for r in rows}


def user_usage_this_month(account_id: str, portal_user_id: int) -> dict[str, Any]:
    """One reader's share: the calls made for the questions they asked this month."""
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with get_db() as conn:
        row = conn.execute(
            f"SELECT {_SUMS} FROM llm_usage WHERE account_id = ? AND SUBSTRING(created_at, 1, 7) = ? "
            "AND question_id IN (SELECT question_id FROM query_log WHERE account_id = ? AND portal_user_id = ? "
            "AND question_id <> '')", (account_id, month, account_id, portal_user_id)).fetchone()
    return _sum_row(row)


def unpriced_models(since: str = "") -> list[dict[str, Any]]:
    """The models that answered with no price set, for the prices card to ask about."""
    where, params = _where(None, None, since)
    where += (" AND " if where else "WHERE ") + "priced = 0"
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT provider, priced_model, deployment_type, COUNT(*) AS calls, MAX(created_at) AS last_call "
            f"FROM llm_usage {where} GROUP BY provider, priced_model, deployment_type "
            "ORDER BY calls DESC, provider, priced_model", params).fetchall()
    return [dict(r) for r in rows]


def deployments_used(provider: str = "azure_openai") -> list[str]:
    """The deployment names calls went to, so one with no model on file can be mapped."""
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT model FROM llm_usage WHERE provider = ? AND model <> ''",
                            (provider,)).fetchall()
    return sorted(str(dict(r)["model"]) for r in rows)


def reprice_unpriced() -> int:
    """Cost the calls made before their model had a price, now that it may have one.

    A call keeps the cost it was recorded with once priced: a later price change is
    not history. Only calls recorded with no price are looked at again, against
    today's table (an Azure deployment's model and deployment type included).
    """
    from core.llm_prices import Usage, cost_of, price_for

    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, provider, model, input_tokens, cached_input_tokens, cache_write_5m_tokens, "
            "cache_write_1h_tokens, output_tokens, reasoning_tokens FROM llm_usage WHERE priced = 0").fetchall()]
    # Priced first, written after: the price table is read on a connection of its own.
    updates = []
    for row in rows:
        found = price_for(row["provider"], row["model"])
        usage = Usage(input=int(row["input_tokens"] or 0), cached_input=int(row["cached_input_tokens"] or 0),
                      cache_write_5m=int(row["cache_write_5m_tokens"] or 0),
                      cache_write_1h=int(row["cache_write_1h_tokens"] or 0),
                      output=int(row["output_tokens"] or 0), reasoning=int(row["reasoning_tokens"] or 0))
        updates.append((row["id"], found.model, found.deployment_type, cost_of(usage, found.price)))
    priced = sum(1 for *_, cost in updates if cost is not None)
    with get_db() as conn:
        for row_id, model, deployment_type, cost in updates:
            # A call still with no price is kept as one, under the model it is now known to run.
            conn.execute("UPDATE llm_usage SET priced_model = ?, deployment_type = ?, cost_usd = ?, priced = ? "
                         "WHERE id = ? AND priced = 0",
                         (model, deployment_type, cost or 0.0, 1 if cost is not None else 0, row_id))
    if priced:
        log.info("llm_usage: %d call(s) made before their model had a price are now costed", priced)
    return priced


# ── The admin's own prices ───────────────────────────────────────────────────

def list_llm_prices() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM llm_price ORDER BY provider, model, deployment_type").fetchall()
    return [dict(r) for r in rows]


def save_llm_price(*, provider: str, model: str, deployment_type: str = "", input: float, output: float,
                   cached_input: Optional[float] = None, cache_write_5m: Optional[float] = None,
                   cache_write_1h: Optional[float] = None, source: str = "Set by an admin") -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO llm_price (provider, model, deployment_type, input, cached_input, cache_write_5m,
                                      cache_write_1h, output, source, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(provider, model, deployment_type) DO UPDATE SET
                   input = excluded.input, cached_input = excluded.cached_input,
                   cache_write_5m = excluded.cache_write_5m, cache_write_1h = excluded.cache_write_1h,
                   output = excluded.output, source = excluded.source, updated_at = excluded.updated_at""",
            (provider.strip(), model.strip(), deployment_type or "", float(input), cached_input, cache_write_5m,
             cache_write_1h, float(output), source))
    from core.llm_prices import forget_cached_prices

    forget_cached_prices()


def delete_llm_price(provider: str, model: str, deployment_type: str = "") -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM llm_price WHERE provider = ? AND model = ? AND deployment_type = ?",
                     (provider, model, deployment_type or ""))
    from core.llm_prices import forget_cached_prices

    forget_cached_prices()
