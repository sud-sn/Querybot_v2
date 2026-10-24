"""Making the copies of metrics a workspace keeps (core2/precompute.py), and handing the ready ones to its answers.

An admin adds a copy on Data > Precomputed copies; it is made at once in the background, then again every few
hours (its schedule), by the minute scheduler (core/notification_scheduler.py). Each making reads the workspace's
database on its own connection, as Learn does, with the same timeouts, and writes a new file; the one before is
deleted once the new one is in place. A making that fails leaves the copy made before in use while it is fresh,
and says why on the admin's page.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core2 import precompute as PC

log = logging.getLogger(__name__)

MAKING_STALE = dt.timedelta(hours=2)        # a making not finished by then stopped with its process
_LIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("core2_copies_live", default=False)


@contextmanager
def live() -> Iterator[None]:
    """Within it, every answer reads the database, never a copy: a scheduled email, a tile alert, a followed
    dashboard -- nobody waits for them, and their figures are the data as it is when they are sent. (Carried into
    the threads that draw a dashboard's tiles, which run in the caller's context.)"""
    token = _LIVE.set(True)
    try:
        yield
    finally:
        _LIVE.reset(token)


def live_only() -> bool:
    return _LIVE.get()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _when(text: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(text or "")[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def copy_name(model: Any, row: dict[str, Any]) -> str:
    """"Net sales by Region and Store, by month": how the admin page and the answers name a copy."""
    measure = model.measures.get(row["metric"]) if model is not None else None
    name = (measure.business_name or measure.slug) if measure is not None else row["metric"]
    labels = []
    for slug in row.get("fields") or []:
        attribute = model.attributes.get(slug) if model is not None else None
        labels.append((attribute.business_name or slug) if attribute is not None else slug)
    by = f" by {', '.join(labels[:-1]) + ' and ' + labels[-1] if len(labels) > 1 else labels[0]}" if labels else ""
    return f"{name}{by}, by {row.get('grain') or 'day'}"


def ready(account_id: str, db_id: int, model: Any) -> list[PC.Ready]:
    """The workspace's copies that may answer now: made, fresh, of this model's version, their file in place."""
    import store

    out: list[PC.Ready] = []
    for row in store.list_copies(account_id, db_id):
        made = _when(row.get("made_at"))
        path = Path(str(row.get("file") or ""))
        if made is None or not row.get("file") or not row.get("fingerprint") or not path.is_file():
            continue
        if not PC.fresh(made, int(row.get("every_hours") or 6)):
            continue
        out.append(PC.Ready(id=int(row["id"]), name=copy_name(model, row), metric=row["metric"],
                            fields=list(row.get("fields") or []), date=str(row.get("date_role") or ""),
                            grain=str(row.get("grain") or "day"), every_hours=int(row.get("every_hours") or 6),
                            made_at=made, path=path, fingerprint=str(row["fingerprint"])))
    return out


def make(account_id: str, copy_id: int) -> dict[str, Any]:
    """Make the copy now, on this thread: {"ok": True, rows, took_ms} or {"ok": False, "error": why}."""
    import store
    from core2.bootstrap.service import answering_model
    from core2.warehouse.querybot import QueryBotWarehouse

    row = store.get_copy(account_id, copy_id)
    if row is None:
        return {"ok": False, "error": "this copy is not kept any more"}
    if not store.copy_making(account_id, copy_id, stale_hours=MAKING_STALE.total_seconds() / 3600):
        return {"ok": False, "error": "it is being made already"}
    started = time.perf_counter()
    try:
        client = store.get_client(account_id) or {}
        db_id = int(client.get("db_config_id") or 0)
        db_config = store.get_db_config(db_id) if db_id else None
        if not db_config or db_id != int(row.get("db_config_id") or 0):
            raise PC.CopyError("the workspace's database is not the one this copy was kept for")
        model = answering_model(account_id, client)
        if model is None:
            raise PC.CopyError("the workspace has not been learned yet")
        with QueryBotWarehouse(str(db_config.get("db_type") or ""), db_config.get("credentials") or {}) as warehouse:
            from core2.compile.compiler import repeats_sql

            def repeats(join: Any) -> bool:
                return bool(warehouse.query(repeats_sql(model, join, warehouse.dialect), max_rows=1).rows)

            defn = PC.define(model, metric=row["metric"], fields=list(row.get("fields") or []),
                             date=str(row.get("date_role") or ""), grain=str(row.get("grain") or "day"),
                             today=dt.date.today(), dialect=warehouse.dialect, link_repeats=repeats)
            for sql, why in defn.checks:
                if warehouse.query(sql, max_rows=1).rows:
                    raise PC.CopyError(why)
            PC.compared_ignoring_case(model, defn, lambda sql: warehouse.query(sql, max_rows=50).rows)
            cap = PC.max_rows()
            result = warehouse.query(defn.sql, max_rows=cap + 1)
        if len(result.rows) > cap:
            raise PC.CopyError(f"it would hold more than {cap:,} rows: keep it by month, or by fewer fields")
        PC.kept_as_compared(defn, result.rows)
        path = PC.file_for(account_id, copy_id)
        size = PC.write(path, defn, result.rows)
        took = int((time.perf_counter() - started) * 1000)
        old = store.copy_made(account_id, copy_id, rows=len(result.rows), took_ms=took, size_bytes=size,
                              file=str(path), fingerprint=defn.fingerprint, model_version=int(model.version or 0))
        if store.get_copy(account_id, copy_id) is None:
            _delete(str(path))             # taken out while it was being made: nothing of it stays on the server
            return {"ok": False, "error": "this copy is not kept any more"}
        # The file it replaces stays until the next making: an answer that listed it a moment ago still reads it.
        for left in PC.files_of(account_id, copy_id):
            if left != path and str(left) != old:
                _delete(str(left))
        log.info("core2: copy %s of %s made: %s rows in %s ms", copy_id, account_id, len(result.rows), took)
        return {"ok": True, "rows": len(result.rows), "took_ms": took}
    except PC.CopyError as exc:
        store.copy_failed(account_id, copy_id, str(exc))
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - said on the admin's page; the copy made before stays in use
        log.warning("core2: copy %s of %s could not be made: %s", copy_id, account_id, exc, exc_info=True)
        store.copy_failed(account_id, copy_id, f"the database could not be read: {str(exc)[:300]}")
        return {"ok": False, "error": str(exc)}


def _delete(file: str) -> None:
    """A copy's old file. One an answer is reading now goes when it closes (Linux); one that cannot be deleted
    yet (Windows) goes at the next making; an answer that finds its file gone is answered by the database."""
    if not file:
        return
    try:
        Path(file).unlink(missing_ok=True)
    except OSError as exc:
        log.info("core2: an old copy file stays for now: %s", exc)


def start(account_id: str, copy_id: int) -> None:
    """Make the copy in the background."""
    threading.Thread(target=make, args=(account_id, copy_id), daemon=True,
                     name=f"core2-copy-{account_id}-{copy_id}").start()


def remove(account_id: str, copy_id: int) -> bool:
    import store

    gone = store.remove_copy(account_id, copy_id)
    if gone is None:
        return False
    _delete(str(gone.get("file") or ""))
    for left in PC.files_of(account_id, copy_id):
        _delete(str(left))
    return True


def due(row: dict[str, Any], now: dt.datetime | None = None) -> bool:
    """Is the copy to be made now: never made, its schedule come round, or a failed try an hour old (a making
    under way is left alone unless it stopped with its process)."""
    now = now or _now()
    if row.get("status") == "making":
        started = _when(row.get("started_at"))
        return started is None or now - started > MAKING_STALE
    every = dt.timedelta(hours=max(1, int(row.get("every_hours") or 6)))
    made, tried = _when(row.get("made_at")), _when(row.get("tried_at"))
    if tried is not None and (made is None or tried > made) and now - tried < min(every, dt.timedelta(hours=1)):
        return False          # it failed a moment ago: tried again within the hour, or at its schedule if sooner
    return made is None or now - made >= every


def run_due_copies(now: dt.datetime | None = None) -> int:
    """Start every making due now, each in the background; how many. Called each minute."""
    import store

    started = 0
    for row in store.all_copies():
        try:
            client = store.get_client(row["account_id"]) or {}
            if int(client.get("db_config_id") or 0) != int(row.get("db_config_id") or 0):
                # A copy of a database the workspace no longer uses: never made again, and its figures not kept.
                for left in PC.files_of(row["account_id"], int(row["id"])):
                    _delete(str(left))
                continue
            if due(row, now):
                start(row["account_id"], int(row["id"]))
                started += 1
        except Exception as exc:  # noqa: BLE001 - one workspace never stops the others
            log.warning("core2: copy %s not started: %s", row.get("id"), exc)
    return started


def choices(model: Any) -> list[dict[str, Any]]:
    """What the admin can keep a copy of: each metric that adds up, its date, and the fields its rows reach."""
    from core2.resolve.resolver import _reaches

    out = []
    for key, m in sorted(model.measures.items(), key=lambda kv: (kv[1].business_name or kv[1].slug).casefold()):
        if m.hidden or m.additivity == "semi_additive" or m.time_aggregation or not m.table:
            continue
        fields = []
        for slug, a in sorted(model.attributes.items(), key=lambda kv: (kv[1].business_name or kv[0]).casefold()):
            column = model.columns.get(a.column)
            if column is None or column.hidden or column.sensitivity != "none" or column.personal != "none" \
                    or a.kind not in ("group",):
                continue
            if column.table == m.table or _reaches(model, m.table, column.table):
                fields.append({"slug": slug, "name": a.business_name or slug})
        dates = [{"key": r.key, "name": r.name or r.slug} for r in model.date_roles_of(m.table) if r.kind != "audit"]
        out.append({"key": key, "name": m.business_name or m.slug, "fields": fields, "dates": dates,
                    "default_date": m.default_date or ""})
    return out
