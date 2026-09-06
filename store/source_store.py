"""The connections a workspace can answer from.

A workspace has held exactly one warehouse connection since the product
started. A tenant whose sales data is in Snowflake and whose finance ledger is
in Azure SQL has to be two workspaces — two knowledge bases, two graphs, and
no way to check one area's number against the other's.

Everything here is additive. ``client.db_config_id`` stays authoritative for a
workspace that has declared no sources, so an existing tenant sees no change:
``resolve_db_config_id`` falls back to it, and the startup backfill gives each
of them one default source pointing at the connection they already had.

One resolver, deliberately. The plan for this said "every existing call site
that reads db_config_id resolves through one helper so the change is
auditable", and that is the whole safety property: there is exactly one place
that decides which connection a question runs against, so a scoping mistake is
one function to read rather than a hundred call sites to audit.
"""

from __future__ import annotations

import logging

from store.db import get_db

log = logging.getLogger("querybot.source_store")


def list_client_sources(account_id: str, *, active_only: bool = True) -> list[dict]:
    """Every source for a workspace, default first, then by name."""
    where = "WHERE account_id = ?" + (" AND is_active = 1" if active_only else "")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM client_source {where} "
            "ORDER BY is_default DESC, name COLLATE NOCASE",
            (account_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_client_source(account_id: str, source_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM client_source WHERE account_id=? AND id=?",
            (account_id, int(source_id)),
        ).fetchone()
    return dict(row) if row else None


def save_client_source(
    account_id: str,
    name: str,
    db_config_id: int,
    *,
    source_id: int | None = None,
    description: str = "",
    domain_id: int | None = None,
    is_default: bool = False,
    is_active: bool = True,
) -> int:
    """Create or update one source. Returns its id.

    The first source a workspace gets is its default whatever the caller
    asked for: a workspace with sources and no default has no answer to
    "which connection does a question with no domain run against?", and the
    resolver would have to invent one.
    """
    name = " ".join(str(name or "").split())[:120]
    if not name:
        raise ValueError("A source needs a name.")

    with get_db() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM client_source WHERE account_id=?", (account_id,)
        ).fetchone()[0]
        make_default = bool(is_default) or existing == 0

        if source_id:
            conn.execute(
                "UPDATE client_source SET name=?, db_config_id=?, description=?, "
                "domain_id=?, is_active=?, updated_at=datetime('now') "
                "WHERE account_id=? AND id=?",
                (name, int(db_config_id), description, domain_id,
                 1 if is_active else 0, account_id, int(source_id)),
            )
            new_id = int(source_id)
        else:
            cursor = conn.execute(
                "INSERT INTO client_source "
                "(account_id, db_config_id, name, description, domain_id, "
                " is_default, is_active) VALUES (?,?,?,?,?,?,?)",
                (account_id, int(db_config_id), name, description, domain_id,
                 0, 1 if is_active else 0),
            )
            new_id = int(cursor.lastrowid)

        if make_default:
            conn.execute(
                "UPDATE client_source SET is_default = CASE WHEN id=? THEN 1 ELSE 0 END, "
                "updated_at=datetime('now') WHERE account_id=?",
                (new_id, account_id),
            )
    return new_id


def set_default_source(account_id: str, source_id: int) -> bool:
    """Make one source the default. Exactly one, always."""
    with get_db() as conn:
        found = conn.execute(
            "SELECT 1 FROM client_source WHERE account_id=? AND id=?",
            (account_id, int(source_id)),
        ).fetchone()
        if not found:
            return False
        conn.execute(
            "UPDATE client_source SET is_default = CASE WHEN id=? THEN 1 ELSE 0 END, "
            "updated_at=datetime('now') WHERE account_id=?",
            (int(source_id), account_id),
        )
    return True


def delete_client_source(account_id: str, source_id: int) -> bool:
    """Remove a source, promoting another to default if this one was it.

    A workspace left with sources and no default cannot answer a question that
    names no domain, so the promotion is not tidiness -- it is the difference
    between a deletion and an outage.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT is_default FROM client_source WHERE account_id=? AND id=?",
            (account_id, int(source_id)),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM client_source WHERE account_id=? AND id=?",
                     (account_id, int(source_id)))
        if row["is_default"]:
            replacement = conn.execute(
                "SELECT id FROM client_source WHERE account_id=? AND is_active=1 "
                "ORDER BY name COLLATE NOCASE LIMIT 1",
                (account_id,),
            ).fetchone()
            if replacement:
                conn.execute(
                    "UPDATE client_source SET is_default=1, updated_at=datetime('now') "
                    "WHERE id=?", (replacement["id"],),
                )
    return True


def source_for_domain(account_id: str, domain_id: int | None) -> dict | None:
    """The source a domain's tables live in, if one is declared for it."""
    if not domain_id:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM client_source WHERE account_id=? AND domain_id=? "
            "AND is_active=1 ORDER BY is_default DESC, id LIMIT 1",
            (account_id, int(domain_id)),
        ).fetchone()
    return dict(row) if row else None


def default_source(account_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM client_source WHERE account_id=? AND is_active=1 "
            "ORDER BY is_default DESC, id LIMIT 1",
            (account_id,),
        ).fetchone()
    return dict(row) if row else None


def resolve_db_config_id(
    account_id: str,
    *,
    source_id: int | None = None,
    domain_id: int | None = None,
) -> int | None:
    """Which connection a question runs against. The only place that decides.

    In order: the source explicitly named, then the source declared for the
    domain the question was routed to, then the workspace's default source,
    then ``client.db_config_id`` for a workspace that has declared none.

    An explicitly named source that does not exist, or is not this account's,
    resolves to None rather than falling through to the default. Falling
    through would run the question against a connection the caller did not
    ask for and report the answer as though it had -- and on a cross-tenant id
    it would run it against somebody else's warehouse.
    """
    from store.config_store import get_client

    if source_id is not None:
        source = get_client_source(account_id, source_id)
        if not source or not source.get("is_active"):
            log.warning("Source %s is not available for %s; refusing to fall back "
                        "to the default connection", source_id, account_id)
            return None
        return int(source["db_config_id"])

    for candidate in (source_for_domain(account_id, domain_id),
                      default_source(account_id)):
        if candidate:
            return int(candidate["db_config_id"])

    client = get_client(account_id) or {}
    return client.get("db_config_id") or None


def db_config_for_domain_name(account_id: str, domain_name: str) -> int | None:
    """The connection a named domain's tables live in, if one is declared.

    Domain routing works in names -- the question matched "Finance", not
    domain 22 -- so the lookup from a routed name to a connection lives here
    rather than in the pipeline, next to the resolver that is the only other
    place a connection is chosen.

    Returns None when the domain has no source of its own, which means "use
    whatever the caller was already using" rather than "refuse": a workspace
    with one connection and several domains is the normal case, and every
    domain in it shares that connection.
    """
    from store.domain_store import get_domain

    name = str(domain_name or "").strip()
    if not name:
        return None
    domain = get_domain(account_id, name)
    if not domain or not domain.get("id"):
        return None
    source = source_for_domain(account_id, int(domain["id"]))
    return int(source["db_config_id"]) if source else None
