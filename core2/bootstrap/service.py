"""Learning a workspace's database: the build an admin starts, run in the background.

Reads what discovery wrote, studies the warehouse through the workspace's own
connection, applies every admin decision, and stores a new model version. A
build that fails says why in the build log the page shows, and leaves the
previous version in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core2 import ids
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import Inventory, from_schema_json
from core2.bootstrap.profiler import ProfileOptions
from core2.model.overrides import apply_overrides
from core2.model.schema import SemanticModel

if TYPE_CHECKING:
    from core2.model.imports import Report

log = logging.getLogger("querybot.core2")


def _state(client: dict[str, Any]) -> dict[str, Any]:
    try:
        state = json.loads(client.get("state_data") or "{}")
    except (TypeError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def schema_path(account_id: str, client: dict[str, Any]) -> Path:
    folder = _state(client).get("schema_dir") or str(Path("clients") / account_id / "schema")
    return Path(folder) / "_schema.json"


def values_gate(account_id: str, inventory: Inventory, state: dict[str, Any]) -> Callable[[str, str], bool]:
    """May common values of this column be read and shown? The value index's own rules, and masking.

    An admin who turned value indexing off gets none: no column's values are read or kept.
    """
    from core.masking import detect_sensitive_columns
    from core.value_index import _clearance_gate, value_index_enabled

    if not value_index_enabled(state):
        return lambda _table, _column: False
    cleared, industry = _clearance_gate(account_id)
    sensitive: dict[str, set[str]] = {}
    for key, table in inventory.tables.items():
        found = detect_sensitive_columns([{"name": c.name, "type": c.raw_type} for c in table.columns], industry)
        sensitive[key] = {c.casefold() for c in found}

    def allowed(table_key: str, column: str) -> bool:
        table = inventory.tables.get(table_key)
        if table is None or table.mask_all or column in table.masked:
            return False
        if column.casefold() in sensitive.get(table_key, set()):
            return False
        return cleared is None or bool(cleared(table.source_key, column))

    return allowed


def _labeler(account_id: str, client: dict[str, Any], inventory: Inventory) -> Callable[[str, str], str] | None:
    """The workspace's AI for naming, or None when it has none (names then come from the names)."""
    from core2.bootstrap.ai import workspace_labeler

    try:
        return workspace_labeler(account_id, client, tables=sorted(t.name for t in inventory.tables.values()),
                                 columns=sorted({c.name for t in inventory.tables.values() for c in t.columns}))
    except Exception as exc:  # noqa: BLE001 - no AI configured still learns everything the data shows
        log.warning("core2: no AI model to name tables for %s: %s", account_id, exc)
        return None


def source_hash(inventory: Inventory) -> str:
    """What discovery found, as a fingerprint: the tables, their columns and types, and what it masks."""
    # A table with nothing masked is written as before masking was part of it, so a model
    # learned then is not called out of date for that alone.
    shape = sorted((k, [(c.name, c.raw_type) for c in t.columns],
                    *([sorted(t.masked), t.mask_all] if t.masked or t.mask_all else []))
                   for k, t in inventory.tables.items())
    return hashlib.sha256(json.dumps(shape).encode()).hexdigest()[:16]


_DISCOVERED: dict[str, tuple[float, Inventory]] = {}


def discovered(account_id: str, client: dict[str, Any]) -> Inventory | None:
    """What discovery wrote last (read again only when its file changes); None before discovery."""
    import store

    path = schema_path(account_id, client)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    kept = _DISCOVERED.get(str(path))
    if kept is not None and kept[0] == stamp:
        return kept[1]
    config = store.get_db_config(int(client.get("db_config_id") or 0)) or {}
    inventory = from_schema_json(json.loads(path.read_text(encoding="utf-8")), str(config.get("db_type") or ""))
    _DISCOVERED[str(path)] = (stamp, inventory)
    return inventory


def behind_discovery(account_id: str, client: dict[str, Any]) -> bool:
    """Has discovery run since the latest Learn, and found other tables, columns or masking?"""
    import store

    versions = store.list_core2_model_versions(account_id, client.get("db_config_id"))
    if not versions or not versions[0]["source_hash"]:
        return False
    try:
        inventory = discovered(account_id, client)
    except Exception as exc:  # noqa: BLE001 - a schema file that cannot be read is discovery's to report
        log.warning("core2: discovery's schema for %s could not be read: %s", account_id, exc)
        return False
    return inventory is not None and source_hash(inventory) != versions[0]["source_hash"]


def mask_as_discovered(model: SemanticModel, inventory: Inventory | None) -> None:
    """Columns discovery masks now keep their values out, though Learn read them before they were masked.

    Masking is set before discovery, and discovery writes it; a column masked after
    Learn would otherwise have its common values put before the AI, and its members
    matched, until QueryBot learned again.
    """
    if inventory is None:
        return
    for key, table in inventory.tables.items():
        if not (table.masked or table.mask_all):
            continue
        masked = {ids.norm(c) for c in table.masked}
        for column in model.columns.values():
            if column.table == key and (table.mask_all or ids.norm(column.name) in masked):
                column.values_allowed = False


def answering_model(account_id: str, client: dict[str, Any]) -> SemanticModel | None:
    """The model questions are answered with: the latest, the admin's decisions, and discovery's masking now."""
    model = load_model(account_id, client.get("db_config_id"))
    if model is not None:
        try:
            mask_as_discovered(model, discovered(account_id, client))
        except Exception as exc:  # noqa: BLE001 - the masking Learn read still holds
            log.warning("core2: discovery's masking for %s could not be read: %s", account_id, exc)
    return model


def propose_classifications(account_id: str, model: SemanticModel) -> int:
    """People's data Learn found, proposed on the Compliance page of a workspace under compliance.

    A name or a detail (an email, a phone, a birth date) becomes a classification to review, with the
    tags the workspace's classifier gives its name (a patient's name is PHI in a pharmacy) or, where the
    name says nothing (an email column called C07, read by its values), as directly identifying PII. An
    admin's reviewed classification is never changed; an automatic one is raised only when it did not
    know the column held people's data. Unreviewed, it already masks, as every classification does.
    Returns how many were written.
    """
    import store
    from core.compliance.classifier import classify_column
    from core2.service import question_scrubber

    if question_scrubber(account_id) is None:
        return 0          # not under compliance: no classifications to keep
    profile = store.get_compliance_profile(account_id) or {}
    industry = str(profile.get("industry") or "")
    existing = store.get_classification_map(account_id)
    written = 0
    for column in model.columns.values():
        table = model.tables.get(column.table)
        if column.personal == "none" or column.parts or table is None:
            continue
        fqn = ".".join(p for p in (table.database, table.schema_name, table.name) if p)
        tail = f".{table.name}.{column.name}".upper()
        current = existing.get(f"{fqn}.{column.name}".upper()) or next(
            (v for k, v in existing.items() if ("." + k).endswith(tail)), None)
        if current and (current.get("reviewed") or set(current.get("tags") or []) & {"PII", "PHI"}):
            continue
        said = classify_column(column.name, industry)
        tags = said["tags"] if set(said["tags"]) & {"PII", "PHI"} else sorted({*said["tags"], "PII"})
        try:
            store.save_classification(
                account_id, fqn, column.name, sensitivity="RESTRICTED", identifiability="DIRECT", tags=tags,
                confidence=0.9, reviewed=False, reviewed_by="", source="learn",
                mask_strategy="safe_alias_name" if column.personal == "name" else "tokenize")
            written += 1
        except Exception as exc:  # noqa: BLE001 - a proposal not written leaves the column masked by core2 itself
            log.warning("core2: the classification of %s.%s was not proposed: %s", table.name, column.name, exc)
    return written


def build_workspace(account_id: str, started: str | None = None) -> int:
    """Learn the workspace's database and store a new model version; returns the version.

    ``started`` is the build the learned page already recorded as running (so the
    page shows it at once); without it the build is recorded here.
    """
    import store
    from core2.warehouse.querybot import QueryBotWarehouse

    client = store.get_client(account_id)
    if not client or not client.get("db_config_id"):
        raise ValueError("This workspace has no database connected.")
    db_id = int(client["db_config_id"])
    started = started or store.start_core2_build(account_id, db_id)
    began = time.monotonic()

    def say(line: str) -> None:
        try:
            store.add_core2_build_line(account_id, db_id, started, line)
        except Exception as exc:  # noqa: BLE001 - the progress line is a help; Learn goes on without it
            log.warning("core2: progress line for %s not saved: %s", account_id, exc)

    try:
        config = store.get_db_config(db_id)
        if not config:
            raise ValueError("The workspace's database connection is missing.")
        path = schema_path(account_id, client)
        if not path.exists():
            raise ValueError("Run discovery first: QueryBot learns from the tables discovery found.")
        inventory = from_schema_json(json.loads(path.read_text(encoding="utf-8")), config["db_type"])
        if not inventory.tables:
            raise ValueError("Discovery found no tables to learn from.")
        options = BuildOptions(profile=ProfileOptions(values_allowed=values_gate(account_id, inventory, _state(client))),
                               labeler=_labeler(account_id, client, inventory), progress=say)
        say(f"Connecting to the {config['db_type'].replace('_', ' ')} database")
        with QueryBotWarehouse(config["db_type"], config.get("credentials") or {}) as warehouse:
            model = build_model(warehouse, inventory, client_id=account_id, db_id=db_id, options=options)
        # Stored as learned: decisions are a layer applied when the model is read, so
        # undoing one brings back exactly what the data said.
        say("Saving what was learned")
        version = store.save_core2_model(account_id, db_id, model.model_dump_json(),
                                         source_hash=source_hash(inventory), stats=stats(model))
        say("Bringing over the decisions already made in today's setup")
        bring_over(account_id, db_id, model, client)
        proposed = propose_classifications(account_id, model)
        if proposed:
            say(f"Proposed {proposed} column(s) of people's data on the Compliance page, masked until reviewed")
        say("Reading the member names questions use (customers, products and the like), so the first "
            "question does not wait for them")
        try:
            from core2.service import read_members

            read_members(account_id)
        except Exception as exc:  # noqa: BLE001 - the first question reads them instead
            log.warning("core2: member names of %s not read after Learn: %s", account_id, exc)
        left_out = sum("was left out" in note for note in model.notes)
        say(f"Done in {_took(began)}: version {version}, {len(model.tables)} tables, {len(model.measures)} measures"
            + (f"; {left_out} left out, each named in the notes below" if left_out else ""))
        store.finish_core2_build(account_id, db_id, started, status="done", version=version)
        return version
    except Exception as exc:
        log.warning("core2 build failed for %s: %s", account_id, exc, exc_info=True)
        say(f"Stopped after {_took(began)}: {exc}")
        store.finish_core2_build(account_id, db_id, started, status="failed", message=str(exc))
        raise


def _took(began: float) -> str:
    seconds = int(time.monotonic() - began)
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def bring_over(account_id: str, db_id: int | None, model: SemanticModel, client: dict[str, Any], *,
               report: Report | None = None) -> dict[str, Any]:
    """Today's admin decisions as this model's overrides (core2.model.imports); never fails the build."""
    import store
    from core2.model.imports import import_approvals

    try:
        done = import_approvals(account_id, db_id, model, str(_state(client).get("kb_dir") or ""), report=report)
    except Exception as exc:  # noqa: BLE001 - the model stands without them; the page says so
        log.warning("core2: today's decisions for %s could not be brought over: %s", account_id, exc, exc_info=True)
        done = {"error": str(exc)}
    store.set_core2_import_report(account_id, db_id, done)
    return done


DECISIONS_EVERY = 10.0       # seconds: how often answers look for decisions made in today's setup since
_LOOKED: dict[tuple[str, int], float] = {}
_LOOKING = threading.Lock()


def keep_decisions_current(account_id: str, db_id: int | None, client: dict[str, Any]) -> bool:
    """Bring today's decisions over again when they changed since they last came over.

    A metric added or a link confirmed in today's setup after Learn reached the new
    core only when an admin pressed "Bring them over again" or learned again. Before
    an answer, at most every DECISIONS_EVERY seconds per workspace, today's setup is
    read and compared with what came over; a change (one undone included) is brought
    over for this answer. True when something came over. Never fails the answer.
    """
    from core2.model.imports import changes, decisions, read_legacy

    key = (account_id, int(db_id or 0))
    now = time.monotonic()
    with _LOOKING:
        if now - _LOOKED.get(key, float("-inf")) < DECISIONS_EVERY:
            return False
        _LOOKED[key] = now
    try:
        model = learned_model(account_id, db_id)
        if model is None:
            return False
        report = decisions(model, read_legacy(account_id, str(_state(client).get("kb_dir") or "")))
        if not changes(account_id, db_id, report):
            return False
    except Exception as exc:  # noqa: BLE001 - the decisions already brought over answer; tried again next time
        log.warning("core2: could not look for new decisions in today's setup for %s: %s", account_id, exc,
                    exc_info=True)
        return False
    log.info("core2: today's setup changed for %s; bringing its decisions over", account_id)
    bring_over(account_id, db_id, model, client, report=report)
    return True


def learned_model(account_id: str, db_id: int | None) -> SemanticModel | None:
    """The latest stored model as learned, before any decision."""
    import store

    row = store.load_core2_model(account_id, db_id)
    if row is None:
        return None
    model = SemanticModel.model_validate_json(row["model_json"])
    model.version = row["version"]
    return model


def load_model(account_id: str, db_id: int | None, version: int | None = None) -> SemanticModel | None:
    """The stored (learned) model with the admin's current decisions applied on top."""
    import store

    row = store.load_core2_model(account_id, db_id, version)
    if row is None:
        return None
    model = SemanticModel.model_validate_json(row["model_json"])
    model.version = row["version"]
    apply_overrides(model, store.list_core2_overrides(account_id, db_id))
    return model


def stats(model: SemanticModel) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for table in model.tables.values():
        kinds[table.kind] = kinds.get(table.kind, 0) + 1
    return {
        "tables": len(model.tables), "kinds": kinds, "joins": len(model.joins),
        "joins_verified": sum(1 for j in model.joins.values() if j.trust in ("verified", "declared", "admin")),
        "date_roles": len(model.date_roles), "measures": len(model.measures), "entities": len(model.entities),
        "quality": len(model.quality), "review": len(model.review),
    }
