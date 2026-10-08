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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import Inventory, from_schema_json
from core2.bootstrap.profiler import ProfileOptions
from core2.model.overrides import apply_overrides
from core2.model.schema import SemanticModel

log = logging.getLogger("querybot.core2")


def schema_path(account_id: str, client: dict[str, Any]) -> Path:
    try:
        state = json.loads(client.get("state_data") or "{}")
    except (TypeError, ValueError):
        state = {}
    folder = state.get("schema_dir") or str(Path("clients") / account_id / "schema")
    return Path(folder) / "_schema.json"


def values_gate(account_id: str, inventory: Inventory) -> Callable[[str, str], bool]:
    """May common values of this column be read and shown? The value index's own rules, and masking."""
    from core.masking import detect_sensitive_columns
    from core.value_index import _clearance_gate

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
    shape = sorted((k, [(c.name, c.raw_type) for c in t.columns]) for k, t in inventory.tables.items())
    return hashlib.sha256(json.dumps(shape).encode()).hexdigest()[:16]


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
        options = BuildOptions(profile=ProfileOptions(values_allowed=values_gate(account_id, inventory)),
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


def bring_over(account_id: str, db_id: int | None, model: SemanticModel, client: dict[str, Any]) -> dict[str, Any]:
    """Today's admin decisions as this model's overrides (core2.model.imports); never fails the build."""
    import store
    from core2.model.imports import import_approvals

    try:
        state = json.loads(client.get("state_data") or "{}")
    except (TypeError, ValueError):
        state = {}
    try:
        report = import_approvals(account_id, db_id, model, str(state.get("kb_dir") or ""))
    except Exception as exc:  # noqa: BLE001 - the model stands without them; the page says so
        log.warning("core2: today's decisions for %s could not be brought over: %s", account_id, exc, exc_info=True)
        report = {"error": str(exc)}
    store.set_core2_import_report(account_id, db_id, report)
    return report


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
