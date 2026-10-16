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
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core2 import ids
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import Inventory, from_schema_json
from core2.bootstrap.profiler import ProfileOptions, TableProfile, profile_from_json, profile_json
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


# Who a table or a column is about when it may be people: their names and cities stay out of the AI's view.
_PEOPLE = {"customer", "cust", "client", "clnt", "consumer", "buyer", "shopper", "owner", "holder", "contact",
           "patient", "person", "employee", "member"}
# Things, never people: a name or a city of one of these is the business's own vocabulary.
_THINGS = {"category", "categ", "cat", "store", "region", "product", "prod", "item", "itm", "warehouse", "whs",
           "division", "dvn", "department", "dept", "segment", "channel", "brand", "family", "group", "grp", "class",
           "type", "status", "plan", "service", "site", "location", "branch", "market", "territory", "country",
           "state", "province", "month", "day", "week", "quarter", "season", "holiday", "account", "ledger", "cost",
           "centre", "center", "project", "campaign", "promotion", "event", "route", "carrier", "shipper",
           "supplier", "vendor", "company", "partner", "genre", "album", "artist", "track", "film", "media",
           "playlist", "language", "unit", "equipment", "device", "model", "part", "material", "zone", "area"}


def _a_things_name(table: str, column: str, strategy: str) -> bool:
    """Is a column the old masking rule reads as a person's name or city ("category_name", "store city") the name
    of a thing? It is when the column names a thing, or when neither it nor its table is about people. People's
    names stay out of the AI's view by this rule and by Learn's own reading of the values (personal.py). A drug's
    or an ingredient's name, which the rule holds back only so that its fake copy looks like a drug, is read the
    same way: an ingredient a pharmacy buys is its own vocabulary; a patient's medication stays held back."""
    from core2.bootstrap import names, personal

    if strategy not in ("name", "city", "drug_name"):
        return False
    own = {names.singular(w) for w in names.tokens(column)}
    if own & (_THINGS | personal.ORG_WORDS) and not own & (_PEOPLE | personal.PERSON_WORDS):
        return True
    about = own | {names.singular(w) for w in names.tokens(table)}
    return not about & (_PEOPLE | personal.PERSON_WORDS)


def held_by_masking(table: Any, column: str) -> bool:
    """Does discovery's masking keep this column's values out? Every column of a table masked whole, and every
    column an admin chose; one discovery's own patterns masked as a name or a city ("store_name", a store's
    "city") only when it is no thing's name (_a_things_name): a person's name, or a city in a table about people."""
    if table.mask_all:
        return True
    hit = next((c for c in table.masked if ids.norm(c) == ids.norm(column)), None)
    if hit is None:
        return False
    if table.mask_mode == "auto":
        kind = table.mask_kinds.get(hit) or table.mask_kinds.get(column) or ""
        return not _a_things_name(table.name, column, kind)
    return True


def _a_name_masked_by_pattern(table: Any, column: str) -> bool:
    """Did discovery's own patterns, not an admin, mask this column as a name? Its values may then still show it
    names businesses ("Hearthstone Home Centres"), which a customer's or a buyer's name often does."""
    if table.mask_all or table.mask_mode != "auto":
        return False
    hit = next((c for c in table.masked if ids.norm(c) == ids.norm(column)), None)
    return hit is not None and (table.mask_kinds.get(hit) or table.mask_kinds.get(column) or "") == "name"


def values_gate(account_id: str, inventory: Inventory, state: dict[str, Any],
                sample: Callable[[str, str], list[str]] | None = None) -> Callable[[str, str], bool]:
    """May common values of this column be read and shown? The value index's own rules, and masking.

    An admin who turned value indexing off gets none: no column's values are read or kept. A name the masking
    rule holds back only because its table or column may be about people ("customer name", "buyer name") is told
    by its own values when ``sample`` reads them: businesses or teams ("Harbor Trading", "Facilities Desk") are
    named, people never (personal.organisations_named). Each such column is read once, in the app.
    """
    from core.masking import detect_sensitive_columns
    from core.value_index import _clearance_gate, value_index_enabled
    from core2.bootstrap import names, personal

    if not value_index_enabled(state):
        return lambda _table, _column: False
    cleared, industry = _clearance_gate(account_id)
    sensitive: dict[str, set[str]] = {}
    by_values: dict[str, set[str]] = {}
    for key, table in inventory.tables.items():
        found = detect_sensitive_columns([{"name": c.name, "type": c.raw_type} for c in table.columns], industry)
        sensitive[key] = {c.casefold() for c, strategy in found.items() if not _a_things_name(table.name, c, strategy)}
        by_values[key] = {c.casefold() for c, strategy in found.items()
                          if strategy == "name" and c.casefold() in sensitive[key]
                          and not {names.singular(w) for w in names.tokens(c)} & personal.PERSON_WORDS
                          and not personal.is_first_name(c) and not personal.is_last_name(c)}
    told: dict[tuple[str, str], bool] = {}

    def businesses(table_key: str, column: str) -> bool:
        if sample is None or column.casefold() not in by_values.get(table_key, set()):
            return False
        if (table_key, column) not in told:
            try:
                told[(table_key, column)] = personal.organisations_named(sample(table_key, column))
            except Exception as exc:  # noqa: BLE001 - unread, the column stays held back
                log.warning("core2: the values of %s.%s could not be read to tell a business's name from a "
                            "person's: %s", table_key, column, exc)
                told[(table_key, column)] = False
        return told[(table_key, column)]

    def allowed(table_key: str, column: str) -> bool:
        table = inventory.tables.get(table_key)
        if table is None:
            return False
        if held_by_masking(table, column) and not (_a_name_masked_by_pattern(table, column)
                                                   and businesses(table_key, column)):
            return False
        if column.casefold() in sensitive.get(table_key, set()) and not businesses(table_key, column):
            return False
        return cleared is None or bool(cleared(table.source_key, column))

    return allowed


def value_sampler(warehouse: Any, inventory: Inventory, limit: int = 200) -> Callable[[str, str], list[str]]:
    """Up to ``limit`` distinct values of one column, read in the warehouse for the app's own use (values_gate):
    never kept, never sent."""
    from sqlglot import exp

    from core2.bootstrap.profiler import _readable
    from core2.warehouse import dialect as D

    def read(table_key: str, column: str) -> list[str]:
        table = inventory.tables[table_key]
        found = next(c for c in table.columns if c.name == column)
        dialect = warehouse.dialect
        value = _readable(found, dialect)
        query = (exp.select(value.copy()).distinct().from_("__SRC__")
                 .where(exp.Not(this=exp.Is(this=value.copy(), expression=exp.Null()))).limit(limit))
        sql = query.sql(dialect=dialect).replace("__SRC__", D.table_sql(table.database, table.schema, table.name,
                                                                         dialect), 1)
        return [str(row[0]) for row in warehouse.query(sql, max_rows=limit).rows if row[0] is not None]

    return read


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
    matched, until QueryBot learned again. A name discovery's own patterns masked that
    Learn let through was weighed by its values there (values_gate: businesses, never
    people), by the same patterns: it stays named.
    """
    if inventory is None:
        return
    for key, table in inventory.tables.items():
        if not (table.masked or table.mask_all):
            continue
        for column in model.columns.values():
            if column.table == key and held_by_masking(table, column.name) \
                    and not (column.values_allowed and _a_name_masked_by_pattern(table, column.name)):
                column.values_allowed = False


# The model each workspace answers with, kept between questions and dashboard tiles: it was read and parsed
# again for every one. Kept per workspace and database with its stamp -- the latest version, every admin
# decision, discovery's file -- and built again when any of them changes. At most ANSWERING_KEPT workspaces.
ANSWERING_KEPT = 64
_ANSWERING: OrderedDict[tuple[str, int], tuple[tuple, SemanticModel]] = OrderedDict()
_ANSWERING_LOCK = threading.Lock()


def _answering_stamp(account_id: str, client: dict[str, Any]) -> tuple | None:
    import store

    made_of = store.core2_model_stamp(account_id, client.get("db_config_id"))
    if made_of is None:
        return None
    path = schema_path(account_id, client)
    try:
        found = path.stat()
        discovery = (found.st_mtime_ns, found.st_size)
    except OSError:
        discovery = None
    # Which store it came from: one process can be pointed at another (a test's own database).
    where = os.environ.get("QUERYBOT_DB_PATH") or os.environ.get("DB_PATH") or ""
    return made_of, str(path), discovery, where


def answering_model(account_id: str, client: dict[str, Any]) -> SemanticModel | None:
    """The model questions are answered with: the latest, the admin's decisions, and discovery's masking now.

    Shared by every question and tile of the workspace until one of those changes: read it, never change it
    (a reader's own metric is answered on a copy, core2.plan.own_metric.with_own)."""
    stamp = _answering_stamp(account_id, client)
    if stamp is None:
        return None
    key = (account_id, int(client.get("db_config_id") or 0))
    with _ANSWERING_LOCK:
        kept = _ANSWERING.get(key)
        if kept is not None and kept[0] == stamp:
            _ANSWERING.move_to_end(key)
            return kept[1]
    model = load_model(account_id, client.get("db_config_id"))
    if model is not None:
        try:
            mask_as_discovered(model, discovered(account_id, client))
        except Exception as exc:  # noqa: BLE001 - the masking Learn read still holds
            log.warning("core2: discovery's masking for %s could not be read: %s", account_id, exc)
            return model          # not kept: the masking is read again for the next question
        with _ANSWERING_LOCK:
            _ANSWERING[key] = (stamp, model)
            _ANSWERING.move_to_end(key)
            while len(_ANSWERING) > ANSWERING_KEPT:
                _ANSWERING.popitem(last=False)
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
        if column.personal == "none" and not column.record_number or column.parts or table is None:
            continue
        fqn = ".".join(p for p in (table.database, table.schema_name, table.name) if p)
        tail = f".{table.name}.{column.name}".upper()
        current = existing.get(f"{fqn}.{column.name}".upper()) or next(
            (v for k, v in existing.items() if ("." + k).endswith(tail)), None)
        if current and (current.get("reviewed") or set(current.get("tags") or []) & {"PII", "PHI"}):
            continue
        said = classify_column(column.name, industry)
        tags = said["tags"] if set(said["tags"]) & {"PII", "PHI"} else sorted({*said["tags"], "PII"})
        if column.record_number:
            tags = sorted({*tags, "PHI"})      # an Rx number identifies a patient's prescription
        try:
            store.save_classification(
                account_id, fqn, column.name, sensitivity="RESTRICTED", identifiability="DIRECT", tags=tags,
                confidence=0.9, reviewed=False, reviewed_by="", source="learn",
                mask_strategy="safe_alias_name" if column.personal == "name" else "tokenize")
            written += 1
        except Exception as exc:  # noqa: BLE001 - a proposal not written leaves the column masked by core2 itself
            log.warning("core2: the classification of %s.%s was not proposed: %s", table.name, column.name, exc)
    return written


def learn_workers() -> int:
    """Tables Learn reads at once: QUERYBOT_LEARN_WORKERS (1 to 16, 4 when unset). Fewer is gentler on a small
    database that others use while Learn runs; each one is a connection and a query running in it."""
    try:
        return max(1, min(16, int(os.getenv("QUERYBOT_LEARN_WORKERS") or 4)))
    except ValueError:
        return 4


class StoredProfiles:
    """Each table a Learn reads, kept in the store as it goes (core2.bootstrap.profiler.KeptProfiles).

    A restart stops a running Learn, and a large database took most of an hour to read: the next Learn then
    starts where the stopped one was, with every table it had read whose columns, row count and masking are
    as they were. Never fails the Learn: a table not kept is read again next time, no more.
    """

    def __init__(self, account_id: str, db_id: int, started: str):
        import store

        self.account_id, self.db_id, self.started = account_id, db_id, started
        try:
            self._kept = store.resume_core2_learn(account_id, db_id, started)
        except Exception as exc:  # noqa: BLE001 - every table is read again
            log.warning("core2: tables kept from a stopped Learn of %s not read: %s", account_id, exc)
            self._kept = {}

    def __len__(self) -> int:
        return len(self._kept)

    def get(self, table_key: str, fingerprint: str) -> TableProfile | None:
        found = self._kept.get(table_key)
        if found is None or found[0] != fingerprint:
            return None
        try:
            return profile_from_json(found[1])
        except Exception as exc:  # noqa: BLE001 - read again
            log.warning("core2: the kept profile of %s could not be read: %s", table_key, exc)
            return None

    def put(self, table_key: str, fingerprint: str, profile: TableProfile) -> None:
        import store

        try:
            store.keep_core2_learn_table(self.account_id, self.db_id, self.started, table_key, fingerprint,
                                         profile_json(profile))
        except Exception as exc:  # noqa: BLE001 - read again after a restart, no more
            log.warning("core2: table %s not kept for a restart: %s", table_key, exc)

    def forget(self) -> None:
        import store

        try:
            store.forget_core2_learn(self.account_id, self.db_id)
        except Exception as exc:  # noqa: BLE001 - removed when the next Learn starts
            log.warning("core2: tables kept while %s learned not removed: %s", self.account_id, exc)


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
            raise ValueError("Run discovery first: Solviz learns from the tables discovery found.")
        inventory = from_schema_json(json.loads(path.read_text(encoding="utf-8")), config["db_type"])
        if not inventory.tables:
            raise ValueError("Discovery found no tables to learn from.")
        kept = StoredProfiles(account_id, db_id, started)
        if len(kept):
            say(f"Starting where the Learn that stopped was: {len(kept)} table(s) it read are kept, unless "
                "their columns or masking changed since")
        say(f"Connecting to the {config['db_type'].replace('_', ' ')} database")
        with QueryBotWarehouse(config["db_type"], config.get("credentials") or {}) as warehouse:
            gate = values_gate(account_id, inventory, _state(client), sample=value_sampler(warehouse, inventory))
            options = BuildOptions(profile=ProfileOptions(values_allowed=gate),
                                   labeler=_labeler(account_id, client, inventory), progress=say,
                                   workers=learn_workers(), kept=kept)
            model = build_model(warehouse, inventory, client_id=account_id, db_id=db_id, options=options)
        # Stored as learned: decisions are a layer applied when the model is read, so
        # undoing one brings back exactly what the data said.
        say("Saving what was learned")
        version = store.save_core2_model(account_id, db_id, model.model_dump_json(),
                                         source_hash=source_hash(inventory), stats=stats(model))
        kept.forget()
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
        say("Checking the business metrics this data holds (approval rate, margin, average order and the like)")
        try:
            from core2.kpi_proposals import propose

            from core2.kpi_proposals import LEARN_BUDGET

            propose(account_id, client, model=load_model(account_id, db_id), say=say, budget=LEARN_BUDGET)
        except Exception as exc:  # noqa: BLE001 - an admin can ask for them on the Requests page
            log.warning("core2: business metrics of %s not proposed after Learn: %s", account_id, exc, exc_info=True)
        say("Mapping every question this data answers, and checking them on the data")
        try:
            from core2.question_map import build_for

            build_for(account_id, client, say=say)
        except Exception as exc:  # noqa: BLE001 - it is built again the first time it is needed
            log.warning("core2: question map of %s not built after Learn: %s", account_id, exc, exc_info=True)
        left_out =sum("was left out" in note for note in model.notes)
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
    """The earlier setup's admin decisions as this model's overrides (core2.model.imports), adopted at once as the
    new core's own (the old engine is switched off; nothing in its setup changes them again); never fails the
    build."""
    import store
    from core2.model.imports import adopt, import_approvals

    try:
        done = import_approvals(account_id, db_id, model, str(_state(client).get("kb_dir") or ""), report=report)
        done["adopted"] = adopt(account_id, db_id)
    except Exception as exc:  # noqa: BLE001 - the model stands without them; the page says so
        log.warning("core2: today's decisions for %s could not be brought over: %s", account_id, exc, exc_info=True)
        done = {"error": str(exc)}
    store.set_core2_import_report(account_id, db_id, done)
    return done


def keep_decisions_current(account_id: str, db_id: int | None, client: dict[str, Any]) -> bool:
    """Nothing to bring over: the new core is the only engine, and the earlier setup's decisions came over at Learn
    as its own (``bring_over``); no page changes them in the earlier setup any more. It used to look, before an
    answer, for decisions made there since. Kept for its callers; always False."""
    return False


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
    good_news(model)
    apply_overrides(model, store.list_core2_overrides(account_id, db_id))
    return model


def good_news(model: SemanticModel) -> None:
    """Which way each learned figure is good news, for a model learned before Learn read it from its name: a cost,
    a refund or churn falls to be good. Before the admin's decisions, which keep theirs."""
    from core2.bootstrap import names

    for m in model.measures.values():
        if m.provenance == "profile" and m.kind in ("column", "count") and m.better == "higher":
            m.better = names.better(m.business_name)  # type: ignore[assignment]


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
