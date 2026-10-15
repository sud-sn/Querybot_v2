"""Business metrics the workspace's data holds, checked on the data and proposed to the admin.

After Learn, and when an admin asks on the Requests page, the business metrics of the workspace's kind of business
(core2/kpis.py) are written over its fields, run over their last months as the metric editor's Check runs a draft,
and each that reads as the metric it is named is filed as a New metric request from Solviz, with its formula and
its last months. Nothing is a metric until an admin accepts it; accepted, it is an admin's metric everyone's
answers use, and executive dashboards show it in its area.

A metric is not proposed when the data does not hold it: it cannot run, it has a value in too few of the last
months, a share falls outside 0 to 100, it is 0 in most months, or it is the same every month at 0 or 100% (its
two fields hold the same numbers). A metric proposed once is never proposed again, whatever was decided: an admin
who rejected it is not asked twice.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core2 import kpis
from core2.model.schema import SemanticModel

log = logging.getLogger("querybot.core2")

SOURCE = "learn"          # a request's source: proposed by Solviz from what Learn found
MOST = 30                 # metrics checked in one run
MIN_MONTHS = 3            # months a metric must have a value in, of the last the check reads
LEARN_BUDGET = 180.0      # seconds Learn spends checking them; the rest wait for Suggest business metrics

# One run at a time per workspace: Learn's last step and an admin's Suggest never file the same metric twice.
_RUNNING: dict[str, threading.Lock] = {}
_RUNNING_LOCK = threading.Lock()


def _lock(account_id: str) -> threading.Lock:
    with _RUNNING_LOCK:
        return _RUNNING.setdefault(account_id, threading.Lock())


@dataclass
class Outcome:
    kind: str = ""                                          # the kind of business the metrics were chosen for
    filed: list[str] = field(default_factory=list)          # the metrics proposed, by name
    skipped: list[dict[str, str]] = field(default_factory=list)   # {"name", "why"}: checked, not proposed
    already: int = 0                                        # proposed before (waiting or decided)
    left: int = 0                                           # not checked in the time given (Learn's)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "filed": self.filed, "skipped": self.skipped, "already": self.already,
                "left": self.left, "error": self.error}


def industry_of(account_id: str) -> str:
    """The workspace's industry on its Compliance page (banking, healthcare_pharmacy), or ""."""
    try:
        import store

        return str((store.get_compliance_profile(account_id) or {}).get("industry") or "")
    except Exception as exc:  # noqa: BLE001 - the data alone decides the kind of business
        log.warning("core2: the industry of %s could not be read: %s", account_id, exc)
        return ""


def governed_warehouse(account_id: str, client: dict[str, Any]):
    """The workspace's database, read-only, every table: what an admin's check runs on."""
    import store
    from core.schema import load_known_tables
    from core2.warehouse.governed import GovernedWarehouse

    db_config = store.get_db_config(int(client.get("db_config_id") or 0)) if client.get("db_config_id") else None
    if not db_config:
        return None
    state = store.get_client_state(account_id) or {}
    return GovernedWarehouse(account_id, None, db_config, known_tables=load_known_tables(state.get("schema_dir", "")),
                             allowed_tables=None)


def _values(result: dict[str, Any]) -> tuple[list[float], bool]:
    months = result.get("months") or []
    single = len(months) == 1 and not months[0].get("period")
    return [m["value"] for m in months if m.get("value") is not None], single


def judge(proposal: kpis.Proposal, result: dict[str, Any]) -> str:
    """Why a checked metric is not proposed, in the admin's words; "" when it is."""
    if result.get("problem"):
        return f"it could not run on the data: {result['problem']}"
    values, single = _values(result)
    months = len(result.get("months") or [])
    if not values:
        return "it has no value in the data"
    if not single and len(values) < MIN_MONTHS:
        return f"it has a value in only {len(values)} of the last {months} months"
    if proposal.share and any(v < -1e-6 or v > 100 + 1e-6 for v in values):
        return "it is meant to be a share between 0 and 100, and is not"
    if all(abs(v) < 1e-9 for v in values):
        return "it is 0 in every month" if not single else "it is 0"
    if proposal.share and not single and sum(abs(v) > 1e-9 for v in values) < MIN_MONTHS:
        return "it is 0 in most of the last months"
    if len(values) > 1 and max(values) - min(values) < 1e-9 and proposal.fmt == "percent" \
            and abs(values[0] - 100) < 1e-6:
        return "it is 100% every month: its two fields hold the same numbers"
    return ""


def shown(value: float | None, fmt: str) -> str:
    if value is None:
        return "no value"
    if fmt == "percent":
        return f"{value:,.1f}%"
    if fmt == "currency":
        return f"{value:,.2f}"
    return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"


def _proposed_before(account_id: str) -> tuple[set[str], set[str]]:
    """The business metrics proposed before, whatever became of them, and the names of new metrics waiting."""
    import store

    keys, names = set(), set()
    for r in store.list_core2_requests(account_id, limit=2000):
        if r.get("kind") != "new_metric":
            continue
        if r.get("source") == SOURCE and isinstance(r.get("definition"), dict) and r["definition"].get("kpi"):
            keys.add(str(r["definition"]["kpi"]))
        if r.get("status") == "waiting":
            names.add(str(r.get("target_name") or "").casefold())
    return keys, names


def _tables_read(definition: dict[str, Any]) -> set[str]:
    """Every table a metric's definition reads: its own, and each one a field it names is on."""
    out = {str(definition.get("table") or "")}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "column" and isinstance(value, str) and "." in value:
                    out.add(value.rsplit(".", 1)[0])
                elif key == "columns" and isinstance(value, list):
                    out.update(v.rsplit(".", 1)[0] for v in value if isinstance(v, str) and "." in v)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({k: v for k, v in definition.items() if k in ("expr", "filters")})
    return {t for t in out if t}


def waiting_metrics(account_id: str, allowed: set[str] | None = None, most: int = 12) -> list[str]:
    """The business metrics suggested from the data and waiting for an admin, by name, the earliest first: of the
    tables a reader may use (``allowed``, the model's table keys; None: all), every table each one reads."""
    import store

    waiting = [r for r in store.list_core2_requests(account_id, status="waiting", limit=500)
               if r.get("kind") == "new_metric" and r.get("source") == SOURCE
               and (allowed is None or _tables_read(r.get("definition") or {}) <= allowed)]
    return [str(r.get("target_name") or "") for r in reversed(waiting)][:most]


def propose(account_id: str, client: dict[str, Any] | None = None, *, model: SemanticModel | None = None,
            warehouse: Any = None, today: dt.date | None = None, lang: str = "en",
            say: Callable[[str], None] | None = None, budget: float | None = None) -> Outcome:
    """Check the business metrics the workspace's data holds and file those that pass as New metric requests.
    ``budget``: seconds to spend checking (Learn's); what is left is checked when an admin asks."""
    with _lock(account_id):
        return _propose(account_id, client, model=model, warehouse=warehouse, today=today, lang=lang, say=say,
                        budget=budget)


def _propose(account_id: str, client: dict[str, Any] | None, *, model: SemanticModel | None, warehouse: Any,
             today: dt.date | None, lang: str, say: Callable[[str], None] | None, budget: float | None) -> Outcome:
    import store
    from core.brand import PRODUCT_NAME
    from core2 import recipes
    from core2.bootstrap.service import load_model
    from core2.model import authoring

    out = Outcome()
    client = client or store.get_client(account_id) or {}
    db_id = client.get("db_config_id")
    model = model if model is not None else load_model(account_id, db_id)
    if model is None:
        out.error = "Solviz has not learned this workspace's data yet."
        return out
    out.kind = recipes.kind_of(model, industry=industry_of(account_id)).kind
    before, waiting = _proposed_before(account_id)
    todo = []
    for p in kpis.proposals(model, out.kind, lang=lang):
        if p.key in before or p.name.casefold() in waiting:
            out.already += 1
        else:
            todo.append(p)
    if not todo:
        return out
    warehouse = warehouse if warehouse is not None else governed_warehouse(account_id, client)
    if warehouse is None:
        out.error = "This workspace has no database to check the metrics on."
        return out
    today = today or dt.date.today()
    checking = model.model_copy(deep=True)     # the check adds its draft to the model it is given
    began = time.monotonic()
    for p in todo[:MOST]:
        if budget is not None and time.monotonic() - began > budget:
            out.left += 1
            continue
        try:
            measure = kpis.measure_of(model, p)
        except Exception as exc:  # noqa: BLE001 - a formula the fields cannot be read in is said so, never filed
            out.skipped.append({"name": p.name, "why": f"its fields cannot be read that way: {exc}"})
            continue
        result = authoring.check(checking, measure, warehouse, today)
        why = judge(p, result)
        if why:
            out.skipped.append({"name": p.name, "why": why})
            continue
        definition = measure.model_dump(mode="json")
        # What was checked, for the admin's card; read by no one else (a Measure ignores it).
        definition["checked"] = {
            "months": [{"period": m.get("period") or "", "value": m.get("value"), "shown": shown(m.get("value"), p.fmt)}
                       for m in result.get("months") or []],
            "date": (result.get("latest") or {}).get("date") or "", "area": p.area, "rank": p.rank,
            "on": today.isoformat()}
        store.add_core2_request(account_id, db_id, kind="new_metric", user_id="", user_name=PRODUCT_NAME,
                                target_name=p.name, note=p.says, source=SOURCE, definition=definition)
        out.filed.append(p.name)
    if say is not None and (out.filed or out.skipped or out.left):
        say(f"Proposed {len(out.filed)} business metric(s) on the Requests page for an admin to accept"
            + (f"; {len(out.skipped)} the data does not hold were left out" if out.skipped else "")
            + (f"; {out.left} more are checked when an admin presses Suggest business metrics" if out.left else ""))
    return out
