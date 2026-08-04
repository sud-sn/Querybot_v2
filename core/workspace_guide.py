"""Governed, user-scoped workspace onboarding and capability answers.

This module deliberately works with metadata only.  It never reads result rows,
generates SQL, or calls an LLM.  The same guide can therefore be rendered in
the web portal and in chat integrations without creating a second, less
governed source of truth about a client's data estate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import store
from core.pipeline_context import client_dir, get_portal_base, get_state
from core.semantic_layer import table_allowed, table_name_variants
from core.semantic_model import load_semantic_model

log = logging.getLogger("querybot.workspace_guide")

GUIDE_KINDS = frozenset({
    "capability_overview",
    "business_overview",
    "data_inventory",
    "table_meanings",
    "semantic_explainer",
    "question_examples",
})

_TECHNICAL_PREFIX_RE = re.compile(
    r"^(?:D|F|DIM|FACT|BR|BRIDGE|TBL|TB)\s+", re.IGNORECASE
)


def _clean_text(value: object, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _humanize(value: object) -> str:
    text = re.sub(r"[_\-]+", " ", str(value or "")).strip()
    text = _TECHNICAL_PREFIX_RE.sub("", text)
    return " ".join(word.upper() if len(word) <= 3 and word.isupper() else word.title()
                    for word in text.split())


def _table_ref(table: dict) -> str:
    return _clean_text(
        table.get("fqn") or table.get("qualified_name") or table.get("table"),
        240,
    ).upper()


def _visible_table(ref: str, allowed_tables: Iterable[str] | None) -> bool:
    return bool(ref) and table_allowed(ref, allowed_tables)


def _overview_from_markdown(content: str) -> str:
    in_overview = False
    lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            if in_overview:
                break
            in_overview = stripped.lower().startswith("## overview")
            continue
        if in_overview and stripped:
            lines.append(stripped.lstrip("- ").strip())
    return _clean_text(" ".join(lines), 360)


def _load_kb_overviews(kb_dir: str) -> dict[str, str]:
    """Index table overview paragraphs by all common table-name variants."""
    root = Path(kb_dir) if kb_dir else None
    if not root or not root.exists():
        return {}
    overviews: dict[str, str] = {}
    for path in sorted(root.glob("*_kb.md")):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        overview = _overview_from_markdown(content)
        if not overview:
            continue
        header_ref = ""
        for line in content.splitlines()[:8]:
            candidate = line.strip().lstrip("#").strip()
            if candidate:
                header_ref = candidate.split()[0]
                break
        refs = table_name_variants(header_ref)
        refs |= table_name_variants(path.stem.removesuffix("_kb"))
        for ref in refs:
            overviews.setdefault(ref, overview)
    return overviews


def _lookup_overview(overviews: dict[str, str], ref: str) -> str:
    for variant in table_name_variants(ref):
        if overviews.get(variant):
            return overviews[variant]
    return ""


def _table_summary(table: dict, overviews: dict[str, str]) -> dict:
    ref = _table_ref(table)
    raw_name = table.get("entity") or table.get("table") or ref.split(".")[-1]
    name = _humanize(raw_name) or ref.split(".")[-1]
    table_type = _clean_text(table.get("type"), 40).lower()
    grain = _clean_text(table.get("grain"), 140)
    meaning = (
        _lookup_overview(overviews, ref)
        or _clean_text(table.get("description") or table.get("overview"), 360)
    )
    if not meaning:
        if grain:
            meaning = f"Represents {name.lower()} at {grain}."
        elif table_type:
            meaning = f"Represents {name.lower()} {table_type} data."
        else:
            meaning = "A business description has not yet been curated for this table."
    parts = ref.split(".")
    schema = _clean_text(table.get("schema") or (parts[-2] if len(parts) >= 2 else "DEFAULT"), 100)
    return {
        "fqn": ref,
        "schema": schema,
        "table": _clean_text(table.get("table") or parts[-1], 120),
        "name": name,
        "type": table_type,
        "grain": grain,
        "meaning": meaning,
    }


def _fallback_tables(state: dict) -> list[dict]:
    refs = state.get("known_tables") or []
    if not refs and state.get("schema_dir"):
        try:
            from core.schema import load_known_tables
            refs = sorted(load_known_tables(state.get("schema_dir", "")))
        except Exception:
            refs = []
    tables = []
    for ref in refs:
        parts = str(ref).split(".")
        tables.append({
            "fqn": str(ref),
            "schema": parts[-2] if len(parts) >= 2 else "DEFAULT",
            "table": parts[-1],
            "entity": _humanize(parts[-1]),
        })
    return tables


def _metric_visible(metric: dict, allowed_tables: set[str] | None) -> bool:
    if allowed_tables is None:
        return True
    base_table = _clean_text(metric.get("base_table"), 240)
    # For restricted users, formula-only metrics without an explicit governed
    # table binding are omitted rather than risking disclosure by name.
    return bool(base_table and table_allowed(base_table, allowed_tables))


def _term_visible(term: dict, allowed_tables: set[str] | None) -> bool:
    if allowed_tables is None:
        return True
    refs = [part.strip() for part in str(term.get("tables_involved") or "").split(",") if part.strip()]
    return bool(refs and all(table_allowed(ref, allowed_tables) for ref in refs))


def _semantic_edge_visible(item: dict, allowed_tables: set[str] | None) -> bool:
    if allowed_tables is None:
        return True
    refs = [
        item.get("from") or item.get("from_table") or item.get("fact_table"),
        item.get("to") or item.get("to_table") or item.get("dimension_table"),
    ]
    refs = [str(ref) for ref in refs if ref]
    return bool(refs and all(table_allowed(ref, allowed_tables) for ref in refs))


def _safe_examples(
    account_id: str,
    kb_dir: str,
    schema_dir: str,
    allowed_tables: set[str] | None,
    limit: int = 6,
) -> list[str]:
    try:
        from core.suggestions import get_suggestions
        suggestions = get_suggestions(
            account_id,
            kb_dir,
            allowed_tables,
            n=limit,
            schema_dir=schema_dir,
        )
        return [
            _clean_text(item.get("question"), 180).rstrip("?") + "?"
            for item in suggestions
            if _clean_text(item.get("question"), 180)
        ][:limit]
    except Exception as exc:
        log.debug("Workspace guide suggestions unavailable: %s", exc)
        return []


def build_workspace_guide(account_id: str, user: dict | None) -> dict:
    """Build a metadata-only guide filtered to the signed-in user's ACL."""
    client = store.get_client(account_id) or {}
    state = get_state(account_id)
    allowed_tables = store.get_allowed_tables(user) if user else set()
    kb_dir = str(state.get("kb_dir") or (client_dir(account_id) / "kb"))
    schema_dir = str(state.get("schema_dir") or (client_dir(account_id) / "schema"))
    try:
        model = load_semantic_model(kb_dir)
    except Exception as exc:
        log.debug("Workspace guide semantic model unavailable: %s", exc)
        model = {}

    raw_tables = list(model.get("tables") or [])
    known_variants: set[str] = set()
    for table in raw_tables:
        known_variants |= table_name_variants(_table_ref(table))
    # A partially curated semantic model must not make an otherwise queryable
    # table disappear from the user's inventory.  Merge schema-known tables as
    # safe metadata-only fallbacks, while preferring richer semantic entries.
    for table in _fallback_tables(state):
        variants = table_name_variants(_table_ref(table))
        if variants & known_variants:
            continue
        raw_tables.append(table)
        known_variants |= variants
    overviews = _load_kb_overviews(kb_dir)
    tables = [
        _table_summary(table, overviews)
        for table in raw_tables
        if _visible_table(_table_ref(table), allowed_tables)
    ]
    tables.sort(key=lambda item: (item["schema"], item["name"], item["table"]))

    metrics: list[dict] = []
    try:
        for metric in store.list_metrics(account_id):
            if not metric.get("is_active", 1) or not _metric_visible(metric, allowed_tables):
                continue
            metrics.append({
                "name": _clean_text(metric.get("name"), 100),
                "description": _clean_text(metric.get("description"), 220),
            })
    except Exception as exc:
        log.debug("Workspace guide metrics unavailable: %s", exc)
    metrics = [item for item in metrics if item["name"]][:12]

    terms: list[dict] = []
    try:
        for term in store.list_terms(account_id, active_only=True):
            if not _term_visible(term, allowed_tables):
                continue
            terms.append({
                "term": _clean_text(term.get("term"), 100),
                "definition": _clean_text(term.get("definition"), 220),
            })
    except Exception as exc:
        log.debug("Workspace guide terms unavailable: %s", exc)
    terms = [item for item in terms if item["term"]][:12]

    relationships = [
        item for item in (model.get("relationships") or [])
        if _semantic_edge_visible(item, allowed_tables)
    ]
    date_roles = [
        item for item in (model.get("date_roles") or [])
        if _semantic_edge_visible(item, allowed_tables)
    ]

    dashboards: list[dict] = []
    if user and user.get("id"):
        try:
            dashboards = store.list_dashboards(account_id, int(user["id"]))
        except Exception as exc:
            log.debug("Workspace guide dashboards unavailable: %s", exc)

    examples = _safe_examples(account_id, kb_dir, schema_dir, allowed_tables)
    business = _clean_text(
        client.get("business_desc")
        or model.get("business_description")
        or state.get("business_desc"),
        700,
    )
    schemas: dict[str, int] = {}
    for table in tables:
        schemas[table["schema"]] = schemas.get(table["schema"], 0) + 1

    return {
        "account_id": account_id,
        "business": business,
        "tables": tables,
        "schemas": schemas,
        "metrics": metrics,
        "terms": terms,
        "relationship_count": len(relationships),
        "date_role_count": len(date_roles),
        "dashboards": dashboards,
        "examples": examples,
        "is_admin": allowed_tables is None,
        "portal_base": get_portal_base(),
    }


def _bullets(values: Iterable[str]) -> str:
    return "\n".join(f"  • {value}" for value in values)


def _examples_block(examples: list[str]) -> str:
    if not examples:
        return "No validated starter questions are available for your current access yet."
    return _bullets(f"_{question}_" for question in examples)


def _table_lines(guide: dict, limit: int = 15) -> list[str]:
    tables = guide.get("tables") or []
    lines = [
        f"*{table['name']}* (`{table['schema']}.{table['table']}`) — {table['meaning']}"
        for table in tables[:limit]
    ]
    if len(tables) > limit:
        lines.append(f"…and {len(tables) - limit} more accessible tables in the Semantic Layer.")
    return lines


def render_workspace_guide(
    kind: str,
    account_id: str,
    user: dict | None,
    *,
    include_examples: bool = True,
) -> tuple[str, list[str]]:
    """Return ``(markdown, suggested questions)`` for one guide intent."""
    guide = build_workspace_guide(account_id, user)
    examples = guide["examples"]
    table_count = len(guide["tables"])
    metric_names = [item["name"] for item in guide["metrics"][:8]]
    schema_text = ", ".join(
        f"{name} ({count} table{'s' if count != 1 else ''})"
        for name, count in sorted(guide["schemas"].items())
    )
    dashboard_url = f"{guide['portal_base']}/portal/dashboard"
    semantic_url = f"{guide['portal_base']}/portal/kb"

    if kind == "capability_overview":
        dashboard_count = len(guide["dashboards"])
        dashboard_note = (
            f"You currently have access to {dashboard_count} named dashboard"
            f"{'s' if dashboard_count != 1 else ''}."
            if dashboard_count else
            "You can create a named dashboard from a result and add suitable KPIs, charts, and tables."
        )
        semantic_note = (
            f"This workspace currently exposes {len(guide['metrics'])} governed metric"
            f"{'s' if len(guide['metrics']) != 1 else ''}, {len(guide['terms'])} business term"
            f"{'s' if len(guide['terms']) != 1 else ''}, {guide['relationship_count']} relationship"
            f"{'s' if guide['relationship_count'] != 1 else ''}, and {guide['date_role_count']} date role"
            f"{'s' if guide['date_role_count'] != 1 else ''} within your access."
        )
        text = (
            "*What I can do in this workspace*\n\n"
            "  • Answer natural-language questions using the business data you are permitted to access.\n"
            "  • Calculate KPIs, trends, comparisons, rankings, distributions, and time-based analysis.\n"
            "  • Present results as KPI cards, charts, or tables and explain how the answer was produced.\n"
            "  • Refine a recent result—filter, sort, limit, reformat dates/currency/decimals, or change its visual.\n"
            f"  • Create and maintain named [dashboards]({dashboard_url}), add results, arrange visuals, apply filters, and subscribe to updates. {dashboard_note}\n"
            f"  • Use the governed [Semantic Layer]({semantic_url}) to resolve business terms, metrics, joins, and dates. {semantic_note}\n\n"
            "Presentation-only follow-ups can reuse your governed recent result. If a request needs new data or a different calculation, I run a new governed query. Access controls and masking still apply."
        )
        if include_examples:
            text += "\n\n*Try one of these validated questions:*\n" + _examples_block(examples)
        return text, examples

    if kind == "business_overview":
        business = guide["business"] or "A business description has not yet been curated for this workspace."
        entity_names = [table["name"] for table in guide["tables"][:10]]
        text = (
            "*Business and data overview*\n\n"
            f"{business}\n\n"
            f"Your access covers {table_count} table{'s' if table_count != 1 else ''}"
            + (f" across {schema_text}." if schema_text else ".")
        )
        if entity_names:
            text += "\n\n*Business entities represented:*\n" + _bullets(entity_names)
        if metric_names:
            text += "\n\n*Governed metrics:* " + ", ".join(metric_names)
        text += "\n\nAsk _which tables are available and what do they mean?_ for table-level descriptions."
        return text, examples

    if kind in {"data_inventory", "table_meanings"}:
        heading = "Available tables and their business meaning" if kind == "table_meanings" else "Data available to you"
        lines = _table_lines(guide)
        text = f"*{heading}*\n\n"
        if guide["business"]:
            text += f"{guide['business']}\n\n"
        if schema_text:
            text += f"*Schemas:* {schema_text}\n\n"
        text += _bullets(lines) if lines else "No queryable tables are assigned to your current access."
        if metric_names:
            text += "\n\n*Governed metrics available:* " + ", ".join(metric_names)
        text += f"\n\nBrowse the [Semantic Layer]({semantic_url}) for the governed catalog."
        return text, examples

    if kind == "semantic_explainer":
        text = (
            "*How the Semantic Layer works*\n\n"
            "The Semantic Layer translates business language into governed database logic. It supplies approved metric definitions, business terms and synonyms, table relationships, date meanings, and access rules. When you ask for something such as _monthly revenue_, QueryBot uses it to determine what revenue means, which business date applies, which tables can be joined, and which governance rules must be enforced before SQL is generated.\n\n"
            f"Within your current access, I can use {len(guide['metrics'])} metric{'s' if len(guide['metrics']) != 1 else ''}, "
            f"{len(guide['terms'])} business term{'s' if len(guide['terms']) != 1 else ''}, "
            f"{guide['relationship_count']} governed relationship{'s' if guide['relationship_count'] != 1 else ''}, and "
            f"{guide['date_role_count']} date role{'s' if guide['date_role_count'] != 1 else ''}. "
            "If wording or a date meaning is ambiguous, I should ask you to clarify instead of guessing. Ad hoc calculations are possible when the accessible schema supports them, while approved metrics are preferred when available.\n\n"
            f"You can inspect the catalog on the [Semantic Layer page]({semantic_url})."
        )
        return text, examples

    if kind == "question_examples":
        text = (
            "*How to ask questions*\n\n"
            "State what you want to measure, then add any breakdown, time range, filter, comparison, or presentation preference. You can ask for totals, KPIs, trends, top/bottom rankings, distributions, period comparisons, tables, and charts. After an answer, you can say things like _show only the top 10_, _format this as currency_, _show month and year_, _change this to a pie chart_, or _add this to a dashboard_. If the intended format or business meaning is unclear, I will ask a follow-up question."
        )
        if include_examples:
            text += "\n\n*Validated questions for your current access:*\n" + _examples_block(examples)
        return text, examples

    return "", []
