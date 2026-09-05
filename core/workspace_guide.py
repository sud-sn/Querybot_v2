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
from core.i18n import plural as _n, t as _t
from core.pipeline_context import client_dir, get_state
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


def _sentence_safe_text(value: object, limit: int = 500, *, first_only: bool = False) -> str:
    """Return readable prose without cutting a word or sentence in half."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().replace("`", "")
    if not text:
        return ""
    sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", text))
    if first_only and sentence_ends:
        return text[:sentence_ends[0].end()].strip()
    if len(text) <= limit:
        return text
    complete = [match for match in sentence_ends if match.end() <= limit]
    if complete:
        return text[:complete[-1].end()].strip()
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (clipped or text[:limit]).strip() + "…"


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
    # A table inventory needs its business purpose, not field lists, row
    # counts, date ranges, or profiling details that often follow in the KB.
    return _sentence_safe_text(" ".join(lines), 280, first_only=True)


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
        or _sentence_safe_text(
            table.get("description") or table.get("overview"),
            280,
            first_only=True,
        )
    )
    if not meaning:
        # name, grain and table_type are the tenant's own semantic model, so
        # they are interpolated; only the sentence around them is copy.
        if grain:
            meaning = _t("guide.table.represents_grain",
                         name=name.lower(), grain=grain)
        elif table_type:
            meaning = _t("guide.table.represents_type",
                         name=name.lower(), type=table_type)
        else:
            meaning = _t("guide.table.no_description")
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
            _clean_text(item.get("question"), 180).rstrip(" ?.!;:") + "?"
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
    business = _sentence_safe_text(
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
    }


def _bullets(values: Iterable[str]) -> str:
    return "\n".join(f"  • {value}" for value in values)


def _examples_block(examples: list[str]) -> str:
    if not examples:
        return _t("guide.examples.none")
    return _bullets(f"_{question}_" for question in examples)


def _table_lines(guide: dict, limit: int = 15) -> list[str]:
    tables = guide.get("tables") or []
    lines = [
        _t("guide.table_line", name=table["name"], schema=table["schema"],
           table=table["table"], meaning=table["meaning"])
        for table in tables[:limit]
    ]
    if len(tables) > limit:
        lines.append(_n("guide.more_tables", len(tables) - limit))
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
    # Every count below goes through plural(). `'s' if n != 1 else ''` is an
    # English rule that reports "0 tables" where French wants "0 table", and
    # an empty workspace is exactly what a new tenant has.
    schema_text = ", ".join(
        _n("guide.schema_entry", count, name=name)
        for name, count in sorted(guide["schemas"].items())
    )
    metric_count = _n("guide.count.metric", len(guide["metrics"]))
    term_count = _n("guide.count.term", len(guide["terms"]))
    if kind == "capability_overview":
        dashboard_count = len(guide["dashboards"])
        dashboard_note = (
            _t("guide.capability.dashboard_note",
               dashboards=_n("guide.count.dashboard", dashboard_count))
            if dashboard_count else _t("guide.capability.dashboard_none")
        )
        semantic_note = _t(
            "guide.capability.semantic_note",
            metrics=metric_count, terms=term_count,
            relationships=_n("guide.count.relationship", guide["relationship_count"]),
            date_roles=_n("guide.count.date_role", guide["date_role_count"]),
        )
        text = (
            f"{_t('guide.capability.title')}\n\n"
            f"  • {_t('guide.capability.answer')}\n"
            f"  • {_t('guide.capability.calculate')}\n"
            f"  • {_t('guide.capability.present')}\n"
            f"  • {_t('guide.capability.refine')}\n"
            f"  • {_t('guide.capability.dashboards', note=dashboard_note)}\n"
            f"  • {_t('guide.capability.semantic', note=semantic_note)}\n\n"
            f"{_t('guide.capability.footer')}"
        )
        if include_examples:
            text += ("\n\n" + _t("guide.capability.try_these") + "\n"
                     + _examples_block(examples))
        return text, examples

    if kind == "business_overview":
        business = guide["business"] or _t("guide.business.no_description")
        entity_names = [table["name"] for table in guide["tables"][:10]]
        tables_phrase = _n("guide.count.table", table_count)
        # One sentence per shape rather than a stem plus an optional clause:
        # French puts the schema list after a participle that agrees with
        # "tables", so the seam English can hide is one French cannot.
        text = (
            f"{_t('guide.business.title')}\n\n"
            f"{business}\n\n"
            + (_t("guide.business.access_covers_schemas",
                  tables=tables_phrase, schemas=schema_text)
               if schema_text else
               _t("guide.business.access_covers", tables=tables_phrase))
        )
        if entity_names:
            text += "\n\n" + _t("guide.business.entities") + "\n" + _bullets(entity_names)
        if metric_names:
            text += "\n\n" + _t("guide.business.metrics", names=", ".join(metric_names))
        text += "\n\n" + _t("guide.business.ask_tables")
        return text, examples

    if kind in {"data_inventory", "table_meanings"}:
        heading = _t("guide.inventory.title_meanings" if kind == "table_meanings"
                     else "guide.inventory.title_data")
        lines = _table_lines(guide)
        text = f"{heading}\n\n"
        if guide["business"]:
            text += f"{guide['business']}\n\n"
        if schema_text:
            text += _t("guide.inventory.schemas", schemas=schema_text) + "\n\n"
        text += _bullets(lines) if lines else _t("guide.inventory.none")
        if metric_names:
            text += "\n\n" + _t("guide.inventory.metrics", names=", ".join(metric_names))
        text += "\n\n" + _t("guide.inventory.footer")
        return text, examples

    if kind == "semantic_explainer":
        text = (
            f"{_t('guide.semantic.title')}\n\n"
            f"{_t('guide.semantic.body')}\n\n"
            + _t("guide.semantic.within_access",
                 metrics=_n("guide.count.plain_metric", len(guide["metrics"])),
                 terms=term_count,
                 relationships=_n("guide.count.governed_relationship",
                                  guide["relationship_count"]),
                 date_roles=_n("guide.count.date_role", guide["date_role_count"]))
            + " " + _t("guide.semantic.clarify") + "\n\n"
            + _t("guide.semantic.inspect")
        )
        return text, examples

    if kind == "question_examples":
        text = f"{_t('guide.questions.title')}\n\n{_t('guide.questions.body')}"
        if include_examples:
            text += ("\n\n" + _t("guide.questions.validated") + "\n"
                     + _examples_block(examples))
        return text, examples

    return "", []
