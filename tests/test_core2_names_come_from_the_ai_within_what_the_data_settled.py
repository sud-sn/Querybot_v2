"""The AI names what the data found, and nothing more.

On a warehouse whose names say nothing (T07.C14), the bootstrap still finds the
structure from the values; the AI then reads profiles and gives business names.
It cannot change what the data settled (a key, a measure, how it adds up), it
sees common values only where they may be shown, and when it fails the names
read from the data stay.
"""

from __future__ import annotations

import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.labels import SYSTEM, table_brief
from core2.bootstrap.profiler import ProfileOptions
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2.domains import retail
from evals.core2.framework import materialize


@pytest.fixture(scope="module")
def generic():
    return materialize(retail.build(), "generic")


def _build(built, **options):
    warehouse = DuckDBWarehouse(built.con)
    return build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                       options=BuildOptions(workers=1, outliers=False, **options))


def test_the_ai_names_what_meaningless_names_hide(generic):
    lines, net, order_date = generic.t("order_lines"), generic.c("order_lines.net_amount")[1], \
        generic.c("order_lines.order_date_key")[1]
    seen: list[str] = []

    def ai(system: str, user: str) -> str:
        seen.append(user)
        assert system == SYSTEM
        if f'"table": "{lines}"' not in user:
            return "{}"
        return json.dumps({"tables": {lines: {
            "name": "Order line", "description": "One line of a customer order.", "grain": "one row per order line",
            "kind": "dimension",   # not the AI's call: ignored
            "columns": {net: {"name": "Net sales", "synonyms_en": ["revenue", "sales"], "synonyms_fr": ["ventes"]},
                        order_date: {"name": "Order date"}}}}})

    model = _build(generic, labeler=ai)
    table = next(t for t in model.tables.values() if t.name == lines)
    assert table.business_name == "Order line" and table.kind == "fact"
    measure = next(m for m in model.measures.values() if getattr(m.expr, "column", None) == f"{table.key}.{net.lower()}")
    assert measure.business_name == "Net sales" and measure.slug == "net_sales"
    assert measure.additivity == "additive" and measure.synonyms["en"] == ["revenue", "sales"]
    # Generic names repeat across tables (C07 is also a customer's column): the table must match too.
    role = model.date_roles[f"{table.key}.{order_date.lower()}"]
    assert role.name == "Order date" and role.slug == "order_date" and role.is_default
    assert any(f'"table": "{lines}"' in u for u in seen)


def test_common_values_reach_the_ai_only_where_they_may_be_shown(generic):
    status = generic.c("order_lines.status_code")[1]
    model = _build(generic, profile=ProfileOptions(values_allowed=lambda table, column: column != status))
    lines = next(k for k, t in model.tables.items() if t.name == generic.t("order_lines"))
    brief = json.dumps(table_brief(model, lines))
    currency = generic.c("order_lines.currency_code")[1]
    assert '"common_values": ["USD"]' in brief, "an allowed code column shows its values"
    entry = next(c for c in table_brief(model, lines)["columns"] if c["column"] == status)
    assert "common_values" not in entry and currency != status


def test_when_the_ai_fails_the_names_from_the_data_stay(generic):
    def broken(system: str, user: str) -> str:
        raise RuntimeError("model unavailable")

    model = _build(generic, labeler=broken)
    assert any("the AI call failed" in note for note in model.notes)
    assert all(t.business_name for t in model.tables.values())
    assert any(m.slug for m in model.measures.values())


def test_an_admin_name_is_never_replaced_by_the_ai(generic):
    from core2.model.overrides import apply_overrides, target

    lines = generic.t("order_lines")

    def ai(system: str, user: str) -> str:
        return json.dumps({"tables": {lines: {"name": "Sales lines"}}})

    model = _build(generic, labeler=ai)
    key = next(k for k, t in model.tables.items() if t.name == lines)
    apply_overrides(model, [{"object_key": target("table", key), "field": "business_name", "value": "Order lines"}])
    assert model.tables[key].business_name == "Order lines"
