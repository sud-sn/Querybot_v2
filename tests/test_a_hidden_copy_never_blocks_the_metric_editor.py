"""A hidden copy of a table never blocks the metric editor, nor a reader's own logic on fields.

On an ERP's reporting database, Learn found a staging copy of the order lines (stg.SalesOrderLine_Load), hid it as
a copy, and named it after the table it copies. The admin then wrote "Net sales" as
SUM([Sales order line · Extended amount (base currency)]), the way the editor suggests it, and the editor refused:
"names more than one field", because the hidden copy's field has the same name. Every metric on the order lines was
refused the same way, and "gross_amount minus discount_amount from order lines" found two fields for each name and
fell back to two totals.

A name the editor suggests reads as the field it suggested: when a name fits a shown table's field and a hidden
table's, the shown one is meant. A field only a hidden table has still reads, and two shown tables of one name are
still asked about. Synthetic data only: the retail warehouse.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.model import authoring
from core2.model.authoring import AuthoringError, field_ref, fields, read
from core2.model.overrides import set_aside
from core2.plan import asks
from core2.plan.ir import Plan
from evals.core2 import domains
from evals.core2.compile_eval import learn

LINES = "memory.main.order_lines"
COPY = "memory.stg.order_lines_load"


def _with_copy(model, *, hidden: bool = True, only: str | None = None):
    """The order lines again under a staging table, named as Learn names a copy: after the table it copies, and set
    aside as the product sets a copy aside (its fields hidden with it)."""
    lines = model.tables[LINES]
    model.tables[COPY] = lines.model_copy(update={"key": COPY, "name": "order_lines_load", "slug": "order_line_2",
                                                  "hidden": False, "evidence": []})
    for column in [c for c in model.columns.values() if c.table == LINES]:
        key = f"{COPY}.{column.name}"
        model.columns[key] = column.model_copy(update={"key": key, "table": COPY})
    if only:                                            # a field only the copy has
        key = f"{COPY}.{only}"
        model.columns[key] = model.columns[f"{COPY}.net_amount"].model_copy(
            update={"key": key, "name": only, "business_name": "Load batch amount"})
    if hidden:
        set_aside(model, COPY, True, "a load copy of Order line")
    return model


@pytest.fixture()
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def test_a_suggested_field_reads_as_the_shown_tables_field(model):
    _with_copy(model)
    net = f"{LINES}.net_amount"
    suggested = next(f["ref"] for f in fields(model) if f["key"] == net)
    assert suggested == field_ref(model, f"{COPY}.net_amount")            # the copy is named like the table
    done = read(model, f"SUM([{suggested}])")
    assert done.table == LINES and done.expr.column == net


def test_a_bare_field_name_reads_as_the_shown_tables(model):
    _with_copy(model)
    name = authoring.field_name(model.columns[f"{LINES}.net_amount"])
    assert read(model, f"SUM([{name}])").expr.column == f"{LINES}.net_amount"


def test_the_tables_name_counts_the_shown_table_whichever_was_learned_last(model):
    _with_copy(model)
    table = authoring.table_name(model, LINES)
    assert read(model, f"COUNT([{table}])").table == LINES
    model.tables = {k: model.tables[k] for k in [COPY, *[k for k in model.tables if k != COPY]]}   # copy first
    assert read(model, f"COUNT([{table}])").table == LINES


def test_a_field_only_the_hidden_copy_has_still_reads(model):
    _with_copy(model, only="load_batch_amount")
    done = read(model, f"SUM([{authoring.table_name(model, LINES)} · Load batch amount])")
    assert done.expr.column == f"{COPY}.load_batch_amount"


def test_two_shown_tables_of_one_name_are_still_asked_about(model):
    _with_copy(model, hidden=False)
    with pytest.raises(AuthoringError, match="more than one field"):
        read(model, f"SUM([{field_ref(model, f'{LINES}.net_amount')}])")


def test_fields_a_reader_joins_are_one_figure_with_a_hidden_copy_learned(model):
    _with_copy(model)
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["gross_amount", "discount_amount"],
                                "derived": [{"name": "Gross less discount", "op": "difference",
                                             "measures": ["gross_amount", "discount_amount"]}],
                                "group_by": ["store.name"]})
    done = asks.fields_combined(plan, model, "gross_amount minus discount_amount from order lines, by store")
    assert [f.formula for f in done.figures] == [
        f"SUM([{field_ref(model, f'{LINES}.gross_amount')}] - [{field_ref(model, f'{LINES}.discount_amount')}])"]
    assert set(read(model, done.figures[0].formula).columns) == {f"{LINES}.gross_amount", f"{LINES}.discount_amount"}


# ── What the review of the change asked: only a copy named after a shown table is set aside ──────────────────


def _another_table(model, name: str, hidden_column: str):
    """The order lines again as another shown table of its own name, one of its fields hidden."""
    key = "memory.main.invoice_lines"
    model.tables[key] = model.tables[LINES].model_copy(update={"key": key, "name": "invoice_lines", "slug": "invoice_line",
                                                                "business_name": name, "evidence": []})
    for column in [c for c in model.columns.values() if c.table == LINES]:
        model.columns[f"{key}.{column.name}"] = column.model_copy(update={"key": f"{key}.{column.name}", "table": key,
                                                                          "hidden": column.name == hidden_column})
    return key


def test_fields_of_another_table_the_question_names_are_never_swapped_for_the_lines(model):
    _another_table(model, "Invoice line", "discount_amount")
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["gross_amount", "discount_amount"],
                                "derived": [{"name": "Gross less discount", "op": "difference",
                                             "measures": ["gross_amount", "discount_amount"]}],
                                "group_by": ["store.name"]})
    done = asks.fields_combined(plan, model, "gross_amount minus discount_amount from invoice lines, by store")
    assert not any(LINES in f.formula or authoring.table_name(model, LINES) in f.formula for f in done.figures)


def test_a_field_hidden_on_the_table_itself_still_reads_with_a_copy_learned(model):
    model.columns[f"{LINES}.cost_amount"].hidden = True
    _with_copy(model)
    ref = field_ref(model, f"{LINES}.cost_amount")
    assert read(model, f"SUM([{ref}])").expr.column == f"{LINES}.cost_amount"


def test_a_copys_metrics_never_take_the_tables_metric_names(model):
    _with_copy(model)
    for m in [m for m in model.measures.values() if m.table == LINES]:          # learned last, as stg.* follows
        key = m.key.replace(LINES, COPY)
        model.measures[key] = m.model_copy(update={"key": key, "table": COPY, "hidden": True,
                                                   "slug": f"{m.slug}_2"})
    net = next(m for m in model.measures.values() if m.slug == "net_amount")
    done = read(model, f"[{net.business_name}] - SUM([{field_ref(model, f'{LINES}.discount_amount')}])")
    assert done.metrics == [net.key]


def test_a_readers_figure_naming_a_hidden_metric_is_refused(model):
    from core2.plan.ir import Plan as _Plan
    from core2.resolve.figures import worked_out
    from core2.resolve.resolver import Context, ResolveError

    net = next(m for m in model.measures.values() if m.slug == "net_amount")
    net.hidden, net.business_name = True, "Kept sales"          # a name no field has: it can only be the metric
    plan = _Plan.model_validate({"kind": "query", "figures": [{"name": "Net twice", "formula":
                                                                f"[{net.business_name}] - SUM([{field_ref(model, f'{LINES}.discount_amount')}])"}]})
    with pytest.raises(ResolveError, match="hidden"):
        worked_out(plan, model, Context(today=dt.date(2026, 6, 15)))


def test_a_field_named_like_a_copys_metric_slug_reads_as_the_field(model):
    # Learn names a copy's metrics by its table ("order_line_net_amount"), which spells "Order line · Net amount".
    from core2.bootstrap.build import assign_slugs

    _with_copy(model)
    for m in [m for m in model.measures.values() if m.table == LINES]:
        key = m.key.replace(LINES, COPY)
        model.measures[key] = m.model_copy(update={"key": key, "table": COPY, "hidden": True})
    assign_slugs(model)
    ref = field_ref(model, f"{LINES}.net_amount")
    with pytest.raises(AuthoringError, match="say how to add it up"):
        read(model, f"[{ref}] - SUM([{field_ref(model, f'{LINES}.discount_amount')}])")
    assert read(model, f"SUM([{ref}])").expr.column == f"{LINES}.net_amount"
