"""The accuracy benchmark counts each learned link, metric and date the way its numbers say.

The benchmark is what the 85% target is measured by, so a scorer that counts wrongly would
report progress that is not there. A model learned from a synthetic warehouse is graded, then
spoiled one belief at a time (a link lost, a link invented, a link held for review, a metric
that adds up wrongly, an identifier learned as a metric, the default date moved): each
spoiling moves exactly the count it should, and nothing else.

Synthetic data only.
"""

from __future__ import annotations

import json

import pytest

from core2.model.schema import AggExpr, Evidence, Join, Measure
from evals.core2 import benchmark, domains
from evals.core2.framework import materialize
from evals.core2.learn_eval import _maps


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    return built, benchmark.learn(built)


def _graded(built, model):
    return benchmark.grade(model, built)


def _truth_join(built, model):
    """A learned link the truth holds, and its key."""
    t_of, c_of = _maps(model, built)
    truth = {(j.from_table, j.from_column, j.to_table) for j in built.domain.truth.joins}
    for key, j in model.joins.items():
        ref = (t_of[j.from_table], c_of[j.from_columns[0]].split(".", 1)[1], t_of[j.to_table])
        if ref in truth and len(j.from_columns) == 1 and not j.to_calendar:
            return key, j
    raise AssertionError("no learned link of the truth")


def test_the_learned_model_is_graded_as_learn_eval_grades_it(retail):
    built, model = retail
    joins = _graded(built, model)["joins"]
    assert joins.right == joins.expected == len(built.domain.truth.joins)
    assert joins.flagged == 0 and joins.auto_right == joins.auto


def test_a_lost_link_counts_against_accuracy(retail):
    built, model = retail
    before = _graded(built, model)["joins"]
    spoiled = model.model_copy(deep=True)
    key, j = _truth_join(built, spoiled)
    del spoiled.joins[key]
    after = _graded(built, spoiled)["joins"]
    assert (after.right, after.expected) == (before.right - 1, before.expected)
    assert any(m.endswith("not found") for m in after.misses)


def test_an_invented_link_counts_against_accuracy_and_precision(retail):
    built, model = retail
    before = _graded(built, model)["joins"]
    spoiled = model.model_copy(deep=True)
    t_of, c_of = _maps(spoiled, built)
    key, j = _truth_join(built, spoiled)
    elsewhere = next(k for k, logical in t_of.items()
                     if logical not in (t_of[j.to_table], t_of[j.from_table]) and spoiled.tables[k].primary_key)
    spoiled.joins["invented"] = j.model_copy(update={
        "key": "invented", "to_table": elsewhere, "to_columns": spoiled.tables[elsewhere].primary_key[:1]})
    after = _graded(built, spoiled)["joins"]
    assert (after.right, after.expected) == (before.right, before.expected + 1)
    assert (after.auto_right, after.auto) == (before.auto_right, before.auto + 1)
    assert any(m.startswith("invented") for m in after.misses)


def test_a_link_held_for_review_is_not_used_and_is_counted_as_review(retail):
    built, model = retail
    before = _graded(built, model)["joins"]
    spoiled = model.model_copy(deep=True)
    key, j = _truth_join(built, spoiled)
    j.trust = "proposed"
    j.evidence.append(Evidence(kind="ambiguous", detail="matches two tables equally well"))
    after = _graded(built, spoiled)["joins"]
    assert after.right == before.right - 1
    assert (after.flagged, after.auto) == (before.flagged + 1, before.auto - 1)


def test_a_metric_that_adds_up_wrongly_counts_against_it(retail):
    built, model = retail
    before = _graded(built, model)["metrics"]
    spoiled = model.model_copy(deep=True)
    _, c_of = _maps(spoiled, built)
    sums = {f"{m.table}.{m.column}" for m in built.domain.truth.measures if m.additivity == "additive" and m.column}
    m = next(m for m in spoiled.measures.values() if c_of.get(getattr(m.expr, "column", None) or "") in sums)
    m.additivity, m.time_aggregation = "semi_additive", "last"
    after = _graded(built, spoiled)["metrics"]
    assert (after.right, after.expected) == (before.right - 1, before.expected)
    assert after.auto_right == before.auto_right - 1


def test_an_identifier_learned_as_a_metric_counts_against_it(retail):
    built, model = retail
    before = _graded(built, model)["metrics"]
    spoiled = model.model_copy(deep=True)
    _, c_of = _maps(spoiled, built)
    column = next(k for k, ref in c_of.items() if ref == "order_lines.line_number")
    spoiled.measures["line_numbers"] = Measure(key="line_numbers", table=spoiled.columns[column].table,
                                               expr=AggExpr(agg="sum", column=column))
    after = _graded(built, spoiled)["metrics"]
    assert (after.right, after.expected) == (before.right, before.expected + 1)
    assert any("line_number: learned as a metric" in m for m in after.misses)


def test_a_moved_default_date_counts_against_both_dates(retail):
    built, model = retail
    before = _graded(built, model)["dates"]
    spoiled = model.model_copy(deep=True)
    _, c_of = _maps(spoiled, built)
    roles = {c_of[r.column]: r for r in spoiled.date_roles.values()}
    roles["order_lines.order_date_key"].is_default = False
    roles["order_lines.ship_date_key"].is_default = True
    after = _graded(built, spoiled)["dates"]
    assert (after.right, after.expected) == (before.right - 2, before.expected)


def test_a_two_column_link_is_right_whichever_order_its_columns_are_in():
    built = materialize(domains.build("compounding_pharmacy"), "descriptive")
    model = benchmark.learn(built)
    before = _graded(built, model)["joins"]
    assert any("claims.fill_number+rx_number -> fills: not found" in m for m in before.misses)
    t_of, c_of = _maps(model, built)
    table = {logical: k for k, logical in t_of.items()}
    column = {ref: k for k, ref in c_of.items()}
    model.joins["claim_fill"] = Join(
        key="claim_fill", from_table=table["claims"], to_table=table["fills"],
        from_columns=[column["claims.fill_number"], column["claims.rx_number"]],
        to_columns=[column["fills.fill_number"], column["fills.rx_number"]], trust="verified")
    after = _graded(built, model)["joins"]
    assert after.right == before.right + 1
    assert not any("claims.fill_number+rx_number" in m for m in after.misses)
    # a distinct count of a table's own key is a count of its rows
    assert not any(m.startswith("batches.*") for m in _graded(built, model)["metrics"].misses)


def test_the_report_and_its_json(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert benchmark.main(["--domains", "retail", "--styles", "descriptive,generic", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "retail" in printed and "targets: accuracy >= 85%" in printed
    data = json.loads(out.read_text())
    assert {r["style"] for r in data["results"]} == {"descriptive", "generic"}
    assert set(data["summary"]["overall"]) == set(benchmark.AREAS)
    for area in benchmark.AREAS:
        assert 0.0 <= data["summary"]["overall"][area]["accuracy"] <= 1.0
