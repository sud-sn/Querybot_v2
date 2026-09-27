"""
An attribute its entity's own table keeps is aggregated there, over its members.

"What is the average item gross weight?" and "average gross weight by item
type" were answered on the sample tenant, in English and French, with a
question: which of two datasets to read -- the daily stock snapshot or the
monthly balances. An item's gross weight is on the item's own row; neither fact
holds it, and read through one it would have been averaged over the fact's
rows, an item counted once for every row it has there. And had the reader
chosen, the item type was joined to the snapshot rather than to the item.

Where the one measure a question names is an attribute of a member table --
not a fact -- and the question asks its average, minimum or maximum, with no
registered metric named, the source is that table: no question is put to the
reader, the aggregate is compiled over its members with its placeholder rows
left out, its breakdowns are joined from it, and its members are no breakdown
unless the question asks by them.

A synthetic tenant (tests/answer_harness.py) whose items each keep a gross
weight on their own row, and two facts keyed on them.
"""

from __future__ import annotations

import json

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("attributes")) as built:
        yield built


def _weights(keys=None) -> list[float]:
    """The weights of the real items, as recorded: the placeholder item is no
    item, and an item with no weight recorded has none to average."""
    return [weight for key, weight in harness.ITEM_WEIGHTS.items()
            if key and weight is not None and (keys is None or key in keys)]


def _average(values: list[float]) -> float:
    return sum(values) / len(values)


def _by_group(aggregate) -> dict:
    groups: dict = {}
    for key, (_code, _name, group, _unit) in harness.ITEMS.items():
        if key:
            groups.setdefault(harness.GROUPS[group][1], []).append(key)
    return {name: aggregate(_weights(keys)) for name, keys in groups.items() if _weights(keys)}


def _asked(answer: dict) -> list:
    return [body for kind, body in answer["replies"] if kind == "clarify"]


def _one_value(answer: dict) -> float:
    assert answer["model_wrote_sql"] is False and _asked(answer) == []
    assert len(answer["rows"]) == 1 and len(answer["rows"][0]) == 1
    return next(iter(answer["rows"][0].values()))


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,expected", [
        ("What is the average item gross weight?", _average(_weights())),
        ("What is the maximum item gross weight?", max(_weights())),
        ("What is the minimum item gross weight?", min(_weights())),
    ])
    def test_over_the_items(self, warehouse, question, expected):
        assert _one_value(harness.ask(warehouse, question)) == pytest.approx(expected)

    def test_by_item_group(self, warehouse):
        answer = harness.ask(warehouse, "Average gross weight by item group")
        assert answer["model_wrote_sql"] is False and _asked(answer) == []
        assert {row["ITEM_GROUP"]: row["AVERAGE_GROSS_WEIGHT"] for row in answer["rows"]} == pytest.approx(
            _by_group(_average))
        # Joined from the item, whose group it is.
        assert "fact_rows.[ITM_GRP_DMS_KEY]" in answer["sql"] and "[ITM_DMS] AS fact_rows" in answer["sql"]

    def test_by_item(self, warehouse):
        answer = harness.ask(warehouse, "Average gross weight by item")
        weights = {name: harness.ITEM_WEIGHTS[key] for key, (_c, name, _g, _u) in harness.ITEMS.items() if key}
        assert {row["ITEM"]: row["AVERAGE_GROSS_WEIGHT"] for row in answer["rows"]} == weights


class TestInFrenchWithThePack:
    """The distribution pack reads "poids brut" as the gross weight, as it does
    on the sample tenant."""

    @pytest.fixture(autouse=True)
    def packs(self, warehouse):
        import store
        from core.vocab_packs import activate_vocab, deactivate_vocab, forget_account_vocab, vocab_for_account

        before = store.get_client(harness.ACCOUNT).get("erp_packs") or "[]"
        store.update_client_meta(harness.ACCOUNT, erp_packs=json.dumps(["wholesale_distribution"]))
        forget_account_vocab(harness.ACCOUNT)
        token = activate_vocab(vocab_for_account(harness.ACCOUNT))
        try:
            yield
        finally:
            deactivate_vocab(token)
            store.update_client_meta(harness.ACCOUNT, erp_packs=before)
            forget_account_vocab(harness.ACCOUNT)

    def test_the_average_of_the_items_is_one_average(self, warehouse):
        answer = harness.ask(warehouse, "Quel est le poids brut moyen des articles ?", "fr")
        assert _one_value(answer) == pytest.approx(_average(_weights()))

    def test_by_item_group(self, warehouse):
        answer = harness.ask(warehouse, "Poids brut moyen par groupe d'articles", "fr")
        assert _asked(answer) == []
        assert {row["ITEM_GROUP"]: row["AVERAGE_GROSS_WEIGHT"] for row in answer["rows"]} == pytest.approx(
            _by_group(_average))


MODEL = {"tables": [
    {"qualified_name": "MART.ITM_DMS", "type": "dimension"},
    {"qualified_name": "MART.ITM_BAL_DLY_FCT", "type": "fact"},
]}


def _plan(*fields: dict) -> dict:
    return {"fields": [dict(field) for field in fields]}


def _field(table: str, column: str, role: str = "measure", **extra) -> dict:
    return {"term": column.lower(), "table": f"WH.{table}", "column": column, "role": role, **extra}


WEIGHT = _field("MART.ITM_DMS", "ITM_GRS_WT", term="gross weight")


class TestTheRule:

    @pytest.mark.parametrize("question,aggregation", [
        ("average item gross weight", "AVG"),
        ("mean gross weight", "AVG"),
        ("the maximum gross weight", "MAX"),
        ("minimum gross weight", "MIN"),
    ])
    def test_an_attribute_of_a_member_table(self, question, aggregation):
        from core.analytical_request_plan import aggregated_attribute

        found = aggregated_attribute(question, _plan(WEIGHT), MODEL)
        assert (found["aggregation"], found["target_table"], found["target_column"]) == (
            aggregation, "MART.ITM_DMS", "ITM_GRS_WT")

    @pytest.mark.parametrize("question,plan,metrics", [
        ("average stock on hand", _plan(_field("MART.ITM_BAL_DLY_FCT", "ON_HND_QTY")), None),
        ("average item gross weight", _plan(WEIGHT), [{"name": "Gross weight"}]),
        ("item gross weight", _plan(WEIGHT), None),
        ("total gross weight", _plan(WEIGHT), None),
        ("average and maximum gross weight", _plan(WEIGHT), None),
        ("average gross weight", _plan(dict(WEIGHT, enforcement="optional")), None),
        ("average gross weight and net weight", _plan(WEIGHT, _field("MART.ITM_DMS", "ITM_NET_WT")), None),
        ("average gross weight", _plan(_field("MART.UNKNOWN_DMS", "GRS_WT")), None),
    ], ids=["a fact's measure", "a metric named", "no aggregate", "a total", "two aggregates", "optional",
            "two measures", "a table the model does not keep"])
    def test_nothing_else(self, question, plan, metrics):
        from core.analytical_request_plan import aggregated_attribute

        assert aggregated_attribute(question, plan, MODEL, metrics) == {}


class TestWhatIsSetAside:

    def _plan(self, *fields, joins=(), **extra) -> dict:
        return {"fields": [dict(field) for field in fields], "joins": [dict(join) for join in joins],
                "source_scope": {"source_kind": "master", "selected_fact": "MART.ITM_DMS"}, **extra}

    ITEM = _field("MART.ITM_DMS", "ITM_NM", "display_dimension", term="items")
    GROUP = _field("MART.ITM_GRP_DMS", "ITM_GRP_DSC", "display_dimension", term="item group")
    FROM_A_FACT = {"from": "MART.ITM_BAL_DLY_FCT", "to": "MART.ITM_GRP_DMS", "enforcement": "required"}
    FROM_THE_ITEM = {"from": "MART.ITM_DMS", "to": "MART.ITM_GRP_DMS", "enforcement": "required"}

    def _demoted(self, plan: dict, dimensions=()) -> tuple[list, list]:
        from core.analytical_request_plan import demote_beside_an_attribute

        demote_beside_an_attribute(plan, {"dimensions": list(dimensions)})
        return ([field.get("enforcement") for field in plan["fields"]],
                [join.get("enforcement") for join in plan["joins"]])

    def test_the_members_and_a_facts_joins(self):
        plan = self._plan(WEIGHT, self.ITEM, self.GROUP, joins=(self.FROM_A_FACT, self.FROM_THE_ITEM))
        assert self._demoted(plan, ["item group"]) == ([None, "optional", None], ["optional", "required"])
        assert plan["fields"][1]["demotion_reason"] == "the members the attribute is aggregated over"

    def test_the_members_the_question_asks_by(self):
        item = dict(self.ITEM, term="item")
        assert self._demoted(self._plan(WEIGHT, item), ["item"]) == ([None, None], [])

    def test_a_population_is_left_to_its_own_rule(self):
        plan = self._plan(WEIGHT, self.ITEM, joins=(self.FROM_A_FACT,), count_target={"status": "selected"})
        assert self._demoted(plan) == ([None, None], ["required"])

    def test_only_where_the_source_is_the_member_table(self):
        # The same table chosen as a fact's source is read as one.
        plan = self._plan(WEIGHT, self.ITEM, joins=(self.FROM_A_FACT,))
        plan["source_scope"] = {"source_kind": "fact", "selected_fact": "MART.ITM_DMS"}
        assert self._demoted(plan) == ([None, None], ["required"])

    def test_with_no_measure_on_the_table_nothing_is(self):
        plan = self._plan(_field("MART.ITM_BAL_DLY_FCT", "ON_HND_QTY"), self.ITEM, joins=(self.FROM_A_FACT,))
        assert self._demoted(plan) == ([None, None], ["required"])
