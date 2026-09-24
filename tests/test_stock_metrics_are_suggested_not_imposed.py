"""
A modelled stock warehouse is offered its everyday metrics; none goes live.

A warehouse nobody has modelled has no metrics, so its first readers get raw
columns, and every question about stock on hand, its value or what was sold is
answered from whatever the SQL model makes of ON_HND_QTY, CUR_ON_HND_QTY and
ITM_CST. The knowledge-base build now files the everyday ones -- stock on hand,
inventory value, allocated and available quantity, purchased quantity, units
sold -- as pending proposals in the administrator's metric queue. Each is
chosen by what the columns mean, and says what it measures, why those columns,
and what it is called in English and French. Accepting one is the only way it
changes an answer.

Drives the real builders: the knowledge-base build from a schema directory
(the model provider stubbed), the semantic model, the proposal store, the
metrics page and its accept route, the metric scope resolver a question goes
through, and the snapshot gate. Synthetic tables in a mart's naming
convention; no customer data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

import store
from core.contextual_dates import question_has_snapshot_intent
from core.metric_scope import resolve_metric_scope
from core.metric_validator import validate_metric
from core.semantic_model import build_semantic_model
from core.starter_metrics import GENERATED_BY, propose_starter_metrics, starter_metrics

DAILY = "MART.ITM_BAL_DLY_FCT"
MONTHLY = "MART.ITM_BAL_PRD_FCT"
INVOICES = "MART.CUS_ORD_IVC_FCT"
MOVEMENTS = "MART.STK_MVT_FCT"


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": True, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


# Each fact carries decoys beside the column a metric should read: parts of the
# stock (on inspection, rejected, allocated), an earlier position (last year's),
# averages, a year-to-date figure, and a second, average unit cost.
SCHEMA = {
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY",
        ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ON_HND_QTY", "decimal"),
        ("ISP_ON_HND_QTY", "decimal"), ("RJC_ON_HND_QTY", "decimal"),
        ("ALC_QTY", "decimal"), ("ALC_ON_HND_QTY", "decimal"), ("RSV_QTY", "decimal"),
        ("LST_YR_ON_HND_QTY", "decimal"), ("ITM_CST", "decimal"), ("AVG_ITM_CST", "decimal"),
    ),
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("PRD_DMS_KEY", "int"), ("CUR_ON_HND_QTY", "decimal"), ("AVG_ON_HND_QTY", "decimal"),
        ("CUR_ALC_ON_HND_QTY", "decimal"), ("AVG_ALC_ON_HND_QTY", "decimal"),
        ("PCH_QTY", "decimal"), ("SLD_QTY", "decimal"), ("YTD_SLD_QTY", "decimal"),
        ("ITM_CST", "decimal"), ("AVG_ITM_CST", "decimal"),
    ),
    # Neither of these holds stock levels, whatever columns they carry.
    "WH.MART.CUS_ORD_IVC_FCT": _table(
        "CUS_ORD_IVC_FCT_KEY",
        ("CUS_ORD_IVC_FCT_KEY", "bigint"), ("CUS_DMS_KEY", "int"),
        ("CUS_IVC_DT_DMS_KEY", "int"), ("IVC_QTY", "decimal"), ("SLD_QTY", "decimal"),
        ("SOP_CUS_IVC_LIN_AMT", "decimal"),
    ),
    "WH.MART.STK_MVT_FCT": _table(
        "STK_MVT_FCT_KEY",
        ("STK_MVT_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("MVT_DT_DMS_KEY", "int"),
        ("PCH_QTY", "decimal"), ("ON_HND_QTY", "decimal"),
    ),
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "varchar"), ("ITM_DSC", "varchar"),
    ),
    "WH.MART.WHS_DMS": _table(
        "WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "varchar"), ("WHS_DSC", "varchar"),
    ),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int"),
    ),
}

EXPECTED = {
    "Stock on hand": (DAILY, "SUM(ON_HND_QTY)"),
    "Inventory value": (DAILY, "SUM(ON_HND_QTY * ITM_CST)"),
    "Allocated quantity": (DAILY, "SUM(ALC_ON_HND_QTY)"),
    "Available quantity": (DAILY, "SUM(ON_HND_QTY) - COALESCE(SUM(ALC_ON_HND_QTY), 0)"),
    "Month-end stock on hand": (MONTHLY, "SUM(CUR_ON_HND_QTY)"),
    "Month-end inventory value": (MONTHLY, "SUM(CUR_ON_HND_QTY * ITM_CST)"),
    "Month-end allocated quantity": (MONTHLY, "SUM(CUR_ALC_ON_HND_QTY)"),
    "Month-end available quantity": (
        MONTHLY, "SUM(CUR_ON_HND_QTY) - COALESCE(SUM(CUR_ALC_ON_HND_QTY), 0)"),
    "Purchased quantity": (MONTHLY, "SUM(PCH_QTY)"),
    "Units sold": (MONTHLY, "SUM(SLD_QTY)"),
}
MOVEMENT_NAMES = {"Purchased quantity", "Units sold"}


def _ops_table(own_key: str, *columns: tuple[str, str]) -> dict:
    return dict(_table(own_key, *columns), schema="OPS")


# Another mart's spelling, where every decoy is SHORTER than the column the
# metric should read -- so only the rule that excludes it keeps it out.
SNAPSHOT_B = "OPS.INV_SNAP_FCT"
SCHEMA_B = {
    "WH.OPS.INV_SNAP_FCT": _ops_table(
        "INV_SNAP_FCT_KEY",
        ("INV_SNAP_FCT_KEY", "bigint"), ("ITEM_KEY", "int"), ("SITE_KEY", "int"),
        ("SNAP_DATE_KEY", "int"),
        ("ON_HAND_QTY", "decimal"),
        ("RSV_OH_QTY", "decimal"),      # a part of the stock
        ("LY_OH_QTY", "decimal"),       # last year's position
        ("ALLOC_QTY", "decimal"), ("ALLOC_OH_QTY", "decimal"),
        ("AVG_CST", "decimal"),         # an average unit cost
        ("STD_UNIT_CST", "decimal"),
        ("PURCHASED_QTY", "decimal"),
        ("PY_PCH_QTY", "decimal"),      # prior year's purchases
        ("SOLD_QTY", "decimal"),
    ),
    "WH.OPS.ITEM_DIM": _ops_table("ITEM_KEY", ("ITEM_KEY", "int"), ("ITEM_CODE", "varchar")),
}
EXPECTED_B = {
    "Stock on hand": (SNAPSHOT_B, "SUM(ON_HAND_QTY)"),
    "Inventory value": (SNAPSHOT_B, "SUM(ON_HAND_QTY * STD_UNIT_CST)"),
    "Allocated quantity": (SNAPSHOT_B, "SUM(ALLOC_OH_QTY)"),
    "Available quantity": (SNAPSHOT_B, "SUM(ON_HAND_QTY) - COALESCE(SUM(ALLOC_OH_QTY), 0)"),
    "Purchased quantity": (SNAPSHOT_B, "SUM(PURCHASED_QTY)"),
    "Units sold": (SNAPSHOT_B, "SUM(SOLD_QTY)"),
}


def _build(tmp_path_factory, schema: dict) -> dict:
    directory = tmp_path_factory.mktemp("starter_model")
    (directory / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return build_semantic_model(str(directory))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return _build(tmp_path_factory, SCHEMA)


@pytest.fixture
def account():
    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _by_name(model) -> dict:
    return {metric.name: metric for metric in starter_metrics(model)}


def _pending(account_id) -> dict:
    return {
        proposal["payload"]["name"]: proposal
        for proposal in store.list_metric_proposals(account_id, status="pending")
    }


class TestTheStarterSetIsChosenByMeaning:

    def test_each_metric_reads_the_column_that_means_it(self, model):
        """Nothing from the invoice fact or the table of movements, no part of
        the stock read as the whole, no average, no earlier position."""
        found = {m.name: (m.base_table, m.sql_template) for m in starter_metrics(model)}
        assert found == EXPECTED

    def test_a_shorter_decoy_never_wins(self, tmp_path_factory):
        """Another spelling. Left to the shortest name, RSV_OH_QTY or LY_OH_QTY
        would be the stock, AVG_CST its cost, ALLOC_QTY the allocation of the
        stock on hand and PY_PCH_QTY this period's purchases."""
        found = {
            m.name: (m.base_table, m.sql_template)
            for m in starter_metrics(_build(tmp_path_factory, SCHEMA_B))
        }
        assert found == EXPECTED_B

    def test_the_end_of_period_position_is_the_level(self, tmp_path_factory):
        schema = {"WH.OPS.SITE_BAL_FCT": _ops_table(
            "SITE_BAL_FCT_KEY", ("SITE_BAL_FCT_KEY", "bigint"), ("SITE_KEY", "int"),
            ("BAL_DATE_KEY", "int"), ("OH_QTY", "decimal"), ("EOP_OH_QTY", "decimal"),
        )}
        metrics = {m.name: m for m in starter_metrics(_build(tmp_path_factory, schema))}
        assert metrics["Stock on hand"].sql_template == "SUM(EOP_OH_QTY)"
        assert "OH_QTY" in " ".join(metrics["Stock on hand"].evidence)

    def test_the_evidence_names_what_it_passed_over(self, model):
        metrics = _by_name(model)
        assert "AVG_ITM_CST" in " ".join(metrics["Inventory value"].evidence)
        assert "ALC_QTY" in " ".join(metrics["Allocated quantity"].evidence)

    def test_available_says_what_it_does_not_subtract(self, model):
        description = _by_name(model)["Available quantity"].description
        assert "ON_HND_QTY less ALC_ON_HND_QTY" in description
        assert "inspection or rejected is not subtracted" in description


class TestNoTwoMetricsShareANameOrASynonym:

    def test_across_the_set(self, model):
        metrics = starter_metrics(model)
        names = [metric.name.casefold() for metric in metrics]
        synonyms = [s.casefold() for metric in metrics for s in metric.synonyms]
        assert len(names) == len(set(names))
        assert len(synonyms) == len(set(synonyms))

    def test_a_second_daily_snapshot_is_named_for_its_table(self, tmp_path_factory):
        weekly = dict(SCHEMA)
        weekly["WH.MART.ITM_BAL_WKL_FCT"] = dict(
            SCHEMA["WH.MART.ITM_BAL_DLY_FCT"], pk_columns=["ITM_BAL_WKL_FCT_KEY"],
        )
        metrics = [m for m in starter_metrics(_build(tmp_path_factory, weekly))
                   if m.key == "on_hand" and "Month-end" not in m.name]
        assert len(metrics) == 2
        plain = [m for m in metrics if m.name == "Stock on hand"]
        renamed = [m for m in metrics if m.name != "Stock on hand"]
        assert len(plain) == 1 and plain[0].synonyms
        assert renamed[0].name == f"Stock on hand ({renamed[0].base_table.split('.')[-1]})"
        assert renamed[0].synonyms == ()

    def test_three_schemas_holding_the_same_table(self, tmp_path_factory):
        """The second is named for its table; the third, for its schema too."""
        triplet = dict(SCHEMA_B)
        for schema in ("EAST", "WEST"):
            triplet[f"WH.{schema}.INV_SNAP_FCT"] = dict(
                SCHEMA_B["WH.OPS.INV_SNAP_FCT"], schema=schema,
            )
        metrics = starter_metrics(_build(tmp_path_factory, triplet))
        on_hand = [m for m in metrics if m.key == "on_hand"]
        assert len(on_hand) == 3
        assert len({m.name.casefold() for m in on_hand}) == 3
        assert sum(m.name == f"Stock on hand ({m.base_table})" for m in on_hand) == 1


class TestTheyMeanWhatTheySay:

    def test_a_level_reads_one_snapshot_and_a_movement_adds_up(self, model):
        """The gate _handle_query_impl calls with the matched metric."""
        for metric in starter_metrics(model):
            reads_one = question_has_snapshot_intent(
                "figures by warehouse", matched_metrics=[metric.as_metric()],
            )
            assert reads_one is (metric.name not in MOVEMENT_NAMES), metric.name

    @pytest.mark.parametrize("db_type", ["azure_sql", "snowflake", "oracle"])
    def test_every_definition_validates_in_every_dialect(self, model, db_type):
        columns = {
            table["qualified_name"]: [field["column"] for field in table["fields"]]
            for table in model["tables"]
        }
        for metric in starter_metrics(model):
            verdict = validate_metric(metric.as_metric(), db_type=db_type, schema_columns=columns)
            assert verdict.valid, (metric.name, verdict.errors)

    @pytest.mark.parametrize("question, expected", [
        ("stock on hand by warehouse", "Stock on hand"),
        ("what is the inventory value by warehouse", "Inventory value"),
        ("month-end stock by warehouse", "Month-end stock on hand"),
        ("units sold by item", "Units sold"),
        ("quelle est la quantité en stock par entrepôt", "Stock on hand"),
        ("valeur du stock par entrepôt", "Inventory value"),
        ("stock disponible par article", "Available quantity"),
        ("quantité vendue par article", "Units sold"),
        ("stock en fin de mois par entrepôt", "Month-end stock on hand"),
    ])
    def test_a_question_in_either_language_finds_it(self, model, question, expected):
        rows = [
            dict(metric.as_metric(), id=index, is_active=1)
            for index, metric in enumerate(starter_metrics(model), 1)
        ]
        scope = resolve_metric_scope(rows, question, None, reader_question=question)
        assert [metric["name"] for metric in scope.metrics][:1] == [expected]


class TestNothingGoesLive:

    def test_each_is_a_pending_proposal_and_none_is_a_metric(self, model, account):
        created = propose_starter_metrics(account, model)
        assert len(created) == len(EXPECTED)
        pending = _pending(account)
        assert set(pending) == set(EXPECTED)
        assert {proposal["generated_by"] for proposal in pending.values()} == {GENERATED_BY}
        assert pending["Stock on hand"]["payload"]["base_table"] == DAILY
        assert "quantité en stock" in pending["Stock on hand"]["payload"]["synonyms"]
        assert store.list_metrics(account, active_only=False) == []

    def test_a_rebuild_files_nothing_new(self, model, account):
        propose_starter_metrics(account, model)
        assert propose_starter_metrics(account, model) == []
        assert len(store.list_metric_proposals(account)) == len(EXPECTED)

    def test_a_rejected_suggestion_is_not_made_again(self, model, account):
        propose_starter_metrics(account, model)
        rejected = _pending(account)["Available quantity"]
        assert store.review_metric_proposal(account, rejected["id"], "rejected")
        assert propose_starter_metrics(account, model) == []
        assert "Available quantity" not in _pending(account)

    def test_a_metric_that_already_exists_is_not_suggested(self, model, account):
        store.save_metric(account, {
            "name": "Stock on hand", "sql_template": "SUM(QTY)",
            "formula_type": "expression", "base_table": DAILY,
        }, db_type="azure_sql")
        propose_starter_metrics(account, model)
        assert set(_pending(account)) == set(EXPECTED) - {"Stock on hand"}

    def test_a_definition_the_validator_refuses_is_not_filed(self, model, account, caplog):
        columns = {
            table["qualified_name"]: [
                field["column"] for field in table["fields"] if field["column"] != "ITM_CST"
            ]
            for table in model["tables"]
        }
        with caplog.at_level(logging.WARNING, logger="querybot.starter_metrics"):
            propose_starter_metrics(account, model, schema_columns=columns)
        pending = set(_pending(account))
        assert "Inventory value" not in pending and "Month-end inventory value" not in pending
        assert "Stock on hand" in pending
        assert "'Inventory value' not proposed" in caplog.text


def _schema_md(fqn: str, table: dict) -> str:
    """A schema document in the shape core/schema.py writes one."""
    rows = "\n".join(
        f"| `{column['name']}` | {column['type']} | Yes |  |" for column in table["columns"]
    )
    return (
        f"# {fqn}\n\n**Type:** BASE TABLE  **Schema:** MART\n\n"
        f"**SQL table name:** `{fqn}`\n\n**Row count:** 1000  **Scale:** Small\n\n"
        f"## Columns\n\n| Column | Type | Nullable | Distinct Values |\n"
        f"|--------|------|:--------:|-----------------|\n{rows}\n"
    )


class TestTheKnowledgeBaseBuildFilesThem:

    def test_a_build_leaves_them_in_the_queue(self, account, tmp_path):
        """From the build's own write to the administrator's queue, with nothing
        handed across by the test."""
        schema_dir, kb_dir, clients = tmp_path / "schema", tmp_path / "kb", tmp_path / "clients"
        for directory in (schema_dir, kb_dir, clients):
            directory.mkdir()
        (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
        for fqn, table in SCHEMA.items():
            bare = fqn.split(".")[-1]
            (schema_dir / f"{bare}.md").write_text(
                _schema_md(fqn.split(".", 1)[1], table), encoding="utf-8")

        async def fake(system, user, provider, model, api_key, max_tokens=1024, **kw):
            return "## Overview\nfinished\n", 100, 200

        from core import knowledge, vocab_packs

        with patch("core.llm.llm_complete", side_effect=fake), \
                patch.object(knowledge, "_embed_kb_files_qdrant", return_value=None), \
                patch.object(vocab_packs, "_CLIENTS_DIR", clients):
            asyncio.run(knowledge.build_kb(
                schema_dir=str(schema_dir), kb_dir=str(kb_dir), chroma_dir=account,
                business_desc="a building-supplies distributor", provider="azure_openai",
                model="deployment", api_key="key",
                extra_kwargs={"azure_endpoint": "https://example.openai.azure.com",
                              "azure_api_version": "2024-02-01"},
                account_id=account,
            ))
        pending = _pending(account)
        assert set(pending) == set(EXPECTED)
        assert pending["Units sold"]["payload"]["sql_template"] == "SUM(SLD_QTY)"
        assert store.list_metrics(account, active_only=False) == []


def _render_metrics_page(account_id: str) -> str:
    from admin import routes

    request = MagicMock()
    request.query_params = {}
    with patch.object(routes, "_is_auth", return_value=True):
        response = asyncio.run(routes.metrics_page(request, account_id))
    return response.body.decode("utf-8", "replace")


class TestTheAdministratorSeesWhy:

    def test_the_card_says_where_it_came_from_what_it_measures_and_why(self, model, account):
        propose_starter_metrics(account, model)
        html = _render_metrics_page(account)
        assert "suggested when the knowledge base was built" in html
        assert "Units in stock at the snapshot (ON_HND_QTY)" in html
        assert "Evidence: ON_HND_QTY: stock on hand by name" in html

    def test_a_request_from_chat_is_not_labelled_a_suggestion(self, account):
        store.create_metric_proposal(
            account, payload={"name": "Revenue Per Customer", "sql_template": "SUM(AMT)"},
            generated_by="portal_chat", source_question="revenue per active customer",
        )
        html = _render_metrics_page(account)
        assert "Revenue Per Customer" in html
        assert "suggested when the knowledge base was built" not in html


class TestTheInboxSaysWhatIsWaiting:

    def test_suggestions_are_counted_and_not_described_as_chat_requests(self, model, account):
        from admin.inbox import build_inbox

        propose_starter_metrics(account, model)
        client = {"account_id": account, "client_name": "Test Ltd",
                  "state": "READY", "db_config_id": "db-1"}
        item = next(
            item for item in build_inbox([client], {"db-1"})
            if item["kind"] == "metric-proposal"
        )
        assert item["total"] == len(EXPECTED)
        assert "suggested by the knowledge-base build" in item["detail"]
        assert "users composed these in chat" not in item["detail"]


class TestAcceptingOneMakesItLive:

    def test_accepted_stock_on_hand_is_a_metric_the_snapshot_gate_reads_as_a_level(
        self, model, account,
    ):
        from admin import routes

        propose_starter_metrics(account, model)
        proposal = _pending(account)["Stock on hand"]
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval"):
            response = asyncio.run(routes.metric_proposal_accept(
                MagicMock(), account, proposal["id"],
            ))
        assert response.status_code == 200, response.body
        live = {metric["name"]: metric for metric in store.list_metrics(account)}
        assert set(live) == {"Stock on hand"}
        assert (live["Stock on hand"]["sql_template"], live["Stock on hand"]["base_table"]) == (
            "SUM(ON_HND_QTY)", DAILY)
        assert question_has_snapshot_intent(
            "figures by warehouse", matched_metrics=[live["Stock on hand"]],
        ) is True
