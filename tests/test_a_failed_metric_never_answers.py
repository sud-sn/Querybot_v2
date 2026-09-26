"""
A metric whose formula failed validation is never used, and never called approved.

save_metric and update_metric run the formula validator on every write and
mark a failure 'draft' -- "validation failed; metric cannot be used", in
core/metric_validator.py's own words. Nothing downstream read that. The
template route ran a draft's stored SQL as the answer; the formula block told
the SQL model it was ADMIN-APPROVED and took ABSOLUTE PRECEDENCE over the
knowledge base; the planner, suggested questions, the workspace guide, the
knowledge-base prompt and scheduled reports all offered or ran it. A draft
also hid any glossary term sharing its name, so the phrase meant nothing at
all. And a metric the reader composed in a thread was printed under the same
ADMIN-APPROVED header, in the prompt and in every trace of it.

These drive the real store against a scratch database (QUERYBOT_DB_PATH):
save a metric that fails validation beside one that passes, then read it back
through each path that turns a metric into an answer, a suggestion or a
report line. The only thing stood in for is the warehouse.
"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import patch

import pytest

TABLE = "MART.SLS_TRX_FCT"
USER = {"id": 1, "role": "admin", "lang": "en"}


@pytest.fixture
def account():
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _save(account_id, name, sql, *, formula_type="query", synonyms=""):
    import store

    metric_id = store.save_metric(account_id, {
        "name": name, "formula_type": formula_type, "sql_template": sql,
        "base_table": TABLE, "synonyms": synonyms, "description": f"{name} as the business reports it",
    })
    return store.get_metric(metric_id)


def _pair(account_id):
    """A metric whose SQL validates, and one whose SQL carries a comment, which
    the validator refuses."""
    good = _save(account_id, "Net Sales", f"SELECT SUM(NET_SLS_AMT) AS net_sales FROM {TABLE}",
                 synonyms="revenue")
    bad = _save(account_id, "Gross Margin",
                f"SELECT SUM(GRS_MRG_AMT) AS margin FROM {TABLE} -- excludes returns",
                synonyms="margin")
    return good, bad


# ── The store says which metrics may answer ──────────────────────────────────

class TestTheStoreSaysWhichMetricsMayAnswer:

    def test_the_validator_marks_the_failed_formula_a_draft(self, account):
        good, bad = _pair(account)
        assert good["metric_status"] == "validated"
        assert bad["metric_status"] == "draft"

    def test_an_answering_read_leaves_the_draft_out_and_the_admin_s_read_keeps_it(self, account):
        import store

        _pair(account)
        assert [m["name"] for m in store.list_metrics(account, answerable_only=True)] == ["Net Sales"]
        assert [m["name"] for m in store.list_metrics(account)] == ["Gross Margin", "Net Sales"]

    def test_an_edit_that_breaks_the_formula_takes_the_metric_out_and_a_fix_puts_it_back(self, account):
        import store

        good, _ = _pair(account)
        store.update_metric(good["id"], {"sql_template": f"SELECT NOT_A_FUNCTION(NET_SLS_AMT) FROM {TABLE}"})
        assert "Net Sales" not in [m["name"] for m in store.list_metrics(account, answerable_only=True)]
        store.update_metric(good["id"], {"sql_template": f"SELECT SUM(NET_SLS_AMT) AS net_sales FROM {TABLE}"})
        assert "Net Sales" in [m["name"] for m in store.list_metrics(account, answerable_only=True)]

    def test_a_retired_metric_does_not_answer(self, account):
        import store

        good, _ = _pair(account)
        store.deprecate_metric(good["id"], account)
        assert store.metric_is_answerable(store.get_metric(good["id"])) is False

    @pytest.mark.parametrize("status,answers", [
        ("validated", True), ("published", True), ("", True), (None, True),
        ("draft", False), ("deprecated", False), ("DRAFT", False),
    ])
    def test_the_rule(self, status, answers):
        import store

        assert store.metric_is_answerable({"is_active": 1, "metric_status": status}) is answers

    def test_an_inactive_metric_does_not_answer_whatever_its_status(self):
        import store

        assert store.metric_is_answerable({"is_active": 0, "metric_status": "validated"}) is False

    def test_a_status_column_added_to_a_table_of_metrics_leaves_them_live(self):
        # Two schema paths add metric_status to an older table: store/db.py's
        # migration and the store's own ensure-schema. Whichever runs first
        # decides the existing rows' status, so both must say "published".
        from store.config_store import _ensure_metric_registry_schema, metric_is_answerable

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE metric_registry (id INTEGER PRIMARY KEY, account_id TEXT, name TEXT, "
                     "synonyms TEXT, sql_template TEXT, is_active INTEGER DEFAULT 1)")
        conn.execute("INSERT INTO metric_registry (account_id, name, synonyms, sql_template) "
                     "VALUES ('a', 'Net Sales', '', 'SELECT 1')")
        _ensure_metric_registry_schema(conn)
        row = dict(conn.execute("SELECT * FROM metric_registry").fetchone())
        assert row["metric_status"] == "published"
        assert metric_is_answerable(row)


# ── The template route ───────────────────────────────────────────────────────

class TestTheTemplateRouteNeverRunsADraft:

    def test_a_question_naming_the_draft_is_planned_instead_of_answered_from_its_sql(self, account):
        import store

        _pair(account)
        assert store.match_metric(account, "what is our gross margin") is None

    def test_a_question_naming_the_validated_metric_still_takes_the_route(self, account):
        import store

        _pair(account)
        assert (store.match_metric(account, "what is our net sales") or {}).get("name") == "Net Sales"


# ── The formula block ────────────────────────────────────────────────────────

class TestTheFormulaBlock:

    def test_the_draft_is_not_a_candidate_from_the_registry(self, account):
        import store

        _pair(account)
        names = [m["name"] for m in store.list_metric_formula_context(account, "gross margin and net sales")]
        assert names == ["Net Sales"]

    def test_the_draft_is_not_a_candidate_from_the_compiled_contract(self, account):
        # The contract snapshots every active metric, drafts included, and the
        # pipeline passes that snapshot in place of the registry read.
        import store

        good, bad = _pair(account)
        snapshot = [dict(good), dict(bad)]
        names = [m["name"] for m in store.list_metric_formula_context(
            account, "gross margin and net sales", metrics=snapshot)]
        assert names == ["Net Sales"]

    def _thread_draft(self):
        return {"name": "Repeat Buyers", "sql_template": "COUNT(DISTINCT CUS_KEY)", "formula_type": "expression",
                "metric_status": "session_draft", "_adhoc": True, "is_active": 1}

    def test_a_metric_from_the_registry_is_printed_as_admin_approved(self):
        from core.pipeline_helpers import _format_metric_formula_context

        block = _format_metric_formula_context([{"name": "Net Sales", "sql_template": "SUM(NET_SLS_AMT)",
                                                 "formula_type": "expression"}])
        assert "ADMIN-APPROVED" in block and "Net Sales" in block
        assert "THIS CONVERSATION" not in block

    def test_a_metric_the_reader_defined_in_the_thread_is_not_printed_as_admin_approved(self):
        from core.pipeline_helpers import _format_metric_formula_context

        block = _format_metric_formula_context([
            {"name": "Net Sales", "sql_template": "SUM(NET_SLS_AMT)", "formula_type": "expression"},
            self._thread_draft(),
        ])
        approved, _, conversation = block.partition("METRICS DEFINED IN THIS CONVERSATION")
        assert "Net Sales" in approved and "Repeat Buyers" not in approved
        assert "Repeat Buyers" in conversation and "NOT" in conversation and "ADMIN-APPROVED" not in conversation
        assert "COUNT(DISTINCT CUS_KEY)" in conversation      # still the formula the model must use

    def test_a_thread_with_only_its_own_metric_gets_no_admin_approved_header(self):
        from core.pipeline_helpers import _format_metric_formula_context

        block = _format_metric_formula_context([self._thread_draft()])
        assert "ADMIN-APPROVED" not in block and "METRICS DEFINED IN THIS CONVERSATION" in block

    def test_the_sql_model_is_told_to_use_both_kinds(self):
        from core.llm import build_sql_system_prompt

        prompt = build_sql_system_prompt("azure_sql", "")
        rule = next(line for line in prompt.splitlines() if "APPROVED METRIC FORMULA RULE" in line)
        assert "METRICS DEFINED IN THIS CONVERSATION" in rule


# ── A glossary term ──────────────────────────────────────────────────────────

class TestADraftHidesNoTerm:

    TERMS = [{"term": "gross margin", "aliases": "margin", "canonical_expression": "SUM(GRS_MRG_AMT)"}]

    def test_a_term_that_shares_only_a_draft_s_name_stays_in_the_prompt(self, account):
        from store.semantic_store import filter_metric_colliding_terms

        _pair(account)
        assert filter_metric_colliding_terms(account, self.TERMS) == self.TERMS

    def test_a_term_that_shares_a_live_metric_s_name_still_gives_way(self, account):
        import store
        from store.semantic_store import filter_metric_colliding_terms

        good, _ = _pair(account)
        store.update_metric(good["id"], {"synonyms": "gross margin"})
        assert filter_metric_colliding_terms(account, self.TERMS) == []


# ── Scheduled reports ────────────────────────────────────────────────────────

class TestAReportDoesNotRunADraft:

    def _report(self, account_id, *metrics):
        from store import report_store

        report = report_store.create_report(account_id, f"Weekly {os.urandom(2).hex()}")
        for order, metric in enumerate(metrics):
            report_store.add_metric_to_report(report["id"], metric["id"], order)
        return report

    def _run(self, account_id, report, lang="en"):
        from core import report_engine

        ran: list[str] = []

        def warehouse(credentials, db_type, sql, **kwargs):
            ran.append(sql)
            raise RuntimeError("the warehouse is not part of this test")

        with patch("core.pipeline_context.get_client_db", return_value={"db_type": "azure_sql", "credentials": {}}), \
                patch("core.compliance.governed_query.execute_governed_query", side_effect=warehouse):
            reply = report_engine.build_report_response(account_id, dict(USER, lang=lang), report, lang=lang)
        return reply, ran

    def test_the_draft_s_sql_never_reaches_the_warehouse_and_its_line_says_why(self, account):
        good, bad = _pair(account)
        reply, ran = self._run(account, self._report(account, good, bad))
        assert all("GRS_MRG_AMT" not in sql for sql in ran)
        assert any("NET_SLS_AMT" in sql for sql in ran)                # the validated one still runs
        line = next(item["text"] for item in reply["items"] if "Gross Margin" in item["text"])
        assert "not computed" in line and "Metrics page" in line

    def test_a_metric_edited_into_a_failing_formula_after_it_joined_the_report_stops_running(self, account):
        import store

        good, _ = _pair(account)
        report = self._report(account, good)
        store.update_metric(good["id"], {"sql_template": f"SELECT NOT_A_FUNCTION(NET_SLS_AMT) FROM {TABLE}"})
        _, ran = self._run(account, report)
        assert ran == []

    def test_a_french_reader_is_told_in_french(self, account):
        good, bad = _pair(account)
        reply, _ = self._run(account, self._report(account, bad), lang="fr")
        assert "non calculé" in reply["items"][0]["text"]

    def test_a_retired_metric_s_line_says_it_was_retired(self):
        from core.report_engine import _format_metric_line, run_metric_for_report

        result = run_metric_for_report("a", USER, {"name": "Old KPI", "is_active": 0, "sql_template": "SELECT 1"})
        assert result == {"ok": False, "reason": "retired", "metric_name": "Old KPI"}
        assert "retired" in _format_metric_line(result, "en")
        assert "retiré" in _format_metric_line(result, "fr")


# ── What readers are offered ─────────────────────────────────────────────────

class TestReadersAreNotOfferedADraft:

    def test_the_workspace_guide_lists_only_the_metric_that_answers(self, account):
        from core.workspace_guide import build_workspace_guide

        _pair(account)
        guide = build_workspace_guide(account, USER)
        assert [m["name"] for m in guide["metrics"]] == ["Net Sales"]

    def test_the_chat_names_only_the_metric_that_answers(self, account):
        from core.conversational import _metric_names

        _pair(account)
        assert _metric_names(account) == ["Net Sales"]

    def test_the_knowledge_base_prompt_calls_only_the_metric_that_answers_approved(self, account):
        # The KB prompt carries formula expressions, so the pair is saved as
        # expressions here; the second has no aggregate, which fails validation.
        from core.knowledge import _format_approved_metrics_for_kb

        good = _save(account, "Net Sales", "SUM(NET_SLS_AMT)", formula_type="expression")
        bad = _save(account, "Gross Margin", "GRS_MRG_AMT * 2", formula_type="expression")
        assert (good["metric_status"], bad["metric_status"]) == ("validated", "draft")
        block = _format_approved_metrics_for_kb(account, TABLE)
        assert "Net Sales" in block and "Gross Margin" not in block


# ── The planner, the analyst, and "did you mean" ─────────────────────────────

class TestThePlannerAndTheAnalystSeeOnlyTheMetricThatAnswers:

    def test_the_planner_is_offered_only_the_metric_that_answers(self, account):
        from core.query_pipeline import planner_metrics_in_scope

        _pair(account)
        assert [m["name"] for m in planner_metrics_in_scope(account, {TABLE})] == ["Net Sales"]

    def test_a_metric_on_a_table_out_of_scope_is_left_out_as_before(self, account):
        from core.query_pipeline import planner_metrics_in_scope

        _pair(account)
        assert planner_metrics_in_scope(account, {"MART.OTHER_FCT"}) == []

    def test_the_analyst_is_told_only_of_the_metric_that_answers(self, account):
        from core.dispatcher import _build_analyst_context

        _pair(account)
        context = _build_analyst_context(account, {})
        assert "Net Sales" in context and "Gross Margin" not in context

    def test_a_dead_end_points_at_the_metric_that_answers_and_not_at_the_draft(self, account):
        from core.failure_messages import suggest_closest_terms

        _pair(account)
        assert not [t for t in suggest_closest_terms("gross margin by week", account) if "margin" in t.lower()]
        assert "net sales" in [t.lower() for t in suggest_closest_terms("net sales by week", account)]

    def test_the_chat_s_fallback_suggestions_name_only_the_metric_that_answers(self, account):
        from portal.routes import _guess_safe_metric_suggestions

        _pair(account)
        assert _guess_safe_metric_suggestions(account) == ["What is our total Net Sales?"]


# ── Report pages ─────────────────────────────────────────────────────────────

class TestTheReportPagesOfferOnlyTheMetricThatAnswers:

    def test_the_admin_report_builder(self, account):
        import asyncio
        from unittest.mock import MagicMock

        import admin.routes as routes

        _pair(account)
        seen: dict = {}

        def _resp(request, name, ctx=None):
            seen.update(ctx or {})
            return MagicMock(status_code=200)

        request = MagicMock()
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_resp", side_effect=_resp):
            asyncio.run(routes.reports_page(request, account))
        assert [m["name"] for m in seen["all_metrics"]] == ["Net Sales"]

    def test_the_reader_s_report_builder(self, account):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import store
        import portal.routes as pr

        _pair(account)
        store.update_client_meta(account, chat_ui_enabled=1)
        user_id, _ = store.create_user(account, "Ada", f"{os.urandom(4).hex()}@x.com",
                                       password="a-password-they-chose", role="admin")
        app = FastAPI()
        app.include_router(pr.router)
        client = TestClient(app)
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        page = client.get("/portal/reports/new").text
        assert "Net Sales" in page and "Gross Margin" not in page


# ── Building a report or a dashboard in chat ─────────────────────────────────

def _reader_socket(account_id, text, *, patch_module, patch_name, monkeypatch):
    """Send one message through the real chat socket, with the planner the
    message reaches replaced at its boundary; return the metrics it was given."""
    import importlib

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import store
    import gateway.webhooks as wh
    import portal.routes as pr

    given: list[list[str]] = []

    async def _planner(text, metrics, complete):
        given.append([m.get("name") for m in metrics or []])
        return None, "stopped here"

    monkeypatch.setattr(importlib.import_module(patch_module), patch_name, _planner)
    monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
    store.update_client_meta(account_id, chat_ui_enabled=1)
    # An admin reader: no table ACL stands between the registry and the planner,
    # so what reaches it is exactly what the store offered.
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com",
                                   password="a-password-they-chose", role="admin")
    app = FastAPI()
    app.include_router(wh.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    with client.websocket_connect(f"/ws/chat/{account_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "message", "text": text})
        for _ in range(20):
            frame = ws.receive_json()
            kind = frame.get("type") if isinstance(frame, dict) else ""
            if kind in ("assistant_error", "clarification_prompt") or (kind == "typing" and frame.get("active") is False):
                break
    return given


class TestBuildingInChatOffersOnlyTheMetricThatAnswers:

    def test_a_report_planned_in_chat(self, account, monkeypatch):
        _pair(account)
        given = _reader_socket(account, "build me a report with net sales and gross margin every Monday",
                               patch_module="core.report_planner", patch_name="parse_report_plan",
                               monkeypatch=monkeypatch)
        assert given == [["Net Sales"]]

    def test_a_dashboard_planned_in_chat(self, account, monkeypatch):
        _pair(account)
        given = _reader_socket(account, "build me a dashboard with net sales, gross margin, and order counts",
                               patch_module="core.dashboard_planner", patch_name="parse_dashboard_plan",
                               monkeypatch=monkeypatch)
        assert given == [["Net Sales"]]


# ── Suggested and example questions ──────────────────────────────────────────

def _workspace_with_a_draft():
    """A small synthetic warehouse, discovered, modelled and compiled the way the
    product does it, with one metric that validates and one that does not."""
    import json
    import tempfile

    import store
    from core.graph_autopopulate import auto_populate_from_schema
    from core.semantic_contract import write_contract
    from core.semantic_model import write_semantic_model

    def table(key, *columns):
        return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
                "pk_columns": [key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"}

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    kb = tempfile.mkdtemp()
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump({"WH.MART.SLS_FCT": table("SLS_KEY", ("SLS_KEY", "bigint"), ("NET_AMT", "decimal(18,2)"),
                                            ("GRS_MRG_AMT", "decimal(18,2)"))}, handle)
    store.update_client_state(account_id, "READY", {"schema_dir": kb, "kb_dir": kb})
    auto_populate_from_schema(account_id, kb)
    write_semantic_model(schema_dir=kb, kb_dir=kb, account_id=account_id)
    for name, template in (("Net sales", "SUM(NET_AMT)"), ("Gross margin", "GRS_MRG_AMT * 2")):
        store.save_metric(account_id, {
            "name": name, "formula_type": "expression", "sql_template": template, "base_table": "MART.SLS_FCT",
            "synonyms": name.lower(), "description": name,
            "example_questions": f"what was the {name.lower()} last month",
        })
    write_contract(account_id, kb)
    return account_id, kb


class TestSuggestedQuestionsNameOnlyTheMetricThatAnswers:

    def test_the_metric_tier_of_the_suggested_questions(self):
        import store
        from core.suggestions import get_suggestions

        account_id, kb = _workspace_with_a_draft()
        assert [m["metric_status"] for m in store.list_metrics(account_id)] == ["draft", "validated"]
        questions = [s["question"] for s in get_suggestions(account_id, kb, None, n=10)]
        assert "What is our total Net sales?" in questions
        assert not [q for q in questions if "margin" in q.lower()]

    def test_the_example_questions_a_greeting_offers(self):
        from core.conversational import _example_questions

        account_id, _ = _workspace_with_a_draft()
        examples = _example_questions(account_id, limit=5)
        assert "what was the net sales last month" in [e.lower().rstrip("?") for e in examples]
        assert not [e for e in examples if "margin" in e.lower()]
