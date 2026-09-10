"""
tests/test_why_answer_admits_its_gaps.py

An answer that did no analysis must not look like one that did.

The "why" pipeline is real: it plans breakdown SQL, validates it, executes it
through the governed executor and narrates over the results. What it never did
was admit when none of that happened. Six different failures --

    the validator rejecting the drill SQL
    the warehouse raising
    a breakdown returning no rows
    the model replying NO_DRILLDOWN
    the planner call itself raising
    no source context to plan from at all

-- all produced `drilldown_briefs=None`, a narration prompt byte-for-byte
identical to a "why" with no pipeline behind it, and a card titled "Why this
pattern?" that a reader could not tell apart from one written over three real
breakdowns. Measured: the failed-case prompts hashed identically to each other
AND to the no-pipeline case.

The last of the six is the one a real user hits. adopt_cached_snapshot blanks
rag_context whenever a result is RESTORED rather than answered fresh -- rightly,
since the previous answer's context is not this result's -- and a page reload is
exactly that restore. So "why did this drop?" on the result still on screen ran
zero warehouse queries, silently, only after a reload, which is why it survived
every test. That one is now fixed rather than merely confessed: the context is
retrieved again for the result's own question.

These execute the real pipeline with the model, the validator and the warehouse
replaced at the boundary, and assert on what the reader is handed.
"""

import asyncio
import hashlib

import pytest

TREND = [
    {"MONTH": "2024-01", "NET_AMOUNT": 900.0},
    {"MONTH": "2024-02", "NET_AMOUNT": 820.0},
    {"MONTH": "2024-03", "NET_AMOUNT": 300.0},
]

DRILL_SQL = "SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY REGION"
DRILL_ROWS = [{"REGION": "North", "NET_AMOUNT": 200.0},
              {"REGION": "South", "NET_AMOUNT": 100.0}]

NARRATIVE = (
    "HEADLINE: Revenue fell in March.\n"
    "BODY: The series drops from 900 to 300 over three months.\n"
    "DETAIL:\n- January 900\n- March 300\n"
    "NEXT: Break March down by region.\n"
)


class _Case:
    """One way for the drill-down to come back with nothing."""

    def __init__(self, name, *, plan=DRILL_SQL, validates=True,
                 warehouse="rows", planner_raises=False):
        self.name = name
        self.plan = plan
        self.validates = validates
        self.warehouse = warehouse
        self.planner_raises = planner_raises


CASES = [
    _Case("succeeds"),
    _Case("declined", plan="NO_DRILLDOWN"),
    _Case("rejected", validates=False),
    _Case("query_failed", warehouse="raise"),
    _Case("empty", warehouse="none"),
    _Case("planner_failed", planner_raises=True),
]


def _run(case, monkeypatch):
    """Execute the real generate_drilldown_insight; report the card and the
    narration prompt the model was sent."""
    import core.insight as insight
    import core.llm
    import core.schema

    seen = {"prompts": [], "warehouse": []}

    async def _complete(system="", user="", *args, **kwargs):
        seen["prompts"].append((str(system), str(user)))
        if "NO_DRILLDOWN" in str(system):          # the planner prompt
            if case.planner_raises:
                raise RuntimeError("planner unavailable")
            return case.plan, 1, 1
        return NARRATIVE, 10, 10

    class _Governed:
        def __init__(self, rows, sql):
            self.rows = rows
            self.sql = sql

    def _executor(_cfg, sql):
        seen["warehouse"].append(sql)
        if case.warehouse == "raise":
            raise RuntimeError("warehouse unreachable")
        return _Governed([] if case.warehouse == "none" else list(DRILL_ROWS), sql)

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(
        insight, "validate_sql", lambda *a, **k: (case.validates, "", "ok"),
        raising=False)
    # The validator is imported inside the function, so patch where it lives.
    import core.validator
    monkeypatch.setattr(
        core.validator, "validate_sql",
        lambda *a, **k: (case.validates, "unknown_table", "unknown_table"))
    monkeypatch.setattr(core.schema, "run_query",
                        lambda *a, **k: list(DRILL_ROWS))

    card = asyncio.run(insight.generate_drilldown_insight(
        rows=list(TREND),
        question="revenue by month",
        follow_up="why did revenue drop in March?",
        original_sql="SELECT MONTH, SUM(NET_AMOUNT) FROM DBO.F_SALES GROUP BY MONTH",
        db_cfg={"db_type": "azure_sql", "credentials": {}},
        context="Net amount is revenue after returns.",
        provider="test", model="test", api_key="k",
        known_tables={"DBO.F_SALES"},
        query_executor=_executor,
    ))
    narration = next((u for sy, u in seen["prompts"]
                      if "NO_DRILLDOWN" not in sy), "")
    return card, narration, seen


@pytest.fixture(autouse=True)
def _allowed(monkeypatch):
    import core.compliance.policy_engine as pe
    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)


FAILURES = [c for c in CASES if c.name != "succeeds"]


class TestTheCardSaysWhenThereWasNoBreakdown:

    def test_a_successful_drilldown_carries_no_caveat(self, monkeypatch):
        card, narration, seen = _run(CASES[0], monkeypatch)
        assert seen["warehouse"], "the drill-down never reached the warehouse"
        assert card.get("grounding_caveat") == "", card.get("grounding_caveat")
        assert "Drill-down breakdowns:" in narration

    @pytest.mark.parametrize("case", FAILURES, ids=lambda c: c.name)
    def test_every_failure_says_so_on_the_card(self, case, monkeypatch):
        card, _narration, _seen = _run(case, monkeypatch)
        assert card.get("grounding_caveat"), (
            f"{case.name} produced a card with no caveat at all")
        assert "No supporting breakdown" in card["grounding_caveat"]

    @pytest.mark.parametrize("case", FAILURES, ids=lambda c: c.name)
    def test_every_failure_tells_the_model_too(self, case, monkeypatch):
        """The narration is written by the model. A caveat the reader can see
        under prose that claims a deep analysis is worse than either alone."""
        _card, narration, _seen = _run(case, monkeypatch)
        assert "No supporting breakdowns were available" in narration, narration
        assert "do not imply that a" in narration.lower()

    def test_the_failures_are_told_apart_from_each_other(self, monkeypatch):
        """The measured symptom: every failed case hashed to one prompt. A
        reader, and the model, were told the same thing whether the validator
        refused the query or the warehouse was down."""
        caveats, prompts = {}, {}
        for case in FAILURES:
            card, narration, _seen = _run(case, monkeypatch)
            caveats[case.name] = card["grounding_caveat"]
            prompts[case.name] = hashlib.sha256(
                narration.encode()).hexdigest()[:12]
        assert len(set(caveats.values())) == len(FAILURES), caveats
        assert len(set(prompts.values())) == len(FAILURES), prompts


class TestTheReasonIsTheRightOne:

    @pytest.mark.parametrize("case,expected", [
        (_Case("declined", plan="NO_DRILLDOWN"), "no safe way"),
        (_Case("rejected", validates=False), "SQL validator"),
        (_Case("query_failed", warehouse="raise"), "could not be completed"),
        (_Case("empty", warehouse="none"), "no rows"),
        (_Case("planner_failed", planner_raises=True), "could not be planned"),
    ], ids=["declined", "rejected", "query_failed", "empty", "planner_failed"])
    def test_it_names_what_actually_went_wrong(self, case, expected, monkeypatch):
        card, _narration, _seen = _run(case, monkeypatch)
        assert expected in card["grounding_caveat"], card["grounding_caveat"]


class TestTheCaveatIsInTheReadersLanguage:

    def test_a_french_reader_gets_it_in_french(self, monkeypatch):
        from core import i18n

        token = i18n.activate_language("fr")
        try:
            card, _n, _s = _run(_Case("rejected", validates=False), monkeypatch)
        finally:
            i18n.deactivate_language(token)
        assert "Aucune ventilation" in card["grounding_caveat"], \
            card["grounding_caveat"]

    def test_the_model_is_addressed_in_the_prompts_own_language(self, monkeypatch):
        """The caveat the READER sees is translated; the one the MODEL is given
        is part of the prompt scaffolding, which is English whatever language
        the reply must be in."""
        from core import i18n

        token = i18n.activate_language("fr")
        try:
            _card, narration, _s = _run(
                _Case("rejected", validates=False), monkeypatch)
        finally:
            i18n.deactivate_language(token)
        assert "No supporting breakdowns were available" in narration


# ══════════════════════════════════════════════════════════════════════════════
# The sixth failure: a reload
# ══════════════════════════════════════════════════════════════════════════════

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_why_gaps.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()


def _client_app():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import webhooks

    app = FastAPI()
    app.include_router(webhooks.router)
    return TestClient(app)


def _reader():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    return account_id, user_id


# The workspace's warehouse connection, as get_client_db would return it.
# Patched rather than seeded: store.save_db_config encrypts the credentials
# with the process's Fernet key, and reading them back in a suite that has
# already reset the store module decrypts with a different one -- an
# InvalidToken that has nothing to do with what is under test.
WAREHOUSE = {"id": 1, "db_type": "azure_sql", "credentials": {}}


def _drain(ws, limit=25, timeout=5.0):
    import anyio

    out = []
    for _ in range(limit):
        async def _recv():
            with anyio.fail_after(timeout):
                return await ws._send_rx.receive()

        try:
            message = ws.portal.call(_recv)
        except Exception:
            break
        if message.get("type") != "websocket.send" or "text" not in message:
            break
        frame = json.loads(message["text"])
        out.append(frame)
        if frame.get("type") == "typing" and frame.get("active") is False:
            break
    return out


class _Retriever:
    """The tenant knowledge base, at the boundary."""

    def __init__(self, docs):
        self.docs = docs
        self.asked = []

    def retrieve(self, question, n=7):
        self.asked.append(question)
        return list(self.docs)

    def retrieve_fact_patterns(self, question, n=2):
        return []

    def _is_global(self, _doc):
        return False


def _ask_why_after_a_reload(monkeypatch, *, kb_docs, retrieval_raises=False):
    """Seed a governed result, connect FRESH -- which is exactly a reload --
    and ask why."""
    import core.compliance.governed_query as gq
    import core.llm
    import gateway.webhooks as wh
    import portal.routes as pr
    from core.result_cache import result_cache

    seen = {"warehouse": [], "prompts": []}
    retriever = _Retriever(kb_docs)

    async def _complete(system="", user="", *a, **k):
        seen["prompts"].append((str(system), str(user)))
        if "NO_DRILLDOWN" in str(system):
            return DRILL_SQL, 1, 1
        return NARRATIVE, 10, 10

    class _Governed:
        def __init__(self, rows, sql):
            self.rows = rows
            self.sql = sql

    def _execute(*args, **kwargs):
        sql = kwargs.get("sql") or (args[1] if len(args) > 1 else "")
        seen["warehouse"].append(sql)
        return _Governed(list(DRILL_ROWS), sql)

    client = _client_app()
    account_id, user_id = _reader()
    session_id = f"{account_id}:web_{user_id}:thread:t1"
    result_cache.store(
        session_id, list(TREND), question="revenue by month",
        sql="SELECT MONTH, SUM(NET_AMOUNT) AS NET_AMOUNT FROM DBO.F_SALES GROUP BY MONTH",
        metadata={"account_id": account_id, "user_id": str(user_id),
                  "db_config_id": 0},
    )
    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(wh, "llm_complete", _complete)
    monkeypatch.setattr(wh, "resolve_provider",
                        lambda *a, **k: ("test", "test", "k", {}))
    monkeypatch.setattr(wh, "get_client_db", lambda *a, **k: dict(WAREHOUSE))
    def _load(_account):
        if retrieval_raises:
            raise RuntimeError("qdrant unreachable")
        return retriever

    monkeypatch.setattr(wh, "load_retriever", _load)
    monkeypatch.setattr(gq, "execute_governed_query", _execute)
    monkeypatch.setattr(wh, "validate_sql", lambda *a, **k: (True, "", "ok"),
                        raising=False)
    import core.validator
    monkeypatch.setattr(core.validator, "validate_sql",
                        lambda *a, **k: (True, "", "ok"))
    frames = []
    try:
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        with client.websocket_connect(f"/ws/chat/{account_id}?thread_id=t1") as ws:
            ws.receive_json()
            ws.send_json({"type": "message",
                          "text": "why did revenue drop in March?"})
            frames = _drain(ws)
    finally:
        result_cache.clear(session_id)
    card = next((f for f in frames
                 if f.get("type") == "assistant_analysis"), {})
    return card, seen, retriever


class TestWhyAfterAReload:
    """The failure a real reader actually hits, and the only one of the six
    that is fixed rather than merely confessed."""

    def test_the_context_is_retrieved_again_for_this_result(self, monkeypatch):
        card, seen, retriever = _ask_why_after_a_reload(
            monkeypatch, kb_docs=["Net amount is revenue after returns."])
        assert retriever.asked == ["revenue by month"], retriever.asked

    def test_the_drilldown_actually_runs(self, monkeypatch):
        """Zero warehouse queries, every time, only after a reload."""
        card, seen, _r = _ask_why_after_a_reload(
            monkeypatch, kb_docs=["Net amount is revenue after returns."])
        assert seen["warehouse"], (
            "the drill-down ran no queries: " + repr([f for f in seen["prompts"]])[:200])

    def test_the_card_it_produces_claims_nothing_it_did_not_do(self, monkeypatch):
        card, _seen, _r = _ask_why_after_a_reload(
            monkeypatch, kb_docs=["Net amount is revenue after returns."])
        assert card, "no analysis card came back"
        assert card.get("grounding_caveat") == "", card.get("grounding_caveat")

    def test_an_empty_knowledge_base_is_admitted_rather_than_hidden(
            self, monkeypatch):
        """A tenant with no KB cannot drill. That is fine; pretending it did
        is not."""
        card, seen, _r = _ask_why_after_a_reload(monkeypatch, kb_docs=[])
        assert seen["warehouse"] == []
        assert card.get("grounding_caveat"), card
        assert "source context" in card["grounding_caveat"], \
            card["grounding_caveat"]

    def test_a_retrieval_failure_is_admitted_too(self, monkeypatch):
        """A vector store that is down is not a reason to answer as though it
        were up."""
        card, seen, _r = _ask_why_after_a_reload(
            monkeypatch, kb_docs=["ignored"], retrieval_raises=True)
        assert seen["warehouse"] == [], seen["warehouse"]
        assert card.get("grounding_caveat"), card
        assert "source context" in card["grounding_caveat"], \
            card["grounding_caveat"]
