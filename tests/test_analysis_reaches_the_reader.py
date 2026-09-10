"""
tests/test_analysis_reaches_the_reader.py

"If it can build a result out of a question it should provide the summary
analysis of its result."

It does, and always did -- but by arithmetic. The answer card's summary is
computed by core/response_builder.py from the data brief; the model-written
narrative is a separate, second message, and the pipeline gated it on
is_causal_question. That is a strict subset of is_insight_question, so:

    "why did revenue drop"      -> narrative (drill-down included)
    "analyse revenue by region" -> nothing. A table, statistics, no analyst.
    "what stands out here"      -> nothing.
    "interpret this split"      -> nothing.

The reader who asked, in as many words, to be told what the result means was
the one reader who did not get an answer to that.

The narrow gate was right about cost -- an LLM round trip on every "revenue by
region" buys nothing over the deterministic summary -- so the fix is a middle
row rather than an open gate: an explicit request for analysis gets ONE call
that interprets the rows already on screen, and issues no further SQL. Only
the causal action drills.

Both halves are executed here: the real decision function over real question
wording, and the real _send_why_insight with core.llm.llm_complete replaced at
the boundary, asserting on what the model was asked and what the reader was
sent.
"""

import asyncio

import pytest

from core.insight import analysis_action_for


class TestWhichQuestionsEarnAnAnalyst:

    @pytest.mark.parametrize("question", [
        "why did revenue drop last quarter?",
        "what drove the increase in North?",
        "what is the root cause of the shortfall?",
        "what changed between March and April?",
    ])
    def test_a_causal_question_gets_the_causal_treatment(self, question):
        assert analysis_action_for(question) == "why"

    @pytest.mark.parametrize("question", [
        "analyse revenue by region",
        "explain the revenue split",
        "give me insight into the margin by warehouse",
    ])
    def test_an_explicit_request_for_analysis_now_gets_one(self, question):
        """Every one of these returned nothing at all."""
        assert analysis_action_for(question) == "analyze"

    @pytest.mark.parametrize("question", [
        "revenue by region",
        "top 10 customers by margin",
        "sales last month",
        "how many open orders are there",
    ])
    def test_a_plain_retrieval_still_costs_nothing_extra(self, question):
        """The half of the old gate that was right. A round trip here buys
        nothing the deterministic summary does not already say."""
        assert analysis_action_for(question) == ""

    def test_a_clarification_reply_is_not_a_new_question(self):
        assert analysis_action_for("why did revenue drop?",
                                   is_clarification=True) == ""

    def test_nothing_at_all_is_not_a_question(self):
        assert analysis_action_for("") == ""


class _Adapter:
    def __init__(self):
        self.analyses = []
        self.messages = []

    async def send_analysis_response(self, event, insight):
        self.analyses.append(insight)

    async def send_message(self, event, text):
        self.messages.append(text)


ROWS = [
    {"REGION": "North", "NET_AMOUNT": 620.0},
    {"REGION": "South", "NET_AMOUNT": 240.0},
    {"REGION": "East", "NET_AMOUNT": 110.0},
]


def _run(action, monkeypatch, *, rows=None, provider_fails=False):
    """Execute the real _send_why_insight; report what the model was asked.

    `provider_fails` makes the model raise. It is a parameter rather than
    something the caller patches beforehand, because this function installs
    its own llm_complete -- a stub set up outside it was simply overwritten,
    so the failure test ran the HAPPY path and passed for the wrong reason.
    """
    import core.llm
    import core.query_pipeline as qp

    seen = {"prompts": [], "drilldown_queries": []}

    NARRATIVE = (
        "HEADLINE: North leads on revenue.\n"
        "BODY: North contributes 620 of the 970 total across 3 regions.\n"
        "DETAIL:\n- North 620\n- South 240\n- East 110\n"
        "NEXT: Break North down by product.\n"
    )
    DRILLDOWN_SQL = (
        "SELECT REGION, SUM(NET_AMOUNT) AS NET_AMOUNT "
        "FROM DBO.F_SALES GROUP BY REGION"
    )

    async def _fake_complete(system="", user="", *args, **kwargs):
        seen["prompts"].append(f"{system}\n{user}")
        if provider_fails:
            raise RuntimeError("provider down")
        # The drill-down planner is the one prompt that asks for SQL; it is
        # the only one that mentions its own refusal token. Everything else
        # wants the narrative. llm_complete returns
        # (text, prompt_tokens, completion_tokens).
        body = DRILLDOWN_SQL if "NO_DRILLDOWN" in str(system) else NARRATIVE
        return body, 100, 40

    monkeypatch.setattr(core.llm, "llm_complete", _fake_complete)
    monkeypatch.setattr(
        qp, "resolve_provider", lambda *a, **k: ("test", "test-model", "key", {})
    )
    # An account with no compliance profile fails closed, which is correct and
    # has its own tests. These are about what the analyst does once it is
    # allowed to run, so the posture is set at the boundary rather than by
    # seeding a tenant -- and the refusal is still asserted, below.
    import core.compliance.policy_engine as pe
    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)

    def _executor(_cfg, sql):
        seen["drilldown_queries"].append(sql)
        raise RuntimeError("no warehouse in this test")

    adapter = _Adapter()
    asyncio.run(qp._send_why_insight(
        adapter, object(),
        action=action,
        question="analyse revenue by region",
        rows=list(ROWS if rows is None else rows),
        sql="SELECT REGION, SUM(NET_AMOUNT) FROM DBO.F_SALES GROUP BY REGION",
        client={},
        account_id=f"acct_{action}",
        db_cfg={"db_type": "azure_sql"},
        rag_context="Revenue is net of returns.",
        known_tables={"DBO.F_SALES"},
        query_executor=_executor,
        question_id="q1",
    ))
    return adapter, seen


class TestTheAnalystActuallyWrites:

    def test_an_analyze_run_reaches_the_reader(self, monkeypatch):
        adapter, _seen = _run("analyze", monkeypatch)
        assert adapter.analyses, "no analysis was sent to the reader"
        sent = adapter.analyses[0]
        # The action decides the card the browser renders and the title on it,
        # so an "analyze" request that arrives labelled "why" is the wrong
        # answer even when the prose is right.
        assert sent["action"] == "analyze", sent
        assert sent["title"] != _run("why", monkeypatch)[0].analyses[0]["title"]
        assert "North" in str(sent)

    def test_an_analyze_run_is_written_from_the_real_numbers(self, monkeypatch):
        """The other half of this change: the contract now carries every
        statistic the result has, so the prompt is not a question with no
        answer in it."""
        _adapter, seen = _run("analyze", monkeypatch)
        prompt = "\n".join(seen["prompts"])
        assert "620" in prompt and "North" in prompt
        assert "South" in prompt, prompt

    def test_an_analyze_run_never_queries_the_warehouse(self, monkeypatch):
        """It interprets rows that are already on the reader's screen. A
        second round of SQL for that is latency the reader pays twice, and
        the whole reason an explicit "analyse this" can be affordable.

        The executor IS passed down -- generate_analysis_response routes the
        drill-down on the action, not on whether it has one -- so this is the
        assertion that keeps that routing honest.
        """
        _adapter, seen = _run("analyze", monkeypatch)
        assert seen["drilldown_queries"] == []

    def test_the_causal_run_still_drills(self, monkeypatch):
        _adapter, seen = _run("why", monkeypatch)
        assert seen["drilldown_queries"], (
            "the causal action stopped drilling down"
        )

    def test_a_failure_falls_back_instead_of_erroring(self, monkeypatch):
        """The factual answer is already on the wire when this runs, so the
        reader must never see a stack trace or an error bubble for it. What
        they get instead is the locally-computed analysis."""
        adapter, seen = _run("analyze", monkeypatch, provider_fails=True)
        assert seen["prompts"], "the model was never called, so nothing failed"
        assert adapter.messages == [], adapter.messages
        assert len(adapter.analyses) == 1
        assert adapter.analyses[0]["action"] == "analyze"

    def test_the_fallback_card_says_the_analysis_did_not_complete(
            self, monkeypatch):
        """A locally-computed stand-in that reads like a real narrative would
        be worse than the failure it replaces."""
        adapter, _seen = _run("analyze", monkeypatch, provider_fails=True)
        card = adapter.analyses[0]
        assert card.get("headline") or card.get("body"), card
        assert "could not" in f"{card.get('headline')} {card.get('body')}".lower()


class TestARegulatedTenantIsStillRefused:

    def test_the_refusal_names_the_action_that_was_blocked(self, monkeypatch):
        """The audit row said "why-insight blocked" whatever ran. An audit
        trail that cannot tell you which analysis was refused is a record of
        the wrong event."""
        import core.compliance.policy_engine as pe
        import core.llm_audit as audit
        import core.query_pipeline as qp

        recorded = []
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        monkeypatch.setattr(
            audit, "record_llm_blocked",
            lambda component, reason: recorded.append((component, reason)))

        adapter = _Adapter()
        asyncio.run(qp._send_why_insight(
            adapter, object(),
            action="analyze",
            question="analyse revenue by region",
            rows=list(ROWS), sql="SELECT 1",
            client={}, account_id="acct_reg", db_cfg={},
            question_id="q1",
        ))
        assert adapter.analyses == [] and adapter.messages == []
        assert recorded and "analyze-insight blocked" in recorded[0][1]
