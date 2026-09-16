# -*- coding: utf-8 -*-
"""A failed turn was explained in the catalogue's words, never the reader's.

When the product could not answer -- the validator refused the SQL, the
database raised, the query returned no rows -- the reader got a diagnostic
card whose sentences came from the catalogue: correct, translated, and the
same every time, whatever they had asked. Nothing on that path read the
question, so the card could not say "there is no 2023 scrap figure for the
Lyon plant in this data", only "no records matched the filters".

core/situation_phraser.py asks the model to tell the reader that same
situation in their own terms, in their language, and checks the prose before
it is shown: every figure must be one the situation states, every quoted term
and column-like identifier must come from the situation or the question, and
both parts must be present. Anything else -- and any regulated tenant -- gets
the catalogue's sentences, exactly as before.

Every test executes the real phraser with the model replaced at its boundary,
the real formatter, and the real zero-row card builder. The wiring inside
_handle_query_impl, which needs a warehouse to run, is read as a syntax tree.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from core.situation_phraser import (
    build_phrasing_prompt,
    parse_parts,
    phrase_failure,
    situation_text,
    verify_phrasing,
)

QUESTION = "scrap rate by plant for 2023"
RCA = {
    "kind": "validation",
    "headline": "I could not run that question safely.",
    "most_likely_reason": "The question names a field the workspace does not hold.",
    "suggested_next_step": "Try one of these terms instead: scrap quantity, yield percent.",
    "technical_notes": ["Validation: unknown_column"],
}
SITUATION = situation_text(RCA, QUESTION)


class TestTheCheck:

    def test_a_faithful_rewording_passes(self):
        ok, why = verify_phrasing(
            "Your 2023 scrap-rate question names a field this workspace does not hold.",
            "Ask for the scrap quantity or the yield percent instead.", SITUATION)
        assert (ok, why) == (True, "")

    def test_an_invented_figure_is_refused(self):
        ok, why = verify_phrasing(
            "About 2,400 rows were checked and none carried the field.",
            "Ask for the scrap quantity instead.", SITUATION)
        assert not ok and why.startswith("uncomputed_figure"), why

    def test_a_small_number_is_not_a_claim(self):
        ok, _ = verify_phrasing("The top 3 terms below are the closest matches.",
                                "Pick one of the 2 terms suggested.", SITUATION)
        assert ok

    def test_a_term_the_situation_never_named_is_refused(self):
        ok, why = verify_phrasing(
            'The field "REBUT_PCT" does not exist here.', "Ask for the yield percent.", SITUATION)
        assert not ok and why.startswith("unknown_term"), why

    def test_a_column_the_situation_never_named_is_refused(self):
        ok, why = verify_phrasing(
            "NET_SLS_AMT is not available for scrap.", "Ask for the yield percent.", SITUATION)
        assert not ok and why.startswith("unknown_identifier"), why

    def test_a_term_the_question_named_is_allowed(self):
        ok, _ = verify_phrasing(
            'There is no "scrap rate" field for 2023 here.', "Ask for the yield percent.", SITUATION)
        assert ok

    def test_both_parts_are_required(self):
        assert verify_phrasing("Something happened.", "", SITUATION) == (False, "missing_part")
        assert verify_phrasing("", "Do this.", SITUATION) == (False, "missing_part")

    def test_an_essay_is_refused(self):
        assert verify_phrasing("word " * 200, "Do this.", SITUATION) == (False, "too_long")


class TestTheParts:

    def test_the_two_labelled_parts_are_read(self):
        assert parse_parts("REASON: The field is missing.\nNEXT: Ask for yield.") == (
            "The field is missing.", "Ask for yield.")

    def test_labels_are_read_in_any_case_and_with_prose_around(self):
        assert parse_parts("Sure.\nreason: A.\n\nnext: B.\n") == ("A.", "B.")

    def test_a_reply_without_the_labels_is_nothing(self):
        assert parse_parts("The field is missing. Ask for yield.") == ("", "")


@pytest.fixture
def model(monkeypatch):
    """The real phraser, with the model at its boundary and an allowing tenant."""
    import core.compliance.policy_engine as pe
    import core.llm

    seen = {"calls": [], "reply": (
        "REASON: Your 2023 scrap-rate question names a field this workspace does not hold.\n"
        "NEXT: Ask for the scrap quantity or the yield percent instead.")}

    async def _complete(system="", user="", *a, **k):
        seen["calls"].append((str(system), str(user), dict(k)))
        return seen["reply"], 10, 10

    monkeypatch.setattr(core.llm, "llm_complete", _complete)
    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)
    return seen


def _phrase(rca=RCA, *, question=QUESTION, lang=None, client=None, account_id="acct"):
    return asyncio.run(phrase_failure(
        dict(rca, technical_notes=list(rca.get("technical_notes") or [])),
        question=question, account_id=account_id, client=client or {"enable_llm_audit": 0},
        provider="test", model="test", api_key="k", lang=lang, azure_endpoint="https://x",
    ))


class TestThePhraser:

    def test_the_reason_and_next_step_are_the_models_and_the_rest_is_the_systems(self, model):
        out = _phrase()
        assert out["most_likely_reason"] == (
            "Your 2023 scrap-rate question names a field this workspace does not hold.")
        assert out["suggested_next_step"] == "Ask for the scrap quantity or the yield percent instead."
        assert out["headline"] == RCA["headline"]
        assert out["kind"] == "validation"
        assert out["phrasing"] == "llm"

    def test_the_note_says_whose_words_they_are(self, model):
        from core.i18n import t
        out = _phrase()
        assert out["technical_notes"][0] == "Validation: unknown_column"
        assert out["technical_notes"][-1] == t("fail.phrased_note", "en")

    def test_the_model_is_told_the_question_and_the_whole_situation(self, model):
        _phrase()
        system, user, kwargs = model["calls"][0]
        assert f"Reader's question: {QUESTION}" in user
        assert RCA["headline"] in user and RCA["most_likely_reason"] in user
        assert RCA["suggested_next_step"] in user
        assert "REASON:" in system and "NEXT:" in system
        assert kwargs.get("azure_endpoint") == "https://x"

    def test_a_french_reader_gets_the_french_rule(self, model):
        from core.i18n import prompt_language_rule
        _phrase(lang="fr")
        system, _user, _k = model["calls"][0]
        assert prompt_language_rule("fr", shape="prose") in system
        assert "stay in English" in system

    def test_an_invented_figure_leaves_the_catalogue_in_place(self, model):
        model["reply"] = ("REASON: 2,400 rows were checked and none had it.\n"
                          "NEXT: Ask for the yield percent.")
        out = _phrase()
        assert out["most_likely_reason"] == RCA["most_likely_reason"]
        assert "phrasing" not in out
        assert out["technical_notes"] == ["Validation: unknown_column"]

    def test_a_reply_without_the_parts_leaves_the_catalogue_in_place(self, model):
        model["reply"] = "The field is missing; ask for yield."
        assert _phrase()["most_likely_reason"] == RCA["most_likely_reason"]

    def test_a_provider_failure_leaves_the_catalogue_in_place(self, monkeypatch, model):
        import core.llm

        async def _boom(*a, **k):
            raise RuntimeError("provider down")

        monkeypatch.setattr(core.llm, "llm_complete", _boom)
        out = _phrase()
        assert out["most_likely_reason"] == RCA["most_likely_reason"]

    def test_a_regulated_tenant_never_reaches_the_model(self, monkeypatch, model):
        import core.compliance.policy_engine as pe
        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        out = _phrase()
        assert out["most_likely_reason"] == RCA["most_likely_reason"]
        assert model["calls"] == []

    def test_the_real_policy_fails_closed_without_a_profile(self, monkeypatch):
        """No policy monkeypatch: an account with no compliance profile is
        treated as regulated, and the model is not asked."""
        import core.llm
        calls = []

        async def _complete(*a, **k):
            calls.append(a)
            return "REASON: x\nNEXT: y", 1, 1

        monkeypatch.setattr(core.llm, "llm_complete", _complete)
        out = _phrase(account_id="acct-without-profile-p3")
        assert out["most_likely_reason"] == RCA["most_likely_reason"]
        assert calls == []

    def test_nothing_to_reword_is_left_alone(self, model):
        assert _phrase({"headline": "x", "most_likely_reason": ""}) == {
            "headline": "x", "most_likely_reason": "", "technical_notes": []}
        assert model["calls"] == []


class TestTheCardCarriesTheWords:

    def test_the_failure_card_renders_the_reworded_parts_under_the_wire_labels(self, model):
        from core.answer_formatter import format_failure_business_response
        text = format_failure_business_response(rca=_phrase(), sql="SELECT 1")
        reason_at = text.index("Most likely reason:")
        next_at = text.index("Suggested next step:")
        assert "Your 2023 scrap-rate question names a field" in text[reason_at:next_at]
        assert "Ask for the scrap quantity or the yield percent instead." in text[next_at:]
        assert "Wording by the model" in text[text.index("Technical details:"):]

    def test_the_zero_row_card_is_the_same_card_in_two_halves(self):
        from core.pipeline_helpers import (
            _build_zero_row_message, build_zero_row_parts, format_zero_row_parts,
        )
        args = ("revenue by region for 2023",
                "SELECT REGION, SUM(NET_AMOUNT) FROM DBO.F_SALES WHERE YR = 2023 GROUP BY REGION",
                None, "ok", 0)
        confidence, rca = build_zero_row_parts(*args, tables_used=["DBO.F_SALES"])
        assert rca.get("most_likely_reason") and rca.get("headline")
        assert isinstance(confidence.get("score"), int)
        whole = _build_zero_row_message(*args, tables_used=["DBO.F_SALES"])
        assert format_zero_row_parts(confidence, rca, args[1]) == whole

    def test_a_reworded_zero_row_card_keeps_its_confidence_and_sql(self, model):
        from core.pipeline_helpers import build_zero_row_parts, format_zero_row_parts
        sql = "SELECT REGION, SUM(NET_AMOUNT) FROM DBO.F_SALES WHERE YR = 2023 GROUP BY REGION"
        confidence, rca = build_zero_row_parts("revenue by region for 2023", sql, None, "ok", 0,
                                               tables_used=["DBO.F_SALES"])
        model["reply"] = ("REASON: Nothing in the data matched revenue by region for 2023.\n"
                          "NEXT: Widen the period or check the region names.")
        reworded = asyncio.run(phrase_failure(
            rca, question="revenue by region for 2023", account_id="acct", client={},
            provider="test", model="test", api_key="k"))
        text = format_zero_row_parts(confidence, reworded, sql)
        assert "Nothing in the data matched revenue by region for 2023." in text
        assert "Confidence:" in text and "SQL tried:" in text and "Kind: empty" in text


class TestThePipelineAsksBeforeItAnswers:
    """_handle_query_impl needs a warehouse and a model to run, so its wiring
    is read as a syntax tree: each failure card it sends is built from an RCA
    that went through phrase_failure first."""

    def _impl(self):
        import core.query_pipeline as qp
        tree = ast.parse(inspect.getsource(qp))
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_query_impl":
                return node
        raise AssertionError("_handle_query_impl not found")

    def test_every_failure_card_is_built_from_a_reworded_rca(self):
        """For each card the function sends, the LAST thing assigned to the
        RCA it renders, in source order, is the awaited rewording -- not the
        translate_failure or build_zero_row_parts that produced it. A site
        that skips the rewording still renders a name some other site
        reworded, which is why the check is per site and in order."""
        impl = self._impl()
        assignments: list[tuple[int, str, bool]] = []
        cards: list[tuple[int, str]] = []
        for node in ast.walk(impl):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                value = node.value
                is_reworded = (isinstance(value, ast.Await) and isinstance(value.value, ast.Call)
                               and getattr(value.value.func, "id", "") == "phrase_failure")
                assignments.append((node.lineno, node.targets[0].id, is_reworded))
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", "")
                if name == "format_failure_business_response":
                    rca_kw = next((k.value for k in node.keywords if k.arg == "rca"), None)
                    cards.append((node.lineno, getattr(rca_kw, "id", "?")))
                elif name == "format_zero_row_parts":
                    cards.append((node.lineno, getattr(node.args[1], "id", "?")
                                  if len(node.args) > 1 else "?"))
        assert len(cards) >= 3, cards
        for line, name in cards:
            before = [a for a in assignments if a[1] == name and a[0] < line]
            assert before, f"line {line}: {name} is never assigned before it is rendered"
            latest = max(before)
            assert latest[2], (
                f"line {line}: the card renders {name}, last assigned at line {latest[0]} "
                "by something other than phrase_failure")

    def test_the_wrapper_the_pipeline_used_to_call_is_no_longer_its_path(self):
        impl = self._impl()
        called = {getattr(n.func, "id", "") for n in ast.walk(impl) if isinstance(n, ast.Call)}
        assert "_build_zero_row_message" not in called
        assert "build_zero_row_parts" in called


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
