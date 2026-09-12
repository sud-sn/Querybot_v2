"""
tests/test_every_validator_refusal_has_a_reader.py

Sixteen validator refusals ended the turn with raw SQL talk and no card.

core/query_pipeline.py's terminal handler chose between two very different
endings, and the condition it chose on was dictionary membership:

    if (last_code or "").lower() in _VALIDATION_REASONS:
        ... translate_failure -> headline, business reason, next step ...
    else:
        await adapter.send_message(event, f"❌ {last_reason}")

The else-branch's own comment says it is for "policy denials and other
non-validator codes [which] already carry a business-written explanation". True
of a PolicyDeniedError, whose reason IS decision.explanation. False of sixteen
codes the validator can emit that were simply absent from the map, whose reason
is a sentence about SQL:

    ❌ Query rejected: the join between EMDW_DMART.CUS_ORD_IVC_FCT and
       EMDW_DMART.CUS_RTN_FCT has no ON condition, which would multiply every
       invoice row by every return row.

No headline, no next step, and in English whatever the reader's language. The
sixteen: cartesian_join, missing_join_condition, graph_join_missing,
graph_join_type_mismatch, join_plan_unresolved, field_plan_join_missing,
bridge_allocation_missing, bridge_allocation_unresolved, multi_fact_not_isolated,
multi_fact_not_aggregated, multi_fact_shared_cte, multi_fact_cte_contract,
multi_fact_missing_subplan, temporal_anchor_ungoverned, observed_period_shape,
select_star.

Several are exactly what an EMCO question hits: a fact joined to another fact,
"sales and returns last quarter" needing each totalled separately, a window that
did not anchor on the approved business date.

Two fixes, because either alone leaves a hole. All sixteen now have a business
reason and, where a reader can act on it, a next step. And the gate no longer
asks whether the code is in a dictionary -- it asks whether the reason is already
prose a reader can read, which only a policy denial is. A seventeenth code gets a
card with the generic reason rather than raw SQL talk.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from core.failure_messages import (
    _VALIDATION_NEXT_STEPS,
    _VALIDATION_REASONS,
    translate_failure,
)
from core.i18n import MESSAGES, activate_language, deactivate_language

ROOT = pathlib.Path(__file__).resolve().parents[1]

# The sixteen this file is named for. Written out rather than derived, so
# deleting one from the map is a failure rather than a smaller test.
PREVIOUSLY_UNREADABLE = [
    "bridge_allocation_missing", "bridge_allocation_unresolved",
    "cartesian_join", "field_plan_join_missing", "graph_join_missing",
    "graph_join_type_mismatch", "join_plan_unresolved",
    "missing_join_condition", "multi_fact_cte_contract",
    "multi_fact_missing_subplan", "multi_fact_not_aggregated",
    "multi_fact_not_isolated", "multi_fact_shared_cte",
    "observed_period_shape", "select_star", "temporal_anchor_ungoverned",
]

RAW_VALIDATOR_SENTENCE = (
    "Query rejected: the join between EMDW_DMART.CUS_ORD_IVC_FCT and "
    "EMDW_DMART.CUS_RTN_FCT has no ON condition, which would multiply every "
    "invoice row by every return row."
)


def emitted_codes() -> set[str]:
    """Every reason code the validation layer can actually produce."""
    found: set[str] = set()
    for name in ("core/validator.py", "core/compliance/sql_guard.py",
                 "core/sql_attempt.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        found |= set(re.findall(r'code\s*=\s*["\']([a-z0-9_]+)["\']', source))
        found |= set(re.findall(
            r'["\']code["\']\s*:\s*["\']([a-z0-9_]+)["\']', source))
    return found


def card(code: str, lang: str) -> dict:
    token = activate_language(lang)
    try:
        return translate_failure(
            kind="validation", code=code, reason=RAW_VALIDATOR_SENTENCE)
    finally:
        deactivate_language(token)


class TestNoValidatorCodeIsWithoutAReason:

    def test_the_scan_finds_the_codes_it_is_named_for(self):
        """If the validator stops spelling codes this way the check below covers
        nothing."""
        codes = emitted_codes()
        assert len(codes) >= 30, sorted(codes)
        assert "raw_fact_to_fact_join" in codes

    def test_every_code_the_validator_emits_has_a_business_reason(self):
        missing = sorted(emitted_codes() - set(_VALIDATION_REASONS))
        assert not missing, (
            "these reach the reader with no business reason:\n  "
            + "\n  ".join(missing))

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    def test_each_of_the_sixteen_is_registered(self, code):
        assert code in _VALIDATION_REASONS, code


class TestTheSixteenReadAsBusinessProse:

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_the_reason_is_specific_not_the_generic_fallback(self, code, lang):
        generic = MESSAGES["fail.v.default.reason"][lang]
        assert card(code, lang)["most_likely_reason"] != generic, code

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_the_reason_is_not_the_raw_validator_sentence(self, code, lang):
        reason = card(code, lang)["most_likely_reason"]
        assert "EMDW_DMART" not in reason, code
        assert "ON condition" not in reason, code

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    def test_the_french_reason_is_french(self, code):
        assert card(code, "fr")["most_likely_reason"] != \
            card(code, "en")["most_likely_reason"], code

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_every_card_still_carries_a_next_step(self, code, lang):
        assert card(code, lang)["suggested_next_step"].strip(), code

    @pytest.mark.parametrize("code", PREVIOUSLY_UNREADABLE)
    def test_the_raw_sentence_is_kept_for_the_technical_notes(self, code):
        """It is still the truth about what happened; it just is not the
        headline."""
        notes = " ".join(card(code, "fr")["technical_notes"])
        assert f"Validation: {code}" in notes
        assert "EMDW_DMART" in notes

    @pytest.mark.parametrize("code", sorted(_VALIDATION_NEXT_STEPS))
    def test_every_next_step_key_exists_in_both_languages(self, code):
        key = _VALIDATION_NEXT_STEPS[code]
        assert key in MESSAGES, key
        assert MESSAGES[key].get("en") and MESSAGES[key].get("fr"), key

    @pytest.mark.parametrize("code", sorted(_VALIDATION_REASONS))
    def test_every_reason_key_exists_in_both_languages(self, code):
        key = _VALIDATION_REASONS[code]
        assert key in MESSAGES, key
        assert MESSAGES[key].get("en"), key
        assert MESSAGES[key].get("fr"), key
        assert MESSAGES[key]["fr"] != MESSAGES[key]["en"], key


class TestTheCardIsNoLongerGatedOnADictionary:
    """The terminal handler sits inside a 6,600-line coroutine that needs a
    warehouse and a websocket, so the branch condition is read as a SYNTAX TREE.
    The question is what the `if` tests, which no substring search can answer.
    """

    @staticmethod
    def _terminal_branch() -> ast.If:
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # The branch whose else-arm sends the bare "❌ {last_reason}".
            for inner in ast.walk(ast.Module(body=node.orelse, type_ignores=[])):
                if (isinstance(inner, ast.JoinedStr)
                        and any(isinstance(part, ast.Constant)
                                and "❌" in str(part.value)
                                for part in inner.values)):
                    return node
        raise AssertionError(
            "the terminal raw-reason fallback is no longer an else-branch of an "
            "if — this check covers nothing")

    def test_the_condition_reads_the_policy_flag(self):
        branch = self._terminal_branch()
        assert isinstance(branch.test, ast.UnaryOp)
        assert isinstance(branch.test.op, ast.Not)
        assert isinstance(branch.test.operand, ast.Name)
        assert branch.test.operand.id == "_reason_is_reader_ready"

    def test_it_does_not_read_the_reason_dictionary(self):
        """Dictionary membership was the defect: it made whether a reader got a
        card at all depend on whether someone had written prose for that code."""
        assert "_VALIDATION_REASONS" not in ast.dump(self._terminal_branch().test)

    def test_the_flag_is_only_raised_for_a_policy_denial(self):
        """A validator sentence must never be passed through as if a person had
        written it."""
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        raised_in: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Assign)
                        and any(isinstance(t, ast.Name)
                                and t.id == "_reason_is_reader_ready"
                                for t in inner.targets)):
                    raised_in.append(
                        getattr(node.type, "id", "") or ast.dump(node.type))
        assert raised_in, "the flag is never set — no reason can pass through"
        assert set(raised_in) == {"PolicyDeniedError"}, raised_in

    def test_and_it_starts_false(self):
        """Initialising it True would send every validator sentence straight to
        the reader -- the defect, with the gate inverted."""
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        inside_except = {
            id(inner) for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler)
            for inner in ast.walk(node)
        }
        initialisers = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and id(node) not in inside_except
            and any(isinstance(target, ast.Name)
                    and target.id == "_reason_is_reader_ready"
                    for target in node.targets)
        ]
        assert initialisers, "the flag is never initialised"
        for value in initialisers:
            assert isinstance(value, ast.Constant) and value.value is False, \
                ast.dump(value)
