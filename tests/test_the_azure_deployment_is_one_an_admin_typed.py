# -*- coding: utf-8 -*-
"""SQL was generated on gpt-4o-mini while the knowledge base used gpt-4o.

Azure OpenAI does not route on a model id. It routes on a DEPLOYMENT name the
tenant chose, and the two are unrelated strings: a resource can serve GPT-4o
under the name ``emco-prod`` and have nothing called ``gpt-4o`` at all.

``resolve_provider`` knew that and preferred the deployment fields. What it did
when they were blank was the defect. It fell through to ``_default_model``,
whose azure row differed BY PURPOSE -- "fast" for a query, "high" for a KB
build -- so the product asked Azure for a deployment literally named:

    purpose="query"  ->  "gpt-4o-mini"     (every SQL generation, repair,
                                            candidate, corroboration, planner)
    purpose="kb"     ->  "gpt-4o"          (every knowledge-base document)

Two outcomes, both bad. On a resource with neither name, every question died
with DeploymentNotFound. On a resource that happens to have both -- which is
the common Azure setup, because mini is the cheap default everyone deploys
first -- the SQL behind every answer was written by the mini model while the
knowledge base it was reading had been written by GPT-4o. Nothing in the UI,
the logs or the audit trail said the tiers differed. The admin believes they
are on GPT-4o because that is what they deployed and what the KB build used.

The guess is gone. Every name ``azure_deployment_name`` can return is one a
human entered, in this order: this purpose's deployment field, this purpose's
configured model, then anything typed for the other purpose (with a warning,
because borrowing a real verified deployment keeps the workspace answering and
cannot change tier behind the admin's back). With nothing typed anywhere it
refuses and names the field to fill.

The tests below execute ``resolve_provider`` on each configuration state and
assert on the deployment name it returns, because that string is the whole
defect -- it is what goes on the wire as Azure's ``model``.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import store  # noqa: E402,F401  (imported so sys.modules["store"] exists to patch)
from core.llm import (  # noqa: E402
    _default_model,
    azure_deployment_name,
    resolve_provider,
)

ENDPOINT = "https://emco.openai.azure.com"
BASE = {
    "default_llm_provider": "azure_openai",
    "azure_openai_api_key": "key",
    "azure_openai_endpoint": ENDPOINT,
}

# The two names _default_model used to hand Azure. Literals, not read back from
# the implementation: the point is that these strings must never reach Azure as
# a deployment name again, whatever the implementation later looks like.
GUESSED_FAST = "gpt-4o-mini"
GUESSED_HIGH = "gpt-4o"


def _resolved(saved: dict, purpose: str, client: dict | None = None) -> str:
    """The deployment name resolve_provider would put on the wire."""
    # resolve_provider does a function-local `import store`, which resolves
    # sys.modules["store"] at call time -- and other modules in this suite
    # replace that entry, so the live object is the only safe patch target.
    with patch.object(sys.modules["store"], "get_all_system",
                      return_value=dict(saved)):
        _provider, model, _key, _extra = resolve_provider(
            dict(client or {"account_id": ""}), purpose=purpose)
    return model


class TestTheDefectItself:
    """A blank deployment field must not silently choose a model tier."""

    def test_only_the_kb_name_is_set_and_queries_no_longer_get_mini(self):
        saved = {**BASE, "azure_kb_deployment_name": "emco-prod"}
        assert _resolved(saved, "query") == "emco-prod"
        assert _resolved(saved, "query") != GUESSED_FAST

    def test_the_two_purposes_never_disagree_by_accident(self):
        """The dangerous case: both tiers deployed, neither field filled. The
        KB is written by one model and the SQL by another, and the product is
        silent about it."""
        saved = dict(BASE)
        with pytest.raises(RuntimeError):
            _resolved(saved, "query")
        with pytest.raises(RuntimeError):
            _resolved(saved, "kb")

    def test_a_deliberate_difference_is_still_honoured(self):
        """An admin who picks two deployments gets two deployments. The fix is
        about accidents, not about forcing one model on everyone."""
        saved = {**BASE,
                 "azure_query_deployment_name": "emco-fast",
                 "azure_kb_deployment_name": "emco-kb"}
        assert _resolved(saved, "query") == "emco-fast"
        assert _resolved(saved, "kb") == "emco-kb"

    def test_the_guess_is_gone_from_the_defaults_table(self):
        """_default_model is still right for providers that route on model ids;
        it has nothing to say about a deployment name."""
        assert _default_model("anthropic", "fast") == "claude-sonnet-4-6"
        assert _default_model("openai", "high") == "gpt-4o"
        # No azure row, so nothing here differs by tier for Azure any more.
        for quality in ("fast", "high"):
            assert _default_model("azure_openai", quality) == GUESSED_FAST


class TestEveryConfigurationThatWorkedStillWorks:
    """Widening the chain is what keeps this from breaking live tenants."""

    def test_both_deployment_names(self):
        saved = {**BASE,
                 "azure_query_deployment_name": "emco-prod",
                 "azure_kb_deployment_name": "emco-prod"}
        assert _resolved(saved, "query") == "emco-prod"
        assert _resolved(saved, "kb") == "emco-prod"

    def test_a_workspace_that_only_ever_set_the_generic_model(self):
        """This configuration worked before -- default_llm_model was already
        being used as the query deployment name -- so it must keep working."""
        saved = {**BASE, "default_llm_model": "emco-prod"}
        assert _resolved(saved, "query") == "emco-prod"

    def test_and_its_kb_build_borrows_that_name_rather_than_refusing(self):
        saved = {**BASE, "default_llm_model": "emco-prod"}
        assert _resolved(saved, "kb") == "emco-prod"

    def test_a_kb_only_model_setting(self):
        saved = {**BASE, "kb_llm_model": "emco-kb"}
        assert _resolved(saved, "kb") == "emco-kb"
        assert _resolved(saved, "query") == "emco-kb"

    def test_a_per_tenant_override_still_wins_over_the_system_default(self):
        saved = {**BASE, "default_llm_model": "emco-shared"}
        assert _resolved(saved, "query",
                         {"account_id": "", "llm_model": "emco-tenant"}) == "emco-tenant"

    def test_the_deployment_field_outranks_the_generic_model(self):
        saved = {**BASE,
                 "azure_query_deployment_name": "emco-prod",
                 "default_llm_model": "gpt-4o"}
        assert _resolved(saved, "query") == "emco-prod"


class TestBorrowingIsAnnounced(unittest.TestCase):
    """A silent fallback is how the original defect survived. This one talks."""

    def test_the_warning_names_the_field_to_fill(self):
        saved = {**BASE, "azure_kb_deployment_name": "emco-prod"}
        with self.assertLogs("querybot.llm", level=logging.WARNING) as captured:
            self.assertEqual(_resolved(saved, "query"), "emco-prod")
        blob = "\n".join(captured.output)
        self.assertIn("azure_query_deployment_name", blob)
        self.assertIn("emco-prod", blob)

    def test_a_fully_configured_workspace_says_nothing(self):
        """A warning on every question would be noise nobody reads."""
        saved = {**BASE,
                 "azure_query_deployment_name": "emco-prod",
                 "azure_kb_deployment_name": "emco-kb"}
        with patch.object(sys.modules["store"], "get_all_system",
                          return_value=dict(saved)):
            with self.assertNoLogs("querybot.llm", level=logging.WARNING):
                resolve_provider({"account_id": ""}, purpose="query")
                resolve_provider({"account_id": ""}, purpose="kb")


class TestTheRefusalIsUseful(unittest.TestCase):

    def test_it_names_where_to_go_and_why_nothing_can_be_guessed(self):
        with self.assertRaises(RuntimeError) as caught:
            _resolved(dict(BASE), "query")
        message = str(caught.exception)
        self.assertIn("Admin", message)
        self.assertIn("deployment name", message)

    def test_an_unconfigured_endpoint_is_still_reported_first(self):
        """Endpoint is step 1 in the UI and the deployment is step 3. An
        operator who has configured nothing should be sent to step 1."""
        saved = {k: v for k, v in BASE.items() if k != "azure_openai_endpoint"}
        with self.assertRaises(RuntimeError) as caught:
            _resolved(saved, "query")
        self.assertIn("endpoint", str(caught.exception).lower())

    def test_a_missing_api_key_still_reports_the_key(self):
        """The key check sits after the branch, so the deployment resolution
        must not have already refused for a workspace that has both names."""
        saved = {k: v for k, v in BASE.items() if k != "azure_openai_api_key"}
        saved["azure_query_deployment_name"] = "emco-prod"
        with self.assertRaises(RuntimeError) as caught:
            _resolved(saved, "query")
        self.assertIn("API key", str(caught.exception))


class TestTheResolverOnItsOwn:
    """azure_deployment_name is public because the admin surface and a
    preflight both want to ask "what would this workspace actually call?"
    without building a provider."""

    @pytest.mark.parametrize("purpose", ["query", "kb"])
    def test_a_blank_config_refuses_for_either_purpose(self, purpose):
        with pytest.raises(RuntimeError):
            azure_deployment_name({}, {}, purpose)

    @pytest.mark.parametrize("whitespace", ["", "   ", "\t\n"])
    def test_whitespace_is_not_a_deployment_name(self, whitespace):
        with pytest.raises(RuntimeError):
            azure_deployment_name(
                {"azure_query_deployment_name": whitespace}, {}, "query")

    def test_a_name_is_stripped_before_it_goes_on_the_wire(self):
        assert azure_deployment_name(
            {"azure_query_deployment_name": "  emco-prod \n"}, {}, "query"
        ) == "emco-prod"

    @pytest.mark.parametrize("purpose", ["query", "kb"])
    def test_an_unknown_purpose_is_treated_as_a_query(self, purpose):
        """Only "kb" is special; everything else is a question. Pinned so a new
        purpose string cannot quietly start reading the KB deployment."""
        cfg = {"azure_query_deployment_name": "emco-fast",
               "azure_kb_deployment_name": "emco-kb"}
        assert azure_deployment_name(cfg, {}, "report") == "emco-fast"
        assert azure_deployment_name(cfg, {}, "") == "emco-fast"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
