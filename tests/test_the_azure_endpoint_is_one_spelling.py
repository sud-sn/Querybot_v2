# -*- coding: utf-8 -*-
"""The admin test button and the product called two different URLs.

Azure routes a completion at ``<base>/openai/deployments/<name>/chat/completions``
and the SDK appends that suffix itself, so the value saved in Admin → System has
to be the resource BASE and nothing else. Admins do not paste the base. They
paste what the Azure portal shows them, which is the target URI:

    https://emco.openai.azure.com/openai/deployments/gpt-4o-emco/chat/completions?api-version=2024-02-01

``admin/routes.py`` knew that. Both of its Azure routes cut the ``/openai`` path
off before testing, so "Test selected deployment" came back green. The runtime
client did not: ``_get_azure_client`` only stripped a trailing slash, so every
real question went to

    https://emco.openai.azure.com/openai/openai/deployments/gpt-4o-emco/...

and got a 404. The admin had a green tick and a bot that could not answer, and
nothing connected the two. A host with no scheme was worse still -- the SDK
treats the base as relative and puts the hostname in the URL twice.

This is the write-site/read-site identity bug in its usual shape: one value,
two readers, each with its own private idea of what it means. The fix is one
function, ``core.llm.normalize_azure_endpoint``, that all three readers call --
the runtime client, the deployment listing route and the connection test.

So the test that matters is the last one in this file: it saves ONE endpoint,
drives the REAL admin route and the REAL client construction, and asserts they
arrive at the same URL. A test that only checked the normaliser's return value
would have passed on the broken tree too, because the normaliser existed there
-- privately, in the one reader that did not need it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import admin.routes as routes  # noqa: E402
from core.llm import (  # noqa: E402
    _get_azure_client,
    _llm_client_cache,
    normalize_azure_endpoint,
    resolve_provider,
)

RESOURCE = "https://emco.openai.azure.com"
DEPLOYMENT = "gpt-4o-emco"

# Every way the same Azure OpenAI resource is written down in the wild. Pinned
# as literals rather than generated from the implementation: a list derived from
# the code under test shrinks whenever the code does.
SPELLINGS = [
    "https://emco.openai.azure.com",
    "https://emco.openai.azure.com/",
    "https://emco.openai.azure.com/openai",
    "https://emco.openai.azure.com/openai/",
    "https://emco.openai.azure.com/OpenAI",
    "https://emco.openai.azure.com/openai/deployments/gpt-4o-emco",
    "https://emco.openai.azure.com/openai/deployments/gpt-4o-emco/chat/completions"
    "?api-version=2024-02-01",
    "emco.openai.azure.com",
    "  https://emco.openai.azure.com/  ",
]


def _url_the_sdk_builds(endpoint: str, *, api_version: str = "2024-02-01") -> str:
    """The URL a completion would actually be POSTed to.

    Built by asking the real client the real question, rather than by reading
    ``_get_azure_client`` and reasoning about it: the whole defect was that the
    base looked right and the assembled URL was not.
    """
    _llm_client_cache.clear()
    client = _get_azure_client("key", endpoint, api_version)
    request = client.chat.completions._client._build_request(
        openai._models.FinalRequestOptions.construct(
            method="post", url="/chat/completions",
            json_data={"model": DEPLOYMENT, "messages": []},
        )
    )
    return str(request.url)


class TestOneResourceOneBase:

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_every_spelling_normalises_to_the_same_base(self, spelling):
        assert normalize_azure_endpoint(spelling) == RESOURCE

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_and_every_spelling_reaches_the_same_completion_url(self, spelling):
        """The assertion the green tick was making and the product was failing."""
        assert _url_the_sdk_builds(spelling) == (
            f"{RESOURCE}/openai/deployments/{DEPLOYMENT}/chat/completions"
            "?api-version=2024-02-01"
        )

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_the_openai_segment_appears_exactly_once(self, spelling):
        """`/openai/openai/deployments/...` is the 404 this fix is named for."""
        assert _url_the_sdk_builds(spelling).count("/openai/") == 1

    def test_a_host_with_no_scheme_does_not_land_in_the_url_twice(self):
        url = _url_the_sdk_builds("emco.openai.azure.com")
        assert url.count("emco.openai.azure.com") == 1
        assert url.startswith("https://")

    def test_one_resource_spelled_two_ways_is_one_cached_client(self):
        """Two entries would mean two connection pools and two TLS handshakes
        for one resource -- the cost the singleton cache exists to avoid."""
        _llm_client_cache.clear()
        first = _get_azure_client("key", f"{RESOURCE}/", "2024-02-01")
        second = _get_azure_client("key", f"{RESOURCE}/openai", "2024-02-01")
        assert first is second
        assert len(_llm_client_cache) == 1


class TestWhatMustSurvive:
    """Normalising too eagerly breaks two real deployments."""

    def test_a_gateway_keeps_its_own_path_prefix(self):
        """A corporate gateway fronting the resource under its own path. Cutting
        at the substring `openai` rather than at the path SEGMENT would have
        eaten `/azure-openai` as well."""
        assert normalize_azure_endpoint(
            "https://gw.corp.example/azure-openai/openai/deployments/x"
        ) == "https://gw.corp.example/azure-openai"

    def test_a_foundry_project_path_is_kept(self):
        """`admin/routes.py`'s deployment listing needs `/api/projects/<name>`;
        dropping it is how the fetch button stops finding anything."""
        endpoint = "https://proj.services.ai.azure.com/api/projects/emco"
        assert normalize_azure_endpoint(endpoint) == endpoint

    def test_a_foundry_v1_base_is_not_doubled_either(self):
        """The Foundry branch appends `/openai/v1` itself."""
        assert _url_the_sdk_builds(
            "https://proj.services.ai.azure.com/openai/v1"
        ) == "https://proj.services.ai.azure.com/openai/v1/chat/completions"

    @pytest.mark.parametrize("empty", ["", None, "   "])
    def test_an_unconfigured_endpoint_stays_empty(self, empty):
        """Both callers have their own "go to Admin → System" message for a
        blank field, and each is reached by testing the value for truth. A
        normaliser that turned "" into "https://" would silence both."""
        assert normalize_azure_endpoint(empty) == ""


class TestResolveProviderHandsOverTheNormalisedBase(unittest.TestCase):

    def test_the_saved_spelling_is_normalised_before_it_leaves(self):
        """resolve_provider's extra_kwargs are what every call site forwards, so
        this is where the value becomes every caller's problem."""
        saved = {
            "default_llm_provider": "azure_openai",
            "azure_openai_api_key": "key",
            "azure_openai_endpoint":
                f"{RESOURCE}/openai/deployments/{DEPLOYMENT}/chat/completions",
            "azure_query_deployment_name": DEPLOYMENT,
            "azure_kb_deployment_name": DEPLOYMENT,
        }
        # resolve_provider does a function-local `import store`, which resolves
        # sys.modules["store"] at call time -- and other modules in this suite
        # replace that entry, so the live object is the only safe target.
        with patch.object(sys.modules["store"], "get_all_system",
                          return_value=dict(saved)):
            _provider, _model, _key, extra = resolve_provider(
                {"account_id": ""}, purpose="query")
        self.assertEqual(extra["azure_endpoint"], RESOURCE)


class TestTheButtonAndTheProductAgree(unittest.IsolatedAsyncioTestCase):
    """The seam. One saved value, both readers, executed."""

    @staticmethod
    def _authenticated_request():
        request = MagicMock()
        request.cookies = {routes._COOKIE: routes._sign_admin_session()}
        request.query_params = {}
        return request

    async def _url_the_admin_button_tests(self, endpoint: str) -> str:
        """Drive the real route and capture where it actually posted."""
        posted: list[str] = []

        class _Response:
            status_code = 200
            is_success = True
            text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": "hi"}}]}

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                posted.append(url)
                return _Response()

            async def get(self, url, **kw):
                posted.append(url)
                return _Response()

        saved = {
            "default_llm_provider": "azure_openai",
            "azure_openai_api_key": "key",
            "azure_openai_endpoint": endpoint,
            "azure_openai_api_version": "2024-02-01",
            "azure_query_deployment_name": DEPLOYMENT,
            "azure_kb_deployment_name": DEPLOYMENT,
        }
        # Patched on the object admin.routes is HOLDING, not on the name
        # "store": that module binds `import store` once at import time, and
        # other modules in this suite replace sys.modules["store"] afterwards,
        # so patching by name reaches a module the route never reads. The
        # symptom is this test passing alone and failing in the full suite.
        with patch.object(routes.store, "get_all_system",
                          return_value=dict(saved)), \
                patch("httpx.AsyncClient", _Client):
            answer = await routes.test_llm_connection(
                self._authenticated_request(), provider="azure_openai")
        # The route's own body when it never got as far as a request: a guard
        # returned early, and "0 != 1" on its own does not say which one.
        self.assertEqual(
            len(posted), 1,
            f"route made {len(posted)} request(s); it answered "
            f"{getattr(answer, 'body', answer)!r}",
        )
        return posted[0]

    async def test_the_portal_target_uri_tests_what_the_product_calls(self):
        """This is the pair that disagreed. Before the fix the button posted to
        .../openai/deployments/gpt-4o-emco/chat/completions and the product
        posted to .../openai/openai/deployments/gpt-4o-emco/chat/completions."""
        endpoint = (f"{RESOURCE}/openai/deployments/{DEPLOYMENT}"
                    "/chat/completions?api-version=2024-02-01")
        tested = await self._url_the_admin_button_tests(endpoint)
        called = _url_the_sdk_builds(endpoint)
        self.assertEqual(tested, called)

    async def test_they_agree_on_every_spelling(self):
        for spelling in SPELLINGS:
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    await self._url_the_admin_button_tests(spelling),
                    _url_the_sdk_builds(spelling),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
