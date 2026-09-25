# -*- coding: utf-8 -*-
"""Admin → System's Azure test checks the deployment each purpose will call.

The button tested the query deployment field and nothing else. The knowledge
base build calls its own deployment, and when that field is blank the runtime's
resolver (core.llm.azure_deployment_name) falls back to the model name the
System form saved -- "gpt-4o" -- which is a deployment only if one happens to
be called that. So the first test of the KB deployment was the first KB build,
and it failed with DeploymentNotFound after the button had said everything
was fine.

Now the test asks the same resolver what each purpose will call, sends one
request per distinct deployment, and reports a line for each: queries and the
knowledge base, what each calls, whether it was set for it or falls back, and
what Azure answered. The page shows those lines.

The route runs for real with Azure replaced at the HTTP boundary; the page's
own rendering code runs in a JavaScript engine over the route's answer.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from admin import routes

ENDPOINT = "https://acme.openai.azure.com"


def _saved(**fields) -> dict:
    return {"default_llm_provider": "azure_openai", "azure_openai_api_key": "key",
            "azure_openai_endpoint": ENDPOINT, "azure_openai_api_version": "2024-02-01", **fields}


class _Azure:
    """Azure at the HTTP boundary: these deployments exist; any status forced."""

    def __init__(self, deployments=(), status: int | None = None):
        self.deployments = set(deployments)
        self.status = status
        self.posted: list[str] = []
        self.got: list[str] = []

    def client(self):
        azure = self

        class _Response:
            def __init__(self, status, body):
                self.status_code = status
                self.is_success = 200 <= status < 300
                self._body = body
                self.text = json.dumps(body)

            def json(self):
                return self._body

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                azure.posted.append(url)
                if azure.status:
                    return _Response(azure.status, {"error": {"message": "refused"}})
                name = url.split("/deployments/", 1)[1].split("/", 1)[0]
                if name in azure.deployments:
                    return _Response(200, {"choices": [{"message": {"content": "hi"}}]})
                return _Response(404, {"error": {"code": "DeploymentNotFound", "message":
                                       "The API deployment for this resource does not exist."}})

            async def get(self, url, **kw):
                azure.got.append(url)
                return _Response(azure.status or 200, {"data": []})

        return _Client


def _test(saved: dict, azure: _Azure) -> dict:
    request = MagicMock()
    request.cookies = {routes._COOKIE: routes._sign_admin_session()}
    request.query_params = {}
    with patch.object(routes.store, "get_all_system", return_value=dict(saved)), \
            patch("httpx.AsyncClient", azure.client()):
        answer = asyncio.run(routes.test_llm_connection(request, provider="azure_openai"))
    return json.loads(answer.body)


def _lines(answer: dict) -> dict:
    return {check["purpose"]: (check["ok"], check["label"], check["error"]) for check in answer["checks"]}


class TestEachPurposeIsTested:

    def test_one_deployment_for_both_is_asked_once(self):
        azure = _Azure({"gpt-4o-prod"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod",
                              azure_kb_deployment_name="gpt-4o-prod"), azure)
        assert answer["ok"] is True
        assert len(azure.posted) == 1
        assert _lines(answer) == {"query": (True, "Queries: gpt-4o-prod", ""),
                                  "kb": (True, "Knowledge base: gpt-4o-prod", "")}

    def test_two_deployments_are_each_asked(self):
        azure = _Azure({"gpt-4o-prod", "gpt-4o-kb"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod",
                              azure_kb_deployment_name="gpt-4o-kb"), azure)
        assert answer["ok"] is True
        assert [url.split("/deployments/")[1].split("/")[0] for url in azure.posted] == ["gpt-4o-prod", "gpt-4o-kb"]

    def test_a_kb_deployment_that_does_not_exist_is_caught_before_the_build(self):
        azure = _Azure({"gpt-4o-prod"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod",
                              azure_kb_deployment_name="gpt4o-kb"), azure)
        assert answer["ok"] is False
        assert _lines(answer)["kb"] == (
            False, "Knowledge base: gpt4o-kb",
            "HTTP 404: The API deployment for this resource does not exist.")
        assert _lines(answer)["query"][0] is True
        assert answer["error"] == ("Knowledge base: gpt4o-kb — HTTP 404: The API deployment for "
                                   "this resource does not exist.")


class TestABlankFieldIsTestedAsWhatItFallsBackTo:

    def test_the_saved_model_name(self):
        """The trap: the System form saves "gpt-4o" as the KB model, and a
        blank KB deployment field calls a deployment by that name."""
        azure = _Azure({"gpt-4o-prod"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod", kb_llm_model="gpt-4o"), azure)
        assert answer["ok"] is False
        ok, label, error = _lines(answer)["kb"]
        assert not ok
        assert label == "Knowledge base: gpt-4o (no deployment set for it; it calls the saved model name)"
        assert error.startswith("HTTP 404")

    def test_the_other_purposes_deployment(self):
        azure = _Azure({"gpt-4o-prod"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod"), azure)
        assert answer["ok"] is True
        assert _lines(answer)["kb"] == (
            True, "Knowledge base: gpt-4o-prod (no deployment set for it; it uses the other one)", "")
        assert len(azure.posted) == 1


class TestWhatDidNotChange:

    def test_nothing_typed_anywhere_checks_the_credentials(self):
        azure = _Azure()
        answer = _test(_saved(), azure)
        assert answer == {"ok": True, "verification": "credentials", "model": "endpoint reachable"}
        assert azure.posted == [] and len(azure.got) == 1

    @pytest.mark.parametrize("status, words", [(401, "API key rejected (401)"), (403, "Permission denied (403)")])
    def test_a_refused_key_or_network_is_one_message(self, status, words):
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod",
                              azure_kb_deployment_name="gpt-4o-kb"), _Azure(status=status))
        assert answer["ok"] is False
        assert answer["error"].startswith(words)
        assert "checks" not in answer


class TestThePage:

    def _render(self, data: dict) -> dict:
        """The page's renderTestResult, run over the route's answer."""
        import dukpy

        from tests.js_lift import function as lift

        source = open("admin/templates/system.html", encoding="utf-8").read()
        script = lift(source, "function renderTestResult(el, data)") + """
var document = {createElement: function(tag){ return {tag: tag, className: '', textContent: ''}; }};
var el = {className: '', textContent: 'old', children: [], appendChild: function(c){ this.children.push(c); }};
renderTestResult(el, dukpy['data']);
var lines = [];
for (var i = 0; i < el.children.length; i++) { lines.push([el.children[i].tag, el.children[i].className, el.children[i].textContent]); }
({className: el.className, textContent: el.textContent, lines: lines});
"""
        return dukpy.evaljs(script, data=data)

    def test_each_deployment_is_its_own_line(self):
        azure = _Azure({"gpt-4o-prod"})
        answer = _test(_saved(azure_query_deployment_name="gpt-4o-prod", kb_llm_model="gpt-4o"), azure)
        shown = self._render(answer)
        assert shown["className"] == "provider-test-result err"
        assert shown["lines"] == [
            ["span", "test-line", "✓ Queries: gpt-4o-prod"],
            ["span", "test-line", "✗ Knowledge base: gpt-4o (no deployment set for it; it calls the saved "
                                  "model name) — HTTP 404: The API deployment for this resource does not exist."],
        ]

    def test_an_answer_without_lines_is_shown_as_before(self):
        shown = self._render({"ok": True, "verification": "model", "model": "claude-haiku-4-5-20251001"})
        assert shown["lines"] == []
        assert shown["textContent"] == "✓ Model request succeeded — claude-haiku-4-5-20251001"

    def test_the_button_says_it_tests_the_deployments(self):
        class _Url:
            path = "/admin/system"

        class _FakeRequest:
            url = _Url()
            query_params: dict = {}
            session: dict = {}
            scope = {"type": "http"}
            cookies: dict = {}
            headers: dict = {}

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_all_system",
                             return_value=_saved(azure_kb_deployment_name="gpt-4o-kb")), \
                patch.object(routes, "_resp", side_effect=lambda request, name, context: context):
            context = asyncio.run(routes.system_page(_FakeRequest()))
        page = routes.templates.get_template("system.html").render(request=_FakeRequest(), **context)
        assert "Test deployments" in page
        assert "Test selected deployment" not in page
