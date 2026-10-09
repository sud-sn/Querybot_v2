"""The admin workspace keeps what runs a workspace, and nothing else in its navigation.

The approved clean-up (October 2026): six areas instead of seven tabs and about
thirty pages. Data keeps the semantic layer an admin corrects -- Setup, what
QueryBot learned, the knowledge base, relationships, dates and metrics.
Questions is the log, the answers readers flagged and usage. The AI egress log
moves to Compliance, where its policy lives. Settings is one page with the
danger zone at its foot. Today's pipeline's own editors and the developer
tooling for its compiler leave the navigation for Diagnostics, a support page
at the foot of Settings, until every workspace answers with the new core.

Also: the counters belong to the Overview alone, the System page no longer
offers to switch the database QueryBot keeps its records in (a deployment
setting), a Standard workspace's Compliance shows only its profile and what
leaves for the AI, and "Needs you" no longer lists warnings that need no one.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from core.static_assets import asset_url

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "admin" / "templates"

DAY_TO_DAY = ["", "setup", "learned", "kb", "relationships", "date-roles", "measures", "queries", "learning-queue",
              "billing", "users", "groups", "pending-users", "compliance", "egress", "settings"]
# The entity graph and the metric registry are today's pipeline's own editors: the new core's are on
# Relationships and Metrics.
DIAGNOSTICS = ["graph", "metrics", "model-health", "evals", "traces", "readiness", "drafts", "mapping", "glossary", "meanings",
               "domains", "reports"]


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=ChainableUndefined)
    env.globals["asset"] = asset_url
    return env


def _nav(segment: str) -> str:
    path = "/admin/clients/demo" + (f"/{segment}" if segment else "")
    request = type("R", (), {"url": type("U", (), {"path": path})()})()
    return _env().get_template("_client_workspace_nav.html").render(
        request=request, client={"account_id": "demo", "client_name": "Sample", "state": "READY"})


def _labels(html: str, level: str) -> list[str]:
    block = re.search(rf'client-workspace-{level}.*?</nav>', html, re.S)
    return [re.sub(r"<[^>]+>.*", "", a).strip() for a in re.findall(r"<a [^>]*>(.*?)</a>", block.group(0), re.S)] \
        if block else []


def _links(html: str) -> set[str]:
    return set(re.findall(r'href="/admin/clients/demo/([a-z-]+)', html))


# ── the navigation ──────────────────────────────────────────────────────────


def test_six_areas():
    assert _labels(_nav(""), "primary") == ["Overview", "Data", "Questions", "People", "Compliance", "Settings"]


def test_data_keeps_the_semantic_layer_an_admin_corrects():
    assert _labels(_nav("setup"), "secondary") == [
        "Setup", "What QueryBot learned", "Knowledge base", "Relationships", "Dates", "Metrics"]


@pytest.mark.parametrize("segment", DAY_TO_DAY)
def test_no_day_to_day_page_links_to_a_diagnostics_page(segment):
    found = _links(_nav(segment)) & set(DIAGNOSTICS + ["diagnostics"])
    assert not found, f"/{segment or '(overview)'} links to {sorted(found)}"


def test_the_egress_log_sits_with_its_policy():
    assert _labels(_nav("egress"), "secondary") == ["Compliance", "AI egress log"]


def test_settings_is_one_page():
    assert _labels(_nav("settings"), "secondary") == []


# ── Diagnostics ─────────────────────────────────────────────────────────────


def _request(path: str = "/admin/clients/demo/diagnostics"):
    request = MagicMock()
    request.url.path = path
    request.query_params = {}
    return request


def test_diagnostics_lists_every_page_that_left_the_navigation():
    from admin import routes

    listed = [seg for seg, _name, _what in routes._DIAGNOSTIC_PAGES]
    assert sorted(listed) == sorted(DIAGNOSTICS)
    registered = {getattr(r, "path", "") for r in routes.router.routes}
    for seg in listed + ["diagnostics"]:
        assert f"/admin/clients/{{account_id}}/{seg}" in registered, seg


def test_the_diagnostics_page_renders_a_link_to_each():
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes.store, "get_client", return_value={"account_id": "demo", "client_name": "Sample",
                                                                    "state": "READY"}), \
            patch.object(routes, "_resp", side_effect=lambda req, name, ctx: (name, ctx)):
        name, ctx = asyncio.run(routes.client_diagnostics_page(_request(), "demo"))
    html = _env().get_template(name).render(request=_request(), **ctx)
    assert {seg for seg in DIAGNOSTICS} <= _links(html)


def test_settings_links_to_diagnostics_at_its_foot():
    detail = (TEMPLATES / "client_detail.html").read_text(encoding="utf-8")
    danger = detail[detail.index('<div id="tab-danger"'):]
    assert "/diagnostics" in danger


def test_the_old_advanced_path_lands_on_the_danger_zone_in_settings():
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        response = asyncio.run(routes.client_advanced_page(_request("/admin/clients/demo/advanced"), "demo"))
    assert response.status_code == 302
    assert response.headers["location"] == "/admin/clients/demo/settings#tab-danger"


@pytest.mark.parametrize("tab,shown", [("overview", True), ("settings", False), ("queries", False), ("audit", False)])
def test_the_counters_belong_to_the_overview(tab, shown):
    request = _request(f"/admin/clients/demo/{'' if tab == 'overview' else tab}")
    html = _env().get_template("client_detail.html").render(
        request=request, active_tab=tab, stats={"total": 3}, token_status={"unlimited": True},
        client={"account_id": "demo", "client_name": "Sample", "state": "READY", "created_at": "2026-01-01",
                "updated_at": "2026-01-01"},
        health_score={"components": [], "pct": 0, "score": 0, "grade": "A", "color": "green"},
        limit_pct=1, monthly_count=3, queries=[], llm_calls=[], failed_queries=[], top_questions=[],
        analysis_subtasks=[], all_dbs=[], query_models=[], cost_rates={}, erp_packs_available=[],
        client_erp_packs=[], teams_platforms=[], latest_sql_eval={}, latest_sql_pass_rate=None,
        kb_file_count=0, schema_file_count=0, semantic_pending_count=0)
    assert ('<div class="metric-label">Total queries</div>' in html) is shown
    if tab == "settings":
        pane = re.search(r'<div id="tab-danger" class="tab-pane([^"]*)"', html).group(1)
        assert "active" in pane, "the danger zone is the foot of Settings"


# ── System, Compliance, Needs you ───────────────────────────────────────────


def test_system_does_not_offer_to_switch_the_database():
    system = (TEMPLATES / "system.html").read_text(encoding="utf-8")
    assert "data-db-btn" not in system and 'name="database_backend"' not in system
    assert "DATABASE_URL" in system, "the page says where the setting lives"


def _compliance_tabs(industry: str) -> list[str]:
    """The page's own profile test and tab bar, rendered on their own: the rest
    of the page needs a live store."""
    source = (TEMPLATES / "client_compliance.html").read_text(encoding="utf-8")
    scope = re.search(r"\{% set _regulated = .*?%\}", source).group(0)
    start = source.index('<div class="compliance-nav qb-tabs"')
    bar = source[start:source.index("</div>", start) + len("</div>")]
    html = _env().from_string(scope + bar).render(profile={"industry": industry})
    return re.findall(r'role="tab" data-tab="([a-z]+)"', html)


def test_a_standard_workspace_sees_its_profile_and_what_leaves_for_the_ai():
    assert _compliance_tabs("standard") == ["profile", "egress"]


def test_a_regulated_workspace_sees_every_control():
    assert len(_compliance_tabs("healthcare")) == 10


def test_needs_you_lists_no_warning_that_needs_no_one():
    source = (ROOT / "admin" / "inbox.py").read_text(encoding="utf-8")
    assert '"conflict-warn"' not in source and "no action required" not in source
