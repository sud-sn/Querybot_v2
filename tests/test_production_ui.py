from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_production_stylesheet_is_loaded_after_page_head_blocks():
    for template in ("admin/templates/base.html", "portal/templates/portal_base.html"):
        source = _read(template)
        assert "production.css" in source
        assert source.index("{% block head %}") < source.index("production.css")


def test_entity_graph_uses_shared_production_layer():
    assert "production.css" in _read("admin/templates/client_graph.html")


def test_brand_tokens_use_enterprise_blue():
    tokens = _read("static/css/tokens.css")
    assert "--primary:       #2563EB;" in tokens
    assert "--primary-hover: #1D4ED8;" in tokens
    assert "--primary-soft:  #EFF6FF;" in tokens


def test_portal_mobile_shell_exposes_theme_and_settings_actions():
    template = _read("portal/templates/portal_base.html")
    assert 'class="portal-mobile-actions"' in template
    assert 'href="/portal/change-password"' in template
    assert "window.__qbToggleTheme()" in template


def test_production_layer_contains_mobile_and_reduced_motion_guards():
    stylesheet = _read("static/css/production.css")
    assert "@media (max-width: 900px)" in stylesheet
    assert "@media (max-width: 640px)" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert "scrollWidth" not in stylesheet


def test_chat_uses_persistent_workspace_rail_and_responsive_drawer():
    import re

    template = _read("portal/templates/portal_chat.html")
    stylesheet = _read("static/css/chat_workspace.css")

    assert "chat_workspace.css" in template
    assert 'class="chat-conversation-rail"' in template
    assert 'class="chat-workspace-main"' in template
    assert 'id="historyPanel"' in template
    assert "loadHistory();" in template
    # Fixed-width rail + flexible main. Width-agnostic on purpose: a previous
    # hard-coded "300px" literal broke when the rail was restyled to 288px —
    # the invariant worth pinning is the grid SHAPE, not the pixel value.
    assert re.search(
        r"\.chat-workspace\s*\{[^}]*grid-template-columns:\s*\d+px minmax\(0, 1fr\)",
        stylesheet,
    )
    # Collapsible rail (added with the workspace refinement): collapsed state
    # zeroes the rail column so the conversation gets the full width.
    assert re.search(
        r"\.chat-workspace\.rail-collapsed\s*\{[^}]*grid-template-columns:\s*0 minmax\(0, 1fr\)",
        stylesheet,
    )
    assert "@media (max-width: 900px)" in stylesheet
    assert "transform: translateX(-101%)" in stylesheet


def test_brand_motion_is_shared_by_admin_and_portal_shells():
    for template in ("admin/templates/base.html", "portal/templates/portal_base.html"):
        source = _read(template)
        assert "brand-motion.css" in source
        assert "brand-motion.js" in source

    stylesheet = _read("static/css/brand-motion.css")
    assert 'data-state="querying"' in stylesheet
    assert 'data-state="success"' in stylesheet
    assert "prefers-reduced-motion: reduce" in stylesheet


def test_login_and_chat_use_query_lens_motion_states():
    admin_login = _read("admin/templates/login.html")
    portal_login = _read("portal/templates/portal_login.html")
    chat = _read("portal/templates/portal_chat.html")

    assert "adminAuthMark" in admin_login
    assert "portalAuthMark" in portal_login
    assert "data-brand-loading" in admin_login
    assert "data-brand-loading" in portal_login
    assert "answerProgressBrand" in chat
    assert "answerStageLabel" in chat
    assert "BRAND_STAGE_STATES" in chat


def test_system_llm_setup_is_saved_and_verified_in_order():
    template = _read("admin/templates/system.html")
    routes = _read("admin/routes.py")

    assert "llm-setup-progress" in template
    assert "Save selection and continue" in template
    assert "provider-change-notice" in template
    assert "Save changes before testing" in template
    assert "Verify saved connection" in template
    assert '"verification": "model"' in routes
    assert '"verification": "credentials"' in routes
    assert "endpoint reachability alone is never called" in routes
