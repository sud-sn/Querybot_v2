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
    template = _read("portal/templates/portal_chat.html")
    stylesheet = _read("static/css/chat_workspace.css")

    assert "chat_workspace.css" in template
    assert 'class="chat-conversation-rail"' in template
    assert 'class="chat-workspace-main"' in template
    assert 'id="historyPanel"' in template
    assert "loadHistory();" in template
    assert "grid-template-columns: 300px minmax(0, 1fr)" in stylesheet
    assert "@media (max-width: 900px)" in stylesheet
    assert "transform: translateX(-101%)" in stylesheet
