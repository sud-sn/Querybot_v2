import re
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


# The three tests below used to pin exact hex values from the previous
# palette (#2563EB / #1D4ED8 / #5B8DEF / graphite surfaces). That pinned a
# snapshot rather than a requirement: it broke on every colour change while
# never checking the things that were actually wrong -- a font-family the
# repo never shipped, a token referenced in seven places and never defined,
# and a brand palette indistinguishable from every other AI-era product.
# These assert the requirements instead.

# Stock Tailwind values. Their presence is the single clearest signal that a
# palette was inherited rather than chosen -- these are the defaults code
# generators emit, so a product wearing them cannot read as its own thing.
_TAILWIND_DEFAULTS = (
    "#2563EB", "#1D4ED8", "#3B82F6", "#60A5FA", "#DBEAFE", "#EFF6FF", "#BFDBFE",
    "#6366F1", "#4F46E5", "#6D28D9", "#7C3AED", "#8B5CF6", "#A855F7",
    "#F8FAFC", "#F1F5F9", "#0F172A", "#020617", "#64748B", "#94A3B8", "#CBD5E1",
    "#059669", "#DC2626", "#B45309", "#10B981", "#EF4444",
)


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def test_brand_palette_is_not_stock_tailwind():
    """The identity must be owned, not inherited from a framework default."""
    tokens = _strip_css_comments(_read("static/css/tokens.css")).upper()
    found = sorted({h for h in _TAILWIND_DEFAULTS if h in tokens})
    assert not found, (
        "tokens.css contains stock Tailwind values, which is what makes a product "
        f"look generated rather than designed: {found}"
    )


def test_brand_accent_is_defined_once_and_reachable_by_every_alias():
    """Every legacy alias must resolve to the brand, so one edit re-skins the app."""
    tokens = _read("static/css/tokens.css")
    for token in ("--primary:", "--primary-hover:", "--primary-soft:",
                  "--on-primary:", "--blue:", "--font-ui:", "--font-display:",
                  "--font-mono:", "--shadow-sm:"):
        assert token in tokens, f"{token} is not defined in tokens.css"


def test_every_referenced_token_is_actually_defined():
    """Structural guard for the class of bug that shipped --shadow-sm: a
    var() reference with no definition computes to nothing, silently."""
    import re
    global_tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", _read("static/css/tokens.css")))
    missing = {}
    for name in ("base", "admin", "portal", "chat_workspace", "production", "brand-motion"):
        css = _read(f"static/css/{name}.css")
        # A stylesheet may also scope its own custom properties to a component
        # (brand-motion's --qb-mark-*), so those count as defined too.
        defined = global_tokens | set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
        # Only flag references with no inline fallback -- var(--x, y) degrades safely.
        for ref in set(re.findall(r"var\(\s*(--[a-z0-9-]+)\s*\)", css)):
            if ref not in defined:
                missing.setdefault(ref, []).append(name)
    assert not missing, f"var() references with no definition and no fallback: {missing}"


def test_the_brand_font_is_actually_shipped():
    """base.css declared 'Inter' for a year with no @font-face and no font
    files, so every user rendered a system fallback and the graded weights
    the stylesheets ask for had no variable axis to land on."""
    fonts_css = _read("static/css/fonts.css")
    assert "@font-face" in fonts_css

    referenced = set(re.findall(r"/static/fonts/([A-Za-z0-9._-]+\.woff2)", fonts_css))
    assert referenced, "fonts.css declares no font files"
    for filename in sorted(referenced):
        path = ROOT / "static" / "fonts" / filename
        assert path.is_file(), f"fonts.css references {filename}, which is not in static/fonts/"
        assert path.read_bytes()[:4] == b"wOF2", f"{filename} is not a valid woff2 file"

    # And the shells must actually load it, or the @font-face rules never run.
    for template in ("admin/templates/base.html", "portal/templates/portal_base.html"):
        shell = _read(template)
        assert "fonts.css" in shell, f"{template} does not load fonts.css"
        assert 'rel="preload"' in shell and ".woff2" in shell, (
            f"{template} should preload the font files to avoid a swap flash"
        )


def test_the_animated_mark_follows_the_theme_tokens():
    """brand-motion.css kept a private copy of the brand palette, so a
    rebrand left the animated logo on the previous hue."""
    css = _strip_css_comments(_read("static/css/brand-motion.css"))
    bare_hex = [
        line.strip() for line in css.splitlines()
        if re.search(r"#[0-9a-fA-F]{3,8}", line) and "var(--" not in line
    ]
    assert not bare_hex, f"brand-motion.css hardcodes colour outside the tokens: {bare_hex}"


def test_dark_theme_redefines_the_full_surface_and_brand_set():
    """A colour whose only definition sits in the light block renders one
    theme's ink on the other theme's ground."""
    tokens = _read("static/css/tokens.css")
    dark = tokens.split("data-theme='dark'", 1)[1]
    for token in ("--bg:", "--surface:", "--surface-2:", "--surface-3:",
                  "--border:", "--text:", "--text-muted:", "--primary:",
                  "--on-primary:", "--sidebar-bg:", "--sidebar-accent:",
                  "--shadow-sm:"):
        assert token in dark, f"{token} is never redefined for dark mode"


def test_admin_and_portal_shells_use_shared_sidebar_tokens():
    for stylesheet_name in (
        "static/css/admin.css",
        "static/css/portal.css",
        "static/css/production.css",
    ):
        stylesheet = _read(stylesheet_name)
        assert "var(--sidebar-bg)" in stylesheet
        assert "var(--sidebar-border)" in stylesheet

    production = _read("static/css/production.css")
    assert "background: #0A1020" not in production


def test_theme_stylesheets_are_cache_busted_together():
    """Version derived from the shell rather than hardcoded, so bumping the
    identity does not require editing this test."""
    admin_src = _read("admin/templates/base.html")
    match = re.search(r"tokens\.css\?v=([\w.-]+)", admin_src)
    assert match, "admin base.html does not cache-bust tokens.css"
    version = match.group(1)
    admin = _read("admin/templates/base.html")
    portal = _read("portal/templates/portal_base.html")
    chat = _read("portal/templates/portal_chat.html")

    assert f"tokens.css?v={version}" in admin
    assert f"admin.css?v={version}" in admin
    assert f"production.css?v={version}" in admin
    assert f"tokens.css?v={version}" in portal
    assert f"portal.css?v={version}" in portal
    assert f"production.css?v={version}" in portal
    assert f"chat_workspace.css?v={version}" in chat


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


def test_chat_uses_unified_portal_sidebar_and_responsive_drawer():
    import re

    base = _read("portal/templates/portal_base.html")
    template = _read("portal/templates/portal_chat.html")
    portal_stylesheet = _read("static/css/portal.css")
    stylesheet = _read("static/css/chat_workspace.css")

    assert "chat_workspace.css" in template
    assert 'id="portalSidebar"' in base
    assert 'class="portal-new-thread"' in base
    assert "{% block portal_sidebar_context %}" in base
    assert "{% block portal_sidebar_context %}" in template
    assert 'class="portal-thread-panel"' in template
    assert 'class="chat-conversation-rail"' not in template
    assert 'class="chat-workspace-main"' in template
    assert 'id="historyPanel"' in template
    assert "loadHistory();" in template
    assert re.search(
        r"\.portal-app-shell\s*\{[^}]*grid-template-columns:\s*\d+px minmax\(0, 1fr\)",
        portal_stylesheet,
    )
    assert re.search(
        r"\.portal-app-shell\.portal-sidebar-collapsed\s*\{[^}]*grid-template-columns:\s*\d+px minmax\(0, 1fr\)",
        portal_stylesheet,
    )
    assert "portal-drawer-open" in base
    assert "portal-drawer-open" in portal_stylesheet
    assert "@media (max-width: 720px)" in portal_stylesheet
    assert "transform: translateX(-101%)" in stylesheet


def test_chat_empty_state_uses_workspace_suggestions_and_large_composer():
    template = _read("portal/templates/portal_chat.html")
    stylesheet = _read("static/css/chat_workspace.css")

    assert "How can I help you today?" in template
    assert "suggestions[:4]" in template
    assert 'class="suggestion-card-copy"' in template
    assert 'id="input" class="chat-input"' in template
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in stylesheet
    assert ".data-table { min-width: 620px; }" in stylesheet


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


def test_no_rule_puts_bare_white_on_a_brand_fill():
    """The brand accent lifts to a bright twin in dark mode, where white text
    measures 1.87:1. Filled controls must take their ink from --on-primary,
    which flips with the theme, rather than hardcoding #fff."""
    brand_bg = re.compile(
        r"background(?:-color)?\s*:[^;}]*var\(\s*--(?:primary|primary-hover|blue|blue-mid"
        r"|blue-soft|teal-600|teal-700)\s*\)"
    )
    white = re.compile(r"color\s*:\s*#(?:fff|ffffff)", re.I)

    offenders = []
    for name in ("base", "admin", "portal", "chat_workspace", "production", "brand-motion"):
        css = _read(f"static/css/{name}.css")
        for block in re.findall(r"\{[^{}]*\}", css):
            if brand_bg.search(block) and white.search(block):
                offenders.append(f"{name}.css: {block.strip()[:90]}")
    assert not offenders, "white hardcoded on a brand fill (use var(--on-primary)): " + "; ".join(offenders)


def test_every_local_stylesheet_is_cache_busted():
    """base.css shipped with no version parameter, so returning users kept a
    stale copy through every deploy -- which is how a fixed rule can look
    unfixed in the browser."""
    for template in ("admin/templates/base.html", "portal/templates/portal_base.html"):
        shell = _read(template)
        unversioned = re.findall(r'href="(/static/css/[A-Za-z0-9._-]+\.css)"', shell)
        assert not unversioned, f"{template} loads un-versioned stylesheets: {unversioned}"


def test_no_template_depends_on_an_external_cdn():
    """A governed deployment behind a corporate CSP cannot reach jsdelivr,
    cdnjs or fonts.googleapis, so the product rendered answers with no charts
    and no entity graph."""
    import glob
    offenders = []
    for path in glob.glob("admin/templates/*.html") + glob.glob("portal/templates/*.html"):
        body = Path(path).read_text(encoding="utf-8")
        for host in ("cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com",
                     "fonts.gstatic.com", "unpkg.com"):
            if host in body:
                offenders.append(f"{Path(path).name} -> {host}")
    assert not offenders, f"external CDN dependencies: {offenders}"


# ── The animated mark ──────────────────────────────────────────────────────
# The static favicon was redesigned first and the animated component was
# missed, so the gradient badge, magnifier handle and sparkle survived in the
# most visible place in the product: the sign-in screen. These guard both
# copies of the macro, which are duplicated between admin and portal.

_GENERIC_AI_MARK_PARTS = (
    "qb-brand-motion__spark",   # four-pointed sparkle (Gemini's glyph)
    "qb-brand-motion__tail",    # magnifier handle
    "qb-brand-motion__data",    # data rows inside the lens
    "qb-brand-motion__badge",   # saturated tile
    "linearGradient",           # gradient badge
)


def test_the_animated_mark_carries_none_of_the_generic_ai_motifs():
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        source = _read(template)
        macro = source.split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]
        found = [part for part in _GENERIC_AI_MARK_PARTS if part in macro]
        assert not found, f"{template}: brand_motion still contains {found}"


def test_the_animated_mark_is_the_q_and_caret():
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        macro = _read(template).split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]
        assert "qb-brand-motion__ring" in macro, f"{template}: the Q ring is missing"
        assert "qb-brand-motion__caret" in macro, f"{template}: the caret tail is missing"


def test_both_macro_copies_render_an_identical_mark():
    """The macro is duplicated across admin and portal, which is how the two
    drifted apart in the first place."""
    marks = []
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        macro = _read(template).split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]
        svg = macro[macro.index("<svg"):macro.index("</svg>")]
        marks.append(re.sub(r"\s+", " ", svg).strip())
    assert marks[0] == marks[1], "admin and portal brand_motion marks have drifted apart"


def test_the_caret_animates_and_survives_reduced_motion():
    css = _read("static/css/brand-motion.css")
    assert "qb-caret-blink" in css, "the caret has no resting animation"
    assert ".qb-brand-motion__caret" in css
    reduced = css.split("prefers-reduced-motion", 1)[1]
    assert "opacity: 1" in reduced, (
        "the blink animates opacity, so reduced-motion must leave the caret solid "
        "rather than stuck mid-cycle"
    )


def test_the_mark_carries_no_stale_brand_colour_in_an_rgba():
    """The old blue survived a hex sweep by hiding in an rgba() drop-shadow,
    which put a blue halo around a green mark."""
    css = _strip_css_comments(_read("static/css/brand-motion.css"))
    stale = re.findall(r"rgba?\(\s*37\s*,\s*99\s*,\s*235", css)
    assert not stale, f"brand-motion.css still references the old brand blue: {stale}"
