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
    """It used to link production.css itself, because it was a standalone
    document. It now inherits the whole stylesheet set from the shell, which is
    what stopped it drifting onto its own cache-busted copy of the tokens."""
    graph = _read("admin/templates/client_graph.html")
    assert graph.lstrip().startswith('{% extends "client_base.html" %}')
    assert "production.css" in _read("admin/templates/base.html")


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


def test_there_is_exactly_one_theme_and_it_is_pinned():
    """The product is light only. `color-scheme: light` is not cosmetic: without
    it a visitor whose OS is dark gets dark scrollbars, dark form controls and a
    dark autofill wash on a light page, because the browser still believes it may
    render UA widgets either way."""
    tokens = _read("static/css/tokens.css")
    assert "color-scheme: light" in tokens, "the single theme is not pinned"
    assert "data-theme" not in tokens, "a theme block came back"
    for stylesheet in ("base.css", "admin.css", "portal.css", "production.css",
                       "chat_workspace.css", "brand-motion.css"):
        css = _read(f"static/css/{stylesheet}")
        assert "data-theme" not in css, f"{stylesheet} still carries theme rules"


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


def test_portal_mobile_shell_exposes_its_account_actions():
    template = _read("portal/templates/portal_base.html")
    assert 'class="portal-mobile-actions"' in template
    assert 'href="/portal/change-password"' in template
    assert "ToggleTheme" not in template, "the theme toggle is back"


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
    "qb-brand-motion__data",    # data rows floating in a lens
    "qb-brand-motion__badge",   # saturated tile
    "linearGradient",           # gradients stay OUT of the component: every
                                # state recolours through one custom property,
                                # which a gradient fill cannot do
    "M24.3 25.3 29 30v-2.2",    # the magnifier handle path
    "M30 3 1.27",               # the sparkle path
)


def _brand_macro(template: str) -> str:
    source = _read(template)
    return source.split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]


def test_the_animated_mark_carries_none_of_the_generic_ai_motifs():
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        found = [part for part in _GENERIC_AI_MARK_PARTS if part in _brand_macro(template)]
        assert not found, f"{template}: brand_motion still contains {found}"


def test_the_mark_is_a_bubble_carrying_three_bars():
    """The 2026-08 redesign: a filled speech bubble (the question) with three
    ascending bars inside it (the answer). The bars are the mark, not an
    indicator -- they rest visible."""
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        macro = _brand_macro(template)
        assert "qb-brand-motion__bowl" in macro, f"{template}: the bubble is missing"
        bars = len(re.findall(r'class="qb-brand-motion__dot"', macro))
        assert bars == 3, f"{template}: expected three bars, found {bars}"
        heights = [float(h) for h in re.findall(
            r'class="qb-brand-motion__dot"[^>]*height="([\d.]+)"', macro)]
        assert heights == sorted(heights) and len(set(heights)) == 3, (
            f"{template}: the bars must ascend -- that is the answer half of the mark"
        )


def test_the_tail_is_part_of_the_closed_fill_never_an_open_stroke():
    """The whole reason the mark has this shape. A straight diagonal leaving a
    circle reads as a magnifying-glass handle however it is tuned; a tail that
    is two bowed curves inside one closed filled path cannot."""
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        macro = _brand_macro(template)
        bubble = re.search(r'class="qb-brand-motion__bowl" d="([^"]+)"', macro)
        assert bubble, f"{template}: no bubble path"
        d = bubble.group(1).strip()
        assert d.endswith("Z"), f"{template}: the bubble must be a closed fill"
        assert "A" in d, f"{template}: the bubble body must be a circular arc"
        assert d.count("Q") >= 2, (
            f"{template}: the tail must be bowed curves, not a straight diagonal"
        )
    css = _read("static/css/brand-motion.css")
    bowl = css.split(".qb-brand-motion__bowl {", 1)[1].split("}", 1)[0]
    assert "fill: var(--qb-mark-glyph)" in bowl, (
        "the bubble must take a flat token fill, so every state can recolour it"
    )
    assert "stroke" not in bowl, "the bubble is a filled shape, never a stroked ring"


def test_both_macro_copies_render_an_identical_mark():
    """The macro is duplicated across admin and portal, which is how the two
    drifted apart in the first place."""
    marks = []
    for template in ("admin/templates/macros.html", "portal/templates/macros.html"):
        macro = _brand_macro(template)
        svg = macro[macro.index("<svg"):macro.index("</svg>")]
        marks.append(re.sub(r"\s+", " ", svg).strip())
    assert marks[0] == marks[1], "admin and portal brand_motion marks have drifted apart"


def test_the_bars_carry_the_motion_across_every_state():
    """A single blinking element is not motion. The bars must crouch into dots
    and type while working, stream while answering, land on success, drop on
    error, pop on hover, and rise on the auth intro."""
    css = _read("static/css/brand-motion.css")
    for keyframes in ("qb-bar-crouch", "qb-bar-bounce", "qb-bar-stream", "qb-bar-land",
                      "qb-bar-drop", "qb-bar-pop", "qb-bar-rise",
                      "qb-bubble-breathe", "qb-bubble-pop", "qb-bubble-shake"):
        assert f"@keyframes {keyframes}" in css, f"{keyframes} is not defined"
        assert css.count(keyframes) >= 2, f"{keyframes} is defined but never applied"

    bar_rule = css.split(".qb-brand-motion__dot {", 1)[1].split("}", 1)[0]
    assert "transform-origin: bottom" in bar_rule, (
        "bars must stretch from their baseline, the way a bar chart does"
    )
    assert "opacity: 1" in bar_rule, (
        "the bars rest VISIBLE -- they are the mark itself, not an indicator"
    )


def test_the_mark_is_interactive_only_at_rest():
    """Hover and press respond at idle, and are scoped so they can never fight
    a working state's animation."""
    css = _read("static/css/brand-motion.css")
    assert '[data-state="idle"]:hover' in css, "the resting mark must respond to hover"
    assert '[data-state="idle"]:active' in css, "the resting mark must respond to press"
    for line in css.splitlines():
        if ":hover" in line and "qb-brand-motion" in line and "@" not in line:
            assert '[data-state="idle"]' in line, (
                f"hover styling outside the idle state would fight the working "
                f"animation: {line.strip()}"
            )


def test_the_typing_bounce_carries_the_crouch_in_every_frame():
    """Two transform animations replace each other rather than composing, so if
    the bounce frames dropped the scaleY crouch the dots would flash back to
    full-height bars every cycle."""
    css = _read("static/css/brand-motion.css")
    bounce = css.split("@keyframes qb-bar-bounce {", 1)[1]
    bounce = bounce[:bounce.index("@keyframes")]
    lines = [ln for ln in bounce.splitlines() if "transform:" in ln]
    assert lines and all("scaleY(var(--qb-squash" in ln for ln in lines), (
        "every qb-bar-bounce frame must keep scaleY(var(--qb-squash)) or the "
        "dots pop back into bars mid-bounce"
    )


def test_reduced_motion_leaves_every_animated_part_at_full_value():
    """The animations drive opacity and scaleY, so switching them off must not
    leave a bar crouched or the bubble mid-breath."""
    css = _read("static/css/brand-motion.css")
    # Split on the @media token, not the phrase: the file header MENTIONS
    # reduced motion in prose, and splitting there reads the base rules instead.
    reduced = css.split("@media (prefers-reduced-motion", 1)[1]
    assert "animation: none" in reduced
    bowl = reduced.split(".qb-brand-motion__bowl", 1)[1].split("}", 1)[0]
    assert "scale(1)" in bowl, "the bubble could be left mid-breath"
    bar = reduced.split(".qb-brand-motion__dot", 1)[1].split("}", 1)[0]
    assert "opacity: 1" in bar and "scaleY(1)" in bar, (
        "with motion off the bars must stand at full height -- the bubble with "
        "three ascending bars IS the static mark"
    )


def test_the_mark_carries_no_stale_brand_colour_in_an_rgba():
    """The old blue survived a hex sweep by hiding in an rgba() drop-shadow,
    which put a blue halo around a green mark."""
    css = _strip_css_comments(_read("static/css/brand-motion.css"))
    stale = re.findall(r"rgba?\(\s*37\s*,\s*99\s*,\s*235", css)
    assert not stale, f"brand-motion.css still references the old brand blue: {stale}"


def test_the_standalone_mark_matches_the_component_geometry():
    """logo-mark.svg is the favicon and every chat avatar; the macro is the
    animated component. If their geometry drifts the product wears two logos."""
    svg = _read("static/img/logo-mark.svg")
    macro = _brand_macro("portal/templates/macros.html")
    svg_bubble = re.search(r'<path[^>]*d="(M31\.32[^"]+)"', svg)
    macro_bubble = re.search(r'class="qb-brand-motion__bowl" d="([^"]+)"', macro)
    assert svg_bubble and macro_bubble
    assert svg_bubble.group(1).strip() == macro_bubble.group(1).strip(), (
        "the standalone SVG and the component draw different bubbles"
    )
    svg_bars = re.findall(
        r'class="qb-bar" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
    macro_bars = re.findall(
        r'class="qb-brand-motion__dot" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"',
        macro)
    assert svg_bars and svg_bars == macro_bars, (
        "the bars have drifted between the SVG and the component"
    )


def test_the_standalone_mark_animates_once_and_respects_reduced_motion():
    """The file is the avatar on every assistant message: it may animate ON
    LOAD only, and must rest still -- forty marks looping out of sync down a
    conversation is noise, not life."""
    svg = _read("static/img/logo-mark.svg")
    assert "infinite" not in svg, "the standalone mark must not loop"
    assert "prefers-reduced-motion" in svg, "the standalone mark must honour reduced motion"
    assert 'gradientUnits="userSpaceOnUse"' in svg, (
        "the bar cut-outs share the tile gradient; objectBoundingBox would "
        "restart the gradient inside every bar and the cuts would not match the tile"
    )


# ── Admin console: colour must come from the tokens ──────────────────────────
# Status colours hardcoded into templates were invisible to the theme, so the
# console had light-only greens/reds/ambers on a dark shell, and white text on
# fills that turn bright in dark mode.

_STATUS_HEXES = (
    # greens
    "#059669", "#10B981", "#16A34A", "#38A169", "#15803D", "#166534", "#065F46",
    "#DCFCE7", "#ECFDF5", "#A7F3D0", "#D1FAE5",
    # reds
    "#DC2626", "#EF4444", "#E53E3E", "#B91C1C", "#991B1B",
    "#FEE2E2", "#FEF2F2", "#FECACA",
    # ambers
    "#D97706", "#F59E0B", "#B45309", "#92400E", "#FEF3C7", "#FFFBEB", "#FDE68A",
    # blues
    "#2563EB", "#1D4ED8", "#3B82F6", "#4F46E5", "#EFF6FF", "#DBEAFE", "#BFDBFE",
)

_STATUS_COLOUR_PAGES = (
    "base.html",
    "client_learning_queue.html",
    "client_compliance.html",
    "client_pending_users.html",
    "client_model_health.html",
)


def test_status_colours_in_admin_come_from_tokens():
    """These pages express success/warning/danger, which the token set already
    defines for both themes. client_graph and client_metrics are deliberately
    excluded: their palettes are categorical (entity type, chart series), and
    mapping 'fact table' onto var(--warning) would be a lie."""
    offenders = {}
    for page in _STATUS_COLOUR_PAGES:
        source = _read(f"admin/templates/{page}").upper()
        found = sorted({h for h in _STATUS_HEXES if h in source})
        if found:
            offenders[page] = found
    assert not offenders, (
        f"hardcoded status colours are invisible to the theme: {offenders}"
    )


def test_the_toast_reads_the_theme():
    """qbToast held a private stock-Tailwind palette with light-only
    backgrounds, so every toast in the console ignored dark mode."""
    base = _read("admin/templates/base.html")
    block = base.split("var colors = {", 1)[1].split("};", 1)[0]
    assert "#" not in block, f"qbToast still hardcodes colour: {block.strip()[:120]}"
    for token in ("--success", "--danger", "--warning", "--primary"):
        assert token in block, f"qbToast does not use {token}"


def test_no_white_text_on_a_token_fill():
    """--primary is the bright verdigris in dark mode, where white text on it
    measures 1.87:1. --on-primary exists for this and flips to near-black."""
    offenders = {}
    for path in sorted((ROOT / "admin" / "templates").glob("*.html")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "svg" in line.lower() or "stroke=" in line or 'fill="' in line:
                continue
            if not re.search(r"background:\s*var\(--", line):
                continue
            if re.search(r"color:\s*#(fff|ffffff)\b", line, re.I):
                offenders.setdefault(path.name, []).append(number)
    assert not offenders, (
        f"white text on a token fill is unreadable in dark mode; use "
        f"var(--on-primary): {offenders}"
    )


# ── Admin console: one dialog system ─────────────────────────────────────────

def test_no_native_browser_dialogs_in_the_admin_console():
    """The console ships its own qbConfirm/qbToast, styled and themed. Mixing
    them with browser-chrome confirm()/alert() was the most visible day-to-day
    inconsistency in the product.

    client_graph.html is excluded on purpose: it is a standalone document that
    never loads base.html, so it has no qbToast to call. Converting there would
    trade a working native dialog for a TypeError. It gets fixed when that page
    comes inside the shell."""
    native = re.compile(r"(?<![\w.{])(confirm|alert|prompt)\s*\(")

    def _blank_comments(text: str) -> str:
        """Blank out comments while preserving line count, so reported line
        numbers stay true. Prose about the old dialogs is not a call to them,
        and a /* */ continuation line has no marker of its own to match on."""
        def blank(match):
            return "\n" * match.group(0).count("\n")

        text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
        text = re.sub(r"<!--.*?-->", blank, text, flags=re.S)
        return re.sub(r"(?m)^\s*//.*$", "", text)

    offenders = {}
    for path in sorted((ROOT / "admin" / "templates").glob("*.html")):
        if path.name in ("client_graph.html", "base.html"):
            continue
        source = _blank_comments(path.read_text(encoding="utf-8"))
        for number, line in enumerate(source.splitlines(), 1):
            if "{{" in line or "{%" in line:      # the Jinja alert() macro
                continue
            if re.search(r"qb(Confirm|Toast|Alert)", line):
                continue
            if native.search(line):
                offenders.setdefault(path.name, []).append(number)
    assert not offenders, f"native browser dialogs remain: {offenders}"


def test_destructive_actions_are_confirmed_not_bare_submits():
    """Converting a form from onsubmit=confirm() to qbConfirm means giving the
    form an id and the button an onclick. Doing only the first half leaves the
    action firing with no confirmation at all -- worse than before."""
    guarded_ids = ("pu-block-", "pu-del-user-", "pu-del-req-", "revoke-att-",
                   "publish-v", "delete-kb-form")
    for path in sorted((ROOT / "admin" / "templates").glob("*.html")):
        source = path.read_text(encoding="utf-8")
        for form_id in guarded_ids:
            if f'id="{form_id}' not in source:
                continue
            # The confirming button must reference this form by id.
            assert f"getElementById('{form_id}" in source, (
                f"{path.name}: form '{form_id}' has no qbConfirm handler pointing at it"
            )


def test_the_compliance_tabs_show_which_panel_is_open():
    """Ten panels behind a tab bar with no active state, and no record of the
    choice, so a refresh silently returned to Profile."""
    source = _read("admin/templates/client_compliance.html")
    assert 'aria-selected' in source, "the compliance tabs expose no selected state"
    assert ".compliance-tab.active" in source, "the active tab is not styled"
    assert "location.hash" in source, (
        "the open panel is not reflected in the URL, so refresh and deep links lose it"
    )


def test_the_widget_wall_pages_state_what_they_are():
    """Both surfaces opened with no h1 at all -- the first strong text was a
    tool's label, which read as the page title."""
    for page in ("client_conflict_inbox.html", "client_learning_queue.html"):
        source = _read(f"admin/templates/{page}")
        assert "page_header(" in source, f"{page} has no page header"


# ── The relationship canvas lives inside the console ─────────────────────────

def test_the_relationship_page_is_not_its_own_document():
    """It shipped as a standalone <!DOCTYPE html> with its own head, its own
    theme script and no navigation, so opening Relationships teleported you out
    of the console and back was two small text links. Being outside the shell is
    also why it had no qbConfirm/qbToast to call."""
    source = _read("admin/templates/client_graph.html")
    assert source.lstrip().startswith("{% extends"), (
        "client_graph.html is a standalone document again"
    )
    for tag in ("<!DOCTYPE html>", "<html", "<head>", "<body"):
        # The explanatory comment may name them; the markup must not contain them.
        markup = re.sub(r"\{#.*?#\}", "", source, flags=re.S)
        assert tag not in markup, f"client_graph.html still emits {tag}"

    # And it must not re-declare stylesheets the shell already loads, which is
    # how it ended up serving a cache-busted version nothing else was on.
    assert "tokens.css" not in source, "the page links its own copy of tokens.css"


def test_the_canvas_palette_is_a_taxonomy_not_a_status():
    """fact / dimension / bridge are categories. Mapping them onto
    --warning/--primary/--violet would be a lie that happens to compile, so they
    get their own tokens."""
    tokens = _read("static/css/tokens.css")
    for name in ("--entity-fact", "--entity-dim", "--entity-bridge"):
        for suffix in ("", "-soft", "-line"):
            assert f"{name}{suffix}:" in tokens, f"{name}{suffix} is missing"


def test_cytoscape_gets_resolved_colours_not_var_references():
    """The canvas is painted with a 2D context, which cannot resolve var().
    A var() reaching cyStyle() renders as no colour at all, silently."""
    source = _read("admin/templates/client_graph.html")
    style_block = source[source.index("function cyStyle()"):]
    style_block = style_block[:style_block.index("function initCytoscape")]
    assert "var(--" not in style_block, (
        "cyStyle() passes a CSS variable to the canvas, which cannot resolve it"
    )
    assert "_tok(" in style_block, "cyStyle() no longer reads the design tokens"


def test_the_canvas_reads_its_palette_from_the_tokens():
    """Cytoscape paints to a canvas and cannot resolve var(), so the colours have
    to be read out of the stylesheet rather than written twice."""
    source = _read("admin/templates/client_graph.html")
    assert "qb-theme-change" not in source, (
        "a listener for an event that can no longer fire is dead code"
    )
    assert "_readTypePalette()" in source, "the palette is never read from the tokens"
    assert "getComputedStyle" in source, "the canvas hardcodes its colours again"


# One ground: the editor surface. There is no second theme to satisfy.
_SYNTAX_GROUND = "#FCFDFC"
_SYNTAX_TOKENS = ("--syntax-kw", "--syntax-fn", "--syntax-col",
                  "--syntax-str", "--syntax-num")


def _relative_luminance(hex_colour: str) -> float:
    hex_colour = hex_colour.lstrip("#")
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _token(block: str, name: str) -> str:
    match = re.search(rf"{name}:\s*(#[0-9a-fA-F]{{6}})", block)
    assert match, f"{name} missing"
    return match.group(1)


def test_sql_syntax_colours_are_legible_on_the_editor_ground():
    css = _strip_css_comments(_read("static/css/tokens.css"))
    failures = []
    for name in _SYNTAX_TOKENS:
        ratio = _contrast(_token(css, name), _SYNTAX_GROUND)
        if ratio < 4.5:
            failures.append(f"{name} {ratio:.2f}")
    assert not failures, (
        "syntax colours below the 4.5 text floor on the editor ground: " + str(failures)
    )


def test_sql_syntax_colours_stay_distinguishable_from_each_other():
    """A categorical set whose members converge stops doing its only job."""
    css = _strip_css_comments(_read("static/css/tokens.css"))
    values = {n: _token(css, n) for n in _SYNTAX_TOKENS}
    names = list(values)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra = [int(values[a][j:j + 2], 16) for j in (1, 3, 5)]
            rb = [int(values[b][j:j + 2], 16) for j in (1, 3, 5)]
            distance = sum((x - y) ** 2 for x, y in zip(ra, rb)) ** 0.5
            assert distance > 40, (
                f"{a} and {b} are {distance:.0f} apart and will read as the same "
                f"token kind"
            )


def test_the_editor_reads_its_syntax_colours_from_the_tokens():
    """A local hex here is how the dark-mode failure happened the first time."""
    metrics = _read("admin/templates/client_metrics.html")
    for name in _SYNTAX_TOKENS:
        assert f"var({name})" in metrics, f"{name} is not used by the editor"
    for rule in ("tok-kw", "tok-fn", "tok-col", "tok-str", "tok-num"):
        match = re.search(rf"\.{rule}\{{color:([^;}}]+)", metrics)
        assert match, rule
        assert match.group(1).startswith("var(--"), (
            f".{rule} hardcodes {match.group(1)} instead of using a token"
        )
