"""
One badge, one status pill, one set of tones.

Badges were five colour classes in base.css plus page after page of its own:
three status colours on the date-role page, "default" and "inactive" badges on
reports, source badges and inline-styled red and amber ones on the glossary,
confidence and review pills twice over on the portal's knowledge page. The
role badge on the users page asked for "badge-indigo", which only that page
defined. Several pairs put a mid-tone ink on its own tint, under 4.5:1 for
12px text.

Now base.css holds the badge and the status pill (a badge with a dot), and one
set of tones -- neutral, brand, info, success, warning, danger, accent -- each
setting only --badge-* properties from tokens. The colour names older pages
use are aliases of those tones. These read the stylesheet as data, compute
each tone's contrast from tokens.css, and render the badge macros.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"


def _strip(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", css)


def _tokens() -> dict[str, str]:
    css = _strip((CSS / "tokens.css").read_text(encoding="utf-8"))
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", css)}


def _block() -> str:
    css = (CSS / "base.css").read_text(encoding="utf-8")
    start = css.index("/* ── Badges")
    return css[start:css.index(".chip-row,", start)]


def _tone_rules() -> dict[str, dict[str, str]]:
    """selector -> its --badge-* properties, one entry per selector in a list."""
    rules: dict[str, dict[str, str]] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(_block())):
        props = dict(re.findall(r"(--badge-[\w-]+):\s*([^;]+);", body))
        for one in selector.split(","):
            rules.setdefault(one.strip(), {}).update(props)
    return rules


def _lum(hexcolour: str) -> float:
    h = hexcolour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


TONES = {  # tone -> the selectors that must give it, semantic name first
    "neutral": [".badge-neutral", ".badge-gray", ".status-pill.neutral"],
    "brand": [".badge-brand", ".badge-blue", ".status-pill.brand"],
    "info": [".badge-info", ".status-pill.info"],
    "success": [".badge-success", ".badge-green", ".status-pill.ok", ".status-pill.success"],
    "warning": [".badge-warning", ".badge-amber", ".status-pill.warning"],
    "danger": [".badge-danger", ".badge-red", ".status-pill.danger", ".status-pill.error"],
    "accent": [".badge-accent", ".badge-violet", ".badge-indigo", ".status-pill.accent"],
}


def _resolved(selector: str) -> dict[str, str]:
    tokens = _tokens()
    rules = _tone_rules()
    assert selector in rules, f"{selector} is not defined"
    props = dict(rules[".badge"])
    props.update(rules[selector])

    def value(expr: str) -> str:
        m = re.fullmatch(r"var\((--[\w-]+)\)", expr.strip())
        assert m and m.group(1) in tokens, f"{selector}: {expr} is not a colour token"
        return tokens[m.group(1)]

    return {k: value(v) for k, v in props.items()}


class TestEveryToneReads:

    @pytest.mark.parametrize("tone", sorted(TONES))
    def test_its_word_reads_on_its_fill(self, tone):
        c = _resolved(TONES[tone][0])
        assert _ratio(c["--badge-ink"], c["--badge-bg"]) >= 4.5, (tone, c)

    @pytest.mark.parametrize("tone", sorted(TONES))
    def test_every_name_for_it_gives_the_same_colours(self, tone):
        first = _resolved(TONES[tone][0])
        for selector in TONES[tone][1:]:
            assert _resolved(selector) == first, (tone, selector)

    def test_the_tones_are_distinct(self):
        fills = {tone: _resolved(names[0])["--badge-bg"] for tone, names in TONES.items()}
        assert len(set(fills.values())) == len(fills), fills

    def test_the_block_is_drawn_from_tokens(self):
        literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\b(?:white|black)\b(?!-)", _strip(_block()))
        assert not literals, literals


def _markup(path: Path) -> str:
    return re.sub(r"<style[^>]*>[\s\S]*?</style>", "", path.read_text(encoding="utf-8"))


def _pages():
    return list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html"))


def test_every_badge_class_a_page_asks_for_exists():
    defined = {m for sel in _tone_rules() for m in re.findall(r"\.(badge-[\w-]+)", sel)}
    asked = {}
    for page in _pages():
        for name in re.findall(r"(?<![\w$-])badge-([a-z]+)(?![\w-])", _markup(page)):
            asked.setdefault(f"badge-{name}", page.name)
    missing = {cls: page for cls, page in asked.items() if cls not in defined}
    assert asked and not missing, missing


@pytest.mark.parametrize("console", ["admin", "portal"])
@pytest.mark.parametrize("colour", ["blue", "green", "amber", "red", "gray"])
def test_the_badge_macro_gives_a_defined_tone(console, colour):
    import admin.routes
    import portal.routes

    env = (admin.routes if console == "admin" else portal.routes).templates.env
    html = env.from_string('{% from "macros.html" import badge %}{{ badge("Ready", colour) }}').render(colour=colour)
    classes = re.search(r'<span class="([^"]+)">Ready</span>', html).group(1).split()
    assert classes[0] == "badge" and _resolved("." + classes[1])


# Badges that are their own thing (a chart-type tag, a live counter) or on a
# page a later phase redoes. None may be added.
SPECIAL_BADGES = {
    "_styles.html": {"cat-badge", "filter-chip", "fn-db-badge", "format-preview-pill", "mc-fx-badge", "mc-mode-tag",
                     "mc-type-badge", "meta-pill", "mp-badge", "syn-pill"},                            # metrics, 6
    "admin.css": {"admin-live-badge", "cs-badge", "inbox-chip", "inbox-chip-n", "semantic-review-chip"},  # shell, 2.6
    "chat_workspace.css": {"hp-badge", "status-badge", "suggestion-chip"},                                # chat, 6
    "client_compliance.html": {"check-pill", "tag"},                                                     # 6
    "client_domains.html": {"chip"},                                                                     # 6
    "client_glossary.html": {"alias-pill"},                                                               # 6
    "client_graph.html": {"bp-tab-badge", "join-type-badge", "review-badge", "rt-scope-pill", "rv-chip",
                          "sb-jbadge", "sb-type-pill", "type-pill"},                                     # 4
    "client_mapping.html": {"mp-chip"},                                                                  # 6
    "client_setup.html": {"mask-item-badge", "status-badge", "td-badge--described", "td-badge--new",
                          "td-badge--undescribed"},                                                      # 6
    "dashboard.css": {"table-pill"},                                                                     # 5
    "portal_chat.html": {"chat-pill", "chat-shell-badge", "citation-chip", "composer-chip", "follow-up-chip",
                         "hp-badge", "query-kpi-pill", "rc-scope-chip", "rc-source-badge", "scope-badge",
                         "status-badge", "suggestion-chip", "token-kpi-pill", "trust-pill"},             # chat, 6
    "portal_kb.html": {"term-chip"},                                                                     # 6
    "production.css": {"chat-shell-badge", "confidence-badge"},                                          # chat, 6
    "system.html": {"az-badge"},                                                                         # 6
}


def test_no_page_adds_a_badge_of_its_own():
    own: dict[str, set[str]] = {}
    sources = [(p.name, "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", p.read_text(encoding="utf-8"))))
               for p in _pages()]
    sources += [(p.name, p.read_text(encoding="utf-8")) for p in CSS.glob("*.css") if p.name != "base.css"]
    for name, css in sources:
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css)):
            # A badge is a fill or an edge; text colour alone, or "border: none",
            # is not one.
            if not re.search(r"(?:^|[;\s])(?:background(?:-color)?|border(?:-color)?)\s*:\s*(?!none\b|0\b|transparent\b)",
                             " " + body):
                continue
            for cls in re.findall(r"\.([\w-]*(?:badge|pill|chip)[\w-]*)", selector):
                own.setdefault(name, set()).add(cls)
    new = {name: sorted(classes - SPECIAL_BADGES.get(name, set())) for name, classes in own.items()}
    assert not {k: v for k, v in new.items() if v}, new
