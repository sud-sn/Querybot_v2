"""
The type system is whole: the faces the tokens name are shipped, nothing is set
below the 12px floor, every weight is one of four, and tabular figures stay on
numbers.

Each of these failed on the product in a way a reader saw:

  * The face. Inter was named in the tokens with no @font-face anywhere, so
    every reader got Segoe UI or whatever system-ui was; the monospace face
    set the headings.
  * The weights. The old face stopped at 700, so a stylesheet could ask for
    800 or 900 and get a plain bold. Inter is variable and renders them as
    asked -- ExtraBold and Black headings, with hyphens that looked spaced.
  * Tabular figures. They were set on <body>. Inter's tabular feature swaps
    the space, hyphen, comma, full stop and brackets for figure-width glyphs,
    so every sentence was spaced out ("Date - like").
  * The floor. Captions, chart axes and legends were 10-11px.

The stylesheets, templates and chart scripts are read as data here; the result
table's numeric cells are checked by running the real renderDataTable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dukpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
FONTS_CSS = STATIC / "css" / "fonts.css"
TOKENS_CSS = STATIC / "css" / "tokens.css"
TYPE_SOURCES = ("fonts.css", "tokens.css")


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", text)


def _stylesheets() -> list[Path]:
    return sorted((STATIC / "css").glob("*.css"))


def _templates() -> list[Path]:
    return sorted(p for d in ("admin/templates", "portal/templates")
                  for p in (ROOT / d).rglob("*.html"))


def _scripts() -> list[Path]:
    return sorted(p for p in (STATIC / "js").glob("*.js") if not p.name.endswith(".min.js"))


def _styled_sources() -> list[Path]:
    return [p for p in _stylesheets() if p.name not in TYPE_SOURCES] + _templates()


def _font_faces() -> list[dict]:
    faces = []
    for block in re.findall(r"@font-face\s*\{([^}]*)\}", _strip_comments(FONTS_CSS.read_text(encoding="utf-8"))):
        family = re.search(r"font-family:\s*'([^']+)'", block).group(1)
        urls = re.findall(r"url\('([^']+)'\)", block)
        ranges = re.search(r"unicode-range:\s*([^;]+);", block)
        faces.append({"family": family, "urls": urls, "range": ranges.group(1) if ranges else ""})
    return faces


def _tokens() -> dict[str, str]:
    css = _strip_comments(TOKENS_CSS.read_text(encoding="utf-8"))
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", css))


def _first_family(stack: str) -> str:
    return stack.split(",")[0].strip().strip("'\"")


def _rules(text: str) -> list[tuple[str, str]]:
    """(selector list, declarations) for every plain rule in a stylesheet or
    a template's <style> blocks and style attributes."""
    text = _strip_comments(text)
    rules = [(sel.strip(), body) for sel, body in re.findall(r"([^{}@;]+)\{([^{}]*)\}", text)]
    rules += [("[style attribute]", body) for body in re.findall(r'style="([^"]*)"', text)]
    return rules


class TestTheFacesShip:

    def test_every_face_is_a_shipped_woff2(self):
        faces = _font_faces()
        assert {face["family"] for face in faces} == {"Inter", "JetBrains Mono"}
        for face in faces:
            for url in face["urls"]:
                path = ROOT / url.lstrip("/")
                assert path.is_file(), url
                assert path.read_bytes()[:4] == b"wOF2", url

    def test_each_family_ships_its_licence(self):
        for family in {face["family"] for face in _font_faces()}:
            licence = STATIC / "fonts" / f"LICENSE-{family.replace(' ', '')}.txt"
            assert licence.is_file(), licence.name
            assert "SIL OPEN FONT LICENSE" in licence.read_text(encoding="utf-8").upper()

    def test_each_family_covers_latin_for_english_and_french(self):
        for family in ("Inter", "JetBrains Mono"):
            ranges = " ".join(face["range"] for face in _font_faces() if face["family"] == family)
            assert "U+0000-00FF" in ranges, family

    @pytest.mark.parametrize("token,family", [
        ("--font-ui", "Inter"), ("--font-display", "Inter"), ("--font-mono", "JetBrains Mono"),
    ])
    def test_the_tokens_lead_with_a_shipped_face(self, token, family):
        assert _first_family(_tokens()[token]) == family

    def test_nothing_points_at_a_font_file_that_is_not_shipped(self):
        for source in _stylesheets() + _templates():
            for url in re.findall(r"/static/fonts/[\w.-]+", source.read_text(encoding="utf-8")):
                assert (ROOT / url.lstrip("/")).is_file(), f"{source.name}: {url}"

    @pytest.mark.parametrize("shell", ["admin/templates/base.html", "portal/templates/portal_base.html"])
    def test_the_preloaded_font_is_the_latin_face_the_page_uses(self, shell):
        page = (ROOT / shell).read_text(encoding="utf-8")
        preloads = re.findall(r'<link rel="preload" href="([^"]+)" as="font"', page)
        assert preloads, shell
        latin = {url for face in _font_faces() if "U+0000-00FF" in face["range"]
                 for url in face["urls"] if face["family"] == "Inter"}
        assert set(preloads) <= latin, preloads


class TestTheFloor:

    def test_no_size_token_is_below_12px(self):
        sizes = {k: v for k, v in _tokens().items() if re.fullmatch(r"--font-(\dxs|xs|sm|base|md|lg|\dxl|xl)", k)}
        assert len(sizes) >= 8
        for token, value in sizes.items():
            assert int(value.rstrip("px")) >= 12, token

    def test_no_stylesheet_or_template_sets_text_below_12px(self):
        small = []
        for source in _styled_sources():
            text = _strip_comments(source.read_text(encoding="utf-8"))
            for px in re.findall(r"font-size\s*:\s*(\d+(?:\.\d+)?)px", text):
                if 0 < float(px) < 12:
                    small.append(f"{source.name}: font-size {px}px")
            for px in re.findall(r"\bfont\s*:\s*(?:[a-z-]+\s+)*(\d+(?:\.\d+)?)px", text):
                if 0 < float(px) < 12:
                    small.append(f"{source.name}: font {px}px")
        assert not small, small

    def test_no_chart_text_is_below_12px(self):
        small = []
        for source in _scripts():
            text = source.read_text(encoding="utf-8")
            small += [f"{source.name}: fontSize {n}" for n in re.findall(r"fontSize\s*:\s*(\d+)\b", text) if int(n) < 12]
            small += [f"{source.name}: font-size {n}px" for n in re.findall(r"font-size:\s*(\d+)px", text) if int(n) < 12]
        assert not small, small


class TestTheWeights:

    def test_there_are_four_and_none_is_heavier_than_bold(self):
        weights = {k: int(v) for k, v in _tokens().items() if k.startswith("--weight-")}
        assert set(weights) == {"--weight-regular", "--weight-medium", "--weight-semibold", "--weight-bold"}
        assert weights["--weight-regular"] < weights["--weight-medium"] < weights["--weight-semibold"] < weights["--weight-bold"] <= 700

    def test_every_stylesheet_and_template_names_a_weight_token(self):
        literal = []
        for source in _styled_sources():
            text = _strip_comments(source.read_text(encoding="utf-8"))
            for value in re.findall(r"font-weight\s*:\s*(\$\{[^}]*\}|[^;}\"']+)", text):
                # A weight chosen in a template literal: each alternative counts.
                choices = re.findall(r"'([^']*)'", value) if value.startswith("${") else [value.strip()]
                for choice in choices:
                    if not (choice.startswith("var(--weight-") or choice in ("inherit", "normal")):
                        literal.append(f"{source.name}: {choice}")
        assert not literal, literal[:20]

    def test_a_weight_set_from_script_is_a_token(self):
        literal = []
        for source in _templates() + _scripts():
            for value in re.findall(r"\.style\.fontWeight\s*=\s*([^;]+);", source.read_text(encoding="utf-8")):
                for quoted in re.findall(r"'([^']*)'", value):
                    if quoted and not quoted.startswith("var(--weight-"):
                        literal.append(f"{source.name}: {quoted}")
        assert not literal, literal

    def test_chart_text_is_never_heavier_than_bold(self):
        bold = int(_tokens()["--weight-bold"])
        for source in _scripts():
            for n in re.findall(r"fontWeight\s*:\s*(\d+)", source.read_text(encoding="utf-8")):
                assert int(n) <= bold, f"{source.name}: fontWeight {n}"


# Inter's tabular feature replaces these with figure-width glyphs, not only the
# digits: set on anything that holds words, it spaces the words out.
_PROSE_CONTAINERS = {"html", "body", "main", "section", "article", "div", "p", "span", "li",
                     "table", "thead", "tbody", "tr", "td", "th", ".data-table", ".rc-table"}


def _last_compound(selector: str) -> str:
    compound = re.split(r"[\s>+~]+", selector.strip())[-1]
    return re.sub(r"::?[\w-]+(\([^)]*\))?", "", compound)


class TestTabularFiguresStayOnNumbers:

    def test_no_rule_sets_them_on_a_container_of_words(self):
        offenders = []
        for source in _stylesheets() + _templates():
            for selectors, body in _rules(source.read_text(encoding="utf-8")):
                if not re.search(r"tabular-nums|'tnum'", body):
                    continue
                for selector in selectors.split(","):
                    if _last_compound(selector) in _PROSE_CONTAINERS:
                        offenders.append(f"{source.name}: {selector.strip()}")
        assert not offenders, offenders

    def test_the_num_class_sets_them(self):
        base = _strip_comments((STATIC / "css" / "base.css").read_text(encoding="utf-8"))
        bodies = [body for selectors, body in _rules(base)
                  if "td.num" in [s.strip() for s in selectors.split(",")]]
        assert any("tabular-nums" in body for body in bodies)


_TABLE_STUBS = """
var window = {qbNum: function (n) { return String(n); }};
var document = {};
function escHtml(s) { return String(s == null ? '' : s); }
function _normaliseColumnKey(k) { return String(k || '').toLowerCase(); }
function _columnFormatMap() { return new Map(); }
function _displayFormatSpec() { return {}; }
function _formatDisplayValue(v) { return String(v == null ? '' : v); }
function _parseDisplayNumber(v) { var n = Number(v); return v === '' || v == null ? NaN : n; }
function t(id) { return id; }
function plural(id, n, v) { return String(n); }
var _dtIdCounter = 0;
"""


class TestTheResultTableMarksItsNumbers:
    """The table that answers a question: its number columns take tabular
    figures and align right; its word columns stay proportional."""

    def render(self, rows):
        from tests.js_lift import function as lift

        page = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        data = {"headers": list(rows[0]), "rows": rows}
        return dukpy.evaljs("\n".join([
            _TABLE_STUBS, lift(page, "function renderDataTable(data,"),
            f"renderDataTable({json.dumps(data)})",
        ]))

    def cells(self, html):
        return re.findall(r"<td( class=\"num\")?>([^<]*)</td>", html)

    def test_a_number_column_is_marked_and_a_word_column_is_not(self):
        html = self.render([{"Region": "North-East", "Net Sales": 1204332.5},
                            {"Region": "South", "Net Sales": 812000}])
        assert self.cells(html) == [("", "North-East"), (' class="num"', "1204332.5"),
                                    ("", "South"), (' class="num"', "812000")]

    def test_a_column_of_labels_that_look_numeric_somewhere_is_still_read_by_value(self):
        html = self.render([{"Code": "A-1", "Units": "12"}, {"Code": "B-2", "Units": "7"}])
        assert self.cells(html) == [("", "A-1"), (' class="num"', "12"), ("", "B-2"), (' class="num"', "7")]
