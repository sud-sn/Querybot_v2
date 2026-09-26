"""
One button, for both consoles.

Buttons were defined three times over: once in base.css, again further down
base.css (the loading spinner, a success hover in a literal green, a warning
hover mixed with black), and overridden by production.css, which loaded after
every page's own styles. Pages then built their own: a filled green "approve",
two copies of an outlined approve/reject pair, a blue fetch button with its own
white spinner, a restart button in a literal brown. The alert macro asked for a
"btn-info" that no stylesheet defined. And each console had its own script
that span a submitted form's first button -- not the one pressed -- and
disabled it, which drops a pressed button's name=value from what the form
sends and left the button spinning for 15 s after a page's own check stopped
the submit.

Now base.css holds the one button: a variant sets only its colours, as tokens,
and every variant has every state. qb-ui.js marks the pressed button busy for
both consoles. These read the stylesheet as data (every variant's ink against
its fill, at rest and on hover, computed from tokens.css), run the real
qb-ui.js in dukpy for the busy state, and render the real macros.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM

ROOT = Path(__file__).resolve().parents[1]
UI_JS = ROOT / "static" / "js" / "qb-ui.js"
CSS = ROOT / "static" / "css"


def _strip(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", css)


def _tokens() -> dict[str, str]:
    css = _strip((CSS / "tokens.css").read_text(encoding="utf-8"))
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", css)}


def _button_block() -> str:
    css = (CSS / "base.css").read_text(encoding="utf-8")
    start = css.index("/* ── Buttons")
    return css[start:css.index("\n/* ── ", start)]   # to the next section


def _rules(block: str) -> dict[str, dict[str, str]]:
    rules: dict[str, dict[str, str]] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(block)):
        props = dict(re.findall(r"(--btn-[\w-]+):\s*([^;]+);", body))
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


# Every variant, the selector that sets it, and the ground a see-through
# button sits on.
VARIANTS = {
    "secondary": (".btn", "--surface"),
    "primary": (".btn-primary", "--surface"),
    "ghost": (".btn-ghost", "--surface"),
    "link": (".btn-link", "--surface"),
    "danger": (".btn-danger", "--surface"),
    "success": (".btn-success", "--surface"),
    "warning": (".btn-warning", "--surface"),
    "soft-success": (".btn-soft-success", "--surface"),
    "soft-danger": (".btn-soft-danger", "--surface"),
    "soft-warning": (".btn-soft-warning", "--surface"),
    "shell": (".btn-shell", "--shell"),
    "pressed": ('.btn[aria-pressed="true"]', "--surface"),
}


def _resolved(variant: str) -> dict[str, str]:
    """The variant's --btn-* colours as hex, the way the browser resolves
    them: .btn's defaults, the variant's own on top, var() followed through
    the --btn-* properties and then tokens.css."""
    tokens = _tokens()
    rules = _rules(_button_block())
    selector, ground = VARIANTS[variant]
    props = dict(rules[".btn"])
    props.update(rules.get(selector, {}))

    def value(expr: str, seen: tuple = ()) -> str:
        expr = expr.strip()
        if expr == "transparent":
            return tokens[ground]
        m = re.fullmatch(r"var\((--[\w-]+)\)", expr)
        assert m, f"{variant}: {expr} is not a token"
        name = m.group(1)
        assert name not in seen, f"{variant}: {name} refers to itself"
        if name in props:
            return value(props[name], seen + (name,))
        assert name in tokens, f"{variant}: {name} is not in tokens.css"
        return tokens[name]

    return {k: value(v) for k, v in props.items()}


class TestEveryVariantReads:

    @pytest.mark.parametrize("variant", sorted(VARIANTS))
    def test_its_label_reads_on_its_fill_at_rest_and_on_hover(self, variant):
        c = _resolved(variant)
        assert _ratio(c["--btn-ink"], c["--btn-bg"]) >= 4.5, (variant, c["--btn-ink"], c["--btn-bg"])
        assert _ratio(c["--btn-ink-hover"], c["--btn-bg-hover"]) >= 4.5, \
            (variant, c["--btn-ink-hover"], c["--btn-bg-hover"])

    def test_a_filled_status_button_is_not_the_primary(self):
        # Delete must not look like Save.
        fills = {v: _resolved(v)["--btn-bg"] for v in ("primary", "danger", "success", "warning")}
        assert len(set(fills.values())) == 4, fills

    def test_the_button_is_drawn_from_tokens_only(self):
        block = _strip(_button_block())
        literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\b(?:white|black)\b(?!-)", block)
        assert not literals, literals


def _page_styles():
    for page in list((ROOT / "admin" / "templates").rglob("*.html")) + \
            list((ROOT / "portal" / "templates").rglob("*.html")):
        yield page.name, "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", page.read_text(encoding="utf-8")))
    for sheet in CSS.glob("*.css"):
        if sheet.name != "base.css":
            yield sheet.name, sheet.read_text(encoding="utf-8")


_COLOUR_PROPS = r"(?:^|[;\s{])(?:background(?:-color)?|color|border(?:-color)?|box-shadow|filter)\s*:"


def _colouring_rules():
    for name, css in _page_styles():
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css)):
            if re.search(_COLOUR_PROPS, " " + body):
                yield name, selector.strip(), body.strip()


def test_no_page_or_later_stylesheet_recolours_the_button():
    # admin.css, a page's <style>: layout (width, margins)
    # is theirs to set, colour is the button's.
    found = [f"{name}: {selector}" for name, selector, _ in _colouring_rules()
             if re.search(r"\.btn(?:-[\w-]+)?(?![\w-])", selector)]
    assert not found, found


# Buttons that are their own control, not an action button, still styled by
# their page. Each is taken in by the phase that redoes its page; none may be
# added.
SPECIAL_BUTTONS = {
    "_styles.html": {"fn-helper-btn", "formula-test-btn", "mr-del-btn"},        # metrics, phase 6
    "admin.css": {"mobile-menu-btn"},                                           # admin shell, 6
    "chat_workspace.css": {"send-btn"},                                         # chat, phase 6
    "portal_chat.html": {"artifact-head-btn", "chart-btn", "ctt-btn", "dt-csv-btn", "feedback-btn", "hero-btn",
                         "hero-dismiss-btn", "hero-info-btn", "history-btn", "rc-csv-btn", "send-btn",
                         "sql-copy-btn"},                                        # chat, phase 6
    "client_detail.html": {"audit-expand-btn"},                                  # phase 6
    "client_graph.html": {"eg-m-btn", "jc-add-btn", "sf-btn", "tb-btn"},         # relationships, phase 4
    "client_setup.html": {"tbl-fields-btn"},                                     # phase 6
    "dashboard.css": {"dctt-btn"},                                               # dashboards, phase 5
    "login.html": {"input-reveal-btn"},                                          # fields, 2.5
    "portal_login.html": {"input-reveal-btn"},
    "portal_change_password.html": {"input-reveal-btn"},
}


def test_no_page_adds_a_button_of_its_own():
    own: dict[str, set[str]] = {}
    for name, selector, _ in _colouring_rules():
        for cls in re.findall(r"\.([\w-]*btn[\w-]*)", selector):
            if not re.fullmatch(r"btn(?:-[\w-]+)?", cls):
                own.setdefault(name, set()).add(cls)
    new = {name: sorted(classes - SPECIAL_BUTTONS.get(name, set())) for name, classes in own.items()}
    assert not {k: v for k, v in new.items() if v}, new


def _classes_in_markup() -> set[str]:
    found: set[str] = set()
    sources = list((ROOT / "admin" / "templates").rglob("*.html")) + \
        list((ROOT / "portal" / "templates").rglob("*.html")) + list((ROOT / "static" / "js").glob("*.js"))
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for attr in re.findall(r"""class(?:Name)?\s*=\s*["'`]([^"'`]*)""", text):
            attr = re.sub(r"\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}", " ", attr)
            found |= {c for c in attr.split() if re.fullmatch(r"btn-[a-z][\w-]*", c)}
    return found


def test_every_button_class_a_page_asks_for_exists():
    rules = _rules(_button_block())
    defined = {m for sel in rules for m in re.findall(r"\.(btn-[\w-]+)", sel)}
    assert {"btn-primary", "btn-soft-danger", "btn-shell", "btn-icon"} <= defined
    missing = sorted(_classes_in_markup() - defined)
    assert not missing, missing


@pytest.mark.parametrize("console", ["admin", "portal"])
@pytest.mark.parametrize("tone,variant", [("info", "btn-secondary"), ("amber", "btn-warning"),
                                          ("error", "btn-danger"), ("success", "btn-success"),
                                          ("blue", "btn-secondary")])
def test_an_alerts_action_is_a_button_that_exists(console, tone, variant):
    import admin.routes
    import portal.routes

    env = (admin.routes if console == "admin" else portal.routes).templates.env
    html = env.from_string('{% from "macros.html" import alert %}'
                           '{{ alert("Check it", type=tone, action_href="/x", action_label="Go") }}').render(tone=tone)
    classes = re.search(r'<a href="/x" class="([^"]+)"', html).group(1).split()
    assert variant in classes, classes
    defined = {m for sel in _rules(_button_block()) for m in re.findall(r"\.(btn-[\w-]+)", sel)}
    assert all(c in defined for c in classes if c.startswith("btn-")), classes


# ── A submitted form's button says it is working ────────────────────────────

_FORM = """
var form = make('form', {'method': 'post'});
var field = make('input', {'name': 'q'}, form);
var approve = make('button', {'type': 'submit', 'name': 'action', 'value': 'approve'}, form); approve.form = form;
var reject = make('button', {'type': 'submit', 'name': 'action', 'value': 'reject'}, form); reject.form = form;
function submit(submitter, pageStops) {
  var ev = {target: form, submitter: submitter || null, defaultPrevented: false};
  ev.preventDefault = function () { ev.defaultPrevented = true; };
  (docListeners.submit || []).forEach(function (f) { f(ev); });
  if (pageStops) ev.preventDefault();       // a page's own listener, after ours
  return ev;
}
function tick(ms) {
  timers.filter(function (t) { return !t.ran && !t.cleared && t.ms <= ms; })
        .forEach(function (t) { t.ran = true; t.f(); });
}
function state() {
  return {approve: approve.getAttribute('aria-busy'), reject: reject.getAttribute('aria-busy'),
          form: form.getAttribute('data-qb-submitting'), disabled: !!(approve.disabled || reject.disabled)};
}
"""


def _submit(script: str) -> object:
    return json.loads(dukpy.evaljs(FAKE_DOM + UI_JS.read_text(encoding="utf-8") + "\n" + _FORM + script))


class TestBusyOnSubmit:

    def test_the_pressed_button_spins_and_stays_enabled(self):
        out = _submit("submit(reject); tick(0); JSON.stringify(state());")
        assert out == {"approve": None, "reject": "true", "form": "1", "disabled": False}

    def test_without_a_submitter_the_first_submit_button_spins(self):
        out = _submit("submit(null); tick(0); JSON.stringify(state());")
        assert out["approve"] == "true" and out["reject"] is None

    def test_a_submit_the_page_stopped_is_not_in_flight(self):
        out = _submit("submit(approve, true); tick(0); JSON.stringify(state());")
        assert out == {"approve": None, "reject": None, "form": None, "disabled": False}

    def test_a_second_submit_while_in_flight_is_held(self):
        out = _submit("submit(approve); tick(0); var again = submit(reject);"
                      "JSON.stringify({held: again.defaultPrevented, s: state()});")
        assert out["held"] is True and out["s"]["reject"] is None

    def test_it_lets_go_after_fifteen_seconds(self):
        out = _submit("""
          submit(approve); tick(0);
          var delays = timers.map(function (t) { return t.ms; });
          tick(15000);
          var again = submit(approve);
          JSON.stringify({delays: delays, s: state(), held: again.defaultPrevented});
        """)
        assert 15000 in out["delays"]
        assert out["s"]["form"] is None and out["held"] is False

    def test_coming_back_to_a_kept_page_lets_go(self):
        out = _submit("""
          submit(approve); tick(0);
          (winListeners.pageshow || []).forEach(function (f) { f({persisted: false}); });
          var still = state();
          (winListeners.pageshow || []).forEach(function (f) { f({persisted: true}); });
          JSON.stringify({still: still, after: state()});
        """)
        assert out["still"]["approve"] == "true"
        assert out["after"] == {"approve": None, "reject": None, "form": None, "disabled": False}

    def test_a_form_or_button_can_opt_out(self):
        out = _submit("""
          reject.setAttribute('data-no-loading', '');
          submit(reject); tick(0); var button = state();
          form.removeAttribute('data-qb-submitting');
          form.setAttribute('data-no-loading', '');
          submit(approve); tick(0);
          JSON.stringify({button: button, form: state()});
        """)
        assert out["button"] == {"approve": None, "reject": None, "form": "1", "disabled": False}
        assert out["form"]["approve"] is None and out["form"]["form"] is None

    @pytest.mark.parametrize("shell", ["admin/templates/base.html", "portal/templates/portal_base.html"])
    def test_neither_shell_keeps_a_script_of_its_own(self, shell):
        page = (ROOT / shell).read_text(encoding="utf-8")
        assert "addEventListener('submit'" not in page and "classList.add('loading')" not in page
