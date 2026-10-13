"""Every dialog, tab, button and toggle in the approved design, and a send button that is a drawn icon.

After the pages took the approved design, the controls on them kept the earlier one: a dialog was a small square box
with a grey head and a grey footer band (Add to dashboard, Share, a schedule, an alert, the admin's Add person, the
metric editor, the Data guide's suggestion); tabs were an underline and a chart's shapes a row of loose chips; the
switches were two sizes and green when on; buttons were small squares; and the chat's send button was a typed "↑".

Now there is one vocabulary, the design's:
- A dialog sits on the card radius with a quiet edge, its title in the display face, no grey band under its buttons;
  it rises in over a blurred page.
- Tabs, a set of choices and an area's menu are one segmented control: a pill track, the chosen one raised on it.
- A switch is 40 by 24, its thumb springing across, the brand's colour when on; a tick box and a round choice take
  the same colour.
- Buttons are pills that give a little under a press; the main one glows softly in its own colour.
- The send button is a round icon button: quiet while the box is empty, filled once there is something to send, the
  arrow lifting under the pointer; while an answer is worked out it is a stop square on red with a ring running
  round it, the question box's edge runs in the brand's colours and the stage being worked on shimmers.

And one thing the page entrance broke, which stays fixed: the entrance kept its transform, so a fixed overlay inside
the page (the metric editor's backdrop) stopped at the page's edge instead of covering the window.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"


def _source(path: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".html"):
        text = "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", text))
    return text


def _rule(path: str, selector: str) -> str:
    """The declarations of every rule (outside @media) whose selector list holds `selector`, joined in order."""
    css = re.sub(r"/\*[\s\S]*?\*/", "", _source(path))
    css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    found = [body for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
             if selector in [s.strip() for s in sel.split(",")]]
    assert found, f"no rule for {selector} in {path}"
    return ";".join(found)


# ── dialogs ─────────────────────────────────────────────────────────────────

DIALOGS = [
    # (where, the dialog, its title, its footer or None)
    ("static/css/base.css", ".qb-dialog", ".qb-dialog-title", ".qb-dialog-footer"),
    ("portal/templates/portal_chat.html", ".dashboard-picker", ".dashboard-picker-title", ".dashboard-picker-footer"),
    ("static/css/dashboard.css", ".dash-dialog", ".dash-dialog-head h2", None),
    ("portal/templates/portal_kb.html", ".ask-dialog", ".ask-dialog-head h2", None),
    ("admin/templates/client_users.html", ".pp-dialog", ".pp-dialog h3", None),
    ("admin/templates/client_graph.html", ".eg-modal", ".eg-modal-head", None),
    ("admin/templates/metrics/_styles.html", "#mc-dialog", ".mc-dialog-hd h2", ".mc-footer"),
]


@pytest.mark.parametrize("where,dialog,title,footer", DIALOGS, ids=[d[1] for d in DIALOGS])
def test_every_dialog_is_the_design_s(where, dialog, title, footer):
    box = _rule(where, dialog)
    assert "var(--radius-card)" in box, "the card radius"
    assert "var(--line-subtle)" in box, "a quiet edge"
    assert "qb-modalIn" in box, "it rises in"
    head = _rule(where, title)
    assert "var(--font-display)" in head and "var(--font-xl)" in head, "its title in the display face"
    if footer:
        foot = _rule(where, footer)
        assert "var(--surface-2)" not in foot and "linear-gradient" not in foot, "no grey band under the buttons"


@pytest.mark.parametrize("where,backdrop", [
    ("static/css/base.css", ".qb-dialog::backdrop"), ("static/css/dashboard.css", ".dash-dialog::backdrop"),
    ("portal/templates/portal_kb.html", ".ask-dialog::backdrop"), ("admin/templates/client_users.html", ".pp-dialog::backdrop"),
    ("admin/templates/client_graph.html", ".eg-modal-overlay"), ("admin/templates/metrics/_styles.html", "#mc-overlay"),
    ("portal/templates/portal_chat.html", ".dashboard-picker-backdrop")])
def test_a_dialog_blurs_the_page_behind_it(where, backdrop):
    rule = _rule(where, backdrop)
    assert "var(--scrim)" in rule and "blur(4px)" in rule


def test_a_long_dialog_s_buttons_sit_on_its_own_padding():
    """The Add person form scrolls on a phone; its buttons stay at the foot. They were pinned 20px below the box's
    edge, which once matched its padding: at the design's 22px a strip of the form showed under them."""
    padding = re.search(r"padding:\s*(\d+)px", _rule("admin/templates/client_users.html", ".pp-dialog")).group(1)
    actions = _rule("admin/templates/client_users.html", ".pp-dialog .form-actions")
    assert f"bottom: -{padding}px" in actions and f"0 -{padding}px" in actions and f"{padding}px; margin" in actions


def test_the_mask_preview_is_the_design_s_dialog_too():
    rule = _rule("admin/templates/client_setup.html", ".mask-preview-modal")
    assert "var(--radius-card)" in rule and "var(--line-subtle)" in rule


# ── tabs and sets of choices ────────────────────────────────────────────────

SEGMENTED = [
    ("static/css/base.css", ".qb-tabs"),
    ("static/css/base.css", ".btn-seg"),
    ("portal/templates/portal_chat.html", ".ctt-seg"),
    ("portal/templates/portal_chat.html", ".dashboard-picker-switch"),
    ("static/css/dashboard.css", ".dctt-seg"),
    ("static/css/admin.css", ".client-workspace-secondary"),
]


@pytest.mark.parametrize("where,track", SEGMENTED, ids=[s[1] for s in SEGMENTED])
def test_every_set_of_choices_is_one_segmented_control(where, track):
    rule = _rule(where, track)
    assert "var(--radius-pill)" in rule and "var(--surface-alt)" in rule and "var(--line-subtle)" in rule


@pytest.mark.parametrize("where,chosen", [
    ("static/css/base.css", '.qb-tab[aria-selected="true"]'),
    ("static/css/base.css", '.btn-seg > .btn[aria-pressed="true"]'),
    ("portal/templates/portal_chat.html", ".ctt-btn.ctt-active"),
    ("portal/templates/portal_chat.html", ".dashboard-picker-switch button.active"),
    ("static/css/dashboard.css", ".dctt-btn.dctt-active"),
    ("static/css/admin.css", ".client-workspace-secondary a.active")])
def test_the_chosen_one_is_raised_on_the_track(where, chosen):
    rule = _rule(where, chosen)
    assert "var(--surface-raised)" in rule and "var(--shadow-sm)" in rule
    assert "var(--primary-fill)" not in rule, "not a filled chip"


def test_tabs_are_no_longer_an_underline():
    assert "border-bottom" not in _rule("static/css/base.css", ".qb-tab")
    assert "border-bottom" not in _rule("static/css/base.css", ".qb-tabs")


def test_the_answer_s_view_and_shape_switches_use_the_segmented_control():
    source = (ROOT / "portal/templates/portal_chat.html").read_text(encoding="utf-8")
    assert source.count('class="answer-seg btn-seg') == 2
    assert '<span class="ctt-seg">${typeHtml}</span>' in source
    assert '<span class="dctt-seg">${typeHtml}</span>' in (ROOT / "portal/templates/portal_dashboard.html").read_text(encoding="utf-8")


# ── switches, tick boxes and round choices ──────────────────────────────────

def test_one_switch_size_and_one_colour_for_on():
    for track, thumb, on in ((".form-switch-track", ".form-switch-track::after",
                              ".form-switch input:checked + .form-switch-track"),
                             (".qb-switch-track", ".qb-switch-thumb", ".qb-switch.on .qb-switch-track")):
        rule = _rule("static/css/base.css", track)
        assert "width: 40px" in rule and "height: 24px" in rule, track
        knob = _rule("static/css/base.css", thumb)
        assert "width: 18px" in knob and "height: 18px" in knob and "cubic-bezier(.34, 1.56, .64, 1)" in knob, thumb
        assert "var(--primary-fill)" in _rule("static/css/base.css", on), on
    assert "var(--success)" not in _rule("static/css/base.css", ".qb-switch.on .qb-switch-track")


def test_a_tick_box_and_a_round_choice_take_the_brand_s_colour():
    assert "accent-color: var(--primary-fill)" in _rule("static/css/base.css", 'input[type="checkbox"]')
    assert "accent-color: var(--primary-fill)" in _rule("static/css/base.css", 'input[type="radio"]')


# ── buttons ─────────────────────────────────────────────────────────────────

def test_buttons_are_pills_that_give_under_a_press():
    assert "var(--radius-pill)" in _rule("static/css/base.css", ".btn")
    pressed = _rule("static/css/base.css", '.btn:active:not(:disabled):not([aria-disabled="true"])')
    assert "scale(.97)" in pressed
    assert "box-shadow" in _rule("static/css/base.css", ".btn-primary"), "the main action glows in its own colour"


# ── the send button ─────────────────────────────────────────────────────────

CHAT = ROOT / "portal/templates/portal_chat.html"
WORKSPACE = "static/css/chat_workspace.css"


def test_the_send_button_is_a_drawn_icon():
    source = CHAT.read_text(encoding="utf-8")
    button = source[source.index('<button class="send-btn"'):]
    button = button[:button.index("</button>")]
    assert '{{ ic("arrow-up", 20) }}' in button and "↑" not in button
    assert 'data-empty="true"' in button, "quiet until there is something to send"
    assert "icon.textContent = isStop ? '' : '↑'" not in source
    assert "qbIcon('arrow-up', 20)" in source
    assert "send.dataset.empty = inp.value.trim() ? 'false' : 'true'" in source


def test_the_send_button_s_three_states():
    ready = _rule(WORKSPACE, ".chat-workspace-main .send-btn")
    assert "border-radius: 50%" in ready and "width: 44px" in ready and "var(--primary-fill)" in ready
    assert "translateY(-2px)" in _rule(WORKSPACE, ".chat-workspace-main .send-btn:hover .qb-icon"), "the arrow lifts"
    assert "scale(.92)" in _rule(WORKSPACE, ".chat-workspace-main .send-btn:active")
    empty = _rule(WORKSPACE, '.chat-workspace-main .send-btn[data-empty="true"]:not(.is-stop)')
    assert "var(--text-muted)" in empty and "box-shadow: none" in empty
    stop = _rule(WORKSPACE, ".chat-workspace-main .send-btn.is-stop")
    assert "var(--danger)" in stop
    ring = _rule(WORKSPACE, ".chat-workspace-main .send-btn.is-stop::after")
    assert "qb-spin" in ring and "pointer-events: none" in ring
    assert "currentColor" in _rule(WORKSPACE, ".chat-workspace-main .send-btn .stop-icon")


def test_while_an_answer_is_worked_out_the_box_and_the_stage_move():
    source = CHAT.read_text(encoding="utf-8")
    assert "document.querySelector('.chat-composer')?.classList.toggle('is-working', !!active);" in source
    assert "qb-edge-run" in _rule(WORKSPACE, ".chat-workspace-main .chat-composer.is-working .chat-input-bar")
    assert "qb-label-shine" in _rule(WORKSPACE, ".answer-progress-label")


def test_the_label_s_shine_does_not_take_the_skeletons_keyframes():
    css = (CSS / "chat_workspace.css").read_text(encoding="utf-8")
    assert "@keyframes qb-shimmer" not in css and "@keyframes qb-spin" not in css


def test_under_reduced_motion_the_send_button_and_the_box_are_still():
    css = (CSS / "chat_workspace.css").read_text(encoding="utf-8")
    reduced = css[css.index("@media (prefers-reduced-motion: reduce) {\n  .chat-workspace-main .send-btn"):]
    reduced = reduced[:reduced.index("\n}\n")]
    for part in (".send-btn.is-stop::after", ".chat-composer.is-working .chat-input-bar", ".answer-progress-label"):
        assert part in reduced, part
    assert "animation: none" in reduced


# ── a page's entrance leaves nothing behind ─────────────────────────────────

@pytest.mark.parametrize("sheet,selector", [("admin.css", ".page"), ("portal.css", ".main"), ("base.css", ".tab-pane.active")])
def test_an_entrance_keeps_no_transform_that_would_hold_an_overlay(sheet, selector):
    css = (CSS / sheet).read_text(encoding="utf-8")
    found = re.findall(re.escape(selector) + r"\s*\{[^}]*animation:\s*([^;}]+)", css)
    assert found, selector
    for animation in found:
        assert " both" not in animation and "forwards" not in animation, (sheet, animation)


# ── words, not capitals ─────────────────────────────────────────────────────

@pytest.mark.parametrize("where,selector", [
    ("portal/templates/portal_chat.html", ".data-table thead th"), ("portal/templates/portal_chat.html", ".rc-table thead th"),
    ("portal/templates/portal_kb.html", ".field-table th"), ("static/css/dashboard.css", ".dash-sched-list-head h3"),
    ("static/css/dashboard.css", ".dash-alert-recent-title")])
def test_a_table_s_head_and_a_dialog_s_section_read_as_words(where, selector):
    """An answer's table said REGION NAME and NET AMOUNT; the schedule and alert dialogs headed their lists in capitals.
    The admin's tables read as words since the console's restyle; these do too."""
    assert "uppercase" not in _rule(where, selector)

