"""On a phone every control is big enough for a fingertip, and the larger view rises from the bottom.

Measured in a real browser at 390 x 844 (tools/ui_check/check.py, which now
fails on a button, field or select under 44px on a phone): the send button was
38px, the answer's action buttons 32 x 22, the menu button 32px, the form
fields 40px, the workspace back arrow 30px.

The fix is one rule in the design system, for touch screens only (a mouse
pointer keeps today's sizes): a control is at least 44px each way. The
measurement also had to be fixed: a Playwright screenshot drops Chromium's
touch emulation, so every page measured after the first screenshot read as a
mouse page; the check restores it after each shot.

These read the stylesheets for the rule; the browser check is what proves it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _coarse_blocks(css: str) -> str:
    return " ".join(m.group(1) for m in re.finditer(r"@media \(pointer: coarse\) \{(.*?)\n\}", css, re.S))


def test_the_design_system_sizes_every_control_for_a_fingertip():
    rules = _coarse_blocks((ROOT / "static/css/base.css").read_text(encoding="utf-8"))
    for control in (".btn", ".icon-btn", "button", "select", '[role="tab"]', ".form-group select"):
        assert control in rules, control
    assert "min-height: 44px" in rules and "min-width: 44px" in rules


def test_both_menus_are_touch_sized():
    assert "min-height: 44px" in _coarse_blocks((ROOT / "static/css/portal.css").read_text(encoding="utf-8"))
    assert "min-height: 44px" in _coarse_blocks((ROOT / "static/css/admin.css").read_text(encoding="utf-8"))


def test_the_larger_view_is_a_sheet_from_the_bottom_on_a_phone():
    page = (ROOT / "portal/templates/portal_chat.html").read_text(encoding="utf-8")
    phone = re.search(r"@media\(max-width:520px\)\{(.*?)\}\}", page, re.S).group(1)
    assert "inset:auto 0 0 0" in phone and "translateY(100%)" in phone
    assert ".chat-workspace.artifact-open .artifact-pane{transform:translateY(0)}" in phone


def test_the_closed_menu_casts_no_shadow_on_the_page():
    """Closed, the phone menu sits just off screen; its shadow fell onto the page's left edge."""
    css = (ROOT / "static/css/portal.css").read_text(encoding="utf-8")
    phone = css[css.index("@media (max-width: 720px) {"):]
    phone = phone[:phone.index("\n}\n")]
    closed = phone[phone.index("  .portal-sidebar {"):]
    closed = closed[:closed.index("}")]
    assert "translateX(-101%)" in closed and "box-shadow: none" in closed
    opened = phone[phone.index(".portal-drawer-open .portal-sidebar {"):]
    assert "box-shadow: 18px" in opened[:opened.index("}")]


def test_the_check_puts_touch_back_after_every_screenshot():
    check = (ROOT / "tools/ui_check/check.py").read_text(encoding="utf-8")
    assert "Emulation.setTouchEmulationEnabled" in check
    shot = check[check.index("    def shot("):check.index("    def check(")]
    assert shot.index("self._touch_again()") < shot.index("found = self.check(label)")
