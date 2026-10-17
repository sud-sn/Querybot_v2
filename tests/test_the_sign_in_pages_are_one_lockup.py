"""The sign-in pages: one lockup, one password reveal, one calm arrival.

Seen on a test server: Microsoft Edge drew its own reveal eye inside the password field beside ours, so the field
had two eyes, in the portal and the admin alike; the mark sat alone above the name, a different size on each page;
and the page simply appeared while only the mark moved, then the mark turned beside the button's own spinner while
signing in.

Now every sign-in page shows the mark beside the wordmark as one lockup (as the sidebar has it), the browser's own
reveal is hidden wherever ours is, the page's blocks rise into place once (never under reduced motion), and the
button alone says it is signing in. These read the real templates and stylesheets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["portal/templates/portal_login.html", "portal/templates/portal_unsubscribe.html",
         "admin/templates/login.html", "admin/templates/admin_invite.html",
         "admin/templates/admin_account_setup.html"]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _rule(css: str, selector_part: str, exact: bool = False) -> str:
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        selector = re.sub(r"/\*.*?\*/", "", selector, flags=re.S).strip()
        if (selector == selector_part) if exact else (selector_part in selector):
            return body
    return ""


@pytest.mark.parametrize("page", PAGES)
def test_the_mark_and_the_name_are_one_lockup(page):
    source = _read(page)
    lockup = source.split('class="qb-lockup', 1)[1].split("</div>", 1)[0]
    assert "brand_motion(" in lockup and "wordmark(" in lockup, page


def test_the_lockup_sizes_the_mark_with_the_letters():
    css = _read("static/css/brand-motion.css")
    lockup = _rule(css, ".qb-lockup", exact=True)
    assert "inline-flex" in lockup and "align-items: center" in lockup
    assert re.search(r"--qb-mark-size:\s*[\d.]+em", _rule(css, ".qb-lockup > .qb-brand-motion"))


def test_the_browser_s_own_reveal_is_hidden_beside_ours():
    css = _read("static/css/base.css")
    rule = _rule(css, "::-ms-reveal")
    assert rule and "display: none" in rule
    selector = re.search(r"([^{}]*::-ms-reveal[^{}]*)\{", css).group(1)
    assert ".input-reveal-wrap" in selector, "only where our own reveal button is"
    # Alone in its rule: Edge would drop a rule whose selector list also names a pseudo-element it does not know.
    assert "," not in selector.split("*/")[-1], selector


@pytest.mark.parametrize("page", ["portal/templates/portal_login.html", "admin/templates/login.html",
                                  "admin/templates/admin_invite.html", "admin/templates/admin_account_setup.html"])
def test_the_page_arrives_once(page):
    assert "qb-enter" in _read(page), page


def test_the_arrival_is_off_under_reduced_motion_and_leaves_nothing_behind():
    css = _read("static/css/base.css")
    block = css.split("@media (prefers-reduced-motion: no-preference)", 1)
    assert len(block) == 2
    enter = block[1].split("@keyframes", 1)[0]
    assert ".qb-enter > *" in enter and "backwards" in enter and "forwards" not in enter and " both" not in enter
    assert ".qb-enter" not in block[0], "no arrival outside the reduced-motion guard"
