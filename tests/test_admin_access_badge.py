"""
tests/test_admin_access_badge.py

Pending access requests were invisible in the admin console, through two
independent failures that each looked fine in isolation.

1. THE POLLER TARGETED A NAV THAT NO LONGER SHIPPED.
   base.html polled `.ws-tab-bar a[href*="/pending-users"]`. That selector
   belonged to the `client_workspace_nav` macro, which had been superseded by
   templates/_client_workspace_nav.html and was called from nowhere. The
   querySelector returned null on every page, the function returned early, and
   the badge never updated once.

2. THE SERVER-RENDERED FALLBACK WAS PASSED FROM THE ONE ROUTE THAT CANNOT
   RENDER IT.
   `_client_workspace_nav.html` renders the badge only when `pending_count` is
   defined. Of ~150 routes, exactly one passed it: `/clients/{id}/graph`. And
   client_graph.html is a standalone document with no workspace nav at all, so
   the value went to a template that never draws the badge, while the three
   routes that DO draw the Access nav (/users, /groups, /pending-users) never
   passed it.

Net effect: an admin could not tell that anyone was waiting for access unless
they already knew to open the page that lists them. Neither half raised an
error; a dead selector and a missing keyword are both silent.

These tests pin the wiring end to end, because that is what nothing did.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# Routes whose pages render the Access secondary nav, and therefore the badge.
_ACCESS_NAV_ROUTES = (
    '@router.get("/clients/{account_id}/users"',
    '@router.get("/clients/{account_id}/groups"',
    '@router.get("/clients/{account_id}/pending-users")',
)


def test_the_badge_poller_targets_the_nav_that_actually_ships():
    base = _read("admin/templates/base.html")
    nav = _read("admin/templates/_client_workspace_nav.html")

    selector = re.search(
        r"querySelector\('([^']*a\[href\*=\"/pending-users\"\])'\)", base
    )
    assert selector, "base.html no longer queries for the pending-users link"

    css_class = selector.group(1).split(" ")[0].lstrip(".")
    assert css_class in nav, (
        f"the poller queries '.{css_class}', which the live workspace nav does not "
        f"render; the badge can never update"
    )


def test_every_route_that_renders_the_access_nav_passes_pending_count():
    """A missing keyword argument leaves no runtime trace: the template's
    `{% if pending_count is defined %}` guard just silently renders nothing."""
    source = _read("admin/routes.py")
    missing = []
    for decorator in _ACCESS_NAV_ROUTES:
        start = source.index(decorator)
        # Scan to the next route decorator, i.e. this handler's body.
        nxt = source.find("\n@router.", start + 1)
        body = source[start: nxt if nxt != -1 else len(source)]
        if "pending_count" not in body:
            missing.append(decorator)
    assert not missing, (
        f"these routes render the Access nav but never pass pending_count, so its "
        f"badge cannot draw: {missing}"
    )


def test_the_superseded_nav_macro_is_gone():
    """Two navigation systems in one console is how the selector drifted apart
    from the markup in the first place."""
    macros = _read("admin/templates/macros.html")
    assert "{% macro client_workspace_nav" not in macros, (
        "the superseded flat nav macro is back; base.html's poller and the live "
        "nav will drift apart again"
    )
    for stylesheet in ("static/css/base.css", "static/css/admin.css"):
        css = _read(stylesheet)
        assert ".ws-tab-bar" not in css and ".ws-nav {" not in css, (
            f"{stylesheet} still styles the deleted nav"
        )


def test_the_count_endpoint_the_poller_calls_exists():
    """The poller fetches a JSON count; if that route is renamed the badge goes
    quiet again, and quietly."""
    base = _read("admin/templates/base.html")
    assert "/pending-users/count" in base, "the poller no longer calls a count endpoint"

    routes = _read("admin/routes.py")
    assert '@router.get("/api/clients/{account_id}/pending-users/count")' in routes, (
        "the count endpoint the poller depends on is missing"
    )


def test_the_nav_renders_a_badge_when_the_count_is_positive():
    nav = _read("admin/templates/_client_workspace_nav.html")
    access_link = next(
        line for line in nav.splitlines() if "/pending-users" in line and "<a " in line
    )
    assert "pending_count" in access_link, "the Access link renders no count"
    assert "client-nav-count" in access_link, "the count has no badge styling"
