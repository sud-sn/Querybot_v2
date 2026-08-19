"""
tests/test_admin_workspace_ia.py

Five of the client workspace's destinations existed only as location.hash tabs
inside client_detail.html: Settings, the schema browser, the query log, the AI
egress log and the destructive actions. They were deep-linkable, but absent from
the workspace nav — so the most-visited support surface in the console had no
presence in its own information architecture, and they switched via a second tab
bar with its own skin and its own state mechanic sitting directly beneath the
real nav.

Each now has a path, the nav lists it, and the active tab is decided server-side
like every other tab in the console.

The nav's active-state checks also moved from substring matching on the whole
path to matching the segment after the account id. Because the account id sits
directly after a slash, `'/schema' in _p` is true for every page of a client
whose account_id *begins with* a tab name — lighting Activity throughout that
workspace, and marking two primaries active at once on the overview.
"""

import re
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "admin" / "templates"

# (path segment, expected primary tab, expected secondary tab or None)
_NAV_EXPECTATIONS = [
    ("",                "Overview",        None),
    ("setup",           "Data &amp; Model", "Setup"),
    ("kb",              "Data &amp; Model", "Knowledge Base"),
    ("graph",           "Data &amp; Model", "Relationships"),
    ("date-roles",      "Data &amp; Model", "Dates"),
    ("metrics",         "Data &amp; Model", "Metrics"),
    ("queries",         "Activity",        "Queries"),
    ("traces",          "Activity",        "Timing"),
    ("egress",          "Activity",        "AI Egress Log"),
    ("schema",          "Activity",        "Schema browser"),
    ("billing",         "Activity",        "Usage &amp; Billing"),
    ("model-health",    "Quality",         "Model Health"),
    ("learning-queue",  "Quality",         "Flagged Answers"),
    ("evals",           "Quality",         "Evaluations"),
    ("users",           "Access",          "Users"),
    ("groups",          "Access",          "Groups"),
    ("pending-users",   "Access",          "Access Requests"),
    ("compliance",      "Compliance",      None),
    ("settings",        "Settings",        "Configuration"),
    ("advanced",        "Settings",        "Advanced"),
]

_NEW_ROUTES = ("settings", "schema", "queries", "egress", "advanced")

# Pre-existing orphans, quarantined rather than hidden. The metrics editor's
# field-browser panel is styled (.col-browser* in client_metrics.html) and its
# handlers are exported to window, but the list element it fills is never
# rendered — `if(!list) return;` swallows it, so the whole panel is dead. That
# template needs its own pass; this list must shrink, never grow.
_KNOWN_ORPHANS = [
    "client_metrics.html: colBrowserList",
    "client_metrics.html: col-browser-status",
]


def _render_nav(account_id: str, segment: str) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    path = f"/admin/clients/{account_id}" + (f"/{segment}" if segment else "")
    request = type("R", (), {"url": type("U", (), {"path": path})()})()
    return env.get_template("_client_workspace_nav.html").render(
        request=request,
        client={"account_id": account_id, "client_name": "EMCO", "state": "READY"},
    )


def _active(html: str, level: str):
    block = re.search(rf'client-workspace-{level}.*?</nav>', html, re.S)
    if not block:
        return None
    return re.findall(r'class="active"\s*>(.*?)(?:<span|</a>)', block.group(0))


@pytest.mark.parametrize("segment,primary,secondary", _NAV_EXPECTATIONS)
def test_exactly_one_tab_is_active_on_every_workspace_path(segment, primary, secondary):
    html = _render_nav("demo", segment)

    active_primary = _active(html, "primary")
    assert active_primary == [primary], (
        f"/{segment or '(overview)'}: expected primary {primary!r}, got {active_primary}"
    )

    active_secondary = _active(html, "secondary")
    if secondary is None:
        assert active_secondary is None, (
            f"/{segment or '(overview)'} should render no secondary nav, got {active_secondary}"
        )
    else:
        assert active_secondary == [secondary], (
            f"/{segment or '(overview)'}: expected secondary {secondary!r}, got {active_secondary}"
        )


def test_an_account_id_containing_a_tab_name_does_not_light_the_wrong_tab():
    """The substring form this replaced marks Activity active on every page of
    this workspace, overview included — where Overview is active too, so two
    primaries light at once. Verified by reintroducing it."""
    for segment, primary, _ in _NAV_EXPECTATIONS:
        html = _render_nav("schema-users-co", segment)
        assert _active(html, "primary") == [primary], (
            f"account_id containing tab names broke /{segment or '(overview)'}"
        )


def test_the_nav_does_not_substring_match_the_whole_path():
    source = (TEMPLATES / "_client_workspace_nav.html").read_text(encoding="utf-8")
    body = re.sub(r"\{#.*?#\}", "", source, flags=re.S)   # the comment explains the old form
    assert "in _p" not in body, (
        "the nav is substring-matching the full path again; match the segment "
        "after the account id instead"
    )


@pytest.fixture(scope="module")
def registered_get_paths():
    """The live router, not a grep of routes.py. A decorator can be present in
    the source and still not register — commented out, inside a dead branch, or
    shadowed by a later route with the same path."""
    from admin import routes as admin_routes
    return {
        r.path
        for r in admin_routes.router.routes
        if "GET" in (getattr(r, "methods", None) or ())
    }


@pytest.mark.parametrize("segment", _NEW_ROUTES)
def test_each_promoted_destination_has_a_real_route(segment, registered_get_paths):
    assert f"/admin/clients/{{account_id}}/{segment}" in registered_get_paths, (
        f"/{segment} is not registered on the router, so the nav links to a 404"
    )


def test_the_nav_links_only_to_paths_that_exist(registered_get_paths):
    """A nav entry pointing at a route nobody registered is a 404 the operator
    finds, not the test suite. Checked for every group, since each renders a
    different secondary nav."""
    missing = []
    for segment, _, _ in _NAV_EXPECTATIONS:
        html = _render_nav("demo", segment)
        for link in sorted(set(re.findall(r'href="/admin/clients/demo/([a-z-]+)"', html))):
            if f"/admin/clients/{{account_id}}/{link}" not in registered_get_paths:
                missing.append(f"/{segment or '(overview)'} -> /{link}")
    assert not missing, f"the nav links to unregistered paths: {sorted(set(missing))}"


def test_the_second_tab_bar_is_gone():
    """Two stacked tab bars with different skins and different state mechanics
    is what made half the product invisible."""
    detail = (TEMPLATES / "client_detail.html").read_text(encoding="utf-8")
    assert 'class="tab-nav"' not in detail, "the in-page tab bar is back"
    assert "function showTab(" not in detail, "showTab() is back"
    for dead in (".tab-btn{", ".tab-nav{"):
        assert dead not in detail, f"dead CSS for the removed bar: {dead}"


def test_the_active_pane_is_decided_server_side():
    detail = (TEMPLATES / "client_detail.html").read_text(encoding="utf-8")
    panes = re.findall(r'id="tab-([a-z-]+)" class="tab-pane([^"]*)"', detail)
    assert panes, "no tab panes found"
    for name, cls in panes:
        assert "active_tab" in cls, (
            f"pane {name!r} does not consult active_tab, so its active state is "
            f"still a client-side default"
        )


def test_old_hash_links_are_redirected_to_the_real_paths():
    """Links and bookmarks in the wild still point at #queries and #audit."""
    detail = (TEMPLATES / "client_detail.html").read_text(encoding="utf-8")
    assert "PATH_FOR_HASH" in detail, "old hash links are no longer handled"
    for legacy in ("queries", "audit", "danger", "schema-browser", "settings"):
        assert f"'{legacy}'" in detail, f"#{legacy} bookmarks are dropped"


def test_no_script_looks_up_an_element_the_page_no_longer_renders():
    """Removing markup orphans every getElementById that pointed at it, and the
    orphan is not quiet: getElementById returns null, .addEventListener on null
    throws, and the rest of that script block never runs.

    Deleting the tab bar left `getElementById('tab-schema-btn')` behind exactly
    this way. Checked across the console, since any template can do it.

    An id counts as rendered if ANY template declares it (base.html's scripts
    legitimately reach into ids a child template draws) or if JS assigns it to a
    node it built."""
    rendered_ids = set()
    for template in TEMPLATES.glob("*.html"):
        src = template.read_text(encoding="utf-8")
        rendered_ids |= set(re.findall(r"""\bid=["']([A-Za-z][\w:-]*)["']""", src))
        # Nodes built in JS: `modal.id = 'diag-modal'`, `dl.id = "base-table-options"`
        rendered_ids |= set(re.findall(r"""\.id\s*=\s*["']([A-Za-z][\w:-]*)["']""", src))

    orphans = set()
    for template in sorted(TEMPLATES.glob("*.html")):
        src = template.read_text(encoding="utf-8")
        for looked_up in re.findall(r"""getElementById\(\s*["']([\w:-]+)["']\s*\)""", src):
            if looked_up not in rendered_ids:
                orphans.add(f"{template.name}: {looked_up}")

    new = sorted(orphans - set(_KNOWN_ORPHANS))
    fixed = sorted(set(_KNOWN_ORPHANS) - orphans)
    assert not new, (
        "these lookups target ids no template renders, so they return null and "
        f"the surrounding script dies: {new}"
    )
    assert not fixed, (
        f"these orphans are fixed -- drop them from _KNOWN_ORPHANS: {fixed}"
    )


def test_the_schema_tree_loads_when_entered_by_its_own_url():
    """The pane is activated server-side now, so nothing clicks the old tab
    button. The load has to fire on DOMContentLoaded instead -- and it has to go
    through an exported symbol: the `_loaded` flag is a `let` inside an IIFE, so
    page-level code reading it directly sees an undeclared name, `typeof` returns
    'undefined', and the guard is false forever. That version shipped, and it
    silently never loaded the tree."""
    detail = (TEMPLATES / "client_detail.html").read_text(encoding="utf-8")

    entry = re.search(
        r"active\.id === 'tab-schema-browser' && ([\w.]+)\)\s*\{\s*([\w.]+)\(\)",
        detail,
    )
    assert entry, "direct entry to /schema no longer triggers the tree load"
    guard, call = entry.groups()
    assert guard == call, f"guarded on {guard!r} but called {call!r}"
    assert guard.startswith("window."), (
        f"{guard!r} is not exported; a closure-scoped name is invisible here, so "
        f"the guard silently evaluates false"
    )
    assert f"{guard} = function" in detail, f"{guard} is never defined"

    # And the flag it gates stays private, so there is one gate rather than two.
    assert "let _loaded" in detail, "the load-once flag was promoted to a global"
