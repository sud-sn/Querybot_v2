"""
Notifications lists the alerts on dashboard tiles that tell the reader: the tile, its dashboard (a link to it), when
it tells them, how often it is checked, and whether its line is crossed for them now. An alert is set on a dashboard's
number tile (its bell); this is the one page that lists them all.

    whose       the ones they made, and the ones others made that name them or a group of theirs
    where       their own workspace only; a dashboard no longer shared with them is not listed (its name included)

Invented data only (the retail domain); no AI.
"""

from __future__ import annotations

from tests.test_a_new_core_answer_can_be_pinned import retail  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner, _person, smtp, inbox  # noqa: F401
from tests.test_tile_alerts_tell_when_a_number_crosses_a_line import _alert, _board, _signed_in, app  # noqa: F401


def _page(app, user) -> str:  # noqa: F811
    """The page's "Your alerts" card alone (Recent above it names a dashboard shared with them too)."""
    reply = _signed_in(app, user).get("/portal/notifications")
    assert reply.status_code == 200, reply.text
    return reply.text.split('id="noticeAlertsTitle"', 1)[1].split('id="noticeReportsTitle"', 1)[0]


def test_the_alerts_a_person_made_are_listed_with_their_tile_and_dashboard(app, retail):  # noqa: F811
    store = app.store
    owner = _owner(store)
    board, tile = _board(store, retail, owner)
    _alert(app, retail, owner, board, tile)

    page = _page(app, owner)
    assert tile["title"] in page and "Sales pulse" in page
    assert "is above $150,000" in page
    assert f"/portal/dashboard?dashboard_id={board['id']}" in page
    assert "Checked every hour" in page


def test_an_alert_that_names_someone_is_listed_for_them_and_not_for_others(app, retail):  # noqa: F811
    store = app.store
    owner = _owner(store)
    board, tile = _board(store, retail, owner)
    ann = _person(store, owner, "Ann Viewer")
    bob = _person(store, owner, "Bob Buyer")
    _alert(app, retail, owner, board, tile, recipients=[{"type": "user", "id": ann["id"]}])

    page = _page(app, ann)
    assert "Sales pulse" in page and "Set by Sam Owner" in page
    assert "Sales pulse" not in _page(app, bob)


def test_a_dashboard_no_longer_shared_is_not_listed(app, retail):  # noqa: F811
    store = app.store
    owner = _owner(store)
    board, tile = _board(store, retail, owner)
    ann = _person(store, owner, "Ann Viewer")
    _alert(app, retail, owner, board, tile, recipients=[{"type": "user", "id": ann["id"]}])
    assert store.remove_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", ann["id"])

    assert "Sales pulse" not in _page(app, ann)


def test_with_no_alerts_it_says_where_they_are_set(app, retail):  # noqa: F811
    store = app.store
    owner = _owner(store)
    page = _page(app, owner)
    assert "bell on a number tile" in page
    assert "Alert me on changes" not in page
