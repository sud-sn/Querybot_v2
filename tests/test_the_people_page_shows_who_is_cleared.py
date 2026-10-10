"""The People page: everyone in the workspace, with what they can do, and who is cleared to see people's data.

In a workspace under compliance each person's row says whether they are cleared to see people's data as stored
(until when, for which data classes, with the signed document), whether that clearance ended or was revoked,
or that they see it masked; and an admin records or renews a signed attestation, and revokes one, from the
row without leaving the page. A workspace not under compliance has no such column or dialog. The grant keeps
to the data classes ticked: "only these" with none ticked is refused, never widened to every class; and it
records only someone in this workspace.

Real routes, real requests, real store; a scratch database per test. Invented people only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from html.parser import HTMLParser
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
from fastapi import HTTPException
from starlette.requests import Request


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from admin import routes

    routes.store.init_db()
    with patch.object(routes, "_is_auth", return_value=True):
        yield routes.store


@pytest.fixture
def regulated():
    """The workspace is under compliance: the new core scrubs its questions."""
    with patch("core2.service.question_scrubber", return_value=lambda text: text):
        yield


def _request(path, query=None, form=None):
    body = urlencode(form or {}, doseq=True).encode()
    headers = [(b"content-type", b"application/x-www-form-urlencoded")] if form is not None else []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": "POST" if form is not None else "GET", "path": path, "root_path": "",
        "scheme": "http", "query_string": urlencode(query or {}).encode(), "headers": headers,
        "server": ("testserver", 80), "client": ("203.0.113.9", 5555),
    }, receive)


def _workspace(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _person(store, account_id, name):
    user_id, _ = store.create_user(account_id, name, f"{os.urandom(4).hex()}@example.com", password="a-password-1")
    return user_id


def _ts(days):
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


class _Page(HTMLParser):
    """Each person's row (its cells' text by label), and every element with its attributes."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self.elements: list[tuple[str, dict]] = []
        self._label: str | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = {k: (v or "") for k, v in attrs}
        self.elements.append((tag, attributes))
        if tag == "tr" and "data-email" in attributes:
            self.rows.append({"name": attributes.get("data-name", "")})
        if tag == "td" and self.rows and "data-label" in attributes:
            self._label = attributes["data-label"]
            self.rows[-1][self._label] = ""

    def handle_endtag(self, tag):
        if tag == "td":
            self._label = None

    def handle_data(self, data):
        if self._label and self.rows:
            self.rows[-1][self._label] += " " + " ".join(data.split())


def _page(account_id, location=None):
    from admin import routes

    query = {k: v[0] for k, v in parse_qs(urlparse(location).query).items()} if location else {}
    page = asyncio.run(routes.users_page(_request(f"/admin/clients/{account_id}/users", query), account_id))
    parsed = _Page()
    parsed.feed(page.body.decode())
    return parsed, page.body.decode()


def _row(page, name):
    return next(row for row in page.rows if row["name"] == name.lower())


def _grant(account_id, **form):
    from admin import routes

    form.setdefault("back", "users")
    response = asyncio.run(routes.compliance_grant_attestation(
        _request(f"/admin/clients/{account_id}/compliance/attestation", form=form), account_id))
    return response.headers["location"]


def test_each_row_says_whether_the_person_is_cleared(fresh_store, regulated):
    account_id = _workspace(fresh_store)
    cleared, ending, ended, revoked, never = (_person(fresh_store, account_id, n)
                                              for n in ("Cleared One", "Ending Soon", "Ended Once",
                                                        "Revoked Once", "Never Signed"))
    fresh_store.save_user_attestation(account_id, str(cleared), expires_at=_ts(200), scope="PII",
                                      document_name="signed-form.pdf", document_sha256="ab" * 32)
    fresh_store.save_user_attestation(account_id, str(ending), expires_at=_ts(10))
    fresh_store.save_user_attestation(account_id, str(ended), expires_at=_ts(-3))
    gone = fresh_store.save_user_attestation(account_id, str(revoked))
    fresh_store.revoke_user_attestation(account_id, gone, "admin")

    page, html = _page(account_id)
    assert "Cleared" in _row(page, "Cleared One")["People's data"]
    assert _ts(200)[:10] in _row(page, "Cleared One")["People's data"]
    assert "PII" in _row(page, "Cleared One")["People's data"]
    assert "signed-form.pdf" in _row(page, "Cleared One")["People's data"]
    assert "renew soon" in _row(page, "Ending Soon")["People's data"]
    assert "renew soon" not in _row(page, "Cleared One")["People's data"]
    assert "Expired" in _row(page, "Ended Once")["People's data"]
    assert "Revoked" in _row(page, "Revoked Once")["People's data"]
    assert "Not cleared" in _row(page, "Never Signed")["People's data"]
    # The clearance can be revoked only where there is one.
    revokes = [a for t, a in page.elements if t == "form" and a.get("action", "").endswith("/revoke")]
    assert len(revokes) == 2
    assert "Clearance to renew within 30 days" in html


def test_the_grant_dialog_records_a_signed_attestation_from_the_page(fresh_store, regulated):
    account_id = _workspace(fresh_store)
    _person(fresh_store, account_id, "Ada Example")
    page, _ = _page(account_id)
    form = next(a for t, a in page.elements if t == "form" and a.get("id") == "attestForm")
    assert form["action"] == f"/admin/clients/{account_id}/compliance/attestation"
    assert form["enctype"] == "multipart/form-data"
    names = {a.get("name") for t, a in page.elements if t in ("input", "select") and a.get("name")}
    assert {"portal_user_id", "back", "attestation_type", "valid_months", "scope", "scope_all", "document",
            "document_ref"} <= names
    assert any(a.get("data-attest-user") for _t, a in page.elements)


def test_a_workspace_not_under_compliance_has_no_clearance_column(fresh_store):
    account_id = _workspace(fresh_store)
    _person(fresh_store, account_id, "Ada Example")
    with patch("core2.service.question_scrubber", return_value=None):
        page, html = _page(account_id)
    assert "People's data" not in _row(page, "Ada Example")
    assert 'id="attestDialog"' not in html
    # Everything else the page did is still there.
    assert 'id="resetDialog"' in html and 'id="createUserForm"' in html and "copySignInLink(this)" in html


def test_a_grant_from_the_page_keeps_its_term_and_classes_and_returns_there(fresh_store, regulated):
    account_id = _workspace(fresh_store)
    user_id = _person(fresh_store, account_id, "Ada Example")
    location = _grant(account_id, portal_user_id=str(user_id), valid_months="6", scope_all="0",
                      scope=["PHI", "PII"])
    assert location == f"/admin/clients/{account_id}/users?saved=attestation"
    assert fresh_store.user_attestation_scope(account_id, str(user_id)) == {"PII", "PHI"}
    status = fresh_store.attestation_status(account_id)[str(user_id)]
    assert status["status"] == "signed"
    assert _ts(180) < status["expires_at"] < _ts(185), "six calendar months"
    page, html = _page(account_id, location)
    assert "Attestation recorded" in html and "Cleared" in _row(page, "Ada Example")["People's data"]


def test_only_these_with_none_ticked_is_refused_never_every_class(fresh_store, regulated):
    account_id = _workspace(fresh_store)
    user_id = _person(fresh_store, account_id, "Ada Example")
    location = _grant(account_id, portal_user_id=str(user_id), valid_months="12", scope_all="0")
    assert location == f"/admin/clients/{account_id}/users?error=attestation_scope_required"
    assert fresh_store.user_attestation_scope(account_id, str(user_id)) is None
    _page_, html = _page(account_id, location)
    assert "Tick at least one kind of personal data" in html
    # "Every kind" grants every class.
    _grant(account_id, portal_user_id=str(user_id), scope_all="1")
    assert fresh_store.user_attestation_scope(account_id, str(user_id)) == {"*"}


def test_a_grant_is_only_for_someone_in_this_workspace(fresh_store, regulated):
    account_id, other = _workspace(fresh_store), _workspace(fresh_store)
    outsider = _person(fresh_store, other, "Someone Else")
    with pytest.raises(HTTPException) as refused:
        _grant(account_id, portal_user_id=str(outsider))
    assert refused.value.status_code == 404
    assert fresh_store.list_user_attestations(account_id) == []
    assert _grant(account_id, portal_user_id="") == f"/admin/clients/{account_id}/users?error=attestation_user_required"


def test_a_revoke_from_the_page_returns_there_and_the_row_says_so(fresh_store, regulated):
    from admin import routes

    account_id = _workspace(fresh_store)
    user_id = _person(fresh_store, account_id, "Ada Example")
    granted = fresh_store.save_user_attestation(account_id, str(user_id))
    response = asyncio.run(routes.compliance_revoke_attestation(
        _request("/revoke", form={"back": "users"}), account_id, granted))
    assert response.headers["location"] == f"/admin/clients/{account_id}/users?saved=attestation_revoked"
    page, html = _page(account_id, response.headers["location"])
    assert "Revoked" in _row(page, "Ada Example")["People's data"]
    assert "masked again" in html
    assert not fresh_store.user_attestation_valid(account_id, str(user_id))


def test_the_compliance_page_lists_each_grant_with_its_state(fresh_store, regulated):
    account_id = _workspace(fresh_store)
    ended = _person(fresh_store, account_id, "Ended Once")
    current = _person(fresh_store, account_id, "Cleared One")
    fresh_store.save_user_attestation(account_id, str(ended), expires_at=_ts(-3))
    fresh_store.save_user_attestation(account_id, str(current), scope="PHI")
    states = {row["user_name"]: row["state"] for row in fresh_store.list_user_attestations(account_id)}
    assert states == {"Ended Once": "expired", "Cleared One": "signed"}
