"""
In a workspace under compliance, adding a person says how they see people's data, and only a role that may
grant it can clear them.

    Add person › People's data: Masked (the default), or Cleared with the attestation they signed: what, for
    how long, which kinds of people's data, a reference and the signed document. It is recorded with the new
    person, named after the admin who granted it, and the People log says so.

Clearing someone is the Compliance page's to grant (admin/access.py: a Super admin's). A Workspace admin adds
people masked, sees who is cleared, and is refused if they try to clear anyone, at creation or after.

Neither the reader's role (Viewer, Asker, Creator) nor "every table" changes what is masked: only a valid
attestation that covers people's data does (core2/warehouse/governed.py).

Invented people only; no AI.
"""

from __future__ import annotations

import io
import os
import types

import pytest
from starlette.testclient import TestClient

GOOD = "a-long-admin-password"


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.chdir(tmp_path)                      # a signed document is kept under clients/<id>/ here
    import main

    main.store.init_db()
    return main


@pytest.fixture
def regulated(app):
    account_id = f"acct{os.urandom(3).hex()}"
    app.store.upsert_client(account_id, "Pharmacy")
    app.store.save_compliance_profile(account_id, mode="regulated", industry="healthcare_pharmacy")
    from admin.routes import _workspace_regulated

    assert _workspace_regulated(account_id)
    return account_id


def _admin(app, role, workspaces=()):
    email = f"{role}-{os.urandom(3).hex()}@example.com"
    app.store.create_admin(email, role.title(), role, password=GOOD, workspaces=list(workspaces))
    client = TestClient(app.app)
    assert client.post("/admin/login", data={"email": email, "password": GOOD},
                       follow_redirects=False).status_code == 303
    return client, email


def _add(client, account_id, *, files=None, **fields):
    data = {"name": "Nora Nurse", "email": f"nora-{os.urandom(2).hex()}@example.com", "group_id": "",
            "role": "analyst", "portal_role": "asker", "password": "", "confirm_password": "", "must_change": "1",
            **fields}
    response = client.post(f"/admin/clients/{account_id}/users/create", data=data, files=files or None,
                           follow_redirects=False)
    return response, data["email"]


class TestASuperAdmin:

    def test_sees_the_choice_when_adding_someone(self, app, regulated):
        client, _ = _admin(app, "super_admin")
        page = client.get(f"/admin/clients/{regulated}/users").text
        assert "People's data" in page and 'name="clearance" value="cleared"' in page
        assert 'enctype="multipart/form-data"' in page

    def test_adds_someone_masked_by_default(self, app, regulated):
        client, _ = _admin(app, "super_admin")
        response, email = _add(client, regulated)
        assert response.status_code == 303 and "error" not in response.headers["location"]
        user = app.store.get_user_by_email(regulated, email)
        assert not app.store.user_attestation_valid(regulated, str(user["id"]))

    def test_adds_someone_cleared_with_what_they_signed(self, app, regulated):
        client, boss = _admin(app, "super_admin")
        signed = io.BytesIO(b"%PDF-1.7 signed confidentiality agreement")
        response, email = _add(client, regulated, clearance="cleared", attestation_type="hipaa_workforce",
                               valid_months="6", scope_all="0", scope=["PHI"], document_ref="HR-2041",
                               files={"document": ("nora-signed.pdf", signed, "application/pdf")})
        assert response.status_code == 303 and "error" not in response.headers["location"], response.headers
        user = app.store.get_user_by_email(regulated, email)
        assert app.store.user_attestation_valid(regulated, str(user["id"]))
        assert app.store.user_attestation_scope(regulated, str(user["id"])) == {"PHI"}
        status = app.store.attestation_status(regulated)[str(user["id"])]
        assert status["status"] == "signed" and status["document_name"] == "nora-signed.pdf"
        assert status["expires_at"] and status["scope"] == "PHI"
        from store.db import get_db

        with get_db() as conn:
            row = conn.execute("SELECT granted_by, attestation_type, document_ref, document_sha256 FROM "
                               "user_attestation WHERE portal_user_id=?", (str(user["id"]),)).fetchone()
        assert row["granted_by"] == boss and row["attestation_type"] == "hipaa_workforce"
        assert row["document_ref"] == "HR-2041" and len(row["document_sha256"]) == 64
        created = [e for e in app.store.list_user_events(regulated) if e["action"] == "created"][0]
        assert "cleared to see people's data (PHI)" in created["detail"] and created["actor"] == boss
        page = client.get(f"/admin/clients/{regulated}/users").text
        assert "Cleared" in page and "nora-signed.pdf" in page

    def test_only_these_with_nothing_ticked_adds_no_one(self, app, regulated):
        client, _ = _admin(app, "super_admin")
        response, email = _add(client, regulated, clearance="cleared", scope_all="0")
        assert "attestation_scope_required" in response.headers["location"]
        assert app.store.get_user_by_email(regulated, email) is None

    def test_a_document_of_another_kind_adds_no_one(self, app, regulated):
        client, _ = _admin(app, "super_admin")
        response, email = _add(client, regulated, clearance="cleared",
                               files={"document": ("macro.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
        assert "attestation_document_type" in response.headers["location"]
        assert app.store.get_user_by_email(regulated, email) is None

    def test_a_workspace_not_under_compliance_asks_nothing(self, app):
        account_id = f"acct{os.urandom(3).hex()}"
        app.store.upsert_client(account_id, "Retail")
        # A new workspace counts as under compliance until it is set up as standard.
        app.store.save_compliance_profile(account_id, mode="standard")
        from admin.routes import _workspace_regulated

        assert not _workspace_regulated(account_id)
        client, _ = _admin(app, "super_admin")
        assert 'name="clearance"' not in client.get(f"/admin/clients/{account_id}/users").text
        response, email = _add(client, account_id, clearance="cleared")
        user = app.store.get_user_by_email(account_id, email)
        assert user and not app.store.user_attestation_valid(account_id, str(user["id"]))

    def test_a_later_clearance_names_who_granted_it_too(self, app, regulated):
        client, boss = _admin(app, "super_admin")
        _, email = _add(client, regulated)
        user = app.store.get_user_by_email(regulated, email)
        client.post(f"/admin/clients/{regulated}/compliance/attestation",
                    data={"portal_user_id": str(user["id"]), "back": "users", "scope_all": "1", "valid_months": "12"})
        from store.db import get_db

        with get_db() as conn:
            granted = conn.execute("SELECT granted_by FROM user_attestation WHERE portal_user_id=?",
                                   (str(user["id"]),)).fetchone()["granted_by"]
        assert granted == boss


class TestAWorkspaceAdmin:

    def test_adds_people_masked_and_cannot_clear_them(self, app, regulated):
        client, _ = _admin(app, "workspace_admin", [regulated])
        page = client.get(f"/admin/clients/{regulated}/users").text
        assert 'name="clearance"' not in page and "A Super admin records a signed attestation" in page
        assert '<dialog id="attestDialog"' not in page and 'data-attest-user="' not in page
        response, email = _add(client, regulated, clearance="cleared")
        assert "Only+a+Super+admin" in response.headers["location"].replace("%20", "+")
        assert app.store.get_user_by_email(regulated, email) is None
        fine, email = _add(client, regulated)
        user = app.store.get_user_by_email(regulated, email)
        assert user and not app.store.user_attestation_valid(regulated, str(user["id"]))
        refused = client.post(f"/admin/clients/{regulated}/compliance/attestation",
                              data={"portal_user_id": str(user["id"]), "scope_all": "1"}, follow_redirects=False)
        assert refused.status_code == 403
        assert not app.store.user_attestation_valid(regulated, str(user["id"]))

    def test_still_sees_who_is_cleared(self, app, regulated):
        boss, _ = _admin(app, "super_admin")
        _add(boss, regulated, clearance="cleared")
        client, _ = _admin(app, "workspace_admin", [regulated])
        page = client.get(f"/admin/clients/{regulated}/users").text
        assert "Cleared" in page and "People's data" in page


# ── A role never widens what is masked ───────────────────────────────────────

class TestRolesDoNotUnmask:
    """The governed warehouse, against each kind of reader, with the workspace's real attestations."""

    @pytest.fixture
    def ask(self, app, regulated, monkeypatch):
        from tests.test_core2_people_shown_masked_until_cleared import (_Governed, _name_slug, _amount, _warehouse)
        from core2.bootstrap.build import BuildOptions, build_model
        from core2.bootstrap.inventory import from_duckdb
        from core2.compile.compiler import compile_query
        from core2.plan.ir import Plan
        from core2.resolve.resolver import Context, resolve
        from core2.service import personal_columns
        from core2.warehouse.governed import GovernedWarehouse
        from core2.warehouse.runner import DuckDBWarehouse
        import datetime as dt

        con = _warehouse()
        model = build_model(DuckDBWarehouse(con), from_duckdb(DuckDBWarehouse(con)), client_id="people",
                            options=BuildOptions(workers=1))
        monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", _Governed(con))
        reader = {"id": None}
        monkeypatch.setattr("core.compliance.policy_engine.resolve_context", lambda *a, **k: types.SimpleNamespace(
            user_id=str(reader["id"]), purpose_id="", channel="portal", policy_version=1))

        def run(user: dict):
            reader["id"] = user["id"]
            warehouse = GovernedWarehouse(regulated, user, {"db_type": "duckdb", "credentials": {}},
                                          known_tables=set(), personal=personal_columns(model))
            plan = Plan(intent="breakdown", measures=[_amount(model)], group_by=[_name_slug(model)])
            compiled = compile_query(resolve(plan, model, Context(today=dt.date(2026, 6, 15), personal_shown=True)),
                                     model, "duckdb")
            names = [str(row[0]) for row in warehouse.query(compiled.sql, max_rows=compiled.row_cap).rows]
            return all(name.startswith("Patient ") for name in names)

        return run

    @pytest.mark.parametrize("portal_role", ["viewer", "asker", "creator"])
    @pytest.mark.parametrize("tables", ["analyst", "admin"])
    def test_without_a_clearance_every_reader_sees_people_masked(self, app, regulated, ask, portal_role, tables):
        user_id, _ = app.store.create_user(regulated, "Rae", f"r-{os.urandom(2).hex()}@example.com",
                                           password="readers-password", role=tables, portal_role=portal_role)
        assert ask(app.store.get_user(user_id)) is True

    def test_a_clearance_unmasks_whatever_the_role(self, app, regulated, ask):
        user_id, _ = app.store.create_user(regulated, "Vic", "vic@example.com", password="readers-password",
                                           portal_role="viewer")
        app.store.save_user_attestation(regulated, str(user_id), granted_by="test", scope="PII")
        assert ask(app.store.get_user(user_id)) is False
