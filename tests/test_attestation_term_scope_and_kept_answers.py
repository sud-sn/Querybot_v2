"""A confidentiality attestation with a term, the data classes it covers, and its signed document; and
answers kept for history under the reader's clearance now, not then.

An attestation ends when revoked or when its term runs out. It releases only the data classes it
names ("PII", "PHI"; "*" every one): a column tagged with a class it does not name stays masked, in
both cores, and only what was released is written to the decision log as released. The People page
reads each user's status from it. A reader who was cleared when they asked, and is not any more,
never sees that answer as it was given again: rows the existing core answered with are masked again
under today's policies, and a new-core answer (its wording may name people) is withheld.

Invented people and accounts only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import types
import unittest
import uuid
from contextlib import ExitStack
from unittest.mock import patch

import pytest

import store


def _ts(days: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def account():
    store.init_db()
    account_id = f"acct-a1-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account_id, "portal")
    yield account_id
    with store.get_db() as conn:
        conn.execute("DELETE FROM user_attestation WHERE account_id=?", (account_id,))
        conn.execute("DELETE FROM client WHERE account_id=?", (account_id,))


def test_an_attestation_ends_with_its_term(account):
    store.save_user_attestation(account, "11", expires_at=_ts(-1))
    store.save_user_attestation(account, "12", expires_at=_ts(30))
    store.save_user_attestation(account, "13")
    assert not store.user_attestation_valid(account, "11")
    assert store.user_attestation_valid(account, "12") and store.user_attestation_valid(account, "13")


def test_an_attestation_names_the_data_classes_it_releases(account):
    store.save_user_attestation(account, "21", scope="pii, phi")
    store.save_user_attestation(account, "22")
    assert store.user_attestation_scope(account, "21") == {"PII", "PHI"}
    assert store.user_attestation_scope(account, "22") == {"*"}
    assert store.user_attestation_scope(account, "23") is None


def test_the_people_page_reads_each_users_status(account):
    first = store.save_user_attestation(account, "31", expires_at=_ts(10), document_name="signed.pdf",
                                        document_sha256="ab" * 32)
    store.save_user_attestation(account, "32", expires_at=_ts(-2))
    gone = store.save_user_attestation(account, "33")
    store.revoke_user_attestation(account, gone, "admin")
    status = store.attestation_status(account)
    assert status["31"]["status"] == "signed" and status["31"]["renew_soon"] is True
    assert status["31"]["id"] == first and status["31"]["document_name"] == "signed.pdf"
    assert status["32"]["status"] == "expired" and status["33"]["status"] == "revoked"
    assert "34" not in status


class ScopedReleaseTests(unittest.TestCase):
    """The existing core's release path, run for real with the store and the warehouse faked."""

    SQL = "SELECT PRESCRIBER_NAME, PATIENT_EMAIL, REVENUE FROM RX.PHARMACY"
    RAW = [{"PRESCRIBER_NAME": "Dr. Alice Real", "PATIENT_EMAIL": "real.person@example.org", "REVENUE": 1250}]

    def _run(self, scope):
        from admin.routes import _default_regulated_rules
        from core.compliance import governed_query, policy_engine, sql_guard
        from core.compliance.models import PolicyContext
        from core.compliance.packs import get_pack

        stores = {policy_engine.store, governed_query.store, sql_guard.store}
        classifications = {
            "RX.PHARMACY.PRESCRIBER_NAME": {"tags": ["PHI"], "sensitivity": "RESTRICTED", "reviewed": 1},
            "RX.PHARMACY.PATIENT_EMAIL": {"tags": ["PII"], "sensitivity": "RESTRICTED", "reviewed": 1},
            "RX.PHARMACY.REVENUE": {"tags": [], "sensitivity": "INTERNAL", "reviewed": 1},
        }
        profile = {"mode": "regulated", "policy_pack_key": "healthcare_pharmacy_v1", "active_policy_version": 1,
                   "enforcement_mode": "enforce"}
        context = PolicyContext(account_id="acct-a1-gq", user_id="7", role="analyst", purpose_id="patient_care",
                                action="query_execution", policy_version=1)
        logged: list[dict] = []
        with ExitStack() as stack:
            for st in stores:
                stack.enter_context(patch.object(st, "get_compliance_profile", return_value=profile))
                stack.enter_context(patch.object(st, "get_classification_map", return_value=classifications))
                stack.enter_context(patch.object(st, "list_policy_rules",
                                                 return_value=_default_regulated_rules(get_pack("healthcare_pharmacy_v1"))))
                stack.enter_context(patch.object(st, "list_purposes", return_value=[]))
                stack.enter_context(patch.object(st, "list_row_policies", return_value=[]))
                stack.enter_context(patch.object(st, "user_attestation_valid", return_value=True))
                stack.enter_context(patch.object(st, "user_attestation_scope", return_value=scope))
                stack.enter_context(patch.object(st, "log_policy_decision",
                                                 side_effect=lambda **kw: logged.append(kw) or "audit"))
            stack.enter_context(patch.object(governed_query, "run_query", return_value=[dict(r) for r in self.RAW]))
            result = governed_query.execute_governed_query({}, "azure_sql", self.SQL, context=context,
                                                           known_tables={"RX.PHARMACY"})
        releases = [entry for entry in logged if entry.get("reason_code") == "attested_unmasked_release"]
        return result.rows[0], releases

    def test_every_class_released_where_it_names_all(self):
        row, releases = self._run({"*"})
        self.assertEqual(row["PRESCRIBER_NAME"], "Dr. Alice Real")
        self.assertEqual(row["PATIENT_EMAIL"], "real.person@example.org")
        self.assertEqual(len(releases), 1)

    def test_a_class_it_does_not_name_stays_masked_and_is_not_logged_as_released(self):
        row, releases = self._run({"PII"})
        self.assertEqual(row["PATIENT_EMAIL"], "real.person@example.org")
        self.assertNotEqual(row["PRESCRIBER_NAME"], "Dr. Alice Real")
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0]["resources"], ["RX.PHARMACY.PATIENT_EMAIL"])

    def test_a_scope_that_cannot_be_read_releases_nothing(self):
        from core.compliance import governed_query

        with patch.object(governed_query.store, "user_attestation_scope", side_effect=RuntimeError("down")):
            self.assertEqual(governed_query._attested_scope("acct", "7"), set())


def test_the_new_core_releases_people_only_within_scope(monkeypatch):
    from core2.warehouse import governed

    monkeypatch.setattr(store, "user_attestation_scope", lambda a, u: {"PCI"})
    assert not governed._covers_people("acct", "7")
    monkeypatch.setattr(store, "user_attestation_scope", lambda a, u: {"PII"})
    assert governed._covers_people("acct", "7")
    monkeypatch.setattr(store, "user_attestation_scope", lambda a, u: (_ for _ in ()).throw(RuntimeError("down")))
    assert not governed._covers_people("acct", "7")


def _trace(route: str, created_at: str, *, released: bool = False, rows=None) -> dict:
    return {"id": 5, "route": route, "created_at": created_at, "generated_sql": "SELECT PATIENT_EMAIL FROM RX.PHARMACY",
            "db_type": "azure_sql", "answer_frame": json.dumps({"released": True}) if released else "",
            "result_rows": json.dumps(rows or [])}


def test_a_new_core_answer_released_then_is_withheld_once_the_clearance_ends(account):
    from core.compliance import kept_answers

    granted = store.save_user_attestation(account, "41")
    trace = _trace("core2", _ts(0), released=True)
    assert not kept_answers.withheld(account, "41", trace), "still cleared: shown as given"
    store.revoke_user_attestation(account, granted, "admin")
    assert kept_answers.withheld(account, "41", trace)
    assert not kept_answers.withheld(account, "42", _trace("core2", _ts(0))), "never released: shown as kept"


def test_an_answer_given_inside_an_attestations_term_counts_as_released(account):
    from core.compliance import kept_answers

    store.save_user_attestation(account, "51", expires_at=_ts(5))
    with store.get_db() as conn:          # granted ten days ago
        conn.execute("UPDATE user_attestation SET granted_at=? WHERE account_id=?", (_ts(-10), account))
    assert kept_answers.was_released(account, "51", _trace("legacy", _ts(-3)))
    assert not kept_answers.was_released(account, "51", _trace("legacy", _ts(-20))), "before it was granted"
    assert not kept_answers.was_released(account, "52", _trace("legacy", _ts(-3))), "someone else's"


def test_rows_released_then_are_masked_again_once_the_clearance_ends(account, monkeypatch):
    from core.compliance import kept_answers, policy_engine

    granted = store.save_user_attestation(account, "61")
    with store.get_db() as conn:
        conn.execute("UPDATE user_attestation SET granted_at=? WHERE account_id=?", (_ts(-10), account))
    store.revoke_user_attestation(account, granted, "admin")
    decision = types.SimpleNamespace(masking={"RX.PHARMACY.PATIENT_EMAIL": "redact"})
    monkeypatch.setattr(policy_engine, "evaluate", lambda context, resources: decision)
    monkeypatch.setattr(policy_engine, "resolve_context", lambda *a, **k: types.SimpleNamespace())
    rows = [{"PATIENT_EMAIL": "real.person@example.org"}]
    trace = _trace("legacy", _ts(-5), rows=rows)
    assert kept_answers.remasked(account, {"id": 61}, trace, rows) == [{"PATIENT_EMAIL": "[REDACTED]"}]
    assert kept_answers.remasked(account, {"id": 62}, trace, rows) == rows, "never released: as kept"


def test_a_term_counts_calendar_months():
    from admin.routes import _attestation_ends

    with patch("datetime.datetime", wraps=dt.datetime) as fake:
        fake.now.return_value = dt.datetime(2026, 1, 31, 9, 30, tzinfo=dt.timezone.utc)
        assert _attestation_ends("1") == "2026-02-28 09:30:00"
        assert _attestation_ends("12") == "2027-01-31 09:30:00"
    assert _attestation_ends("") is None and _attestation_ends("0") is None and _attestation_ends("x") is None


def test_the_signed_document_is_kept_by_its_digest(tmp_path, monkeypatch):
    import hashlib

    from admin import routes

    monkeypatch.chdir(tmp_path)

    class Upload:
        def __init__(self, name: str, data: bytes):
            self.filename, self._data = name, data

        async def read(self) -> bytes:
            return self._data

    data = b"%PDF-1.4 signed by an invented person"
    name, digest = asyncio.run(routes._attestation_document("acct-a1-doc", Upload("Signed form.pdf", data)))
    assert (name, digest) == ("Signed form.pdf", hashlib.sha256(data).hexdigest())
    assert (tmp_path / "clients" / "acct-a1-doc" / "attestations" / f"{digest}.pdf").read_bytes() == data
    assert asyncio.run(routes._attestation_document("acct-a1-doc", Upload("form.exe", data))) == \
        "attestation_document_type"
    assert asyncio.run(routes._attestation_document("acct-a1-doc", Upload("big.pdf", b"x" * 5_000_001))) == \
        "attestation_document_size"


def test_an_attestation_that_ends_between_the_checks_releases_nothing(monkeypatch):
    """Valid when checked, gone when its classes are read (its term ran out in between): masked, in both cores."""
    from core.compliance import governed_query
    from core2.warehouse import governed

    monkeypatch.setattr(store, "user_attestation_scope", lambda a, u: None)
    assert governed_query._attested_scope("acct", "7") == set()
    assert not governed._covers_people("acct", "7")


def test_a_new_core_answer_given_while_cleared_is_withheld_only_where_policies_mask_what_it_read(account,
                                                                                                monkeypatch):
    """No people's data in it (nothing its query read is masked today): shown as kept after the clearance
    ends. Something it read is masked today: withheld. Unable to check: withheld."""
    from core.compliance import kept_answers, policy_engine

    granted = store.save_user_attestation(account, "71")
    with store.get_db() as conn:
        conn.execute("UPDATE user_attestation SET granted_at=? WHERE account_id=?", (_ts(-10), account))
    store.revoke_user_attestation(account, granted, "admin")
    trace = _trace("core2", _ts(-5))
    monkeypatch.setattr(policy_engine, "resolve_context", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(policy_engine, "evaluate", lambda context, resources: types.SimpleNamespace(masking={}))
    assert not kept_answers.withheld(account, {"id": 71}, trace)
    monkeypatch.setattr(policy_engine, "evaluate",
                        lambda context, resources: types.SimpleNamespace(masking={"RX.PHARMACY.PATIENT_EMAIL": "redact"}))
    assert kept_answers.withheld(account, {"id": 71}, trace)
    monkeypatch.setattr(policy_engine, "evaluate", lambda context, resources: (_ for _ in ()).throw(RuntimeError()))
    assert kept_answers.withheld(account, {"id": 71}, trace)
    assert not kept_answers.withheld(account, {"id": 72}, trace), "never cleared: masked when given"
