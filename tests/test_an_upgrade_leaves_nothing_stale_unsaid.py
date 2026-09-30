"""
An upgrade cannot silently leave stale settings.

Nothing said which build was running: /health reported "2.0.0" whatever was
deployed. A knowledge base is built once and read by every answer after it, so
an upgrade that changed what a build writes left each workspace answering from
the old kind until someone thought to rebuild, and nothing told them to. A key
file lost in an upgrade left every saved credential unreadable, and the first
anyone heard of it was a question that could not reach the warehouse -- or the
dashboard failing outright, since listing the databases decrypted each one and
raised on the first it could not read.

Now the version is VERSION, CHANGELOG.md has an entry for it that names the
knowledge-base format it builds, and /health and the admin sidebar give it. A
finished build records the format and the version that built it, and a
workspace another format built is told to rebuild -- at startup, in the
dashboard's inbox and on its setup page. At startup and on the dashboard every
saved credential is read with the server's key, and each one that cannot be is
named, and marked on its own Databases or Platforms page. Only a save writes a
key: a missing key file stays missing, and is called that, until someone
restores it or enters the credentials again.

A scratch store and key file; the product's own startup checks, health page,
inbox and admin routes, and the knowledge-base build itself, with its model,
vector store, example validation, repair and quality report stubbed at their
boundaries.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest
from starlette.background import BackgroundTasks
from starlette.requests import Request

ACCOUNT = "acct-upgrade-checks"
ROOT = Path(__file__).resolve().parents[1]


def _stores_in_use() -> list:
    """Every store the code under test reads through. Tests elsewhere delete
    the store modules and import them again, so the admin routes, the startup
    checks and the inbox can each hold an older store -- with its own key file
    setting -- than the one this test imports."""
    import sys

    import admin.inbox
    import admin.routes
    import main

    return [sys.modules["store"], admin.routes.store, admin.inbox.store, main.store]


def _use_key_file(monkeypatch, key_file: Path) -> None:
    """Point every store in use at ``key_file``."""
    namespaces = {}
    for store in _stores_in_use():
        for namespace in (store.crypto.__dict__, store.config_store.decrypt.__globals__):
            namespaces[id(namespace)] = namespace
    for namespace in namespaces.values():
        monkeypatch.setitem(namespace, "KEY_FILE", key_file)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A store and a key file of the test's own, and the working directory
    the knowledge base is written under."""
    from cryptography.fernet import Fernet

    import store

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    key_file = tmp_path / ".querybot_key"
    key_file.write_bytes(Fernet.generate_key())
    _use_key_file(monkeypatch, key_file)
    for used in _stores_in_use():
        monkeypatch.setattr(used.config_store, "_TOLD_UNREADABLE", set())
    monkeypatch.chdir(tmp_path)
    store.init_db()
    store.upsert_client(ACCOUNT, "portal")
    return store


def _built_by_an_earlier_release(store) -> None:
    """READY, as a build before 2.1.0 left a workspace: no format recorded."""
    store.update_client_state(ACCOUNT, "READY", {"kb_dir": f"clients/{ACCOUNT}/kb", "schema_dir": "schema"})


def _press_rebuild(store, validation: dict) -> int:
    """The setup page's Rebuild Knowledge Base, to the end of its background
    work. The id of the database it builds from."""
    from admin import routes

    schema = Path("schema")
    schema.mkdir(exist_ok=True)
    (schema / "dbo.ORDERS.md").write_text("# dbo.ORDERS\n", encoding="utf-8")
    db_id = store.save_db_config("azure_sql", "Warehouse", {
        "server": "dw.example.net", "database": "SALES_DW", "user": "reader", "password": "a-password"})
    store.update_client_meta(ACCOUNT, db_config_id=db_id)
    store.update_client_state(ACCOUNT, "SCHEMA_READY", {"schema_dir": str(schema)})

    tasks = BackgroundTasks()
    request = _request(f"/admin/clients/{ACCOUNT}/setup/build-kb", "POST", {"business_desc": "Orders."})
    with patch.object(routes, "_is_auth", return_value=True), \
            patch("core.llm.resolve_provider", return_value=("azure_openai", "gpt-4o", "k", {})), \
            patch("core.knowledge.build_kb", AsyncMock(return_value=1)), \
            patch("core.dispatcher._run_example_validation", AsyncMock(return_value=validation)), \
            patch("core.knowledge.repair_failed_query_patterns", AsyncMock(return_value={"files_rewritten": 0})), \
            patch("core.kb_quality.load_kb_quality_report", return_value={"status": "ready"}), \
            patch.object(routes, "_after_semantic_approval"), \
            patch.object(routes, "_sync_all_log_exports_bg"):
        asyncio.run(routes.admin_build_kb(request, ACCOUNT, tasks, business_desc="Orders."))
        asyncio.run(tasks())
    return db_id


def _build_the_knowledge_base(schema: Path, kb: Path) -> int:
    """The knowledge-base build, with the model and the vector store stubbed.
    How many table documents the model was asked to write."""
    from core import knowledge, release

    written = []

    async def model(system, user, provider, model, api_key, max_tokens=1024, **kwargs):
        if "Table schema for:" in user:
            written.append(user)
        return f"## Overview\nWritten under knowledge-base format {release.KB_FORMAT}.\n", 10, 20

    with patch("core.llm.llm_complete", side_effect=model), \
            patch.object(knowledge, "_embed_kb_files_qdrant", return_value=None):
        asyncio.run(knowledge.build_kb(
            schema_dir=str(schema), kb_dir=str(kb), chroma_dir=ACCOUNT, business_desc="A distributor.",
            provider="azure_openai", model="gpt-4o", api_key="k", extra_kwargs={}, account_id=ACCOUNT))
    return len(written)


def _request(path: str, method: str = "GET", form: dict | None = None) -> Request:
    body = urlencode(form or {}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/x-www-form-urlencoded"),
               (b"content-length", str(len(body)).encode())] if form is not None else []
    return Request({
        "type": "http", "method": method, "path": path, "root_path": "", "scheme": "http",
        "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1), "headers": headers,
    }, receive)


class TestTheVersion:

    def test_the_version_running_has_a_changelog_entry(self):
        from core.release import product_version

        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        assert product_version() == version
        assert re.search(rf"^## {re.escape(version)}$", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)

    def test_its_entry_names_the_knowledge_base_format_it_builds(self):
        from core.release import KB_FORMAT, product_version

        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        entry = changelog.split(f"## {product_version()}\n", 1)[1].split("\n## ", 1)[0]
        assert f"knowledge-base format {KB_FORMAT}." in entry

    def test_health_says_which_build_is_running(self, scratch):
        import main
        from core.release import KB_FORMAT, code_release, product_version

        health = asyncio.run(main.health())
        assert (health["version"], health["release"], health["kb_format"]) == (
            product_version(), code_release(), KB_FORMAT)
        assert main.app.version == product_version()


class TestAKnowledgeBaseAnotherReleaseBuilt:

    def test_the_startup_log_says_to_rebuild(self, scratch, caplog):
        import main

        _built_by_an_earlier_release(scratch)
        main._upgrade_checks()
        told = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
        assert any(ACCOUNT in line and "format 0" in line and "Rebuild Knowledge Base" in line for line in told)

    def test_the_dashboard_inbox_has_it(self, scratch):
        from admin.inbox import build_inbox

        _built_by_an_earlier_release(scratch)
        (item,) = [item for item in build_inbox(scratch.list_clients(), set()) if item["kind"] == "kb-release"]
        assert (item["severity"], item["cta"], item["href"]) == (
            "action", "Rebuild the KB", f"/admin/clients/{ACCOUNT}/setup")

    def test_one_a_newer_release_built_is_named_too(self, scratch):
        from admin.inbox import build_inbox
        from core.release import KB_FORMAT

        scratch.update_client_state(ACCOUNT, "READY", {"kb_dir": "kb", "kb_format": KB_FORMAT + 1,
                                                       "kb_built_version": "9.0.0"})
        (item,) = [item for item in build_inbox(scratch.list_clients(), set()) if item["kind"].startswith("kb-")]
        assert item["label"] == "knowledge base built by a newer release"

    def test_health_counts_it(self, scratch):
        import main

        _built_by_an_earlier_release(scratch)
        assert asyncio.run(main.health())["kb_rebuild_needed"] == 1

    def test_a_workspace_with_no_knowledge_base_has_none_to_rebuild(self, scratch):
        from core.release import kb_rebuild_needed

        scratch.update_client_state(ACCOUNT, "SCHEMA_READY", {"schema_dir": "schema"})
        assert kb_rebuild_needed(scratch.get_client(ACCOUNT)) is None

    def test_a_build_under_this_release_clears_it(self, scratch):
        """From the build route to the notice, with nothing handed between."""
        from admin.inbox import build_inbox
        from core.release import KB_FORMAT, kb_rebuild_needed, product_version

        _built_by_an_earlier_release(scratch)
        assert kb_rebuild_needed(scratch.get_client(ACCOUNT))
        db_id = _press_rebuild(scratch, {"status": "passed", "validated": 1, "failed": 0, "total": 1,
                                         "pass_rate": 100.0})

        client = scratch.get_client(ACCOUNT)
        state = json.loads(client["state_data"])
        assert client["state"] == "READY"
        assert (state["kb_format"], state["kb_built_version"]) == (KB_FORMAT, product_version())
        assert kb_rebuild_needed(client) is None
        assert not [item for item in build_inbox(scratch.list_clients(), {db_id}) if item["kind"].startswith("kb-")]

    def test_a_rebuild_under_a_new_format_rewrites_every_table(self, scratch, monkeypatch):
        """A rebuild skips a table whose inputs have not changed. A release
        that changes what a build writes changes none of them, so the rebuild
        the notice asked for skipped every table, kept the old documents and
        recorded the new format over them."""
        from core import release

        schema, kb = Path("schema"), Path("kb")
        schema.mkdir()
        for table in ("ORDERS", "CUSTOMERS"):
            columns = "\n".join(f"| `COL_{i}` | int | No |  |" for i in range(4))
            (schema / f"{table}.md").write_text(
                f"# dbo.{table}\n\n**Type:** BASE TABLE  **Schema:** dbo\n\n**SQL table name:** `dbo.{table}`\n\n"
                f"**Row count:** 500  **Scale:** Small\n\n## Columns\n\n| Column | Type | Nullable | Distinct Values |\n"
                f"|--------|------|:--------:|-----------------|\n{columns}\n", encoding="utf-8")

        assert _build_the_knowledge_base(schema, kb) == 2
        assert _build_the_knowledge_base(schema, kb) == 0
        monkeypatch.setattr(release, "KB_FORMAT", release.KB_FORMAT + 1)
        assert _build_the_knowledge_base(schema, kb) == 2
        for table in ("ORDERS", "CUSTOMERS"):
            assert f"format {release.KB_FORMAT}." in (kb / f"{table}_kb.md").read_text(encoding="utf-8")

    def test_accepting_a_blocked_build_after_an_upgrade_keeps_the_release_that_built_it(self, scratch, monkeypatch):
        """A build blocked at 82% of its examples, the server upgraded, then
        the admin accepts: the documents are the earlier release's, and the
        notice to rebuild stands."""
        from admin import routes
        from core import release
        from core.release import kb_rebuild_needed

        kb = Path("clients") / ACCOUNT / "kb"
        kb.mkdir(parents=True)
        (kb / "_sql_validation_report.json").write_text(json.dumps({"summary": {"total": 100, "validated": 82}}))
        _press_rebuild(scratch, {"status": "failed", "validated": 82, "failed": 18, "total": 100, "pass_rate": 82.0,
                                 "minimum_pass_rate": 85.0, "report_file": str(kb / "_sql_validation_report.json")})
        client = scratch.get_client(ACCOUNT)
        assert client["state"] == "SCHEMA_READY"
        built_under = json.loads(client["state_data"])["kb_format"]

        monkeypatch.setattr(release, "KB_FORMAT", built_under + 1)
        tasks = BackgroundTasks()
        with patch.object(routes, "_is_auth", return_value=True), \
                patch("core.kb_quality.load_kb_quality_report", return_value={"status": "ready"}), \
                patch("core.dispatcher._run_log_harvest", AsyncMock()), \
                patch.object(routes, "_sync_graph_after_build", AsyncMock(return_value={})), \
                patch("core.semantic_contract.write_contract", return_value={"meta": {}}), \
                patch.object(routes, "notify_kb_build_changed", AsyncMock()), \
                patch.object(routes, "_after_semantic_approval"), \
                patch.object(routes, "_sync_all_log_exports_bg"):
            asyncio.run(routes.admin_accept_kb_validation(_request("/accept", "POST"), ACCOUNT, tasks))
            asyncio.run(tasks())

        client = scratch.get_client(ACCOUNT)
        assert client["state"] == "READY"
        assert json.loads(client["state_data"])["kb_format"] == built_under
        assert kb_rebuild_needed(client)["built_format"] == built_under


class TestSavedCredentials:

    def _save_all(self, store) -> int:
        store.set_system("azure_openai_api_key", "sk-a-secret-key")
        store.save_platform("teams", "Company Teams", {"app_id": "app", "app_password": "a-bot-password", "tenant_id": "tenant"})
        return store.save_db_config("azure_sql", "Warehouse", {
            "server": "dw.example.net", "database": "SALES_DW", "user": "reader", "password": "a-password"})

    def test_all_read_with_the_key_they_were_saved_with(self, scratch):
        self._save_all(scratch)
        assert scratch.unreadable_credentials() == []

    def test_under_another_key_each_is_named_and_none_is_shown(self, scratch, caplog):
        from cryptography.fernet import Fernet

        import main
        from store import crypto

        from admin import credentials as admin_credentials

        self._save_all(scratch)
        admin_credentials.set_password("an-admin-password")
        crypto.KEY_FILE.write_bytes(Fernet.generate_key())
        unreadable = scratch.unreadable_credentials()
        assert {(item["kind"], item["name"], item["key_file"]) for item in unreadable} == {
            ("setting", "azure_openai_api_key", "different"), ("platform", "Company Teams", "different"),
            ("database", "Warehouse", "different")}
        main._upgrade_checks()
        (logged,) = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"
                     and "cannot be read with the key" in record.getMessage()]
        assert "'Warehouse'" in logged and "a-password" not in logged and "sk-a-secret-key" not in logged
        assert asyncio.run(main.health())["unreadable_credentials"] == 3

    def test_the_lists_still_list_them(self, scratch):
        from cryptography.fernet import Fernet

        from store import crypto

        self._save_all(scratch)
        crypto.KEY_FILE.write_bytes(Fernet.generate_key())
        (database,) = scratch.list_db_configs()
        (platform,) = scratch.list_platforms()
        assert (database["name"], database["credentials"], database["credentials_unreadable"]) == (
            "Warehouse", {}, True)
        assert platform["credentials_unreadable"] is True

    def test_the_dashboard_names_them(self, scratch):
        from cryptography.fernet import Fernet

        from admin import routes
        from store import crypto

        self._save_all(scratch)
        crypto.KEY_FILE.write_bytes(Fernet.generate_key())
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_first_run", return_value=False):
            page = asyncio.run(routes.dashboard(_request("/admin"))).body.decode()
        assert "Saved credentials cannot be read" in page
        assert "azure_openai_api_key, Company Teams, Warehouse" in page

    @pytest.mark.parametrize("page,name", [("databases_page", "Warehouse"), ("platforms_page", "Company Teams")])
    def test_their_own_pages_mark_them(self, scratch, page, name):
        from cryptography.fernet import Fernet

        from admin import routes
        from store import crypto

        self._save_all(scratch)

        def render() -> str:
            with patch.object(routes, "_is_auth", return_value=True):
                return asyncio.run(getattr(routes, page)(_request("/admin/page"))).body.decode()

        assert name in render() and "Credentials cannot be read" not in render()
        crypto.KEY_FILE.write_bytes(Fernet.generate_key())
        assert name in render() and "Credentials cannot be read" in render()

    def test_with_no_key_file_the_check_writes_none(self, scratch, caplog):
        import main
        from store import crypto

        self._save_all(scratch)
        crypto.KEY_FILE.unlink()
        unreadable = scratch.unreadable_credentials()
        assert len(unreadable) == 3 and all(item["key_file"] == "missing" for item in unreadable)
        main._upgrade_checks()
        assert not crypto.KEY_FILE.exists()
        assert any(record.levelname == "CRITICAL" and "does not exist" in record.getMessage()
                   for record in caplog.records)

    def test_with_no_key_file_no_page_writes_one(self, scratch):
        """Every page that listed a saved credential wrote a new key when the
        file was missing, so the dashboard's banner said the key had changed,
        never that it had gone -- and sign-in wrote one too."""
        from admin import credentials as admin_credentials
        from admin import routes
        from store import crypto

        self._save_all(scratch)
        admin_credentials.set_password("an-admin-password")
        crypto.KEY_FILE.unlink()
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_first_run", return_value=False):
            dashboard = asyncio.run(routes.dashboard(_request("/admin"))).body.decode()
            for page in ("databases_page", "platforms_page"):
                asyncio.run(getattr(routes, page)(_request("/admin/page")))
        form = {"password": "an-admin-password"}
        asyncio.run(routes.login_submit(_request("/admin/login", "POST", form), password=form["password"]))
        assert not crypto.KEY_FILE.exists()
        assert "key file is missing, so it cannot read 3 saved credentials" in dashboard.replace("&#39;", "'")

    def test_a_save_writes_the_new_key(self, scratch):
        """Entering a credential again is the other way back: the save writes
        a new key, and the ones saved under the lost key say so."""
        from store import crypto

        self._save_all(scratch)
        crypto.KEY_FILE.unlink()
        scratch.set_system("azure_openai_api_key", "sk-entered-again")
        assert crypto.KEY_FILE.exists()
        assert scratch.get_system("azure_openai_api_key") == "sk-entered-again"
        assert {(item["name"], item["key_file"]) for item in scratch.unreadable_credentials()} == {
            ("Company Teams", "different"), ("Warehouse", "different")}

    def test_a_key_file_the_service_cannot_open(self, scratch, monkeypatch, caplog):
        """The service moved to another user, and the key is in the old
        user's home: /health, the dashboard and the startup checks each raised
        the PermissionError, and the knowledge-base warnings after it were
        never logged."""
        import main
        from admin import routes
        from store import crypto

        class InAnotherUsersHome(type(Path())):
            def exists(self, *args, **kwargs):
                raise PermissionError(13, "Permission denied", str(self))

            def read_bytes(self):
                raise PermissionError(13, "Permission denied", str(self))

        self._save_all(scratch)
        _built_by_an_earlier_release(scratch)
        _use_key_file(monkeypatch, InAnotherUsersHome(str(crypto.KEY_FILE)))
        assert asyncio.run(main.health())["unreadable_credentials"] == 3
        with patch.object(routes, "_is_auth", return_value=True), patch.object(routes, "_first_run", return_value=False):
            page = asyncio.run(routes.dashboard(_request("/admin"))).body.decode()
        assert "This server cannot open its key file, so it cannot read 3 saved credentials" in page
        main._upgrade_checks()
        told = {record.levelname: record.getMessage() for record in caplog.records
                if record.name == "querybot" and record.levelname in {"CRITICAL", "WARNING"}}
        assert "cannot be opened by this service's user" in told["CRITICAL"]
        assert "Rebuild Knowledge Base" in told["WARNING"]

    def test_an_unreadable_row_is_logged_once(self, scratch, caplog):
        """The log-export loop lists every database each minute: one error
        line a minute per row buried everything else in the log."""
        from cryptography.fernet import Fernet

        from store import crypto

        self._save_all(scratch)
        crypto.KEY_FILE.write_bytes(Fernet.generate_key())
        for _ in range(3):
            scratch.list_db_configs()
            scratch.get_system("azure_openai_api_key")
        errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
        assert sum("'Warehouse'" in line for line in errors) == 1
        assert sum("azure_openai_api_key" in line for line in errors) == 1
