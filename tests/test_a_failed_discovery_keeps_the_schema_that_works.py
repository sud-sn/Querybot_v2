"""
A discovery that fails keeps the schema that worked, and the Setup page says why.

Re-discover emptied the workspace's schema folder before it connected. When the connection then failed (a
password changed, a warehouse suspended, a timeout), the folder stayed empty, the workspace stayed "ready", the
Setup page said "No schema files yet" next to "Connected", and every answer failed on tables that were no longer
there. Discovery also ran on the server's event loop, so every page and question waited while it read the tables.

Now discovery reads into a folder of its own and puts the result in place only when it found tables; a failure
leaves the previous schema as it was and is shown on the Setup page with the reason, until a discovery works.
It reads in a worker thread.

The admin's routes run for real against a fresh store; the warehouse read is a stand-in.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.chdir(tmp_path)
    import store
    from admin import routes

    for module in {id(m): m for m in (store, routes.store)}.values():
        for namespace in (module.crypto.__dict__, module.config_store.decrypt.__globals__):
            monkeypatch.setitem(namespace, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    store.save_compliance_profile(account, mode="standard")
    with store.get_db() as conn:
        db_id = int(conn.execute(
            "INSERT INTO db_config (name, db_type, credentials_encrypted) VALUES (?,?,?)",
            (f"wh-{account}", "snowflake", "read-by-the-stub"),
        ).lastrowid)
    store.update_client_meta(account, db_config_id=db_id)
    schema = Path("clients") / account / "schema"
    schema.mkdir(parents=True)
    (schema / "ORDERS.md").write_text("# ORDERS\n", encoding="utf-8")
    (schema / "_schema.json").write_text(json.dumps({"ORDERS": {"columns": [{"name": "ORDER_ID"}]}}),
                                         encoding="utf-8")
    store.update_client_state(account, "SCHEMA_READY", {"schema_dir": str(schema)})
    return account, schema


def _rediscover(account: str, discover) -> str:
    """Runs Re-discover to its end; the id of the run the route started."""
    from admin import routes

    def saved_connection(db_id):
        return {"id": int(db_id), "db_type": "snowflake", "credentials": {"account": "org-acct"}}

    tasks = routes.BackgroundTasks()
    request = MagicMock()
    request.headers = {"accept": "application/json"}
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "get_db_config", side_effect=saved_connection), \
            patch("store.get_db_config", side_effect=saved_connection), \
            patch("core.schema.discover_and_write", side_effect=discover), \
            patch.object(routes, "_profile_suggested_joins", new=AsyncMock(return_value={})), \
            patch.object(routes, "_sync_all_log_exports_bg", return_value=None):
        started = json.loads(asyncio.run(routes.admin_discover_schema(request, account, tasks)).body)
        asyncio.run(tasks())
    return started["run"]


def _state(account: str) -> tuple[str, dict]:
    import store

    client = store.get_client(account)
    return client["state"], json.loads(client.get("state_data") or "{}")


def _files(schema: Path) -> dict[str, str]:
    return {f.name: f.read_text(encoding="utf-8") for f in schema.iterdir()}


class TestAFailedDiscovery:

    def test_keeps_the_tables_that_worked_and_says_why(self, workspace):
        account, schema = workspace
        before = _files(schema)

        def refused(*_args, **_kwargs):
            raise RuntimeError("250001: JWT token is invalid.")

        _rediscover(account, refused)
        assert _files(schema) == before
        state, data = _state(account)
        assert state == "SCHEMA_READY"
        assert "JWT token is invalid" in data["discovery_error"]["message"]
        assert data["discovery_error"]["at"]
        assert not schema.with_name("schema.discovering").exists()

    def test_finding_no_table_changes_nothing(self, workspace):
        account, schema = workspace
        before = _files(schema)
        _rediscover(account, lambda *_a, **_k: 0)
        assert _files(schema) == before
        assert "no tables" in _state(account)[1]["discovery_error"]["message"]


class TestADiscoveryThatWorks:

    def test_replaces_the_files_and_clears_the_last_failure(self, workspace):
        account, schema = workspace
        _rediscover(account, lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("timed out")))
        assert _state(account)[1]["discovery_error"]

        def found(creds, db_type, folder, **_kwargs):
            folder = Path(folder)
            (folder / "CUSTOMERS.md").write_text("# CUSTOMERS\n", encoding="utf-8")
            (folder / "_schema.json").write_text(json.dumps({"CUSTOMERS": {"columns": []}}), encoding="utf-8")
            return 1

        _rediscover(account, found)
        assert sorted(_files(schema)) == ["CUSTOMERS.md", "_schema.json"]
        state, data = _state(account)
        assert state == "SCHEMA_READY" and "discovery_error" not in data

    def test_reads_off_the_event_loop(self, workspace):
        account, _schema = workspace
        threads = []

        def found(creds, db_type, folder, **_kwargs):
            threads.append(threading.current_thread())
            (Path(folder) / "_schema.json").write_text("{}", encoding="utf-8")
            return 1

        _rediscover(account, found)
        assert threads and threads[0] is not threading.main_thread()


class TestTheSetupPage:

    def test_shows_the_failure_and_that_the_old_tables_still_answer(self, workspace):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from admin import routes

        account, _schema = workspace
        _rediscover(account, lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("Warehouse 'WH' is suspended.")))
        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, "_is_auth", return_value=True):
            html = TestClient(app).get(f"/admin/clients/{account}/setup").text
        assert "The last discovery failed" in html
        assert "Warehouse &#39;WH&#39; is suspended." in html or "Warehouse 'WH' is suspended." in html
        assert "still come from the discovery before it" in html


class TestSetupLeadsWhereItSays:
    """Two traps on the way to a first discovery."""

    def test_go_to_settings_opens_the_settings_tab(self, workspace):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from admin import routes
        import store

        account, _schema = workspace
        with store.get_db() as conn:          # no database assigned yet
            conn.execute("UPDATE client SET db_config_id=NULL WHERE account_id=?", (account,))
        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, "_is_auth", return_value=True):
            html = TestClient(app).get(f"/admin/clients/{account}/setup").text
        assert f'href="/admin/clients/{account}/settings" class="btn btn-secondary btn-sm"' in html

    def test_web_portal_only_hides_the_chat_platform_not_the_database(self):
        import re

        page = (Path(__file__).resolve().parents[1] / "admin/templates/client_new.html").read_text(encoding="utf-8")
        # The whole element the switch hides: from its opening tag to the </div> that closes it.
        start = page.index('id="platform-section"')
        depth, at = 1, page.index(">", start) + 1
        for tag in re.finditer(r"<(/?)div\b", page[at:]):
            depth += -1 if tag.group(1) else 1
            if depth == 0:
                hidden = page[start:at + tag.start()]
                break
        assert 'name="platform_type"' in hidden and 'name="db_config_id"' not in hidden

    def test_an_industry_profile_never_saved_is_shown_as_not_chosen(self, workspace):
        # The policy engine treats a workspace with no saved profile as regulated (fail-closed); the page
        # showed Standard as chosen, and the member-name index stayed empty until Save was pressed anyway.
        import re
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from admin import routes
        import store

        account, _schema = workspace
        with store.get_db() as conn:
            conn.execute("DELETE FROM compliance_profile WHERE account_id=?", (account,))
        app = FastAPI()
        app.include_router(routes.router)

        def standard_radio(html: str) -> str:
            return re.search(r'<input type="radio" name="industry" value="standard"[^>]*>', html).group(0)

        with patch.object(routes, "_is_auth", return_value=True):
            html = TestClient(app).get(f"/admin/clients/{account}/setup").text
            assert "No industry profile chosen yet." in html and "checked" not in standard_radio(html)
            store.save_compliance_profile(account, mode="standard")
            html = TestClient(app).get(f"/admin/clients/{account}/setup").text
            assert "No industry profile chosen yet." not in html and "checked" in standard_radio(html)


class TestTheSetupPageFollowsTheRun:
    """The progress bar waited for "schema ready": a re-run read as finished at once, and a first run that failed
    spun for five minutes and then said "timed out". It now follows the run it started to its end."""

    def _status(self, account: str) -> dict:
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from admin import routes

        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, "_is_auth", return_value=True):
            return TestClient(app).get(f"/admin/clients/{account}/setup/status").json()["discovery_run"]

    def test_a_failed_run_is_reported_with_its_reason(self, workspace):
        account, _schema = workspace
        run = _rediscover(account, lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("Warehouse 'WH' is suspended.")))
        status = self._status(account)
        assert status["id"] == run and status["status"] == "failed"
        assert "Warehouse 'WH' is suspended." in status["message"]

    def test_a_run_that_worked_is_done_with_its_table_count(self, workspace):
        account, _schema = workspace

        def found(creds, db_type, folder, **_kwargs):
            (Path(folder) / "ORDERS.md").write_text("# ORDERS\n", encoding="utf-8")
            (Path(folder) / "_schema.json").write_text(json.dumps({"ORDERS": {"columns": []}}), encoding="utf-8")
            return 1

        run = _rediscover(account, found)
        assert self._status(account) == {**self._status(account), "id": run, "status": "done", "tables": 1}

    def test_the_page_follows_the_run_it_started(self):
        page = (Path(__file__).resolve().parents[1] / "admin/templates/client_setup.html").read_text(encoding="utf-8")
        assert "_pollUntilDiscoveryDone(runId)" in page and "run.status === 'failed'" in page
