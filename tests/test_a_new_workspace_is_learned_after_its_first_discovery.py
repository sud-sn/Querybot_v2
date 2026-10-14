"""A new workspace's first discovery starts its first Learn; a later one leaves Learn to the admin.

The new core is the only engine, and a workspace it has not learned tells its readers it is being set up. An admin
who ran discovery then had to find "Learn" on another page before anyone could ask a question. Now the first
discovery that works starts Learn in the background; a later discovery only marks what was learned out of date
(learning again replaces it, which is the admin's call), and nothing starts while Learn already runs.

The real store; the Learn itself is stood in for.
"""

from __future__ import annotations

import threading
import uuid

import pytest

import store
from admin import core2_routes


@pytest.fixture()
def workspace(monkeypatch):
    store.init_db()
    account = f"learn-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "zoom")
    db_id = store.save_db_config("azure_sql", "invented", {"server": "invented", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    started: list[tuple[str, str]] = []
    done = threading.Event()

    def learn(account_id, when=None):
        started.append((account_id, when))
        done.set()

    monkeypatch.setattr(core2_routes, "_run_build", learn)
    monkeypatch.setenv("SOLVIZ_LEARN_AFTER_DISCOVERY", "1")
    yield account, db_id, started, done


def test_the_first_discovery_starts_learn(workspace):
    account, db_id, started, done = workspace
    assert core2_routes.learn_after_first_discovery(account)
    assert done.wait(5) and started[0][0] == account
    assert store.latest_core2_build(account, db_id)["status"] == "running", "shown as learning at once"


def test_a_workspace_already_learned_is_left_to_the_admin(workspace):
    account, db_id, started, _ = workspace
    store.save_core2_model(account, db_id, "{}")
    assert not core2_routes.learn_after_first_discovery(account) and not started


def test_nothing_starts_while_learn_runs(workspace):
    account, db_id, started, _ = workspace
    store.start_core2_build(account, db_id)
    assert not core2_routes.learn_after_first_discovery(account) and not started


def test_no_database_no_learn(workspace):
    account, _db_id, started, _ = workspace
    with store.get_db() as conn:
        conn.execute("UPDATE client SET db_config_id=NULL WHERE account_id=?", (account,))
    assert not core2_routes.learn_after_first_discovery(account) and not started


def test_it_can_be_turned_off(workspace, monkeypatch):
    account, _db_id, started, _ = workspace
    monkeypatch.setenv("SOLVIZ_LEARN_AFTER_DISCOVERY", "0")
    assert not core2_routes.learn_after_first_discovery(account) and not started


def test_discovery_calls_it_after_it_worked():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "admin/routes.py").read_text(encoding="utf-8")
    body = source[source.index("async def _do_discover():"):source.index("bg.add_task(_do_discover)")]
    success, failure = body.split("except Exception as e:\n            log.error(\"Admin schema discovery failed")[:2]
    assert "learn_after_first_discovery" in success and "learn_after_first_discovery" not in failure
    assert success.index('save_state(account_id, "SCHEMA_READY"') < success.index("learn_after_first_discovery")
