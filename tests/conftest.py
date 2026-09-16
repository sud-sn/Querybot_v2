"""Cross-test isolation for the one piece of state that is not a fixture.

`core.i18n.activate_language` sets a ContextVar that stays set until it is
deactivated, and it is deliberately process-wide: the pipeline activates the
reader's language once and every bare `_t()` for the rest of that answer picks
it up. In a test process that same design makes a language a test leaves behind
the next module's default, and the failure lands somewhere else entirely -- a
test asserting an English phrase, three modules later, reading French.

That happened while this branch was being written: a new test looped over
SUPPORTED_LANGUAGES with a bare `activate_language(lang)`, left "fr" active, and
broke `test_the_provenance_says_it_is_not_an_approved_default` in another file.
Nothing was wrong with either test on its own, and both passed when run alone.

So the check is here, once, as an assertion rather than a silent reset: a reset
would let the leak keep happening invisibly, and the leak is the defect. Use a
scoped activation (`activate_language` paired with `deactivate_language`, or a
`with` helper) or pass `lang=` explicitly.

The second piece is the database. The store opens every connection at the path
the environment names at that moment (QUERYBOT_DB_PATH, then DB_PATH, then the
application's own data/querybot.db), and a test that names no path of its own
would open the application's. Dozens of modules point themselves at a mkdtemp()
of their own; the ones that do not used to leave their throwaway clients in
data/querybot.db. So before collection, and again before every test, any path
that is not under the temp directory is replaced with one database for the
run -- the application's file is never opened by a test process, whatever the
module does or fails to do. A path a module set under the temp directory is
left alone.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import pytest

_DB_ENV = ("QUERYBOT_DB_PATH", "DB_PATH")
_APP_DEFAULT = "data/querybot.db"
_run_db: dict = {"path": None, "initialised": False}


def _effective_db_path() -> tuple[str, bool]:
    """(path the store would open now, whether something set it explicitly)."""
    for key in _DB_ENV:
        value = os.environ.get(key)
        if value:
            return value, True
    return _APP_DEFAULT, False


def _point_tests_away_from_the_application_database() -> bool:
    path, explicit = _effective_db_path()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if Path(path).resolve().is_relative_to(temp_root):
        return False
    if _run_db["path"] is None:
        _run_db["path"] = os.path.join(tempfile.mkdtemp(prefix="qb-tests-"), "querybot.db")
    if explicit:
        warnings.warn(
            f"tests never open {path!r}; this run uses {_run_db['path']!r} instead "
            "(set QUERYBOT_DB_PATH under the temp directory to choose your own)",
            stacklevel=2,
        )
    for key in _DB_ENV:
        os.environ[key] = _run_db["path"]
    return True


def pytest_configure(config):
    _point_tests_away_from_the_application_database()


@pytest.fixture(autouse=True)
def _never_the_application_database():
    _point_tests_away_from_the_application_database()
    if not _run_db["initialised"] and os.environ.get("QUERYBOT_DB_PATH") == _run_db["path"]:
        # A test that relied on the application's database already having
        # its tables finds them in this one too.
        import store

        store.init_db()
        _run_db["initialised"] = True
    yield


@pytest.fixture(autouse=True)
def _no_language_leaks_between_tests():
    yield
    from core import i18n

    active = i18n.get_active_language()
    if active not in (None, "", "en"):
        i18n.activate_language("en")
        pytest.fail(
            f"this test left {active!r} as the process-wide active language, so "
            "the next test's bare _t() calls render in it -- activate the "
            "language in a scope that ends (deactivate_language with the token "
            "activate_language returned) or pass lang= explicitly"
        )
