"""The installers' settings file (deploy/envfile.py): a setting is added when it has no value, and a value already
there is never changed, since a session secret that changes signs everyone out and a pseudonym secret that changes
turns every alias shown before into a different one. Read back as the app reads it (python-dotenv, which uvicorn's
--env-file uses)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import envfile  # noqa: E402

SECRETS = ["SESSION_SECRET", "PORTAL_SESSION_SECRET", "ADMIN_SESSION_SECRET", "PII_PSEUDONYM_SECRET"]


def test_a_new_file_gets_each_secret_once_and_a_strong_one(tmp_path):
    path = tmp_path / ".env"
    added = envfile.ensure(path, secrets_wanted=SECRETS, defaults={"QUERYBOT_KEY_FILE": "/home/qb/.querybot_key"})
    assert added == SECRETS + ["QUERYBOT_KEY_FILE"]
    values = dotenv_values(path)
    assert all(len(values[n]) == 64 and int(values[n], 16) >= 0 for n in SECRETS)
    assert len({values[n] for n in SECRETS}) == 4            # independent, never one value reused
    assert values["QUERYBOT_KEY_FILE"] == "/home/qb/.querybot_key"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_running_it_again_changes_nothing(tmp_path):
    path = tmp_path / ".env"
    envfile.ensure(path, secrets_wanted=SECRETS)
    before = path.read_bytes()
    assert envfile.ensure(path, secrets_wanted=SECRETS, defaults={"SESSION_SECRET": "other"}) == []
    assert path.read_bytes() == before


def test_a_value_already_there_is_kept_and_a_placeholder_is_filled_where_it_stands(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# QueryBot\nSESSION_SECRET=keep-me\nPORTAL_SESSION_SECRET=CHANGE_ME\nADMIN_SESSION_SECRET=\n"
                    "QUERYBOT_QUERY_TIMEOUT_SECONDS=300\n", encoding="utf-8")
    added = envfile.ensure(path, secrets_wanted=SECRETS)
    assert added == ["PORTAL_SESSION_SECRET", "ADMIN_SESSION_SECRET", "PII_PSEUDONYM_SECRET"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# QueryBot" and lines[1] == "SESSION_SECRET=keep-me"
    assert lines[2].startswith("PORTAL_SESSION_SECRET=") and lines[3].startswith("ADMIN_SESSION_SECRET=")
    assert lines[4] == "QUERYBOT_QUERY_TIMEOUT_SECONDS=300" and lines[5].startswith("PII_PSEUDONYM_SECRET=")
    assert dotenv_values(path)["SESSION_SECRET"] == "keep-me"


def test_windows_paths_and_spaces_read_back_as_written(tmp_path):
    path = tmp_path / ".env"
    wanted = {"QUERYBOT_KEY_FILE": r"C:\Users\qbot\.querybot_key",
              "PLAYWRIGHT_BROWSERS_PATH": r"C:\Users\qbot\AppData\Local\ms-playwright",
              "PORTAL_BASE_URL": "https://bot.example.com:8001", "NOTE": 'two words "quoted"'}
    envfile.ensure(path, defaults=wanted)
    assert dotenv_values(path) == wanted


def test_no_byte_order_mark_and_an_existing_one_is_read(tmp_path):
    path = tmp_path / ".env"
    envfile.ensure(path, secrets_wanted=["SESSION_SECRET"])
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    bom = tmp_path / "bom.env"
    bom.write_bytes(b"\xef\xbb\xbfSESSION_SECRET=keep-me\n")       # as Windows PowerShell 5 writes UTF-8
    assert envfile.ensure(bom, secrets_wanted=["SESSION_SECRET"]) == []


def test_the_command_line_prints_names_never_values(tmp_path, capsys):
    path = tmp_path / "querybot.env"
    assert envfile.main([str(path), "--secret", "SESSION_SECRET", "--default", "QUERYBOT_KEY_FILE=/k"]) == 0
    out = capsys.readouterr().out
    assert "SESSION_SECRET" in out and dotenv_values(path)["SESSION_SECRET"] not in out
