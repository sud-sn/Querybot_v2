"""
QueryBot signs in to Snowflake as a service user, with a key pair.

Snowflake is blocking single-factor password sign-in for users without MFA,
and a server cannot answer an MFA prompt. A Snowflake connection could only
sign in with a user name and password: the form required a password, the
store refused a connection without one, and the connector passed nothing else.

A connection now has a sign-in method: a key pair (a service user, the
recommended way), a programmatic access token, or a password for connections
made before. A key can be pasted or generated; the private half is stored
encrypted with the other credentials (unlocked, so no passphrase is kept),
never shown or sent back to the browser, and the page shows the public key and
the statement that sets it on the Snowflake user. A generated key goes into the
form, so the connection test signs in with exactly the key that Save stores,
and a working connection changes only when it is saved. The test also says when
the warehouse or database can't be used. Only the secret of the connection's
own method is kept and reaches the connector.

The real connector arguments, form parsing, store, routes and template; the
Snowflake connector itself is a stand-in.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from starlette.requests import Request

from core import snowflake_auth
from core.snowflake_auth import SnowflakeAuthError, connect_kwargs


def _rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key, *, passphrase: str = "", traditional: bool = False) -> str:
    encryption = (serialization.BestAvailableEncryption(passphrase.encode())
                  if passphrase else serialization.NoEncryption())
    fmt = serialization.PrivateFormat.TraditionalOpenSSL if traditional else serialization.PrivateFormat.PKCS8
    return key.private_bytes(serialization.Encoding.PEM, fmt, encryption).decode()


SM2_KEY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqBHM9VAYItBG0wawIBAQQgosicd4kvU/hUJnRR
eenI9FOegudtOir8SlwKD1OBogahRANCAASVqKWj9WjDeqG55Jhm76UJ2pbNvVek
4Gbd36GdMdIXs/q/AwY5OYp9D5vyPAhWDOUNaxpBJ/2rhw52SJIhsgWd
-----END PRIVATE KEY-----"""

BASE = {"account": "org-acct", "user": "QUERYBOT_SVC", "warehouse": "WH", "database": "DW",
        "schema": "SALES", "role": "QUERYBOT_READER"}


class TestTheConnectorArguments:

    def test_a_key_pair_connection_sends_its_key_and_no_password(self):
        key = _rsa_key()
        kwargs = connect_kwargs({**BASE, "auth_method": "keypair", "private_key": _pem(key),
                                 "password": "an-old-password", "token": "an-old-token",
                                 "log_export_enabled": "0", "selected_schema_tables": ["DW.SALES.T"]})
        assert kwargs["authenticator"] == "SNOWFLAKE_JWT"
        loaded = serialization.load_der_private_key(kwargs["private_key"], password=None)
        assert loaded.private_numbers() == key.private_numbers()
        assert "password" not in kwargs and "token" not in kwargs
        assert "log_export_enabled" not in kwargs and "selected_schema_tables" not in kwargs
        assert {k: kwargs[k] for k in BASE} == BASE
        assert kwargs["session_parameters"] == {"QUERY_TAG": "QueryBot"}

    def test_an_encrypted_key_opens_with_its_passphrase(self):
        key = _rsa_key()
        kwargs = connect_kwargs({**BASE, "auth_method": "keypair",
                                 "private_key": _pem(key, passphrase="s3cret"), "private_key_passphrase": "s3cret"})
        assert serialization.load_der_private_key(kwargs["private_key"], password=None).private_numbers() \
            == key.private_numbers()

    def test_an_older_rsa_private_key_format_is_read_too(self):
        kwargs = connect_kwargs({**BASE, "auth_method": "keypair", "private_key": _pem(_rsa_key(), traditional=True)})
        assert kwargs["private_key"]

    @pytest.mark.parametrize("pem, passphrase, said", [
        ("pem-encrypted", "", "encrypted. Enter its passphrase"),
        ("pem-encrypted", "wrong", "passphrase does not open"),
        ("pem-plain", "unneeded", "takes no passphrase"),
        ("not a key at all", "", "PEM format"),
        ("-----BEGIN PRIVATE KEY-----\nnot base64\n-----END PRIVATE KEY-----", "", "could not be read"),
    ])
    def test_a_key_that_cannot_be_used_says_why(self, pem, passphrase, said):
        key = _rsa_key()
        pem = {"pem-encrypted": _pem(key, passphrase="right"), "pem-plain": _pem(key)}.get(pem, pem)
        with pytest.raises(SnowflakeAuthError, match=said):
            connect_kwargs({**BASE, "auth_method": "keypair", "private_key": pem,
                            "private_key_passphrase": passphrase})

    def test_a_key_type_the_library_cannot_read_is_refused_the_same_way(self):
        # An SM2 key (openssl genpkey -algorithm SM2): a server error before.
        with pytest.raises(SnowflakeAuthError, match="needs an RSA key"):
            connect_kwargs({**BASE, "auth_method": "keypair", "private_key": SM2_KEY})

    def test_a_key_that_is_not_rsa_is_refused(self):
        key = ec.generate_private_key(ec.SECP256R1())
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
        with pytest.raises(SnowflakeAuthError, match="needs an RSA key"):
            connect_kwargs({**BASE, "auth_method": "keypair", "private_key": pem})

    def test_a_token_connection_sends_its_token(self):
        kwargs = connect_kwargs({**BASE, "auth_method": "pat", "token": " the-token ", "password": "old"})
        assert kwargs["authenticator"] == "PROGRAMMATIC_ACCESS_TOKEN" and kwargs["token"] == "the-token"
        assert "password" not in kwargs and "private_key" not in kwargs
        with pytest.raises(SnowflakeAuthError, match="programmatic access token"):
            connect_kwargs({**BASE, "auth_method": "pat"})

    def test_a_connection_saved_before_sign_in_methods_still_uses_its_password(self):
        kwargs = connect_kwargs({**BASE, "password": "p", "authenticator": "snowflake"})
        assert kwargs["password"] == "p" and kwargs["authenticator"] == "snowflake"
        assert "private_key" not in kwargs

    def test_every_connection_goes_through_these_arguments(self):
        from core import schema

        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return object()

        with patch("snowflake.connector.connect", side_effect=fake_connect):
            schema._sf_connect({**BASE, "auth_method": "keypair", "private_key": _pem(_rsa_key()),
                                "password": "old"}, max_retries=1)
        assert captured["authenticator"] == "SNOWFLAKE_JWT" and isinstance(captured["private_key"], bytes)
        assert "password" not in captured


class TestTheMethodDecidesWhatIsRequired:

    @pytest.mark.parametrize("creds, secret", [
        ({"auth_method": "keypair"}, "private_key"),
        ({"auth_method": "pat"}, "token"),
        ({"auth_method": "password"}, "password"),
        ({}, "password"),
        ({"private_key": "-----BEGIN"}, "private_key"),
    ])
    def test_the_secret_of_the_method(self, creds, secret):
        assert snowflake_auth.required_fields(creds) == ["account", "user", "warehouse", secret]

    def test_the_store_saves_a_key_pair_connection_without_a_password(self, admin):
        db_id = admin.store.save_db_config("snowflake", "Snowflake", {**BASE, "auth_method": "keypair",
                                                                      "private_key": _pem(_rsa_key())})
        assert admin.store.get_db_config(db_id)["credentials"]["auth_method"] == "keypair"
        with pytest.raises(ValueError, match="private_key"):
            admin.store.save_db_config("snowflake", "Snowflake", {**BASE, "auth_method": "keypair"})


class TestThePublicKey:

    def test_matches_the_fingerprint_snowflake_shows(self):
        from snowflake.connector.auth.keypair import AuthByKeyPair

        key = _rsa_key()
        details = snowflake_auth.public_key_details(_pem(key, passphrase="p"), "p")
        der = key.public_key().public_bytes(serialization.Encoding.DER,
                                            serialization.PublicFormat.SubjectPublicKeyInfo)
        assert details["public_key"] == base64.b64encode(der).decode()
        # The fingerprint the connector puts in its sign-in token, which Snowflake
        # compares with RSA_PUBLIC_KEY_FP.
        assert details["fingerprint"] == AuthByKeyPair.calculate_public_key_fingerprint(key)

    def test_the_statement_names_the_user(self):
        assert snowflake_auth.alter_user_statement("QUERYBOT_SVC", "MIIB") == \
            "ALTER USER QUERYBOT_SVC SET RSA_PUBLIC_KEY='MIIB';"
        assert snowflake_auth.alter_user_statement('odd"user name', "K") == \
            "ALTER USER \"odd\"\"user name\" SET RSA_PUBLIC_KEY='K';"

    def test_a_key_shorter_than_snowflake_accepts_is_refused(self):
        short = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        with pytest.raises(SnowflakeAuthError, match="1024 bits; Snowflake needs 2048"):
            snowflake_auth.public_key_details(_pem(short))

    def test_a_generated_key_signs_in(self):
        pem = snowflake_auth.generate_private_key_pem()
        assert "BEGIN PRIVATE KEY" in pem
        assert connect_kwargs({**BASE, "auth_method": "keypair", "private_key": pem})["private_key"]


@pytest.fixture
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store.crypto

    # The store reads the module's KEY_FILE, fixed when it was first imported.
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    from admin import routes

    routes.store.init_db()
    return routes


def _request(path: str, fields: dict | None = None, method: str = "POST") -> Request:
    body = urlencode(fields or {}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": method, "path": path, "root_path": "",
        "scheme": "http", "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
        "headers": [(b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode())],
    }, receive)


def _same_key(pem_a: str, pem_b: str, passphrase: str = "") -> bool:
    a = serialization.load_pem_private_key(pem_a.encode(), password=None)
    b = serialization.load_pem_private_key(pem_b.encode(), password=passphrase.encode() if passphrase else None)
    return a.private_numbers() == b.private_numbers()


FORM = {"db_type": "snowflake", "name": "Snowflake DW", "sf_account": "org-acct", "sf_user": "QUERYBOT_SVC",
        "sf_warehouse": "WH", "sf_database": "DW", "sf_schema": "SALES", "sf_role": "QUERYBOT_READER"}


def _save(routes, fields: dict):
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(routes.database_save(_request("/admin/databases/save", fields), routes.BackgroundTasks()))


def _page(routes) -> str:
    import html

    with patch.object(routes, "_is_auth", return_value=True):
        page = asyncio.run(routes.databases_page(_request("/admin/databases", method="GET"))).body.decode()
    return html.unescape(page)  # as the browser shows it


class TestTheAdminConsole:

    def test_a_generated_key_is_kept_and_only_its_public_half_shown(self, admin):
        generated = _new_key(admin)["private_key"]
        response = _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": generated})
        assert response.status_code == 303 and "error" not in response.headers["location"]
        stored = admin.store.list_db_configs()[0]["credentials"]
        assert stored["auth_method"] == "keypair" and "BEGIN PRIVATE KEY" in stored["private_key"]

        html = _page(admin)
        details = snowflake_auth.public_key_details(stored["private_key"])
        assert details["fingerprint"] in html
        assert snowflake_auth.alter_user_statement("QUERYBOT_SVC", details["public_key"]) in html
        assert stored["private_key"] not in html
        body_lines = [line for line in stored["private_key"].splitlines() if "-----" not in line]
        assert not any(line in html for line in body_lines)

    def test_an_edit_keeps_the_stored_key_when_none_is_pasted(self, admin):
        pem = _pem(_rsa_key())
        _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": pem})
        db_id = admin.store.list_db_configs()[0]["id"]
        _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "keypair", "sf_role": "OTHER_ROLE"})
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert _same_key(stored["private_key"], pem) and stored["role"] == "OTHER_ROLE"

    def test_a_key_that_cannot_be_read_is_refused_at_save(self, admin):
        response = _save(admin, {**FORM, "sf_auth_method": "keypair",
                                 "sf_private_key": _pem(_rsa_key(), passphrase="right"),
                                 "sf_private_key_passphrase": "wrong"})
        assert "passphrase" in response.headers["location"] and not admin.store.list_db_configs()

    def test_a_password_connection_is_marked_for_switching(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "password", "sf_password": "a-password-123"})
        html = _page(admin)
        assert "Password: switch to a key pair" in html and "a-password-123" not in html

    def test_the_connection_test_explains_a_rejected_key(self, admin):
        fields = {**FORM, "sf_auth_method": "keypair", "sf_private_key": _pem(_rsa_key())}
        with patch.object(admin, "_is_auth", return_value=True), \
                patch("core.schema.test_connection",
                      side_effect=RuntimeError("250001: JWT token is invalid.")):
            import json
            answer = json.loads(asyncio.run(admin.database_test(_request("/admin/databases/test", fields))).body)
        assert answer["status"] == "error"
        assert "RSA_PUBLIC_KEY" in answer["message"] and "JWT token is invalid" in answer["message"]

    def test_a_token_connection_needs_no_password(self, admin):
        fields = {**FORM, "sf_auth_method": "pat", "sf_token": "the-token"}
        with patch.object(admin, "_is_auth", return_value=True), \
                patch("core.schema.test_connection", return_value={"user": "QUERYBOT_SVC"}) as tested:
            import json
            answer = json.loads(asyncio.run(admin.database_test(_request("/admin/databases/test", fields))).body)
        assert answer["status"] == "ok"
        assert tested.call_args.args[0]["token"] == "the-token"


def _post(routes, handler, path: str, fields: dict | None = None, **kwargs):
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(handler(_request(path, fields), **kwargs))


def _only(routes) -> dict:
    (cfg,) = routes.store.list_db_configs()
    return cfg


class TestAStoredKey:

    def test_an_encrypted_key_is_stored_unlocked_without_its_passphrase(self, admin):
        pem = _pem(_rsa_key(), passphrase="s3cret")
        _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": pem,
                      "sf_private_key_passphrase": "s3cret"})
        stored = _only(admin)["credentials"]
        assert _same_key(stored["private_key"], pem, "s3cret")
        assert "ENCRYPTED" not in stored["private_key"] and not stored.get("private_key_passphrase")

    def test_a_new_key_replaces_an_encrypted_one_with_the_passphrase_left_blank(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": _pem(_rsa_key(), passphrase="p1"),
                      "sf_private_key_passphrase": "p1"})
        db_id = _only(admin)["id"]
        replacement = _pem(_rsa_key())
        response = _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "keypair",
                                 "sf_private_key": replacement})
        assert "error" not in response.headers["location"]
        assert _same_key(_only(admin)["credentials"]["private_key"], replacement)

    def test_a_passphrase_typed_while_keeping_the_stored_key_is_ignored(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": _pem(_rsa_key())})
        db_id = _only(admin)["id"]
        response = _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "keypair",
                                 "sf_private_key_passphrase": "typed-by-mistake"})
        assert "error" not in response.headers["location"]


def _new_key(routes) -> dict:
    import json

    with patch.object(routes, "_is_auth", return_value=True):
        answer = asyncio.run(routes.database_new_snowflake_key(_request("/admin/databases/snowflake/new-key")))
    assert answer.headers["cache-control"] == "no-store"
    return json.loads(answer.body)


class TestAGeneratedKey:

    def test_a_new_key_pair_is_handed_to_the_form_and_not_stored(self, admin):
        key = _new_key(admin)
        loaded = serialization.load_pem_private_key(key["private_key"].encode(), password=None)
        assert isinstance(loaded, rsa.RSAPrivateKey) and loaded.key_size == 2048
        public = snowflake_auth.public_key_details(key["private_key"])
        assert (key["public_key"], key["fingerprint"]) == (public["public_key"], public["fingerprint"])
        assert admin.store.list_db_configs() == []

    def test_only_an_admin_gets_one(self, admin):
        with patch.object(admin, "_is_auth", return_value=False):
            answer = asyncio.run(admin.database_new_snowflake_key(_request("/admin/databases/snowflake/new-key")))
        assert answer.status_code == 401 and b"PRIVATE KEY" not in answer.body

    def test_the_test_signs_in_with_the_key_that_save_then_stores(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "password", "sf_password": "old-password"})
        db_id = _only(admin)["id"]
        generated = _new_key(admin)["private_key"]
        fields = {**FORM, "db_id": str(db_id), "sf_auth_method": "keypair", "sf_private_key": generated}

        with patch("core.schema.test_connection", return_value={}) as tested:
            _post(admin, admin.database_test, "/admin/databases/test", fields)
        assert _same_key(tested.call_args.args[0]["private_key"], generated)
        # Testing changes nothing: the connection still signs in as it did.
        assert admin.store.get_db_config(db_id)["credentials"]["auth_method"] == "password"

        _save(admin, fields)
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert stored["auth_method"] == "keypair" and _same_key(stored["private_key"], generated)
        assert "password" not in stored

    def test_an_edit_without_a_new_key_tests_the_stored_key(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "keypair", "sf_private_key": _new_key(admin)["private_key"]})
        cfg = _only(admin)
        with patch("core.schema.test_connection", return_value={}) as tested:
            _post(admin, admin.database_test, "/admin/databases/test",
                  {**FORM, "db_id": str(cfg["id"]), "sf_auth_method": "keypair"})
        assert tested.call_args.args[0]["private_key"] == cfg["credentials"]["private_key"]

    def test_the_form_offers_the_key_pair_and_the_grants(self, admin):
        html = _page(admin)
        assert "Generate a key pair" in html and "GRANT CREATE SCHEMA ON DATABASE" in html


class _Session:
    """A Snowflake session that reports what CURRENT_*() returns."""

    def __init__(self, row):
        self.row = row

    def cursor(self):
        return self

    def execute(self, sql):
        self.sql = sql

    def fetchone(self):
        return self.row

    def close(self):
        pass


class TestTheConnectionTest:

    KEY_PAIR = {**BASE, "auth_method": "keypair"}

    def _test(self, row):
        from core import schema

        with patch("snowflake.connector.connect", return_value=_Session(row)):
            return schema.test_connection({**self.KEY_PAIR, "private_key": _pem(_rsa_key())}, "snowflake")

    def test_a_usable_session_passes_and_says_how_it_signed_in(self):
        details = self._test(("DW", "SALES", "QUERYBOT_SVC", "QUERYBOT_READER", "WH"))
        assert details["warehouse"] == "WH" and details["role"] == "QUERYBOT_READER"
        assert details["sign-in"] == "key pair"

    @pytest.mark.parametrize("row, said", [
        (("DW", "SALES", "QUERYBOT_SVC", "QUERYBOT_READER", None), "cannot use the warehouse WH"),
        ((None, None, "QUERYBOT_SVC", "QUERYBOT_READER", "WH"), "cannot use the database DW"),
    ])
    def test_a_warehouse_or_database_the_role_cannot_use_is_said(self, row, said):
        # Snowflake signs in anyway and leaves the session without one; every
        # question would then fail.
        with pytest.raises(SnowflakeAuthError, match=said):
            self._test(row)

    def test_the_admin_sees_it_on_test_connection(self, admin):
        import json

        fields = {**FORM, "sf_auth_method": "keypair", "sf_private_key": _pem(_rsa_key())}
        with patch.object(admin, "_is_auth", return_value=True), \
                patch("snowflake.connector.connect",
                      return_value=_Session(("DW", "SALES", "QUERYBOT_SVC", "QUERYBOT_READER", None))):
            answer = json.loads(asyncio.run(admin.database_test(_request("/admin/databases/test", fields))).body)
        assert answer["status"] == "error" and "cannot use the warehouse WH" in answer["message"]


class TestWhatAnEditKeeps:

    def test_only_the_secret_of_the_method_in_use_is_kept(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "password", "sf_password": "old-password"})
        db_id = _only(admin)["id"]
        _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "keypair", "sf_private_key": _pem(_rsa_key())})
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert stored["auth_method"] == "keypair" and "password" not in stored

        _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "pat", "sf_token": "the-token"})
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert stored["token"] == "the-token" and "private_key" not in stored

    def test_switching_back_to_password_does_not_reuse_a_removed_one(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "password", "sf_password": "old-password"})
        db_id = _only(admin)["id"]
        _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "pat", "sf_token": "the-token"})
        response = _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "password"})
        assert "error=" in response.headers["location"]
        assert admin.store.get_db_config(db_id)["credentials"]["auth_method"] == "pat"

    @pytest.mark.parametrize("method, secret", [
        ("password", {"sf_password": "old-password"}),
        ("keypair", {"sf_private_key": "KEY"}),
        ("pat", {"sf_token": "the-token"}),
    ])
    def test_a_page_from_before_sign_in_methods_keeps_the_stored_method(self, admin, method, secret):
        secret = {k: (_pem(_rsa_key()) if v == "KEY" else v) for k, v in secret.items()}
        _save(admin, {**FORM, "sf_auth_method": method, **secret})
        db_id = _only(admin)["id"]
        before = admin.store.get_db_config(db_id)["credentials"]
        # Such a page has no method field and leaves the secret blank.
        response = _save(admin, {**FORM, "db_id": str(db_id), "sf_warehouse": "WH2"})
        assert "error" not in response.headers["location"]
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert stored["auth_method"] == method and stored["warehouse"] == "WH2"
        field = {"password": "password", "keypair": "private_key", "pat": "token"}[method]
        assert stored[field] == before[field]

    def test_a_new_connection_from_such_a_page_signs_in_with_its_password(self, admin):
        response = _save(admin, {**FORM, "sf_password": "a-password"})
        assert "error" not in response.headers["location"]
        assert _only(admin)["credentials"]["auth_method"] == "password"

    def test_another_databases_password_is_never_sent(self, admin):
        _save(admin, {"db_type": "azure_sql", "name": "Azure", "az_server": "srv", "az_user": "u",
                      "az_password": "azure-secret"})
        db_id = _only(admin)["id"]
        # The edit form switched from Azure SQL to Snowflake, password left blank.
        with patch("core.schema.test_connection", return_value={}) as tested:
            answer = _post(admin, admin.database_test, "/admin/databases/test",
                           {**FORM, "db_id": str(db_id), "sf_auth_method": "password"})
        assert answer.status_code == 400 and not tested.called

    def test_role_and_database_can_be_cleared(self, admin):
        _save(admin, {**FORM, "sf_auth_method": "password", "sf_password": "old-password"})
        db_id = _only(admin)["id"]
        _save(admin, {**FORM, "db_id": str(db_id), "sf_auth_method": "password", "sf_role": "", "sf_database": ""})
        stored = admin.store.get_db_config(db_id)["credentials"]
        assert stored["role"] == "" and stored["database"] == "" and stored["password"] == "old-password"


class TestErrorsInPlainLanguage:

    @pytest.mark.parametrize("account", ["teamfast-prod", "acme-mfa", "duo-retail"])
    def test_an_account_name_is_not_read_as_an_mfa_prompt(self, account):
        raw = (f"250001 (08001): Failed to connect to DB: {account}.snowflakecomputing.com:443. "
               "Incorrect username or password was specified.")
        assert "did not accept the user name and password" in snowflake_auth.friendly_error(raw)

    def test_an_mfa_prompt_is_explained(self):
        raw = ("250001 (08001): Failed to connect to DB: org-acct.snowflakecomputing.com:443. "
               "Multi-factor authentication is required for this account.")
        assert "multi-factor sign-in" in snowflake_auth.friendly_error(raw)
