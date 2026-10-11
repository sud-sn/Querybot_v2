"""
Email goes out the way a Super admin set it up, and every message is in the delivery log.

    core/mail/settings.py   Microsoft 365 (Graph) or SMTP, kept encrypted; allowed domains
    core/mail/transport.py  one message to the service; what failed for good, what is worth another try
    core/mail/outbox.py     queued, sent each minute, tried again 1, 5, 30, 120, 360 minutes later, then failed
    core/mail/links.py      a signed link that stops an email, without signing in
    admin/email_routes.py   the settings page, a test email, the log, Send again

Nothing here reaches a real mail service: an SMTP server runs in this process, and Microsoft's endpoints are
answered by a stand-in (httpx.MockTransport).
"""

from __future__ import annotations

import base64
import datetime as dt
import email as email_lib
import json
import os
import socketserver
import sqlite3
import threading
from email import policy

import httpx
import pytest
from starlette.testclient import TestClient

GOOD = "a-long-admin-password"


# ── An SMTP server in this process ───────────────────────────────────────────

class _Inbox:
    def __init__(self):
        self.messages: list[email_lib.message.EmailMessage] = []
        self.refuse: dict[str, str] = {}          # address -> reply to RCPT, e.g. "550 no such user"
        self.logins: list[str] = []
        self.lock = threading.Lock()

    def to(self, address):
        return [m for m in self.messages if address in str(m["To"])]


def _smtp_server(inbox: _Inbox, *, auth: tuple[str, str] | None = None):
    class Handler(socketserver.StreamRequestHandler):
        def _say(self, line):
            self.wfile.write((line + "\r\n").encode())

        def handle(self):
            self._say("220 test ESMTP")
            rcpts, mail_from = [], None
            while True:
                raw = self.rfile.readline()
                if not raw:
                    return
                line = raw.decode().rstrip("\r\n")
                verb = line.split(" ", 1)[0].upper()
                if verb in ("EHLO", "HELO"):
                    self.wfile.write(b"250-test\r\n" + (b"250-AUTH PLAIN\r\n" if auth else b"") + b"250 8BITMIME\r\n")
                elif verb == "AUTH":
                    parts = line.split(" ")
                    decoded = base64.b64decode(parts[2]).decode().split("\x00") if len(parts) > 2 else []
                    inbox.logins.append(decoded[1] if len(decoded) > 1 else "")
                    if auth and decoded[1:] == list(auth):
                        self._say("235 ok")
                    else:
                        self._say("535 authentication failed")
                elif verb == "MAIL":
                    mail_from, rcpts = line, []
                    self._say("250 ok")
                elif verb == "RCPT":
                    address = line.split(":", 1)[1].strip().strip("<>")
                    if address in inbox.refuse:
                        self._say(inbox.refuse[address])
                    else:
                        rcpts.append(address)
                        self._say("250 ok")
                elif verb == "DATA":
                    self._say("354 go on")
                    lines = []
                    while True:
                        chunk = self.rfile.readline().decode()
                        if chunk in (".\r\n", ".\n"):
                            break
                        lines.append(chunk[1:] if chunk.startswith("..") else chunk)
                    with inbox.lock:
                        inbox.messages.append(email_lib.message_from_string("".join(lines), policy=policy.default))
                    self._say("250 queued")
                elif verb == "RSET":
                    rcpts = []
                    self._say("250 ok")
                elif verb == "QUIT":
                    self._say("221 bye")
                    return
                else:
                    self._say("502 not here")

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("PORTAL_BASE_URL", "https://bi.example.com")
    import main
    from core.mail import transport

    main.store.init_db()
    transport.forget_tokens()
    return main


@pytest.fixture
def inbox():
    return _Inbox()


@pytest.fixture
def smtp(app, inbox):
    server = _smtp_server(inbox)
    from core.mail import settings

    settings.save({"provider": "smtp", "from_address": "reports@example.com", "from_name": "Example BI",
                   "smtp_host": "127.0.0.1", "smtp_port": str(server.server_address[1]), "smtp_security": "none"})
    yield server
    server.shutdown()
    server.server_close()


def _browser(app) -> TestClient:
    """Over https, as PORTAL_BASE_URL says the server is: its session cookies are Secure."""
    return TestClient(app.app, base_url="https://testserver")


def _outbox(**kw):
    from core.mail import outbox

    return outbox.queue(**{"kind": "test", "to": "ada@example.com", "subject": "Hello", "text": "Hi", **kw})


def _email(email_id):
    from store import email_store

    return email_store.get_email(email_id)


def _shift(email_id, minutes):
    """The message's next try brought forward (the clock moved on)."""
    from store.db import get_db

    with get_db() as conn:
        conn.execute("UPDATE email_outbox SET next_attempt_at=datetime('now', ?) WHERE id=?", (f"-{minutes} minutes",
                                                                                              int(email_id)))


# ── Settings ─────────────────────────────────────────────────────────────────

class TestTheSettings:

    def test_nothing_is_sent_until_email_is_set_up(self, app):
        from core.mail import outbox

        email_id = _outbox()
        assert _email(email_id)["status"] == "skipped"
        assert "not set up" in _email(email_id)["last_error"]
        assert outbox.deliver_due() == {"sent": 0, "queued": 0, "failed": 0}

    def test_what_is_still_needed_is_named(self, app):
        from core.mail import settings

        settings.save({"provider": "graph", "from_address": "reports@example.com", "graph_tenant_id": "t"})
        assert settings.load().missing() == ["the application (client) ID", "the client secret"]
        assert not settings.configured()

    def test_secrets_are_kept_encrypted_and_kept_when_left_blank(self, app):
        from core.mail import settings

        settings.save({"provider": "smtp", "from_address": "r@example.com", "smtp_host": "h",
                       "smtp_password": "the-smtp-secret"})
        settings.save({"provider": "smtp", "smtp_password": ""})
        assert settings.load().smtp_password == "the-smtp-secret"
        raw = sqlite3.connect(os.environ["QUERYBOT_DB_PATH"]).execute(
            "SELECT value_encrypted FROM system_config WHERE key='email_smtp_password'").fetchone()[0]
        assert "the-smtp-secret" not in str(raw)

    def test_allowed_domains_cover_their_subdomains_only(self, app):
        from core.mail import settings

        settings.save({"allowed_domains": "Example.com; @partner.org, nonsense"})
        config = settings.load()
        assert config.allowed_domains == ("example.com", "partner.org")
        assert config.allows("a@example.com") and config.allows("b@eu.example.com") and config.allows("c@partner.org")
        assert not config.allows("d@notexample.com") and not config.allows("e@gmail.com") and not config.allows("x")


# ── SMTP ─────────────────────────────────────────────────────────────────────

class TestSmtp:

    def test_a_message_goes_with_its_html_text_and_attachment(self, app, smtp, inbox):
        from core.mail import outbox

        email_id = _outbox(subject="Weekly sales", text="Plain words", html="<p>Rich words</p>",
                           attachments=(("Sales.pdf", "application/pdf", b"%PDF-1.7 tiny"),),
                           unsubscribe_url="https://bi.example.com/portal/unsubscribe/x")
        assert outbox.deliver_due()["sent"] == 1
        sent = _email(email_id)
        assert sent["status"] == "sent" and sent["provider"] == "smtp" and sent["sent_at"] and sent["attempts"] == 1
        message = inbox.to("ada@example.com")[0]
        assert message["Subject"] == "Weekly sales" and "Example BI" in message["From"]
        assert message["List-Unsubscribe"] == "<https://bi.example.com/portal/unsubscribe/x>"
        assert message.get_body(("plain",)).get_content().strip() == "Plain words"
        assert "Rich words" in message.get_body(("html",)).get_content()
        attached = [part for part in message.iter_attachments()]
        assert attached[0].get_filename() == "Sales.pdf" and attached[0].get_content() == b"%PDF-1.7 tiny"

    def test_a_refused_address_fails_at_once(self, app, smtp, inbox):
        from core.mail import outbox

        inbox.refuse["nobody@example.com"] = "550 no such user"
        email_id = _outbox(to="nobody@example.com")
        assert outbox.deliver_due()["failed"] == 1
        assert _email(email_id)["status"] == "failed" and "refused the address" in _email(email_id)["last_error"]

    def test_a_busy_server_is_tried_again_later_then_given_up(self, app, smtp, inbox):
        from core.mail import outbox

        inbox.refuse["busy@example.com"] = "451 try later"
        email_id = _outbox(to="busy@example.com")
        assert outbox.deliver_due()["queued"] == 1
        waiting = _email(email_id)
        assert waiting["status"] == "queued" and waiting["attempts"] == 1 and "put off" in waiting["last_error"]
        assert outbox.deliver_due()["queued"] == 0                       # not due for a minute
        for _ in range(5):
            _shift(email_id, 400)
            outbox.deliver_due()
        given_up = _email(email_id)
        assert given_up["status"] == "failed" and given_up["attempts"] == 6
        assert given_up["last_error"].startswith("Gave up after 5 tries")

    def test_a_put_off_message_goes_when_the_server_takes_it(self, app, smtp, inbox):
        from core.mail import outbox

        inbox.refuse["busy@example.com"] = "421 later"
        email_id = _outbox(to="busy@example.com")
        outbox.deliver_due()
        del inbox.refuse["busy@example.com"]
        _shift(email_id, 2)
        assert outbox.deliver_due()["sent"] == 1 and _email(email_id)["attempts"] == 2

    def test_a_wrong_password_fails_for_good(self, app, inbox):
        from core.mail import outbox, settings

        server = _smtp_server(inbox, auth=("bot", "right"))
        try:
            settings.save({"provider": "smtp", "from_address": "r@example.com", "smtp_host": "127.0.0.1",
                           "smtp_port": str(server.server_address[1]), "smtp_security": "none",
                           "smtp_username": "bot", "smtp_password": "wrong"})
            email_id = _outbox()
            outbox.deliver_due()
            assert _email(email_id)["status"] == "failed" and "user and password" in _email(email_id)["last_error"]
            settings.save({"smtp_password": "right"})
            from store import email_store

            assert email_store.requeue_email(email_id)
            assert outbox.deliver_due()["sent"] == 1 and inbox.logins[-1] == "bot"
        finally:
            server.shutdown()
            server.server_close()

    def test_starttls_asked_of_a_server_without_it_says_so(self, app, inbox):
        from core.mail import outbox, settings

        server = _smtp_server(inbox)
        try:
            settings.save({"provider": "smtp", "from_address": "r@example.com", "smtp_host": "127.0.0.1",
                           "smtp_port": str(server.server_address[1]), "smtp_security": "starttls"})
            email_id = _outbox()
            outbox.deliver_due()
            assert "does not offer STARTTLS" in _email(email_id)["last_error"]
        finally:
            server.shutdown()
            server.server_close()

    def test_an_unreachable_server_is_tried_again(self, app):
        from core.mail import outbox, settings

        settings.save({"provider": "smtp", "from_address": "r@example.com", "smtp_host": "127.0.0.1",
                       "smtp_port": "1", "smtp_security": "none"})
        email_id = _outbox()
        outbox.deliver_due()
        assert _email(email_id)["status"] == "queued" and "could not be reached" in _email(email_id)["last_error"]


# ── Microsoft Graph ──────────────────────────────────────────────────────────

class _Microsoft:
    def __init__(self, replies=None):
        self.calls: list[httpx.Request] = []
        self.replies = list(replies or [])
        self.tokens = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if "login.microsoftonline.com" in str(request.url):
            self.tokens += 1
            return httpx.Response(200, json={"access_token": f"token-{self.tokens}", "expires_in": 3600})
        if self.replies:
            status, body = self.replies.pop(0)
            return httpx.Response(status, json=body)
        return httpx.Response(202, headers={"request-id": "graph-request-1"})


@pytest.fixture
def graph(app):
    from core.mail import settings

    settings.save({"provider": "graph", "from_address": "reports@example.com", "from_name": "Example BI",
                   "graph_tenant_id": "tenant-1", "graph_client_id": "client-1", "graph_client_secret": "s3cret"})


def _http(microsoft):
    return httpx.Client(transport=httpx.MockTransport(microsoft))


class TestMicrosoftGraph:

    def test_the_app_signs_in_and_sends_as_the_mailbox(self, app, graph):
        from core.mail import outbox

        microsoft = _Microsoft()
        email_id = _outbox(html="<p>Hi</p>", attachments=(("a.pdf", "application/pdf", b"PDF"),))
        assert outbox.deliver_due(http=_http(microsoft))["sent"] == 1
        login, send = microsoft.calls
        assert "tenant-1" in str(login.url) and b"client_secret=s3cret" in login.content
        assert str(send.url) == "https://graph.microsoft.com/v1.0/users/reports@example.com/sendMail"
        assert send.headers["authorization"] == "Bearer token-1"
        body = json.loads(send.content)
        assert body["saveToSentItems"] is False
        message = body["message"]
        assert message["toRecipients"] == [{"emailAddress": {"address": "ada@example.com"}}]
        assert message["body"] == {"contentType": "HTML", "content": "<p>Hi</p>"}
        assert base64.b64decode(message["attachments"][0]["contentBytes"]) == b"PDF"
        assert _email(email_id)["provider_message_id"] == "graph-request-1"

    def test_the_sign_in_is_reused_until_it_expires(self, app, graph):
        from core.mail import outbox

        microsoft = _Microsoft()
        _outbox(), _outbox(to="bob@example.com")
        outbox.deliver_due(http=_http(microsoft))
        assert microsoft.tokens == 1

    def test_no_permission_to_send_fails_for_good_and_says_why(self, app, graph):
        from core.mail import outbox

        microsoft = _Microsoft([(403, {"error": {"code": "ErrorAccessDenied", "message": "Access is denied."}})])
        email_id = _outbox()
        outbox.deliver_due(http=_http(microsoft))
        error = _email(email_id)["last_error"]
        assert _email(email_id)["status"] == "failed" and "give it Mail.Send" in error and "ErrorAccessDenied" in error

    def test_throttling_is_tried_again(self, app, graph):
        from core.mail import outbox

        microsoft = _Microsoft([(429, {"error": {"code": "ApplicationThrottled", "message": "slow down"}})])
        email_id = _outbox()
        outbox.deliver_due(http=_http(microsoft))
        assert _email(email_id)["status"] == "queued"
        _shift(email_id, 2)
        assert outbox.deliver_due(http=_http(microsoft))["sent"] == 1

    def test_a_wrong_secret_fails_for_good(self, app, graph):
        from core.mail import outbox

        def refuse(request):
            return httpx.Response(401, json={"error": "invalid_client", "error_description": "AADSTS7000215: bad"})

        email_id = _outbox()
        outbox.deliver_due(http=httpx.Client(transport=httpx.MockTransport(refuse)))
        assert _email(email_id)["status"] == "failed" and "AADSTS7000215" in _email(email_id)["last_error"]


# ── The outbox ───────────────────────────────────────────────────────────────

class TestTheOutbox:

    def test_outside_the_allowed_domains_is_not_sent(self, app, smtp, inbox):
        from core.mail import settings

        settings.save({"allowed_domains": "example.com"})
        email_id = _outbox(to="someone@gmail.com")
        assert _email(email_id)["status"] == "skipped" and "outside the domains" in _email(email_id)["last_error"]

    def test_the_same_message_is_queued_once(self, app, smtp):
        assert _outbox(dedupe_key="follow:1:2026-10-01") is not None
        assert _outbox(dedupe_key="follow:1:2026-10-01") is None

    def test_two_senders_never_send_one_message_twice(self, app, smtp, inbox):
        from core.mail import outbox

        for i in range(6):
            _outbox(to=f"p{i}@example.com")
        threads = [threading.Thread(target=outbox.deliver_due) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(str(m["To"]) for m in inbox.messages) == [f"p{i}@example.com" for i in range(6)]

    def test_a_message_left_sending_by_a_restart_goes_back_in_the_queue(self, app, smtp, inbox):
        from core.mail import outbox
        from store import email_store
        from store.db import get_db

        email_id = _outbox()
        assert email_store.claim_email(email_id)
        with get_db() as conn:
            conn.execute("UPDATE email_outbox SET claimed_at=datetime('now','-20 minutes') WHERE id=?", (email_id,))
        assert outbox.deliver_due()["sent"] == 1

    def test_someone_who_stopped_an_email_is_not_sent_it(self, app, smtp):
        from store import email_store

        email_store.opt_out("Ada@Example.com", "follow", "7")
        assert _email(_outbox(opt_out_kind="follow", ref="7"))["status"] == "skipped"
        assert _email(_outbox(opt_out_kind="follow", ref="8"))["status"] == "queued"
        assert _email(_outbox())["status"] == "queued"            # an email they cannot stop (a test, an invitation)


class TestTheLinks:

    def test_a_link_cannot_be_made_up_or_changed(self, app):
        from core.mail import links

        token = links.unsubscribe_token("ada@example.com", "follow", "7")
        assert links.read(token) == {"a": "ada@example.com", "k": "follow", "r": "7"}
        body, mac = token.split(".")
        forged = base64.urlsafe_b64encode(json.dumps({"a": "bob@example.com", "k": "follow", "r": "7"}).encode())
        assert links.read(forged.decode().rstrip("=") + "." + mac) is None
        assert links.read("nonsense") is None and links.read("") is None

    def test_the_signing_key_survives_a_restart(self, app):
        from core.mail import links

        token = links.unsubscribe_token("ada@example.com", "follow", "7")
        assert app.store.get_system("email_link_secret")
        assert links.read(token) is not None


# ── The admin page ───────────────────────────────────────────────────────────

def _super(app):
    email = f"{os.urandom(3).hex()}@example.com"
    app.store.create_admin(email, "Sam", "super_admin", password=GOOD)
    client = _browser(app)
    assert client.post("/admin/login", data={"email": email, "password": GOOD},
                       follow_redirects=False).status_code == 303
    return client, email


class TestTheAdminPage:

    def test_a_super_admin_sets_up_smtp_and_sends_a_test(self, app, inbox):
        server = _smtp_server(inbox)
        try:
            client, me = _super(app)
            page = client.get("/admin/system/email")
            assert page.status_code == 200 and "Not set up yet" in page.text and f'value="{me}"' in page.text
            client.post("/admin/system/email/save", data={
                "provider": "smtp", "from_address": "reports@example.com", "from_name": "Example BI",
                "smtp_host": "127.0.0.1", "smtp_port": str(server.server_address[1]), "smtp_security": "none",
                "smtp_username": "", "smtp_password": "", "reply_to": "", "allowed_domains": "example.com"})
            tested = client.post("/admin/system/email/test", data={"to": me})
            assert "Sent</b> to " + me in tested.text
            assert inbox.to(me)[0]["Subject"] == "QueryBot can send email"
            log_page = client.get("/admin/system/email?status=sent").text
            assert "QueryBot can send email" in log_page and "SMTP 127.0.0.1" in log_page
        finally:
            server.shutdown()
            server.server_close()

    def test_a_failed_email_is_sent_again_from_the_log(self, app, smtp, inbox):
        from core.mail import outbox

        inbox.refuse["x@example.com"] = "550 nope"
        email_id = _outbox(to="x@example.com")
        outbox.deliver_due()
        del inbox.refuse["x@example.com"]
        client, _ = _super(app)
        assert "Send again" in client.get(f"/admin/system/email/{email_id}").text
        client.post(f"/admin/system/email/{email_id}/retry")
        assert _email(email_id)["status"] == "sent"

    def test_the_secret_is_never_shown(self, app):
        client, _ = _super(app)
        client.post("/admin/system/email/save", data={"provider": "graph", "from_address": "r@example.com",
                                                      "graph_tenant_id": "t", "graph_client_id": "c",
                                                      "graph_client_secret": "very-secret-value"})
        page = client.get("/admin/system/email").text
        assert "very-secret-value" not in page and "Saved: leave blank to keep it" in page

    def test_only_a_super_admin_opens_it(self, app):
        account_id = f"acct{os.urandom(3).hex()}"
        app.store.upsert_client(account_id, "Retail")
        app.store.create_admin("ws@example.com", "W", "workspace_admin", password=GOOD, workspaces=[account_id])
        client = _browser(app)
        client.post("/admin/login", data={"email": "ws@example.com", "password": GOOD})
        assert client.get("/admin/system/email", follow_redirects=False).status_code == 403
        assert client.post("/admin/system/email/save", data={"provider": "smtp"},
                           follow_redirects=False).status_code == 403


class TestInvitations:

    def test_an_invitation_is_emailed_with_its_link(self, app, smtp, inbox):
        account_id = f"acct{os.urandom(3).hex()}"
        app.store.upsert_client(account_id, "Retail")
        client, _ = _super(app)
        page = client.post("/admin/admins/invite", data={"email": "newbie@example.com", "role": "data_steward",
                                                         "workspaces": [account_id]}).text
        assert "Emailed to newbie@example.com" in page
        message = inbox.to("newbie@example.com")[0]
        link = next(line for line in message.get_body(("plain",)).get_content().splitlines() if "/admin/invite/" in line)
        token = link.rsplit("/admin/invite/", 1)[1].strip()
        assert app.store.admin_for_invite(token)["email"] == "newbie@example.com"

    def test_with_email_off_the_link_is_shown_to_copy(self, app):
        account_id = f"acct{os.urandom(3).hex()}"
        app.store.upsert_client(account_id, "Retail")
        client, _ = _super(app)
        page = client.post("/admin/admins/invite", data={"email": "n2@example.com", "role": "data_steward",
                                                         "workspaces": [account_id]}).text
        assert "Send this link to n2@example.com" in page


# ── A dashboard followed by email ────────────────────────────────────────────

class TestAFollowByEmail:

    @pytest.fixture
    def follower(self, app, smtp):
        account_id = f"acct{os.urandom(3).hex()}"
        app.store.upsert_client(account_id, "Retail")
        user_id, _ = app.store.create_user(account_id, "Ada", "ada@example.com", password="readers-password")
        board = app.store.create_dashboard(account_id, user_id, "t", "Weekly sales")
        client = _browser(app)
        client.post("/portal/login", data={"account_id": account_id, "email": "ada@example.com",
                                           "password": "readers-password"})
        return client, account_id, user_id, board

    def _due(self, app, user_id, days=8):
        """The follow's last update ``days`` ago: one is due (each period is due from its own moment)."""
        from store.db import get_db

        with get_db() as conn:
            conn.execute("UPDATE dashboard_subscription SET last_sent_at=datetime('now', ?) WHERE user_id=?",
                         (f"-{days} days", user_id))

    def test_the_update_comes_by_email_with_a_link_to_stop_it(self, app, inbox, follower):
        from core.dashboard_follow import run_due_follows
        from core.mail import outbox

        client, account_id, user_id, board = follower
        assert "By email too" in client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        client.post(f"/portal/dashboard/{board['id']}/subscribe", data={"cadence": "weekly", "by_email": "1"})
        assert app.store.get_dashboard_subscription(board["id"], user_id, account_id)["channel"] == "email"
        self._due(app, user_id)
        assert run_due_follows() == 1
        outbox.deliver_due()
        message = inbox.to("ada@example.com")[0]
        assert "Weekly sales" in message["Subject"]
        text = message.get_body(("plain",)).get_content()
        assert f"https://bi.example.com/portal/dashboard?dashboard_id={board['id']}" in text
        stop = next(line for line in text.splitlines() if "/portal/unsubscribe/" in line).split(": ", 1)[1]
        assert message["List-Unsubscribe"] == f"<{stop}>"
        # Opening the link asks; a mail scanner opening every link stops nothing.
        reader = _browser(app)
        path = stop.replace("https://bi.example.com", "")
        asked = reader.get(path)
        assert asked.status_code == 200 and "Weekly sales" in asked.text and "Stop these emails" in asked.text
        from store import email_store

        assert not email_store.opted_out("ada@example.com", "follow", str(board["id"]))
        assert "will not get these emails" in reader.post(path).text
        assert email_store.opted_out("ada@example.com", "follow", str(board["id"]))
        # The next update stays in the app.
        self._due(app, user_id, days=9)
        run_due_follows()
        newest = email_store.list_emails(kind="follow", limit=1)[0]
        assert newest["status"] == "skipped" and "stopped this email" in newest["last_error"]
        assert len(app.store.list_notices(account_id, user_id)) == 2          # both updates in the app

    def test_a_follow_in_the_app_sends_no_email(self, app, inbox, follower):
        from core.dashboard_follow import run_due_follows
        from core.mail import outbox

        client, account_id, user_id, board = follower
        client.post(f"/portal/dashboard/{board['id']}/subscribe", data={"cadence": "daily"})
        self._due(app, user_id)
        run_due_follows()
        outbox.deliver_due()
        assert inbox.messages == []

    def test_a_tampered_link_stops_nothing(self, app):
        reader = _browser(app)
        assert "This link does not work" in reader.get("/portal/unsubscribe/abc.def").text
        assert "This link does not work" in reader.post("/portal/unsubscribe/abc.def").text


def test_the_scheduler_sends_what_is_due(app, smtp, inbox):
    from core.notification_scheduler import run_due_notifications_once

    _outbox(to="sched@example.com")
    run_due_notifications_once()
    assert inbox.to("sched@example.com")
