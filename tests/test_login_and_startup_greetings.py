"""
tests/test_login_and_startup_greetings.py

Two proactive-greeting features, replacing/complementing the reactive
"session greeting" that used to fire only after the user's first message:

  1. Portal: gateway/webhooks.py's ws_chat now sends the QueryBot greeting
     at WS-connect time (login), gated by store.touch_user_activity's
     session-boundary signal, instead of waiting for the user's first
     non-conversational message to reach core/dispatcher.py's reactive
     block. core/dispatcher.py's reactive block is correspondingly skipped
     for the "web" platform to avoid double-greeting a portal user.
  2. Teams: main.py's startup handler proactively notifies every approved
     Teams user that the service is back up, symmetric to the existing
     shutdown handler's "signing off" notification.

Marker/wiring tests — same convention as WebhooksWiringTests in
tests/test_stop_query.py — since dispatch()/app-startup pull in the full
LLM/DB/adapter stack and aren't practically unit-testable in isolation.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PortalLoginGreetingWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_touches_session_activity_on_connect(self):
        anchor = self.src.index('async def ws_chat(')
        body = self.src[anchor:anchor + 3500]
        self.assertIn("await websocket.accept()", body)
        self.assertIn("store.touch_user_activity(user_id)", body)
        # The touch must happen after accept() -- it's the login moment.
        self.assertLess(body.index("await websocket.accept()"), body.index("store.touch_user_activity(user_id)"))

    def test_new_session_gets_full_greeting_as_a_chat_message(self):
        anchor = self.src.index("_is_new_portal_session = store.touch_user_activity")
        body = self.src[anchor:anchor + 700]
        self.assertIn("if _is_new_portal_session:", body)
        self.assertIn("from core.conversational import build_reply", body)
        self.assertIn('build_reply("greeting", account_id, portal_user)', body)
        # Rendered as a normal bot chat bubble (type "message"), matching
        # the exact same rendering as the typed "Hi" greeting reply.
        self.assertIn('"type":    "message"', body)

    def test_reconnect_within_session_keeps_plain_connected_line(self):
        anchor = self.src.index("_is_new_portal_session = store.touch_user_activity")
        body = self.src[anchor:anchor + 900]
        self.assertIn("else:", body)
        self.assertIn("Connected as {portal_user.get(", body)

    def test_touch_failure_does_not_crash_the_connection(self):
        anchor = self.src.index("_is_new_portal_session = False")
        body = self.src[anchor:anchor + 300]
        self.assertIn("try:", body)
        self.assertIn("except Exception as _touch_exc:", body)

    def test_new_session_also_offers_the_login_report_prompt(self):
        # The report prompt is a follow-up to the greeting, inside the same
        # "genuinely new session" branch -- never on a plain reconnect.
        anchor = self.src.index("_is_new_portal_session = store.touch_user_activity")
        body = self.src[anchor:anchor + 900]
        self.assertIn("_offer_login_report_prompt", body)
        self.assertIn("except Exception as _report_prompt_exc:", body)

    def test_clarification_response_handles_login_report_prompt_source(self):
        anchor = self.src.index('if msg_type == "clarification_response":')
        body = self.src[anchor:anchor + 3000]
        self.assertIn('cmeta.get("source") == "login_report_prompt"', body)
        self.assertIn("_deliver_report_via_adapter", body)
        self.assertIn("list_promptable_reports", body)
        # Declining (or resolving nothing) must not fall into the generic
        # combine_with_clarification + handle_query path built for refining
        # a data question.
        login_branch_end = body.index("continue", body.index('"login_report_prompt"'))
        login_branch = body[:login_branch_end]
        self.assertNotIn("combine_with_clarification(", login_branch)


class DispatcherSkipsWebPlatformSessionGreetingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "core" / "dispatcher.py").read_text(encoding="utf-8")

    def test_session_greeting_gate_excludes_web_platform(self):
        self.assertIn(
            'if portal_user and not _conv_kind and event.platform != "web":',
            self.src,
        )

    def test_gate_change_is_explained_for_future_readers(self):
        anchor = self.src.index('if portal_user and not _conv_kind and event.platform != "web":')
        head = self.src[max(0, anchor - 700):anchor]
        self.assertIn("ws_chat now sends this greeting proactively at WS-connect time", head)


class TeamsStartupNotificationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_startup_handler_notifies_approved_teams_users(self):
        start_anchor = self.src.index('@app.on_event("startup")')
        shutdown_anchor = self.src.index('@app.on_event("shutdown")')
        body = self.src[start_anchor:shutdown_anchor]

        self.assertIn(
            "WHERE p.status='approved' AND p.platform_type='teams'",
            body,
        )
        self.assertIn("TeamsAdapter", body)
        self.assertIn("up and running", body)

    def test_startup_and_shutdown_exclude_temporarily_stopped_users(self):
        # A user with "Temporarily Stop Access" toggled (portal_user.is_active
        # = 0, admin/routes.py's toggle-active route) must not get proactive
        # startup/shutdown pings -- confirmed live: status stays 'approved'
        # the whole time (that's a separate reject/block flow), so the old
        # query (status='approved' only) kept notifying a user an admin had
        # explicitly, if temporarily, silenced.
        start_anchor = self.src.index('@app.on_event("startup")')
        shutdown_anchor = self.src.index('@app.on_event("shutdown")')
        startup_body = self.src[start_anchor:shutdown_anchor]
        shutdown_body = self.src[shutdown_anchor:]

        for body in (startup_body, shutdown_body):
            self.assertIn("LEFT JOIN portal_user pu ON p.portal_user_id = pu.id", body)
            self.assertIn("COALESCE(pu.is_active, 1) = 1", body)

    def test_startup_notification_never_blocks_or_crashes_startup(self):
        start_anchor = self.src.index('@app.on_event("startup")')
        shutdown_anchor = self.src.index('@app.on_event("shutdown")')
        body = self.src[start_anchor:shutdown_anchor]

        anchor = body.index("Notify active Teams users the service is back up")
        block = body[anchor:anchor + 2400]
        self.assertIn("try:", block)
        self.assertIn("except Exception as exc:", block)
        self.assertIn("asyncio.gather(", block)
        self.assertIn("return_exceptions=True", block)

    def test_startup_and_shutdown_use_the_same_query_shape(self):
        # Symmetric feature -- confirm both handlers query the identical
        # approved-Teams-user set, so "who gets notified" can't silently
        # drift between the two lifecycle events.
        start_anchor = self.src.index('@app.on_event("startup")')
        shutdown_anchor = self.src.index('@app.on_event("shutdown")')
        startup_body = self.src[start_anchor:shutdown_anchor]
        shutdown_body = self.src[shutdown_anchor:]

        # Each fragment below is one complete Python string-literal line in
        # main.py (adjacent literals spanning multiple lines) -- asserted
        # individually rather than as one joined string, since the raw
        # source text (unlike the runtime-concatenated SQL) has a quote/
        # newline/indent break between each fragment.
        query_fragments = (
            "FROM pending_platform_user p ",
            "LEFT JOIN portal_user pu ON p.portal_user_id = pu.id ",
            "WHERE p.status='approved' AND p.platform_type='teams' ",
            "AND COALESCE(pu.is_active, 1) = 1",
        )
        for fragment in query_fragments:
            self.assertIn(fragment, startup_body)
            self.assertIn(fragment, shutdown_body)


if __name__ == "__main__":
    unittest.main()
