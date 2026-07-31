"""
Portal session resolution must re-check portal_user.is_active on every
request/connection, not just at login.

Live bug: an admin's "Temporarily Stop Access" toggle (admin/routes.py's
toggle-active route -> store.update_user(is_active=0)) only ever blocked
fresh logins -- store.get_user_by_email (used by portal_login_submit)
already filters is_active=1. But every already-issued session cookie kept
resolving successfully, because portal.routes._get_portal_user,
portal.routes._get_portal_user_from_socket, and gateway/webhooks.py's
ws_chat all called the unfiltered store.get_user(user_id) lookup -- so a
user an admin had explicitly, if temporarily, deactivated kept using an
already-open portal tab (or reconnecting the chat WebSocket with the same
cookie) indefinitely. Symmetric to the Teams "Temporarily Stop Access"
notification fix in this same session (main.py's startup/shutdown
handlers) -- same root cause class, different surface.

Real SQLite (store.init_db()) so is_active toggling exercises the actual
store.update_user/get_user round trip, not a mock.
"""

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store


def _arun(coro):
    return asyncio.run(coro)


class PortalSessionDeactivationTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        self.account_id = f"acct-deactivate-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.user_id, _ = store.create_user(
            self.account_id, "Test User", f"{uuid.uuid4().hex[:8]}@test.com",
        )

    def tearDown(self):
        with store.get_db() as conn:
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _cookie(self):
        import portal.routes as routes
        return routes._sign_session_value(self.user_id)

    def test_get_portal_user_returns_none_once_deactivated(self):
        import portal.routes as routes
        cookie = self._cookie()
        request = MagicMock()
        request.cookies.get.return_value = cookie

        self.assertIsNotNone(routes._get_portal_user(request))

        store.update_user(self.user_id, is_active=0)
        self.assertIsNone(routes._get_portal_user(request))

    def test_get_portal_user_from_socket_returns_none_once_deactivated(self):
        import portal.routes as routes
        cookie = self._cookie()
        websocket = MagicMock()
        websocket.cookies.get.return_value = cookie

        self.assertIsNotNone(routes._get_portal_user_from_socket(websocket))

        store.update_user(self.user_id, is_active=0)
        self.assertIsNone(routes._get_portal_user_from_socket(websocket))

    def test_reactivating_restores_access(self):
        # Symmetric to the "Re-activate Access" admin action -- must not be
        # a one-way lock.
        import portal.routes as routes
        cookie = self._cookie()
        request = MagicMock()
        request.cookies.get.return_value = cookie

        store.update_user(self.user_id, is_active=0)
        self.assertIsNone(routes._get_portal_user(request))

        store.update_user(self.user_id, is_active=1)
        self.assertIsNotNone(routes._get_portal_user(request))

    def test_ws_chat_closes_deactivated_users_socket(self):
        import gateway.webhooks as webhooks

        store.update_user(self.user_id, is_active=0)

        websocket = MagicMock()
        websocket.cookies.get.return_value = self._cookie()
        websocket.close = AsyncMock()

        # Real signed cookie (self._cookie()), real is_active=0 in SQLite --
        # ws_chat's own `from portal.routes import _read_session_value`
        # decodes it for real, then store.get_user(...).get("is_active")
        # must reject the connection.
        _arun(webhooks.ws_chat(websocket, self.account_id))

        websocket.close.assert_called_once_with(code=4003)


if __name__ == "__main__":
    unittest.main()
