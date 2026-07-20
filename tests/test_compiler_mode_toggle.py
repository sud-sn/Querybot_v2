"""Semantic compiler mode-toggle admin control.

Sprint 2's nine detectors and the enforce-mode blocking behavior are only
provably correct in tests until an admin can actually flip a real tenant
from shadow to enforce - store.set_semantic_compiler_mode existed and was
tested from Sprint 1 onward, but no route in admin/routes.py ever called
it. This closes that gap: a route + a Model Health form control.
"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arun(coro):
    import asyncio
    return asyncio.run(coro)


class ModeToggleRouteTests(unittest.TestCase):
    def setUp(self):
        import store

        store.init_db()
        self.store = store
        self.account_id = f"acct-mode-toggle-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute(
                "DELETE FROM semantic_compiler_state WHERE account_id=?", (self.account_id,)
            )
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _post(self, mode: str):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.model_health_set_compiler_mode(object(), self.account_id, mode=mode))

    def test_setting_enforce_persists_and_redirects_to_model_health(self):
        resp = self._post("enforce")
        self.assertEqual(resp.status_code, 303)
        self.assertIn(f"/admin/clients/{self.account_id}/model-health", resp.headers["location"])
        self.assertIn("saved=compiler_mode", resp.headers["location"])
        self.assertEqual(
            self.store.get_semantic_compiler_state(self.account_id)["mode"], "enforce",
        )

    def test_setting_off_and_shadow_also_persist(self):
        self._post("off")
        self.assertEqual(self.store.get_semantic_compiler_state(self.account_id)["mode"], "off")
        self._post("shadow")
        self.assertEqual(self.store.get_semantic_compiler_state(self.account_id)["mode"], "shadow")

    def test_invalid_mode_is_rejected_without_persisting(self):
        resp = self._post("YOLO")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("error=", resp.headers["location"])
        # Untouched — still whatever the tenant started at (default 'shadow').
        self.assertEqual(self.store.get_semantic_compiler_state(self.account_id)["mode"], "shadow")

    def test_unknown_client_returns_404(self):
        import admin.routes as routes
        from fastapi import HTTPException

        with patch.object(routes, "_is_auth", return_value=True):
            with self.assertRaises(HTTPException) as ctx:
                _arun(routes.model_health_set_compiler_mode(
                    object(), "no-such-account-xyz", mode="enforce",
                ))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_unauthenticated_redirects_to_login(self):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=False):
            resp = _arun(routes.model_health_set_compiler_mode(
                object(), self.account_id, mode="enforce",
            ))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/admin/login", resp.headers["location"])

    def test_switching_mode_does_not_trigger_a_recompile(self):
        # Confirmed by design: this route only writes state — it must not
        # touch semantic_compile_run or contract versions on its own.
        before = self.store.list_semantic_conflicts(self.account_id)
        self._post("enforce")
        after = self.store.list_semantic_conflicts(self.account_id)
        self.assertEqual(before, after)


class ModeToggleWiringTests(unittest.TestCase):
    def test_route_registered(self):
        src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        self.assertIn(
            '@router.post("/clients/{account_id}/model-health/compiler/mode")', src,
        )
        self.assertIn("set_semantic_compiler_mode", src)

    def test_template_has_mode_form_with_all_three_options(self):
        src = (ROOT / "admin/templates/client_model_health.html").read_text(encoding="utf-8")
        self.assertIn('id="compiler-mode-form"', src)
        self.assertIn('action="/admin/clients/{{ client.account_id }}/model-health/compiler/mode"', src)
        for option in ('value="off"', 'value="shadow"', 'value="enforce"'):
            self.assertIn(option, src)

    def test_switching_to_enforce_is_confirmed_client_side(self):
        src = (ROOT / "admin/templates/client_model_health.html").read_text(encoding="utf-8")
        script = src[src.index("compiler-mode-form"):]
        self.assertIn("confirm(", script)
        self.assertIn("enforce", script.split("confirm(", 1)[1][:400].lower())

    def test_open_error_count_warning_shown_before_switching(self):
        src = (ROOT / "admin/templates/client_model_health.html").read_text(encoding="utf-8")
        panel = src[src.index('id="compiler-mode-form"'):src.index("</form>")]
        self.assertIn("compile_counts.errors", panel)


if __name__ == "__main__":
    unittest.main()
