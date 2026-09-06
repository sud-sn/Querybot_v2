# -*- coding: utf-8 -*-
"""The front door speaks the reader's language — core/dispatcher.py.

The dispatcher is the first thing every turn hits, and it already had the
language: gateway/webhooks.py activates it once per websocket task, two frames
before dispatch, and the module already imported `t` and used it in two places.
Twenty-two other replies were English literals sitting inside that activation —
welcome and access messages, `status` and `whoami`, the report offers, the
"choose an option" retries, the not-set-up and still-building notices.

The guard here is structural on purpose. Most of these sit inline in a
600-line `dispatch_message`, so there is no unit to call; what can be checked
is that no literal reaches a send at all, which is the property that broke and
the one that will break again next time someone adds a reply.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.i18n import MESSAGES, activate_language, deactivate_language, t  # noqa: E402

MODULE = Path(__file__).resolve().parents[1] / "core" / "dispatcher.py"

# The calls that put text in front of a user.
_SENDS = {"send_message", "_send_sq", "send_clarification_prompt", "send_prompt"}


def _literals_reaching_a_send() -> list[tuple[int, str]]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
        if name not in _SENDS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for lit in ast.walk(arg):
                if (isinstance(lit, ast.Constant)
                        and isinstance(lit.value, str)
                        and len(lit.value) > 12):
                    found.append((lit.lineno, lit.value))
    return found


def _is_message_id(value: str) -> bool:
    return value in MESSAGES or value.replace(".one", ".x").replace(".other", ".x") in MESSAGES


class TestNoReplyIsHardcodedEnglish(unittest.TestCase):

    def test_every_string_reaching_a_send_is_a_catalogue_id(self):
        offenders = [
            f"{line}: {value[:70]!r}"
            for line, value in _literals_reaching_a_send()
            if not _is_message_id(value)
        ]
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_scan_is_actually_finding_the_sends(self):
        # A scanner that matched nothing would make the assertion above
        # vacuous — and it was vacuous once, when every one of these was
        # a literal.
        found = _literals_reaching_a_send()
        self.assertGreater(len(found), 15)
        self.assertTrue(any(v.startswith("dispatch.") for _l, v in found))

    def test_the_dispatcher_reads_the_catalogue(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("from core.i18n import t as _t", source)


class TestTheRepliesRenderInBothLanguages(unittest.TestCase):

    def _both(self, msg_id, **kw):
        out = {}
        for lang in ("en", "fr"):
            token = activate_language(lang)
            try:
                out[lang] = t(msg_id, **kw)
            finally:
                deactivate_language(token)
        return out

    def test_every_dispatch_id_differs_between_the_languages(self):
        ids = [k for k in MESSAGES if k.startswith("dispatch.")]
        self.assertGreaterEqual(len(ids), 20)
        for msg_id in ids:
            self.assertTrue(MESSAGES[msg_id].get("fr"), msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], msg_id)

    def test_status_keeps_its_data_and_translates_its_labels(self):
        rendered = self._both("dispatch.status", state="READY", database="Sales DW",
                              used=12, limit=500, user="Camille")
        for text in rendered.values():
            for value in ("READY", "Sales DW", "12/500", "Camille"):
                self.assertIn(value, text)
        self.assertNotEqual(rendered["en"], rendered["fr"])
        self.assertIn("Base de données", rendered["fr"])

    def test_whoami_keeps_its_data_and_translates_its_labels(self):
        rendered = self._both("dispatch.whoami", name="Camille", role="analyst",
                              group="Finance", tables="FIN.GL")
        for text in rendered.values():
            for value in ("Camille", "analyst", "Finance", "FIN.GL"):
                self.assertIn(value, text)
        self.assertIn("Groupe", rendered["fr"])

    def test_the_report_offers_keep_the_report_names(self):
        one = self._both("dispatch.report_offer_one", name="Daily Sales")
        for text in one.values():
            self.assertIn("Daily Sales", text)
        self.assertNotEqual(one["en"], one["fr"])

        missing = self._both("dispatch.report_not_found", name="Foo",
                             available='"Daily Sales", "Stock"')
        for text in missing.values():
            self.assertIn("Foo", text)
            self.assertIn("Daily Sales", text)

    def test_no_placeholder_is_left_unfilled(self):
        for msg_id, kw in (
            ("dispatch.status", dict(state="R", database="D", used=1, limit=2, user="U")),
            ("dispatch.whoami", dict(name="N", role="R", group="G", tables="T")),
            ("dispatch.report_offer_one", dict(name="N")),
            ("dispatch.report_not_found", dict(name="N", available="A")),
            ("dispatch.which_report", dict(available="A")),
        ):
            for lang, text in self._both(msg_id, **kw).items():
                self.assertNotIn("{", text, f"{lang} {msg_id}")


if __name__ == "__main__":
    unittest.main()
