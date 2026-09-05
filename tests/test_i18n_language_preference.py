"""
tests/test_i18n_language_preference.py

The per-user language preference and the catalogue that serves it.

This is the foundation the whole French feature sits on, and its failure modes
are all quiet ones, so every test here executes the real path and asserts on
what comes back.

Three of them exist because of a specific hazard:

  * The migration is declared ONLY in store/db.py's migrations list and NOT in
    _SCHEMA. init_db() runs migrations on fresh databases too, which is how
    last_active_at reaches a brand-new DB. Declaring a column in both places
    means two definitions that drift, so the fresh-database case is tested
    rather than assumed.

  * update_user() builds its SET clause from a four-field whitelist and
    silently ignores anything else. A language passed to it would look accepted
    and never be written, so the language has its own setter and the setter is
    what is tested.

  * Some strings look like copy and are actually protocol. "redacted segment"
    is produced by core/insight.py and compared BY EQUALITY in two places; a
    translated sentinel makes the comparison fail and the unredacted label
    reaches the user. The catalogue refuses those values, checked here rather
    than trusted to reviewer memory.
"""

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_i18n_pref.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

from core import i18n  # noqa: E402


def _account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _user(lang=None):
    # create_user is (account_id, name, email) -- name first. Getting that
    # backwards stores the email as the name and get_user_by_email then finds
    # nothing, which is exactly how this helper was wrong the first time.
    account_id = _account()
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
    if lang:
        store.set_user_language(user_id, lang)
    return user_id


# ══════════════════════════════════════════════════════════════════════════════
# The column
# ══════════════════════════════════════════════════════════════════════════════

class TestTheLanguageColumn:

    def test_a_fresh_database_has_the_column(self):
        """init_db() ran the migrations above. If lang had been declared in
        _SCHEMA instead of the migrations list this would still pass, which is
        why the next test exists too."""
        from store.db import get_db
        with get_db() as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(portal_user)")]
        assert "lang" in columns

    def test_the_default_is_english_not_french(self):
        """'fr' as the column default would silently flip every existing tenant
        on upgrade. The French client gets a per-client default separately."""
        assert store.get_user(_user()).get("lang") == "en"

    def test_the_setter_writes_and_the_reader_surfaces_it(self):
        """get_user does SELECT u.*, so nothing in the read path changes."""
        user_id = _user()
        assert store.set_user_language(user_id, "fr") == "fr"
        assert store.get_user(user_id)["lang"] == "fr"

    def test_regional_tags_are_accepted_and_folded(self):
        user_id = _user()
        for value in ("fr-FR", "FR", "fr_FR", "  fr  "):
            store.set_user_language(user_id, "en")
            assert store.set_user_language(user_id, value) == "fr", value
            assert store.get_user(user_id)["lang"] == "fr"

    def test_an_unsupported_language_is_refused_not_stored(self):
        """The column carries no CHECK constraint by design -- SQLite enforces
        one on ADD COLUMN but cannot drop it, so a third language would need a
        table rebuild. The setter is the validation."""
        user_id = _user("fr")
        assert store.set_user_language(user_id, "klingon") == "en"
        assert store.get_user(user_id)["lang"] == "en"
        assert store.set_user_language(user_id, None) == "en"

    def test_update_user_still_cannot_write_the_language(self):
        """Pins why a separate setter exists. If update_user ever grows a lang
        field, this test says so instead of leaving two write paths."""
        user_id = _user("fr")
        store.update_user(user_id, name="Ada Lovelace")
        assert store.get_user(user_id)["lang"] == "fr"

    def test_a_row_read_never_needs_a_none_guard(self):
        """NOT NULL DEFAULT 'en' means every row carries a real value, so
        (user or {}).get('lang') or 'en' is belt-and-braces rather than load
        bearing on a migrated database."""
        user_id = _user()
        assert store.get_user(user_id)["lang"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# The catalogue
# ══════════════════════════════════════════════════════════════════════════════

class TestTheCatalogueIsWellFormed:

    def test_every_id_has_every_supported_language(self):
        missing = [
            f"{msg_id}:{lang}"
            for msg_id, entry in i18n.MESSAGES.items()
            for lang in i18n.SUPPORTED_LANGUAGES
            if not entry.get(lang)
        ]
        assert missing == []

    def test_placeholders_match_across_languages(self):
        """The bug this catches: a French translation that drops {title} or
        renames it. t() would then silently emit a template with a literal
        brace in it, or lose the value entirely."""
        for msg_id, entry in i18n.MESSAGES.items():
            english = i18n.placeholders(msg_id)
            for lang in i18n.SUPPORTED_LANGUAGES:
                found = set(i18n._PLACEHOLDER_RE.findall(entry[lang]))
                assert found == english, f"{msg_id} [{lang}]: {found} != {english}"

    def test_no_positional_placeholders_anywhere(self):
        """Word order changes between languages; {0} and {} silently swap
        arguments when it does."""
        import re
        bad = re.compile(r"\{\}|\{\d+\}")
        for msg_id, entry in i18n.MESSAGES.items():
            for lang, template in entry.items():
                assert not bad.search(template), f"{msg_id} [{lang}]"

    def test_no_message_collides_with_a_protocol_sentinel(self):
        """'redacted segment' is compared by equality in core/insight.py and
        core/response_builder.py. A translated sentinel fails that comparison
        and the real, unredacted label reaches the user."""
        for msg_id, entry in i18n.MESSAGES.items():
            for lang, template in entry.items():
                for sentinel in i18n.FORBIDDEN_VALUES:
                    assert template.strip() != sentinel, f"{msg_id} [{lang}]"

    def test_the_supported_set_has_not_drifted_from_the_store(self):
        """Two modules name the languages. This is the only thing stopping them
        disagreeing, which is how a user could be stored 'fr' and rendered
        'en'."""
        assert tuple(i18n.SUPPORTED_LANGUAGES) == tuple(store.SUPPORTED_LANGUAGES)
        assert i18n.normalise_language("fr-FR") == store.normalise_language("fr-FR")
        assert i18n.normalise_language("klingon") == store.normalise_language("klingon")

    # Ids whose two languages legitimately coincide. Every entry here is a word
    # that is spelled the same in French, and listing them explicitly is what
    # makes the test below able to catch a copy-paste that was never translated.
    IDENTICAL_BY_DESIGN = {
        "answer.total",                 # the same word in French
        "ui.chat.table_count.one",      # "table" is the same word, singular
        "ui.chat.table_count.other",    # and plural
        "ui.chat.trust.source",         # "source" is the same word
        "ui.chat.trust.sources",        # and so is its plural
        "ui.pin.description_label",
        "ui.dash.version",              # "Version" is the same word
        "ui.enum.charttype.kpi",        # an international acronym
        "ui.shell.chat",                # borrowed into French unchanged
        "ui.shell.notifications",       # same spelling in both
    }

    def test_french_is_actually_french(self):
        """A guard against a catalogue whose 'fr' entries were copied from 'en'
        and never translated -- which reads as done and is not."""
        identical = {
            msg_id for msg_id, entry in i18n.MESSAGES.items()
            if entry["en"] == entry["fr"]
        }
        assert identical == self.IDENTICAL_BY_DESIGN, (
            f"untranslated: {sorted(identical - self.IDENTICAL_BY_DESIGN)}; "
            f"now translated, drop from the allowlist: "
            f"{sorted(self.IDENTICAL_BY_DESIGN - identical)}"
        )


class TestLookupDegradesInsteadOfRaising:

    def test_an_unknown_id_returns_the_id(self):
        """Visible in review. An empty string would read as a layout bug."""
        assert i18n.t("ui.nope.missing") == "ui.nope.missing"

    def test_an_unknown_language_falls_back_to_english(self):
        assert i18n.t("ui.pin.title", lang="de") == "Add to dashboard"

    def test_interpolation_works_in_both_languages(self):
        assert i18n.t("ui.pin.added", lang="en", dashboard="Ops") == "Added to Ops."
        assert i18n.t("ui.pin.added", lang="fr", dashboard="Ops") == "Ajouté à Ops."

    def test_a_missing_keyword_leaves_the_placeholder_and_does_not_raise(self):
        """This runs inside answer construction. A KeyError there costs the
        user the whole answer to save one word."""
        out = i18n.t("ui.pin.added", lang="en")
        assert "{dashboard}" in out

    def test_an_extra_keyword_is_ignored(self):
        assert i18n.t("ui.pin.title", lang="en", unused=1) == "Add to dashboard"


class TestTheActiveLanguageIsPerContext:

    def test_activate_and_deactivate_round_trip(self):
        assert i18n.get_active_language() == "en"
        token = i18n.activate_language("fr")
        try:
            assert i18n.get_active_language() == "fr"
            assert i18n.t("ui.pin.title") == "Ajouter au tableau de bord"
        finally:
            i18n.deactivate_language(token)
        assert i18n.get_active_language() == "en"
        assert i18n.t("ui.pin.title") == "Add to dashboard"

    def test_activation_normalises(self):
        token = i18n.activate_language("fr-FR")
        try:
            assert i18n.get_active_language() == "fr"
        finally:
            i18n.deactivate_language(token)

    def test_two_concurrent_contexts_do_not_leak(self):
        """The reason this is a ContextVar and not a module global. Executed
        with real concurrent tasks rather than argued from the docs."""
        import asyncio

        async def answer(lang):
            token = i18n.activate_language(lang)
            try:
                await asyncio.sleep(0)
                return i18n.t("ui.pin.title")
            finally:
                i18n.deactivate_language(token)

        async def both():
            return await asyncio.gather(*[answer(l) for l in ("fr", "en") * 10])

        results = asyncio.run(both())
        assert results[0::2] == ["Ajouter au tableau de bord"] * 10
        assert results[1::2] == ["Add to dashboard"] * 10

    def test_a_bound_translator_ignores_the_active_language(self):
        """Templates render outside the pipeline's activation, so they must not
        depend on the ContextVar."""
        french = i18n.translator_for("fr")
        token = i18n.activate_language("en")
        try:
            assert french("ui.pin.title") == "Ajouter au tableau de bord"
        finally:
            i18n.deactivate_language(token)


class TestTheBrowserCatalogue:

    def test_it_is_flat_and_all_strings(self):
        catalogue = i18n.catalogue_for("fr")
        assert catalogue["ui.pin.title"] == "Ajouter au tableau de bord"
        assert all(isinstance(v, str) for v in catalogue.values())
        assert len(catalogue) == len(i18n.MESSAGES)

    def test_a_prefix_narrows_it(self):
        ui_only = i18n.catalogue_for("fr", prefix="ui.pin.")
        assert ui_only
        assert all(k.startswith("ui.pin.") for k in ui_only)
        assert len(ui_only) < len(i18n.MESSAGES)

    def test_it_survives_json(self):
        import json
        catalogue = i18n.catalogue_for("fr")
        assert json.loads(json.dumps(catalogue)) == catalogue


# ══════════════════════════════════════════════════════════════════════════════
# Delivery — the context processor and the switcher, executed
# ══════════════════════════════════════════════════════════════════════════════

def _probe_client():
    """A real app that renders a real template through the untouched _resp.

    The point of the test is that _resp is NOT modified: the language reaches a
    template rendered with an empty context, so none of the portal's ~24 render
    call sites had to change.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from jinja2 import ChoiceLoader, DictLoader
    from starlette.testclient import TestClient

    import portal.routes as pr

    if not getattr(pr.templates.env, "_probe_installed", False):
        pr.templates.env.loader = ChoiceLoader([
            DictLoader({"__lang_probe__.html":
                        "lang={{ lang }}|t={{ t('ui.pin.title') }}|n={{ i18n_catalogue|length }}"}),
            pr.templates.env.loader,
        ])
        pr.templates.env._probe_installed = True

    app = FastAPI()

    @app.get("/probe", response_class=HTMLResponse)
    async def _probe(request: Request):
        return pr._resp(request, "__lang_probe__.html", {})

    app.include_router(pr.router)
    return TestClient(app), pr


class TestTheLanguageReachesEveryTemplate:

    def test_the_default_render_is_english(self):
        client, _ = _probe_client()
        body = client.get("/probe").text
        assert "lang=en" in body
        assert "t=Add to dashboard" in body

    def test_the_cookie_switches_the_whole_render(self):
        client, _ = _probe_client()
        body = client.get("/probe", cookies={"qb_lang": "fr"}).text
        assert "lang=fr" in body
        assert "t=Ajouter au tableau de bord" in body

    def test_the_catalogue_arrives_too(self):
        """Without this the inline scripts have nothing to read and would keep
        their own hardcoded English -- which is how _fmtNum and the stage
        labels already drifted between two templates."""
        client, _ = _probe_client()
        body = client.get("/probe", cookies={"qb_lang": "fr"}).text
        assert f"n={len(i18n.MESSAGES)}" in body

    def test_a_browser_preference_is_honoured_before_login(self):
        """The login and registration pages render with no user in context. A
        French customer would otherwise meet an English login screen every time,
        and it is the first screen they ever see."""
        client, _ = _probe_client()
        body = client.get("/probe",
                          headers={"accept-language": "fr-FR,fr;q=0.9,en;q=0.8"}).text
        assert "lang=fr" in body

    def test_a_junk_cookie_falls_back_rather_than_breaking_the_page(self):
        client, _ = _probe_client()
        assert "lang=en" in client.get("/probe", cookies={"qb_lang": "../../etc"}).text

    def test_the_cookie_beats_the_header(self):
        client, _ = _probe_client()
        body = client.get("/probe", cookies={"qb_lang": "en"},
                          headers={"accept-language": "fr-FR"}).text
        assert "lang=en" in body


class TestTheSwitcher:

    def _signed_in(self, lang=None):
        client, pr = _probe_client()
        user_id = _user(lang)
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        return client, user_id

    def test_an_anonymous_switch_is_refused(self):
        client, _ = _probe_client()
        assert client.post("/portal/api/language", json={"lang": "fr"}).status_code == 401

    def test_it_writes_the_row_and_the_cookie(self):
        client, user_id = self._signed_in()
        response = client.post("/portal/api/language", json={"lang": "fr"})
        assert response.status_code == 200
        assert response.json()["lang"] == "fr"
        # The row: what the answer pipeline reads.
        assert store.get_user(user_id)["lang"] == "fr"
        # The cookie: what the page chrome reads.
        assert client.cookies.get("qb_lang") == "fr"

    def test_the_next_page_render_is_french(self):
        """The end-to-end assertion: switch, then render, and the page is
        French with nothing passed between the two by the test."""
        client, _ = self._signed_in()
        client.post("/portal/api/language", json={"lang": "fr"})
        assert "t=Ajouter au tableau de bord" in client.get("/probe").text

    def test_a_form_post_works_without_javascript(self):
        client, user_id = self._signed_in()
        response = client.post("/portal/api/language", data={"lang": "fr"})
        assert response.status_code == 200
        assert store.get_user(user_id)["lang"] == "fr"

    def test_a_form_post_with_a_return_path_redirects_back(self):
        """Without JavaScript the reader must land back on the page they were
        on, in the new language -- not on a JSON body."""
        client, user_id = self._signed_in()
        response = client.post(
            "/portal/api/language",
            data={"lang": "fr", "next": "/portal/chat?thread=abc"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/portal/chat?thread=abc"
        assert store.get_user(user_id)["lang"] == "fr"
        assert client.cookies.get("qb_lang") == "fr"

    def test_the_redirect_refuses_to_leave_the_portal(self):
        """The field is attacker-reachable: anyone can craft a link that posts
        this form. Each of these is a shape a browser would follow off-site."""
        client, _ = self._signed_in()
        for hostile in ("https://evil.example/",
                        "//evil.example/",
                        "/\\evil.example",
                        "/admin/clients",
                        "/portal/chat\r\nSet-Cookie: a=b"):
            response = client.post(
                "/portal/api/language",
                data={"lang": "fr", "next": hostile},
                follow_redirects=False,
            )
            assert response.status_code == 200, hostile
            assert response.json()["lang"] == "fr", hostile

    def test_a_json_post_is_still_answered_in_json(self):
        """A fetch() caller that followed a redirect would replace its own page
        with the response body."""
        client, _ = self._signed_in()
        response = client.post("/portal/api/language", json={"lang": "fr"},
                               follow_redirects=False)
        assert response.status_code == 200
        assert response.json()["lang"] == "fr"

    def test_a_storage_failure_still_returns_the_reader_to_their_page(self):
        """A 503 body is unreadable without JavaScript, and losing the page is
        a worse outcome than losing the preference."""
        from unittest.mock import patch
        client, _ = self._signed_in()
        import portal.routes as pr
        with patch.object(pr.store, "set_user_language",
                          side_effect=RuntimeError("no such column: lang")):
            response = client.post(
                "/portal/api/language",
                data={"lang": "fr", "next": "/portal/chat"},
                follow_redirects=False,
            )
        assert response.status_code == 303
        assert response.headers["location"] == "/portal/chat"

    def test_an_unsupported_language_is_folded_not_stored(self):
        client, user_id = self._signed_in("fr")
        assert client.post("/portal/api/language", json={"lang": "klingon"}).json()["lang"] == "en"
        assert store.get_user(user_id)["lang"] == "en"

    def test_a_missing_column_reports_unavailable_rather_than_pretending(self):
        """store/db.py logs a failed ALTER at debug and continues, so a
        deployment can boot with the column missing. Without this the switcher
        would return 200 and change nothing."""
        import portal.routes as pr

        def _boom(user_id, lang):
            raise Exception("no such column: lang")

        client, _ = self._signed_in()
        original = pr.store.set_user_language
        pr.store.set_user_language = _boom
        try:
            response = client.post("/portal/api/language", json={"lang": "fr"})
        finally:
            pr.store.set_user_language = original
        assert response.status_code == 503

    def _login(self, *, temp_password: bool):
        client, _ = _probe_client()
        account_id = _account()
        email = f"{os.urandom(4).hex()}@x.com"
        user_id, password = store.create_user(
            account_id, "Ada", email,
            password=None if temp_password else "correct-horse-battery",
        )
        store.set_user_language(user_id, "fr")
        response = client.post(
            "/portal/login",
            data={"account_id": account_id, "email": email, "password": password},
            follow_redirects=False,
        )
        return client, response

    def test_logging_in_refreshes_the_cookie_from_the_row(self):
        """A preference set on one device follows the user to the next."""
        client, response = self._login(temp_password=False)
        assert response.status_code == 303, response.text[:400]
        assert client.cookies.get("qb_lang") == "fr"

    def test_the_forced_password_change_branch_sets_it_too(self):
        """There are two redirects out of login and only one of them is the
        common path. A user created with a temp password meets the other one on
        their very first login -- the first page they ever see."""
        client, response = self._login(temp_password=True)
        assert response.status_code == 303
        assert "change-password" in response.headers["location"]
        assert client.cookies.get("qb_lang") == "fr"
