"""
tests/test_portal_shell_language.py

The portal shell -- sidebar, mobile bar, confirm dialog, live toast -- in the
reader's language.

Every test here renders the real template through the portal's own Jinja
environment, or EXECUTES the shell's own JavaScript in duktape. Nothing asserts
on template source text: the catalogue is injected into the page as JSON, so
every English string is present in the source whatever the page displays, and a
source assertion would pass against a page that renders nothing at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import i18n
from tests.js_lift import function as _function
from tests.portal_render import render, visible

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "portal" / "templates" / "portal_base.html"

USER = {"id": 1, "name": "Ada Lovelace", "account_id": "a",
        "role": "analyst", "group_name": "Analysts"}


def _shell(lang=None, user=USER, path="/portal/notifications") -> str:
    """The shell around the lightest page that extends it."""
    return render("portal_notifications.html", lang=lang, path=path,
                  user=user, alerts=[], reports=[], subscriptions={},
                  saved=None, error=None)


# ══════════════════════════════════════════════════════════════════════════════
# The document
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDocumentDeclaresItsLanguage:

    def test_the_html_lang_attribute_follows_the_reader(self):
        """It was hardcoded to "en". A screen reader announces French text with
        English phonemes when this is wrong, and it is wrong silently."""
        assert '<html lang="fr">' in _shell(lang="fr")
        assert '<html lang="en">' in _shell(lang="en")

    def test_an_unknown_language_still_produces_a_valid_attribute(self):
        assert '<html lang="en">' in _shell(lang="de")


# ══════════════════════════════════════════════════════════════════════════════
# The sidebar
# ══════════════════════════════════════════════════════════════════════════════

class TestTheSidebarIsTranslated:

    def test_the_navigation_reads_in_french(self):
        markup = visible(_shell(lang="fr"))
        for expected in ("Nouvelle conversation", "Tableau de bord",
                         "Couche sémantique", "Notifications",
                         "Paramètres", "Déconnexion"):
            assert expected in markup, expected

    def test_the_english_labels_are_gone_when_the_reader_is_french(self):
        """The catalogue is injected as JSON, so this only means anything
        against markup with the <script> blocks removed."""
        markup = visible(_shell(lang="fr"))
        for absent in (">New Thread<", ">Semantic Layer<", ">Settings<",
                       ">Logout<", "Collapse sidebar", "Open navigation",
                       "Close navigation", "Account actions",
                       "Portal navigation"):
            assert absent not in markup, absent

    def test_the_accessible_names_are_translated_too(self):
        """A sighted French user would not notice these; the one person who
        depends on them is the one who cannot see the icon."""
        markup = _shell(lang="fr")
        assert 'aria-label="Navigation du portail"' in markup
        assert 'aria-label="Ouvrir la navigation"' in markup
        assert 'aria-label="Fermer la navigation"' in markup
        assert 'aria-label="Actions du compte"' in markup
        assert 'aria-label="Réduire le menu latéral"' in markup

    def test_the_role_is_a_label_not_a_raw_column_value(self):
        """It rendered `{{ user.role }}` -- the lowercase database value."""
        assert "Analyste" in visible(_shell(lang="fr"))
        assert "Analyst" in visible(_shell(lang="en"))
        assert "&middot; analyst" not in visible(_shell(lang="en"))

    def test_a_role_the_catalogue_does_not_know_still_renders_readably(self):
        """New roles get added to the database before they get added here."""
        markup = visible(_shell(lang="fr", user={**USER, "role": "auditor"}))
        assert "Auditor" in markup
        assert "ui.enum.role" not in markup

    def test_the_group_fallback_is_translated(self):
        markup = visible(_shell(lang="fr", user={**USER, "group_name": ""}))
        assert "Aucun groupe" in markup
        assert "No group" not in markup

    def test_a_real_group_name_is_never_translated(self):
        """Customer data is not a message id."""
        markup = visible(_shell(lang="fr", user={**USER, "group_name": "Pharmacy"}))
        assert "Pharmacy" in markup


# ══════════════════════════════════════════════════════════════════════════════
# One catalogue for the whole document
# ══════════════════════════════════════════════════════════════════════════════

class TestTheCatalogueIsInjectedOnce:

    def _dashboard(self, lang=None):
        from tests.dashboard_render import render as render_dashboard
        return render_dashboard(lang=lang)

    def test_a_page_that_extends_the_shell_ships_one_copy(self):
        """Three copies of 172 strings in one document is the drift the page
        comments warn about, in payload form."""
        page = self._dashboard(lang="fr")
        assert page.count('"ui.shell.toast_field"') == 1

    def test_the_page_alias_resolves_to_the_injected_catalogue(self):
        """`const I18N = window.QB_I18N || {}` is only an improvement if the
        window property is actually there and actually populated."""
        page = self._dashboard(lang="fr")
        catalogue = _extract_catalogue(page)
        assert catalogue["ui.shell.logout"] == "Déconnexion"
        assert catalogue["ui.dash.kicker"] == i18n.t("ui.dash.kicker", lang="fr")

    def test_the_two_declarations_do_not_collide(self):
        """A second top-level `const I18N` is a SyntaxError that takes the
        page's WHOLE script with it -- the sidebar would still work and the
        dashboard would silently do nothing. Executed, not read."""
        dukpy = pytest.importorskip("dukpy")
        page = self._dashboard(lang="fr")
        # `var window` because duktape has no browser globals; the point of
        # the test is the DECLARATIONS colliding, not what they read.
        program = ("var window = {QB_I18N: {}, QB_CHART_THEME: {}};\n"
                   + "\n".join(_top_level_lexical_declarations(page))
                   + "\ntypeof I18N;")
        assert dukpy.evaljs(program) == "object"


def _extract_catalogue(page: str) -> dict:
    marker = "window.QB_I18N = "
    start = page.index(marker) + len(marker)
    end = page.index(";\n", start)
    return json.loads(page[start:end])


def _top_level_lexical_declarations(page: str) -> list[str]:
    """Every `const NAME = ...;` written at the top level of a <script>.

    Browsers share one global lexical environment across classic scripts, so
    these have to be unique across the whole document, not per file.
    """
    out = []
    for line in page.splitlines():
        if line.startswith("const ") or line.startswith("window.QB_I18N = "):
            out.append(line if line.rstrip().endswith(";") else line + ";")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The shell's own JavaScript
# ══════════════════════════════════════════════════════════════════════════════

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the shell's own "
           "functions; asserting on their source text instead is the failure "
           "mode this file exists to avoid",
)

SOURCE = BASE.read_text(encoding="utf-8")


def _run(script: str, *, lang="en", collapsed=False) -> dict:
    """Execute the shell's real functions against a DOM thin enough for them.

    The catalogue is the REAL one from core/i18n.py, so a message id the shell
    uses and the catalogue does not have shows up here rather than as a raw id
    on a customer's screen.
    """
    lifted = "\n".join((
        _function(SOURCE, "window.qbT = function (id, vars)"),
        _function(SOURCE, "function syncToggleLabel()"),
        _function(SOURCE, "function handleSemanticMessage(msg,fromPoll)"),
        _function(SOURCE, "function handleSystemMessage(msg)"),
        _function(SOURCE, "window.qbConfirm = function(opts)"),
    ))
    harness = f"""
const _log = {{toasts: [], events: []}};
function _el(id) {{
  return {{
    _id: id, textContent: '', className: '', _attrs: {{}},
    style: {{}},
    classList: {{_c: {{}},
      add(n) {{ this._c[n] = true; }},
      remove(n) {{ delete this._c[n]; }},
      toggle(n, on) {{ if (on) this._c[n] = true; else delete this._c[n]; }},
      contains(n) {{ return !!this._c[n]; }}}},
    setAttribute(k, v) {{ this._attrs[k] = String(v); }},
    getAttribute(k) {{ return this._attrs[k] === undefined ? null : this._attrs[k]; }},
    focus() {{}},
    querySelector() {{ return _toggleButton; }},
  }};
}}
const _toggleButton = _el('toggle');
const _nodes = {{}};
for (const id of ['qbDialogBackdrop','qbDialogTitle','qbDialogBody',
                  'qbDialogCancel','qbDialogConfirm'])
  _nodes[id] = _el(id);
var document = {{getElementById: id => _nodes[id] || null}};
var window = {{
  QB_I18N: {json.dumps(i18n.catalogue_for(lang))},
  dispatchEvent(e) {{ _log.events.push(e); }},
}};
function CustomEvent(name, init) {{ return {{type: name, detail: (init||{{}}).detail}}; }}
function setTimeout(fn) {{ fn(); return 0; }}
var shell = {{classList: {{_c: {json.dumps({'portal-sidebar-collapsed': True} if collapsed else {})},
  contains(n) {{ return !!this._c[n]; }}}}}};
var sidebar = {{querySelector: () => _toggleButton}};
// Seen-tracking is sessionStorage bookkeeping, not language. Stubbed so the
// toast under test is the only thing this harness can fail on.
function markSeen() {{}}
function isSeen() {{ return false; }}
function showPortalToast(title, body, level) {{ _log.toasts.push({{title, body, level}}); }}
const backdrop = _nodes.qbDialogBackdrop;
const titleEl = _nodes.qbDialogTitle;
const bodyEl = _nodes.qbDialogBody;
const cancelBtn = _nodes.qbDialogCancel;
const confirmBtn = _nodes.qbDialogConfirm;
var _onConfirm = null;
function close() {{ backdrop.style.display = 'none'; _onConfirm = null; }}

{lifted}

{script}

JSON.stringify({{
  log: _log,
  toggle: _toggleButton._attrs,
  dialog: {{title: titleEl.textContent, body: bodyEl.textContent,
            cancel: cancelBtn.textContent, confirm: confirmBtn.textContent}},
}});
"""
    return json.loads(dukpy.evaljs(harness))


class TestTheBrowserSideTranslator:

    def test_it_resolves_an_id(self):
        out = dukpy.evaljs(
            _function(SOURCE, "window.qbT = function (id, vars)").replace("window.qbT =", "var qbT =")
            + f"\nvar window = {{QB_I18N: {json.dumps(i18n.catalogue_for('fr'))}}};"
            + "\nqbT('ui.shell.logout');"
        )
        assert out == "Déconnexion"

    def test_an_unknown_id_returns_the_id(self):
        """An empty string would read as a layout bug rather than a missing
        string, and nobody would find it."""
        out = dukpy.evaljs(
            _function(SOURCE, "window.qbT = function (id, vars)").replace("window.qbT =", "var qbT =")
            + "\nvar window = {QB_I18N: {}};\nqbT('ui.shell.nope');"
        )
        assert out == "ui.shell.nope"

    def test_it_interpolates_named_placeholders(self):
        out = _run("var _r = window.qbT('ui.shell.toast_approved_body', {field: 'Marge'});"
                   "_log.toasts.push({title: _r, body: '', level: ''});", lang="fr")
        assert out["log"]["toasts"][0]["title"] == "Marge a été approuvé par un administrateur."


class TestTheSidebarToggleLabel:

    def test_it_is_french_when_the_reader_is(self):
        out = _run("syncToggleLabel();", lang="fr", collapsed=False)
        assert out["toggle"]["aria-label"] == "Réduire le menu latéral"
        assert out["toggle"]["title"] == "Réduire le menu latéral"

    def test_the_collapsed_state_flips_the_verb_not_the_language(self):
        out = _run("syncToggleLabel();", lang="fr", collapsed=True)
        assert out["toggle"]["aria-label"] == "Développer le menu latéral"

    def test_english_still_works(self):
        out = _run("syncToggleLabel();", lang="en", collapsed=True)
        assert out["toggle"]["aria-label"] == "Expand sidebar"


class TestTheConfirmDialogDefaults:

    def test_the_defaults_are_translated(self):
        """These are set in JavaScript, so translating the markup alone leaves
        every caller that omits them in English."""
        out = _run("window.qbConfirm({});", lang="fr")
        assert out["dialog"] == {
            "title": "Confirmer l'action ?", "body": "",
            "cancel": "Annuler", "confirm": "Confirmer",
        }

    def test_a_caller_supplied_label_still_wins(self):
        out = _run("window.qbConfirm({title: 'Supprimer ?', confirm: 'Supprimer'});",
                   lang="fr")
        assert out["dialog"]["title"] == "Supprimer ?"
        assert out["dialog"]["confirm"] == "Supprimer"
        assert out["dialog"]["cancel"] == "Annuler"


class TestTheLiveToast:

    def test_the_approval_sentence_is_one_string_not_three(self):
        """It was `(column||'Field') + ' was ' + statusText + ' by admin.'`.
        French puts the participle after the auxiliary and agrees it with the
        subject, so there is no seam to concatenate at."""
        out = _run("handleSemanticMessage({type:'semantic_feedback_reviewed',"
                   "status:'approved',feedback_id:1,column_name:'Marge'}, false);",
                   lang="fr")
        toast = out["log"]["toasts"][0]
        assert toast["title"] == "Modification de la couche sémantique approuvée"
        assert toast["body"] == "Marge a été approuvé par un administrateur."
        assert toast["level"] == "success"

    def test_the_rejection_sentence_too(self):
        out = _run("handleSemanticMessage({type:'semantic_feedback_reviewed',"
                   "status:'rejected',feedback_id:1,column_name:'Marge'}, false);",
                   lang="fr")
        toast = out["log"]["toasts"][0]
        assert toast["title"] == "Modification de la couche sémantique refusée"
        assert toast["body"] == "Marge a été refusé par un administrateur."
        assert toast["level"] == "warning"

    def test_the_missing_column_fallback_is_translated(self):
        out = _run("handleSemanticMessage({type:'semantic_feedback_reviewed',"
                   "status:'approved',feedback_id:1}, false);", lang="fr")
        assert out["log"]["toasts"][0]["body"] == "Champ a été approuvé par un administrateur."

    def test_a_column_name_is_never_translated(self):
        """Customer schema is data. It goes through the placeholder, not the
        catalogue."""
        out = _run("handleSemanticMessage({type:'semantic_feedback_reviewed',"
                   "status:'approved',feedback_id:1,column_name:'ui.shell.logout'},"
                   "false);", lang="fr")
        assert out["log"]["toasts"][0]["body"].startswith("ui.shell.logout a été")

    def test_the_system_notice_falls_back_in_french(self):
        out = _run("handleSystemMessage({type:'system_message'});", lang="fr")
        assert out["log"]["toasts"][0]["title"] == "Message système"
        assert out["log"]["toasts"][0]["body"] == "Mise à jour du système"

    def test_a_server_supplied_message_is_shown_as_sent(self):
        out = _run("handleSystemMessage({type:'system_message',message:'Maintenance'});",
                   lang="fr")
        assert out["log"]["toasts"][0]["body"] == "Maintenance"

    def test_english_is_unchanged(self):
        out = _run("handleSemanticMessage({type:'semantic_feedback_reviewed',"
                   "status:'approved',feedback_id:1,column_name:'Margin'}, false);",
                   lang="en")
        toast = out["log"]["toasts"][0]
        assert toast["title"] == "Semantic Layer change approved"
        assert toast["body"] == "Margin was approved by an administrator."


# ══════════════════════════════════════════════════════════════════════════════
# The monthly allowance, built server-side and shown by the shell's toast
# ══════════════════════════════════════════════════════════════════════════════

class TestTheQueryLimitToast:

    def _status(self, used, limit, lang):
        import portal.routes as pr
        from unittest.mock import patch
        with patch.object(pr.store, "get_client", return_value={"query_limit_monthly": limit}), \
             patch.object(pr.store, "get_monthly_query_count", return_value=used):
            return pr._query_limit_status("acct", lang)

    def test_the_normal_state_is_french(self):
        status = self._status(used=3, limit=500, lang="fr")
        assert status["title"] == "Limite mensuelle de requêtes"
        assert status["message"] == "497 requêtes restantes ce mois-ci."

    def test_the_warning_state_is_french(self):
        status = self._status(used=450, limit=500, lang="fr")
        assert status["level"] == "warning"
        assert status["title"] == "Alerte de limite mensuelle de requêtes"
        assert "450/500 requêtes utilisées ce mois-ci." in status["message"]

    def test_the_blocked_state_is_french(self):
        status = self._status(used=500, limit=500, lang="fr")
        assert status["level"] == "blocked"
        assert status["title"] == "Limite mensuelle de requêtes atteinte"
        assert status["message"].startswith("500/500 requêtes utilisées ce mois-ci.")

    def test_french_takes_the_singular_at_zero(self):
        """English says "0 queries remaining"; French says "0 requête". The
        inline `{% if n != 1 %}s{% endif %}` this replaces got that wrong."""
        assert self._status(used=500, limit=500, lang="fr")["remaining"] == 0
        status = self._status(used=10, limit=500, lang="fr")
        status["remaining"] = 0
        assert i18n.plural("ui.shell.limit_remaining_body", 0, lang="fr") == \
            "0 requête restante ce mois-ci."
        assert i18n.plural("ui.shell.limit_remaining_body", 0, lang="en") == \
            "0 queries remaining this month."

    def test_english_is_unchanged(self):
        status = self._status(used=3, limit=500, lang="en")
        assert status["title"] == "Monthly query limit"
        assert status["message"] == "497 queries remaining this month."

    def test_the_default_is_english_so_no_caller_breaks(self):
        assert self._status(used=3, limit=500, lang=None)["title"] == "Monthly query limit"
