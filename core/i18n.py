"""
core/i18n.py
────────────
One message catalogue, delivered three ways: to Python prose producers, to Jinja
markup, and to the browser's inline scripts.

Why a catalogue and not gettext
───────────────────────────────
The toolchain is the only real argument for gettext, and it does not apply here.
``xgettext`` cannot extract an f-string, and 426 of this product's ~1,300
user-facing strings are f-strings (263 of them interpolating mid-sentence), so a
gettext project would begin with exactly the same manual rewrite of every
producer that this needs -- and then add a .po/.mo compile step to a deploy that
ships plain files. ``babel`` is also not a dependency here.

Its locale model is wrong for this server besides. ``gettext.install()`` and
``locale.setlocale`` are process- or thread-global, and this is an async FastAPI
app with two users in different languages in flight on one event loop. That
class of problem is already solved in this codebase with a ContextVar --
core/vocab_packs.py, core/llm_audit.py, core/pipeline_trace.py -- so this module
copies that shape rather than inventing a fourth one.

Why post-hoc translation of the payload cannot work
───────────────────────────────────────────────────
``sanitize_response_text_fields`` in core/response_builder.py already walks the
whole outgoing payload keyed by field name and looks like the perfect single
hook. By the time the payload reaches it the field holds

    "Acme SA leads at 1,250,000."

-- copy, tenant entity name and a locale-formatted number fused into one opaque
string. There is no way to separate the translatable third, and an LLM pass over
it would rewrite tenant data values. Message ids have to be introduced at the
producer.

Rules this module enforces
──────────────────────────
1. **Whole sentences, never fragments.** A caller that glues clauses together
   and then sentence-cases the result cannot be translated: French word order
   and agreement do not survive it. Register one id per whole sentence, with a
   separate id for each shape.
2. **Named placeholders only.** ``{count}``, never ``{0}`` or ``{}``. Word order
   changes between languages; positional placeholders silently swap arguments.
3. **English is the fallback for everything.** An unknown id, an unknown
   language, or a missing keyword all degrade to something readable rather than
   raising. A missing translation is a cosmetic bug; an exception in an answer
   path is an outage.
4. **Opt-in per call site.** Nothing is scanned or auto-extracted, so SQL, log
   lines, audit records, column names and tenant data values are excluded by
   construction -- they are only translated if someone rewrites a producer to
   call ``t()``.
5. **Sentinels are refused.** Some strings look like copy and are actually
   protocol: ``"redacted segment"`` is produced by core/insight.py and compared
   BY EQUALITY in two places, so translating it makes the comparison fail and
   the real label reaches the user. ``FORBIDDEN_VALUES`` below is checked by a
   test rather than trusted to reviewer memory.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

DEFAULT_LANGUAGE = "en"

# Kept in step with store.user_store.SUPPORTED_LANGUAGES; asserted by a test
# rather than imported, because store imports are heavier than this module wants
# and a drift between the two is exactly what the test is for.
SUPPORTED_LANGUAGES = ("en", "fr")

# Strings that must never appear as a catalogue value in ANY language, because
# something compares them by equality:
#   "redacted segment"  core/insight.py::_display_label, compared at
#                       core/response_builder.py::_safe_category_label and
#                       core/insight.py; a translated sentinel fails the
#                       comparison and the unredacted label is rendered.
#   "[REDACTED]"        core/compliance/result_guard.py::_mask output, matched
#                       by the _MASKED_MARKERS sets in core/result_planner.py,
#                       core/result_commands.py and core/forecast_gate.py.
#   "CANNOT_GENERATE"   the model's refusal token, compared in core/validator.py.
FORBIDDEN_VALUES = ("redacted segment", "[REDACTED]", "CANNOT_GENERATE")

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


# ══════════════════════════════════════════════════════════════════════════════
# The catalogue
# ══════════════════════════════════════════════════════════════════════════════
#
# msg_id -> {language: template}. Ids are namespaced by surface so the browser
# can be sent one subtree rather than the whole catalogue.
#
#   ui.*        portal chrome (Jinja markup and inline scripts)
#   answer.*    the deterministic answer surface
#   caveat.*    coverage caveats
#
MESSAGES: dict[str, dict[str, str]] = {
    # ── The add-to-dashboard picker ──────────────────────────────────────────
    "ui.pin.title": {
        "en": "Add to dashboard",
        "fr": "Ajouter au tableau de bord",
    },
    "ui.pin.subtitle": {
        "en": "Choose where this live, governed result should appear.",
        "fr": "Choisissez où ce résultat gouverné et actualisé doit apparaître.",
    },
    "ui.pin.mode_existing": {
        "en": "Existing dashboard",
        "fr": "Tableau de bord existant",
    },
    "ui.pin.mode_new": {"en": "Create new", "fr": "Créer un tableau de bord"},
    "ui.pin.search_placeholder": {
        "en": "Search your dashboards",
        "fr": "Rechercher dans vos tableaux de bord",
    },
    "ui.pin.loading": {
        "en": "Loading dashboards…",
        "fr": "Chargement des tableaux de bord…",
    },
    "ui.pin.none_yet": {
        "en": "You have no dashboards yet. Create one to pin this result.",
        "fr": "Vous n'avez pas encore de tableau de bord. Créez-en un pour épingler ce résultat.",
    },
    "ui.pin.no_match": {
        "en": "No dashboard matches “{query}”.",
        "fr": "Aucun tableau de bord ne correspond à « {query} ».",
    },
    "ui.pin.name_label": {"en": "Dashboard name", "fr": "Nom du tableau de bord"},
    "ui.pin.name_placeholder": {
        "en": "e.g. Pharmacy performance",
        "fr": "ex. Performance pharmacie",
    },
    "ui.pin.description_label": {"en": "Description", "fr": "Description"},
    "ui.pin.optional": {"en": "(optional)", "fr": "(facultatif)"},
    "ui.pin.description_placeholder": {
        "en": "What this dashboard is used for",
        "fr": "À quoi sert ce tableau de bord",
    },
    "ui.pin.visibility_label": {"en": "Visibility", "fr": "Visibilité"},
    "ui.pin.visibility_personal": {"en": "Personal", "fr": "Personnel"},
    "ui.pin.visibility_team": {"en": "Team draft", "fr": "Brouillon d'équipe"},
    "ui.pin.adding_what": {
        "en": "Adding: {title}",
        "fr": "À ajouter : {title}",
    },
    "ui.pin.submit": {"en": "Add chart", "fr": "Ajouter le graphique"},
    "ui.pin.submit_new": {
        "en": "Create and add",
        "fr": "Créer et ajouter",
    },
    "ui.pin.submitting": {"en": "Adding…", "fr": "Ajout en cours…"},
    "ui.pin.cancel": {"en": "Cancel", "fr": "Annuler"},
    "ui.pin.close": {"en": "Close", "fr": "Fermer"},
    "ui.pin.added": {
        "en": "Added to {dashboard}.",
        "fr": "Ajouté à {dashboard}.",
    },
    "ui.pin.err_name_required": {
        "en": "Give the new dashboard a name.",
        "fr": "Donnez un nom au nouveau tableau de bord.",
    },
    "ui.pin.err_pick_one": {
        "en": "Choose a dashboard, or create a new one.",
        "fr": "Choisissez un tableau de bord ou créez-en un.",
    },
    "ui.pin.err_expired": {
        "en": "This result can no longer be pinned. Run the question again and retry.",
        "fr": "Ce résultat ne peut plus être épinglé. Relancez la question puis réessayez.",
    },
    "ui.pin.err_forbidden": {
        "en": "You do not have access to that dashboard.",
        "fr": "Vous n'avez pas accès à ce tableau de bord.",
    },
    "ui.pin.err_generic": {
        "en": "The chart could not be added, and nothing was changed. "
              "Run the question again to retry.",
        "fr": "Le graphique n'a pas pu être ajouté et rien n'a été modifié. "
              "Relancez la question pour réessayer.",
    },
    "ui.pin.err_offline": {
        "en": "Could not reach the server. Nothing was changed — check your "
              "connection and try again.",
        "fr": "Serveur injoignable. Rien n'a été modifié — vérifiez votre "
              "connexion et réessayez.",
    },
    "ui.pin.err_load": {
        "en": "Your dashboards could not be loaded. You can still create a new one.",
        "fr": "Vos tableaux de bord n'ont pas pu être chargés. Vous pouvez tout de "
              "même en créer un.",
    },
    "ui.pin.err_session": {
        "en": "Your session ended. Sign in again to add this result.",
        "fr": "Votre session a expiré. Reconnectez-vous pour ajouter ce résultat.",
    },
    "ui.pin.already_added": {
        "en": "Already added to {dashboard}. Run the question again to add it a second time.",
        "fr": "Déjà ajouté à {dashboard}. Relancez la question pour l'ajouter une seconde fois.",
    },
    # ── The language switcher ────────────────────────────────────────────────
    "ui.lang.label": {"en": "Language", "fr": "Langue"},
    "ui.lang.en": {"en": "English", "fr": "Anglais"},
    "ui.lang.fr": {"en": "French", "fr": "Français"},
}


# ══════════════════════════════════════════════════════════════════════════════
# Lookup
# ══════════════════════════════════════════════════════════════════════════════

_ACTIVE: ContextVar[str] = ContextVar("querybot_active_language", default=DEFAULT_LANGUAGE)


def normalise_language(value) -> str:
    """A supported language tag, or "en". Accepts "fr-FR" and "FR"."""
    tag = str(value or "").strip().lower().replace("_", "-").split("-")[0]
    return tag if tag in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def get_active_language() -> str:
    return _ACTIVE.get()


def activate_language(lang: str):
    """Set the language for this context; returns a token for deactivate.

    Does NOT survive ``loop.run_in_executor``. core/vocab_packs.py's docstring
    documents that, and core/query_pipeline.py records a real incident where a
    consumer read the default and degraded silently. Anything reached through an
    executor takes the language explicitly instead.
    """
    return _ACTIVE.set(normalise_language(lang))


def deactivate_language(token) -> None:
    try:
        _ACTIVE.reset(token)
    except Exception:
        pass


def lookup(msg_id: str, lang: str | None = None) -> str:
    """The raw template for an id, un-formatted.

    Falls back English, then to the id itself. Returning the id rather than ""
    is deliberate: a missing string shows up as ``ui.pin.title`` on the page,
    which is obvious in review, where an empty string looks like a layout bug.
    """
    entry = MESSAGES.get(msg_id)
    if not entry:
        return msg_id
    tag = normalise_language(lang if lang is not None else get_active_language())
    return entry.get(tag) or entry.get(DEFAULT_LANGUAGE) or msg_id


def t(msg_id: str, /, lang: str | None = None, **kw) -> str:
    """Translate and interpolate.

    ``lang`` is keyword-only in practice and normally omitted -- the ContextVar
    carries it. Pass it explicitly on any path that crosses an executor.

    A missing keyword leaves its placeholder visibly in the output rather than
    raising. This runs inside answer construction, and a KeyError there costs
    the user their whole answer to save a word.
    """
    template = lookup(msg_id, lang)
    if not kw:
        return template
    try:
        return template.format(**kw)
    except (KeyError, IndexError, ValueError):
        out = template
        for name, value in kw.items():
            out = out.replace("{" + name + "}", str(value))
        return out


def translator_for(lang: str):
    """A ``t`` bound to one language, for Jinja and for any executor-side caller.

    Templates render outside the pipeline's activation, so they must not depend
    on the ContextVar.
    """
    tag = normalise_language(lang)

    def _bound(msg_id: str, **kw) -> str:
        return t(msg_id, lang=tag, **kw)

    return _bound


def catalogue_for(lang: str, prefix: str = "") -> dict[str, str]:
    """A flat ``{msg_id: template}`` for one language, for JSON injection.

    The browser gets this as ``const I18N = {{ i18n_catalogue|tojson }}`` and
    does its own ``{name}`` interpolation, so the inline scripts and the markup
    read from one source instead of drifting -- which is what happened to
    ``_fmtNum``, and to the stage labels, already.
    """
    tag = normalise_language(lang)
    return {
        msg_id: (entry.get(tag) or entry.get(DEFAULT_LANGUAGE) or msg_id)
        for msg_id, entry in MESSAGES.items()
        if not prefix or msg_id.startswith(prefix)
    }


def placeholders(msg_id: str) -> set[str]:
    """The named placeholders in an id's English template."""
    return set(_PLACEHOLDER_RE.findall(lookup(msg_id, DEFAULT_LANGUAGE)))
